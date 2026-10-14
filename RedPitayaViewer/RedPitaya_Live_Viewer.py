"""Red Pitaya live signal viewer: time-domain trace on top, PSD below.

No ESN / estimation, just the signal. Streams IN1, IN2 or both through the
official Red Pitaya streaming python_lib (../python_lib by default), shows the
latest samples and a Welch PSD with configurable FFT points, overlap and window.
PSD settings apply live; stream settings apply on Start.

Run:  python RedPitaya_Live_Viewer.py
      python RedPitaya_Live_Viewer.py --self-test     (DSP checks, no GUI/hardware)
Requires numpy, scipy, PyQt5, pyqtgraph and a _python_lib3_XX.so matching your
Python version (3.9-3.13 are supplied).
"""
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
import argparse
import csv
import importlib
import math
import queue
import sys
import threading
import time

import numpy as np
from scipy import signal

HERE = Path(__file__).resolve().parent
BASE_RATE = 125e6                                    # STEMlab 125-14 ADC clock
DECIMATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
FFT_SIZES = tuple(2**k for k in range(6, 18))        # 64 .. 131072
WINDOWS = ("Hann", "Hamming", "Blackman", "Blackman-Harris", "Flat top", "Rectangular", "Kaiser")
CHANNEL_CHOICES = {"IN1": (1,), "IN2": (2,), "IN1 + IN2": (1, 2)}
COLORS = {1: "#56c7ff", 2: "#ffce76"}
MAX_RING_SAMPLES = 1 << 25                           # 32 M samples (128 MB) per channel at most


# ============================================================
# SETTINGS
# ============================================================
@dataclass
class StreamSettings:
    source: str = "Red Pitaya"           # "Red Pitaya", "Replay file", "Simulation"
    host: str = ""                       # empty = auto-discover
    channels: str = "IN1"
    decimation: int = 32
    volts_per_fs: float = 1.0            # 1.0 for LV jumper (+-1 V); ~20 for HV (+-20 V)
    extra_config: str = ""               # key=value lines sent to the streaming server
    replay_path: str = ""                # .npy (1-D or N x channels) or raw float32 .bin
    streaming_lib_dir: str = str(HERE.parent / "python_lib")
    block_size: int = 65536
    adc_size: int = 1638400
    buffer_seconds: float = 2.0          # ring buffer kept per channel

    @property
    def fs(self):
        return BASE_RATE / self.decimation

    @property
    def channel_list(self):
        return CHANNEL_CHOICES[self.channels]

    def validate(self):
        if self.source not in ("Red Pitaya", "Replay file", "Simulation"):
            raise ValueError("Choose Red Pitaya, Replay file or Simulation.")
        if self.channels not in CHANNEL_CHOICES:
            raise ValueError("Choose IN1, IN2 or IN1 + IN2.")
        if self.decimation not in DECIMATIONS:
            raise ValueError("Unsupported decimation.")
        if not (math.isfinite(self.volts_per_fs) and self.volts_per_fs > 0):
            raise ValueError("Volts per full scale must be positive.")
        for line in self.extra_config.splitlines():
            if line.strip() and "=" not in line:
                raise ValueError(f"Extra config line must be key=value: {line!r}")
        if self.source == "Replay file" and not Path(self.replay_path).is_file():
            raise ValueError("Choose an existing replay file (.npy or float32 .bin).")
        return self


@dataclass
class PSDSettings:
    nfft: int = 4096
    overlap_percent: float = 50.0
    window: str = "Hann"
    kaiser_beta: float = 8.6
    psd_seconds: float = 0.05            # newest data used for each Welch estimate
    averaging: str = "Exponential"       # "None", "Exponential", "Max hold"
    averages: int = 8                    # exponential time constant in updates
    scaling: str = "density"             # "density" (per Hz) or "spectrum" (power per bin)
    remove_dc: bool = True


def make_window(settings, n):
    name = settings.window
    if name == "Kaiser":
        return signal.get_window(("kaiser", float(settings.kaiser_beta)), n, fftbins=True)
    key = {"Hann": "hann", "Hamming": "hamming", "Blackman": "blackman", "Blackman-Harris": "blackmanharris",
           "Flat top": "flattop", "Rectangular": "boxcar"}[name]
    return signal.get_window(key, n, fftbins=True)


def enbw_bins(window):
    """Equivalent noise bandwidth of a window, in FFT bins."""
    return window.size * np.sum(window**2) / np.sum(window)**2


def compute_psd(x, fs, settings):
    """One-sided Welch estimate. Returns (freqs, psd, rbw_hz, segments)."""
    nfft = int(settings.nfft)
    if x.size < nfft:
        raise ValueError(f"Need at least {nfft} samples for the PSD; have {x.size}.")
    window = make_window(settings, nfft)
    noverlap = min(nfft - 1, int(round(nfft * settings.overlap_percent / 100.0)))
    f, p = signal.welch(x, fs=fs, window=window, nperseg=nfft, noverlap=noverlap,
                        detrend="constant" if settings.remove_dc else False,
                        scaling=settings.scaling, return_onesided=True)
    step = nfft - noverlap
    segments = 1 + (x.size - nfft) // step
    return f, p, fs / nfft * enbw_bins(window), segments


def minmax_decimate(t, y, max_points=4000):
    """Envelope-preserving reduction for plotting: keep min and max per bin."""
    if y.size <= max_points:
        return t, y
    bins = max_points // 2
    n = (y.size // bins) * bins
    yb = y[:n].reshape(bins, -1)
    tb = t[:n].reshape(bins, -1)
    lo, hi = yb.argmin(axis=1), yb.argmax(axis=1)
    first = np.minimum(lo, hi)
    second = np.maximum(lo, hi)
    rows = np.arange(bins)
    tt = np.column_stack([tb[rows, first], tb[rows, second]]).ravel()
    yy = np.column_stack([yb[rows, first], yb[rows, second]]).ravel()
    return tt, yy


class Ring:
    """Fixed-size float32 ring buffer; latest(n) returns the newest n samples in order."""
    def __init__(self, capacity):
        self.data = np.zeros(int(capacity), dtype=np.float32)
        self.pos = 0
        self.filled = 0
        self.lock = threading.Lock()

    def push(self, x):
        x = np.asarray(x, dtype=np.float32)
        with self.lock:
            cap = self.data.size
            if x.size >= cap:
                self.data[:] = x[-cap:]
                self.pos, self.filled = 0, cap
                return
            end = self.pos + x.size
            if end <= cap:
                self.data[self.pos:end] = x
            else:
                k = cap - self.pos
                self.data[self.pos:] = x[:k]
                self.data[:end - cap] = x[k:]
            self.pos = end % cap
            self.filled = min(cap, self.filled + x.size)

    def latest(self, n):
        with self.lock:
            n = min(int(n), self.filled)
            start = (self.pos - n) % self.data.size
            if start + n <= self.data.size:
                return self.data[start:start + n].copy()
            return np.concatenate((self.data[start:], self.data[:self.pos]))

    def clear(self):
        with self.lock:
            self.pos = self.filled = 0


# ============================================================
# SOURCES: read(stop) -> ({channel: samples in FS}, gap_flag, report)
# ============================================================
class RedPitayaSource:
    def __init__(self, settings):
        self.settings = settings
        self.adc = None
        self.packets = queue.Queue(maxsize=512)
        self.lock = threading.Lock()
        self.host_overflow = False
        self.events = deque(maxlen=32)
        self.checked_rate = False
        lib_dir = str(Path(settings.streaming_lib_dir).resolve())
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)
        try:
            streaming = importlib.import_module("streaming")
            streaming.ADCStreamClient
        except (ImportError, OSError, AttributeError) as error:
            raise RuntimeError(f"Cannot import the Red Pitaya streaming python_lib from {lib_dir}. It needs "
                               f"_python_lib{sys.version_info.major}_{sys.version_info.minor}.so; the repository "
                               "supplies Python 3.9-3.13.") from error
        source = self
        channels = settings.channel_list

        class ADCHandler(streaming.ADCCallback):
            def receivePack(self, client, pack):
                item = {}
                for ch in channels:
                    c = pack.channel1 if ch == 1 else pack.channel2
                    raw = c.raw
                    if raw:
                        item[ch] = (np.array(raw, dtype=np.int16), int(c.fpgaLost), int(c.baseRate),
                                    int(c.bitsPerSample or c.adcBaseBits or 16))
                if not item:
                    return
                try:
                    source.packets.put_nowait(item)
                except queue.Full:
                    with source.lock:
                        source.host_overflow = True

            def connected(self, client, host):
                source.events.append(f"Stream connected to {host}")

            def disconnected(self, client, host):
                source.events.append(f"Stream disconnected from {host}")

            def error(self, client, host, code):
                source.events.append(f"Stream error {code} on {host}")

        class ConfigHandler(streaming.ConfigCallback):
            def configError(self, client, host, code):
                source.events.append(f"Config error {code} on {host}")

        try:
            self.conf = streaming.ConfigStreamClient()
            self.conf_handler = ConfigHandler()
            self.conf.addCallback(self.conf_handler)
            self.adc = streaming.ADCStreamClient(self.conf)
            self.adc_handler = ADCHandler()
            self.adc.setCallback(self.adc_handler)
            host = settings.host.strip()
            if not (self.conf.connect([host]) if host else self.conf.connect()):
                raise RuntimeError("Red Pitaya streaming server not found" + (f" at {host}" if host else
                                   " on the LAN") + ". Start the streaming app on the board.")
            config = [("adc_pass_mode", "NET"), ("adc_decimation", str(settings.decimation)),
                      ("block_size", str(settings.block_size)), ("adc_size", str(settings.adc_size))]
            config += [(f"channel_state_{ch}", "ON" if ch in channels else "OFF") for ch in (1, 2)]
            config += [tuple(p.strip() for p in line.split("=", 1))
                       for line in settings.extra_config.splitlines() if line.strip()]
            for key, value in config:
                if not self.conf.sendConfig(key, value):
                    raise RuntimeError(f"Red Pitaya rejected config {key}={value}.")
            if not self.adc.startStreaming():
                raise RuntimeError("Red Pitaya ADC streaming did not start.")
            self.waiter = threading.Thread(target=self.adc.wait, name="rp-wait", daemon=True)
            self.waiter.start()
            self.description = (f"Red Pitaya {', '.join(self.conf.getHosts()) or host} | {settings.channels} | "
                                f"decimation {settings.decimation} = {settings.fs/1e6:.6g} MSPS")
        except Exception:
            self.close()
            raise

    def read(self, stop):
        while not stop.is_set():
            try:
                item = self.packets.get(timeout=0.2)
                break
            except queue.Empty:
                continue
        else:
            return {}, False, {}
        out, lost = {}, 0
        for ch, (raw, ch_lost, base_rate, bits) in item.items():
            if not self.checked_rate and base_rate:
                # baseRate is the decimated rate on some firmware and the ADC clock on others.
                if not (math.isclose(base_rate, self.settings.fs, rel_tol=1e-6) or
                        math.isclose(base_rate / self.settings.decimation, self.settings.fs, rel_tol=1e-6)):
                    raise RuntimeError(f"Board reports {base_rate/1e6:g} MHz; expected {self.settings.fs/1e6:g} MSPS "
                                       f"at decimation {self.settings.decimation}.")
                self.checked_rate = True
            out[ch] = raw.astype(np.float32) / float(2 ** (bits - 1))
            lost = max(lost, ch_lost)
        with self.lock:
            overflow, self.host_overflow = self.host_overflow, False
        return out, bool(lost or overflow), dict(lost=lost, overflow=overflow)

    def close(self):
        if self.adc is not None:
            try:
                self.adc.notifyStop()
            except Exception:
                pass
            waiter = getattr(self, "waiter", None)
            if waiter is not None:
                waiter.join(timeout=3)
            self.adc = None


class ReplaySource:
    """Replays a .npy (1-D, or samples x channels) or raw float32 .bin file in real time."""
    def __init__(self, settings, chunk=16384):
        path = Path(settings.replay_path)
        data = np.load(path, mmap_mode="r") if path.suffix == ".npy" else np.memmap(path, dtype="<f4", mode="r")
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        if data.ndim != 2 or data.shape[0] < chunk:
            raise ValueError("Replay data must be 1-D or (samples x channels) with at least 16384 samples.")
        self.data, self.chunk, self.settings = data, chunk, settings
        self.pos, self.next_read = 0, time.monotonic()
        self.events = deque()
        self.description = f"Replay {path.name} | {data.shape[0]:,} samples x {data.shape[1]} ch | {settings.fs/1e6:.6g} MSPS"

    def read(self, stop):
        delay = self.next_read - time.monotonic()
        if delay > 0 and stop.wait(delay):
            return {}, False, {}
        idx = (self.pos + np.arange(self.chunk)) % self.data.shape[0]
        self.pos = (self.pos + self.chunk) % self.data.shape[0]
        self.next_read = max(self.next_read + self.chunk / self.settings.fs, time.monotonic())
        cols = {ch: min(i, self.data.shape[1] - 1) for i, ch in enumerate(self.settings.channel_list)}
        return {ch: self.data[idx, col] for ch, col in cols.items()}, False, {}

    def close(self):
        pass


class SimulationSource:
    """Test signal at the selected rate: a pulsed chirp in noise on IN1, its envelope on IN2."""
    def __init__(self, settings, chunk=16384):
        self.settings, self.chunk = settings, chunk
        self.n0, self.next_read = 0, time.monotonic()
        self.rng = np.random.default_rng(0)
        self.events = deque()
        self.description = f"Simulation | pulsed chirp + noise | {settings.fs/1e6:.6g} MSPS"

    def read(self, stop):
        delay = self.next_read - time.monotonic()
        if delay > 0 and stop.wait(delay):
            return {}, False, {}
        fs = self.settings.fs
        t = (self.n0 + np.arange(self.chunk)) / fs
        self.n0 += self.chunk
        self.next_read = max(self.next_read + self.chunk / fs, time.monotonic())
        pri, pw = 512e-6, 325e-6
        tp = t % pri
        on = tp < pw
        f0, f1 = 0.05 * fs, 0.30 * fs                     # chirp across 5-30 % of the sample rate
        phase = 2 * np.pi * (f0 * tp + 0.5 * (f1 - f0) / pw * tp**2)
        in1 = np.where(on, 0.3 * np.cos(phase), 0.0) + 0.02 * self.rng.standard_normal(t.size)
        in2 = np.where(on, 0.5, 0.05) + 0.01 * self.rng.standard_normal(t.size)
        data = {1: in1, 2: in2}
        return {ch: data[ch].astype(np.float32) for ch in self.settings.channel_list}, False, {}

    def close(self):
        pass


# ============================================================
# ACQUISITION THREAD
# ============================================================
class Acquisition:
    def __init__(self, settings):
        self.settings = settings.validate()
        cap = min(max(int(settings.fs * settings.buffer_seconds), 1 << 18), MAX_RING_SAMPLES)
        self.rings = {ch: Ring(cap) for ch in settings.channel_list}
        self.data_lock = threading.Lock()          # all channels are written/read together
        self.stop_event = threading.Event()
        self.messages = queue.Queue(maxsize=128)
        self.lock = threading.Lock()
        self.stats = dict(state="Starting", received=0, gaps=0, lost=0, overflows=0, error="")
        self.started_at = None
        self.thread = threading.Thread(target=self._run, name="viewer-capture", daemon=True)

    @property
    def alive(self):
        return self.thread.is_alive()

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def note(self, text):
        try:
            self.messages.put_nowait(text)
        except queue.Full:
            pass

    def latest(self, n):
        """Newest n samples of every channel, from the same instant (IN1/IN2 stay aligned)."""
        with self.data_lock:
            return {ch: ring.latest(n) for ch, ring in self.rings.items()}

    def snapshot(self):
        with self.lock:
            stats = dict(self.stats)
        elapsed = max(time.monotonic() - self.started_at, 1e-9) if self.started_at else 1.0
        stats["rate_msps"] = stats["received"] / len(self.rings) / elapsed / 1e6
        return stats

    def _run(self):
        source = None
        try:
            factory = {"Red Pitaya": RedPitayaSource, "Replay file": ReplaySource,
                       "Simulation": SimulationSource}[self.settings.source]
            source = factory(self.settings)
            self.note(source.description)
            self.started_at = time.monotonic()
            with self.lock:
                self.stats["state"] = "Receiving"
            while not self.stop_event.is_set():
                data, gap, report = source.read(self.stop_event)
                while source.events:
                    self.note(source.events.popleft())
                with self.data_lock:
                    for ch, x in data.items():
                        self.rings[ch].push(x)
                with self.lock:
                    self.stats["received"] += sum(x.size for x in data.values())
                    if gap:
                        self.stats["gaps"] += 1
                        self.stats["lost"] += report.get("lost", 0)
                        self.stats["overflows"] += bool(report.get("overflow"))
        except Exception as error:
            with self.lock:
                self.stats["state"], self.stats["error"] = "Error", str(error)
            self.note("ERROR: " + str(error))
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception as error:
                    self.note("Close error: " + str(error))
            with self.lock:
                if self.stats["state"] != "Error":
                    self.stats["state"] = "Stopped"


# ============================================================
# GUI
# ============================================================
def run_viewer():
    try:
        from PyQt5 import QtCore, QtWidgets
        import pyqtgraph as pg
    except ImportError as error:
        raise SystemExit("Install GUI packages: python -m pip install PyQt5 pyqtgraph") from error
    pg.setConfigOptions(antialias=False, background="#101d30", foreground="#b8c7db")

    class Viewer(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.acq = None
            self.paused = False
            self.psd_state = {}
            self.last = {}
            self.setWindowTitle("Red Pitaya · Live Signal Viewer")
            self.resize(1400, 920)
            self.setStyleSheet("""
                QMainWindow, QWidget { background: #091221; color: #e1eaf5; font-size: 13px; }
                QGroupBox { border: 1px solid #253750; border-radius: 8px; margin-top: 14px; padding: 10px; }
                QGroupBox::title { subcontrol-origin: margin; left: 12px; color: #93aac7; }
                QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QPlainTextEdit {
                    background: #142238; border: 1px solid #304660; border-radius: 5px; padding: 5px; }
                QPushButton { background: #1b3554; border: 1px solid #335576; border-radius: 6px; padding: 8px 12px; }
                QPushButton:hover { background: #285078; }
                QPushButton:disabled { color: #637387; background: #101d30; }
            """)
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            outer = QtWidgets.QVBoxLayout(central)
            title = QtWidgets.QLabel("RED PITAYA  /  LIVE SIGNAL  /  TIME + PSD")
            title.setStyleSheet("font-size: 22px; font-weight: bold; color: #eaf3ff; padding: 6px;")
            outer.addWidget(title)
            split = QtWidgets.QHBoxLayout()
            outer.addLayout(split, 1)

            panel = QtWidgets.QWidget()
            controls = QtWidgets.QVBoxLayout(panel)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFixedWidth(360)
            scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            scroll.setWidget(panel)
            split.addWidget(scroll)

            # ---- stream (applies on Start)
            stream = QtWidgets.QGroupBox("Stream (applies on Start)")
            form = QtWidgets.QFormLayout(stream)
            defaults = StreamSettings()
            self.source = QtWidgets.QComboBox()
            self.source.addItems(["Red Pitaya", "Replay file", "Simulation"])
            self.host = QtWidgets.QLineEdit(defaults.host)
            self.host.setPlaceholderText("auto-discover")
            self.channels = QtWidgets.QComboBox()
            self.channels.addItems(list(CHANNEL_CHOICES))
            self.decimation = QtWidgets.QComboBox()
            for d in DECIMATIONS:
                self.decimation.addItem(f"{d}  ·  {BASE_RATE/d/1e6:.6g} MSPS", d)
            self.decimation.setCurrentIndex(DECIMATIONS.index(defaults.decimation))
            self.vpfs = QtWidgets.QDoubleSpinBox()
            self.vpfs.setRange(0.001, 1000)
            self.vpfs.setDecimals(3)
            self.vpfs.setValue(defaults.volts_per_fs)
            self.vpfs.setToolTip("Volts per ADC full scale: 1 for the LV jumper (±1 V), ~20 for HV (±20 V).")
            self.replay = QtWidgets.QLineEdit()
            self.replay.setPlaceholderText(".npy or float32 .bin")
            browse = QtWidgets.QPushButton("…")
            browse.setFixedWidth(34)
            browse.clicked.connect(self.browse_replay)
            replay_row = QtWidgets.QWidget()
            rl = QtWidgets.QHBoxLayout(replay_row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.addWidget(self.replay)
            rl.addWidget(browse)
            self.extra = QtWidgets.QPlainTextEdit()
            self.extra.setPlaceholderText("optional key=value per line")
            self.extra.setMaximumHeight(52)
            form.addRow("Source", self.source)
            form.addRow("Board IP", self.host)
            form.addRow("Inputs", self.channels)
            form.addRow("Decimation", self.decimation)
            form.addRow("Volts / FS", self.vpfs)
            form.addRow("Replay file", replay_row)
            form.addRow("Extra config", self.extra)
            controls.addWidget(stream)
            self.stream_box = stream
            buttons = QtWidgets.QHBoxLayout()
            self.start_button = QtWidgets.QPushButton("Start")
            self.start_button.setStyleSheet("background: #167c6b; font-weight: bold;")
            self.stop_button = QtWidgets.QPushButton("Stop")
            self.stop_button.setEnabled(False)
            self.pause_button = QtWidgets.QPushButton("Pause")
            self.pause_button.setCheckable(True)
            self.start_button.clicked.connect(self.start)
            self.stop_button.clicked.connect(self.stop)
            self.pause_button.toggled.connect(self.toggle_pause)
            for b in (self.start_button, self.stop_button, self.pause_button):
                buttons.addWidget(b)
            controls.addLayout(buttons)

            # ---- time display (live)
            display = QtWidgets.QGroupBox("Time domain (live)")
            dform = QtWidgets.QFormLayout(display)
            self.span = QtWidgets.QDoubleSpinBox()
            self.span.setRange(0.001, 2000)
            self.span.setDecimals(3)
            self.span.setValue(2.0)
            self.span.setSuffix(" ms")
            self.autoscale = QtWidgets.QCheckBox("Auto-scale Y")
            self.autoscale.setChecked(True)
            dform.addRow("Time span", self.span)
            dform.addRow(self.autoscale)
            controls.addWidget(display)

            # ---- PSD (live)
            psd = QtWidgets.QGroupBox("PSD (live, Welch)")
            pform = QtWidgets.QFormLayout(psd)
            p = PSDSettings()
            self.nfft = QtWidgets.QComboBox()
            for n in FFT_SIZES:
                self.nfft.addItem(f"{n:,}", n)
            self.nfft.setCurrentIndex(FFT_SIZES.index(p.nfft))
            self.overlap = QtWidgets.QDoubleSpinBox()
            self.overlap.setRange(0, 95)
            self.overlap.setDecimals(1)
            self.overlap.setValue(p.overlap_percent)
            self.overlap.setSuffix(" %")
            self.window = QtWidgets.QComboBox()
            self.window.addItems(WINDOWS)
            self.beta = QtWidgets.QDoubleSpinBox()
            self.beta.setRange(0, 40)
            self.beta.setValue(p.kaiser_beta)
            self.beta.setEnabled(False)
            self.window.currentTextChanged.connect(lambda name: self.beta.setEnabled(name == "Kaiser"))
            self.psd_len = QtWidgets.QDoubleSpinBox()
            self.psd_len.setRange(0.01, 2000)
            self.psd_len.setDecimals(2)
            self.psd_len.setValue(p.psd_seconds * 1e3)
            self.psd_len.setSuffix(" ms")
            self.psd_len.setToolTip("Newest data used per estimate; more data = more Welch segments averaged.")
            self.averaging = QtWidgets.QComboBox()
            self.averaging.addItems(["None", "Exponential", "Max hold"])
            self.averaging.setCurrentText(p.averaging)
            self.averages = QtWidgets.QSpinBox()
            self.averages.setRange(1, 1000)
            self.averages.setValue(p.averages)
            self.scaling = QtWidgets.QComboBox()
            self.scaling.addItems(["dB/Hz (density)", "dB (power per bin)"])
            self.remove_dc = QtWidgets.QCheckBox("Remove DC per segment")
            self.remove_dc.setChecked(p.remove_dc)
            self.log_freq = QtWidgets.QCheckBox("Log frequency axis")
            reset = QtWidgets.QPushButton("Reset averaging")
            reset.clicked.connect(lambda: self.psd_state.clear())
            for w in (self.nfft, self.window, self.averaging, self.scaling):
                w.currentIndexChanged.connect(lambda *_: self.psd_state.clear())
            for w in (self.overlap, self.beta, self.psd_len):
                w.valueChanged.connect(lambda *_: self.psd_state.clear())
            self.remove_dc.toggled.connect(lambda *_: self.psd_state.clear())
            self.log_freq.toggled.connect(lambda on: self.psd_plot.setLogMode(x=on, y=False))
            pform.addRow("FFT points", self.nfft)
            pform.addRow("Overlap", self.overlap)
            pform.addRow("Window", self.window)
            pform.addRow("Kaiser β", self.beta)
            pform.addRow("Data per estimate", self.psd_len)
            pform.addRow("Averaging", self.averaging)
            pform.addRow("Averages", self.averages)
            pform.addRow("Units", self.scaling)
            pform.addRow(self.remove_dc)
            pform.addRow(self.log_freq)
            pform.addRow(reset)
            controls.addWidget(psd)

            export = QtWidgets.QHBoxLayout()
            save_trace = QtWidgets.QPushButton("Save trace")
            save_psd = QtWidgets.QPushButton("Save PSD")
            save_trace.clicked.connect(self.save_trace)
            save_psd.clicked.connect(self.save_psd)
            export.addWidget(save_trace)
            export.addWidget(save_psd)
            controls.addLayout(export)
            self.stats_label = QtWidgets.QLabel("Ready. Choose a source and press Start.")
            self.stats_label.setWordWrap(True)
            self.stats_label.setStyleSheet("color: #a9bbd3; padding: 6px;")
            controls.addWidget(self.stats_label)
            controls.addStretch()

            right = QtWidgets.QVBoxLayout()
            split.addLayout(right, 1)
            self.readout = QtWidgets.QLabel("—")
            self.readout.setWordWrap(True)
            self.readout.setStyleSheet("background: #142238; padding: 8px; color: #b8dce8;")
            right.addWidget(self.readout)
            plots = pg.GraphicsLayoutWidget()
            right.addWidget(plots, 1)
            self.time_plot = plots.addPlot(row=0, col=0, title="Time domain · newest samples")
            self.time_plot.setLabel("bottom", "Time", units="s")
            self.time_plot.setLabel("left", "Amplitude", units="V")
            self.time_plot.showGrid(x=True, y=True, alpha=0.15)
            self.time_plot.addLegend(offset=(10, 10))
            self.psd_plot = plots.addPlot(row=1, col=0, title="Power spectral density")
            self.psd_plot.setLabel("bottom", "Frequency", units="Hz")
            self.psd_plot.setLabel("left", "PSD", units="dB")
            self.psd_plot.getAxis("left").enableAutoSIPrefix(False)
            self.psd_plot.showGrid(x=True, y=True, alpha=0.15)
            self.psd_plot.addLegend(offset=(10, 10))
            self.time_curves, self.psd_curves = {}, {}
            for ch in (1, 2):
                pen = pg.mkPen(COLORS[ch], width=1)
                self.time_curves[ch] = self.time_plot.plot(pen=pen, name=f"IN{ch}")
                self.psd_curves[ch] = self.psd_plot.plot(pen=pg.mkPen(COLORS[ch], width=1.5), name=f"IN{ch}")
            self.peak_marker = pg.ScatterPlotItem(size=9, brush=pg.mkBrush("#ff819c"), pen=None)
            self.psd_plot.addItem(self.peak_marker)
            self.log = QtWidgets.QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumBlockCount(100)
            self.log.setMaximumHeight(80)
            outer.addWidget(self.log)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.refresh)
            self.timer.start(100)

        # ---------------- settings ----------------
        def stream_settings(self):
            return replace(StreamSettings(), source=self.source.currentText(), host=self.host.text().strip(),
                           channels=self.channels.currentText(), decimation=int(self.decimation.currentData()),
                           volts_per_fs=float(self.vpfs.value()), extra_config=self.extra.toPlainText(),
                           replay_path=self.replay.text().strip())

        def psd_settings(self):
            return PSDSettings(nfft=int(self.nfft.currentData()), overlap_percent=float(self.overlap.value()),
                               window=self.window.currentText(), kaiser_beta=float(self.beta.value()),
                               psd_seconds=float(self.psd_len.value()) / 1e3,
                               averaging=self.averaging.currentText(), averages=int(self.averages.value()),
                               scaling="density" if self.scaling.currentIndex() == 0 else "spectrum",
                               remove_dc=self.remove_dc.isChecked())

        def browse_replay(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Replay file", self.replay.text(),
                                                            "Signals (*.npy *.bin);;All files (*)")
            if path:
                self.replay.setText(path)
                self.source.setCurrentText("Replay file")

        # ---------------- run control ----------------
        def start(self):
            if self.acq is not None and self.acq.alive:
                return
            try:
                self.acq = Acquisition(self.stream_settings())
            except (ValueError, RuntimeError) as error:
                QtWidgets.QMessageBox.warning(self, "Stream settings", str(error))
                return
            self.psd_state.clear()
            self.last = {}
            for ch in (1, 2):
                self.time_curves[ch].setData([], [])
                self.psd_curves[ch].setData([], [])
            s = self.acq.settings
            self.log.appendPlainText(f"Starting {s.source}: {s.channels}, decimation {s.decimation} "
                                     f"({s.fs/1e6:.6g} MSPS).")
            self.stream_box.setEnabled(False)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.acq.start()

        def stop(self):
            if self.acq is not None:
                self.acq.stop()
            self.stop_button.setEnabled(False)

        def toggle_pause(self, on):
            self.paused = on
            self.pause_button.setText("Resume" if on else "Pause")

        # ---------------- refresh ----------------
        def refresh(self):
            if self.acq is None:
                return
            while True:
                try:
                    self.log.appendPlainText(self.acq.messages.get_nowait())
                except queue.Empty:
                    break
            st = self.acq.snapshot()
            s = self.acq.settings
            self.stats_label.setText(f"{st['state']}  ·  {s.fs/1e6:.6g} MSPS nominal\n"
                                     f"Throughput: {st['rate_msps']:.3f} MSPS per channel\n"
                                     f"Gaps: {st['gaps']}  ·  FPGA lost samples: {st['lost']:,}  ·  "
                                     f"host overflows: {st['overflows']}"
                                     + (f"\n{st['error']}" if st["error"] else ""))
            if not self.acq.alive:
                self.stream_box.setEnabled(True)
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
            if self.paused:
                return
            fs, vscale = s.fs, s.volts_per_fs
            p = self.psd_settings()
            span_n = max(2, int(self.span.value() * 1e-3 * fs))
            psd_n = max(p.nfft, int(p.psd_seconds * fs))
            unit = "dBV²/Hz" if p.scaling == "density" else "dBV²"
            lines = []
            latest = self.acq.latest(max(span_n, psd_n))
            n_common = min(v.size for v in latest.values())
            for ch in (1, 2):
                if ch not in latest:
                    self.time_curves[ch].setData([], [])
                    self.psd_curves[ch].setData([], [])
                    continue
                x = latest[ch][latest[ch].size - n_common:] * vscale
                if x.size < 2:
                    continue
                y = x[-span_n:]
                t = np.arange(y.size) / fs
                tt, yy = minmax_decimate(t, y)
                self.time_curves[ch].setData(tt, yy)
                stats = f"IN{ch}: mean {np.mean(y):+.4g} V · RMS {np.sqrt(np.mean(y**2)):.4g} V · p-p {np.ptp(y):.4g} V"
                if x.size >= p.nfft:
                    f, pxx, rbw, segs = compute_psd(x[-psd_n:], fs, p)
                    state = self.psd_state.get(ch)
                    if p.averaging == "Exponential" and state is not None and state.shape == pxx.shape:
                        a = 1.0 / max(1, p.averages)
                        pxx = (1 - a) * state + a * pxx
                    elif p.averaging == "Max hold" and state is not None and state.shape == pxx.shape:
                        pxx = np.maximum(state, pxx)
                    self.psd_state[ch] = pxx
                    db = 10 * np.log10(np.maximum(pxx, 1e-30))
                    keep = f > 0 if self.log_freq.isChecked() else slice(None)
                    self.psd_curves[ch].setData(f[keep], db[keep])
                    k = int(np.argmax(db[1:]) + 1)
                    self.last[ch] = (f, pxx, x[-span_n:], fs)
                    stats += (f" · peak {db[k]:.1f} {unit} at {f[k]/1e3:.3f} kHz · median floor "
                              f"{np.median(db):.1f} {unit}")
                    if ch == s.channel_list[0]:
                        self.peak_marker.setData([np.log10(f[k]) if self.log_freq.isChecked() else f[k]], [db[k]])
                        self.psd_plot.setTitle(f"Power spectral density · {p.nfft:,}-pt {p.window}"
                                               f"{' β=' + format(p.kaiser_beta, 'g') if p.window == 'Kaiser' else ''}"
                                               f" · {p.overlap_percent:g}% overlap · RBW {rbw:,.1f} Hz · "
                                               f"{segs} segments/estimate · {p.averaging.lower()} averaging")
                else:
                    stats += f" · PSD needs {p.nfft:,} samples (have {x.size:,})"
                lines.append(stats)
            self.psd_plot.setLabel("left", "PSD", units=unit)
            if self.autoscale.isChecked():
                self.time_plot.enableAutoRange(axis="y")
            else:
                self.time_plot.disableAutoRange(axis="y")
            self.readout.setText("\n".join(lines) or "Waiting for samples…")

        # ---------------- export ----------------
        def save_trace(self):
            if not self.last:
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save trace", "rp_trace.npy", "NumPy (*.npy)")
            if path:
                chans = sorted(self.last)
                n = min(self.last[c][2].size for c in chans)
                np.save(path, np.column_stack([self.last[c][2][-n:] for c in chans]))
                self.log.appendPlainText(f"Saved {n:,} samples x {len(chans)} channel(s) (volts) to {path}")

        def save_psd(self):
            if not self.last:
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save PSD", "rp_psd.csv", "CSV (*.csv)")
            if path:
                chans = sorted(self.last)
                f = self.last[chans[0]][0]
                with open(path, "w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["frequency_hz"] + [f"IN{c}_psd" for c in chans])
                    for i in range(f.size):
                        writer.writerow([f[i]] + [self.last[c][1][i] if self.last[c][1].size == f.size else ""
                                                  for c in chans])
                self.log.appendPlainText("Saved PSD to " + path)

        def closeEvent(self, event):
            if self.acq is not None:
                self.acq.stop()
                if self.acq.thread.is_alive():
                    self.acq.thread.join(timeout=3)
            self.timer.stop()
            event.accept()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = Viewer()
    window.show()
    return app, window


def self_test():
    fs = 1e6
    t = np.arange(1 << 16) / fs
    rng = np.random.default_rng(0)
    x = 0.5 * np.sin(2 * np.pi * 123.0e3 * t) + 0.01 * rng.standard_normal(t.size)
    for window in WINDOWS:
        p = PSDSettings(nfft=4096, overlap_percent=50, window=window, scaling="spectrum", remove_dc=True)
        f, pxx, rbw, segs = compute_psd(x, fs, p)
        assert abs(f[np.argmax(pxx)] - 123e3) <= fs / 4096, window
        assert segs == 1 + (x.size - 4096) // 2048
    # 'spectrum' scaling with a flat-top window reads the tone power correctly: A^2/2 = 0.125.
    f, pxx, _, _ = compute_psd(x, fs, PSDSettings(nfft=4096, window="Flat top", scaling="spectrum"))
    assert abs(pxx.max() / 0.125 - 1) < 0.02, pxx.max()
    # 'density' integrates to the signal variance (Parseval).
    f, pxx, _, _ = compute_psd(x, fs, PSDSettings(nfft=4096, window="Hann", scaling="density"))
    assert abs(np.sum(pxx) * (f[1] - f[0]) / np.var(x) - 1) < 0.05
    assert abs(enbw_bins(make_window(PSDSettings(window="Hann"), 4096)) - 1.5) < 1e-3
    ring = Ring(10)
    ring.push(np.arange(7))
    ring.push(np.arange(7, 15))
    assert list(ring.latest(10)) == list(range(5, 15)) and list(ring.latest(3)) == [12, 13, 14]
    tt, yy = minmax_decimate(np.arange(100000.0), np.r_[np.zeros(50000), 1.0, np.zeros(49999)])
    assert yy.max() == 1.0 and tt.size <= 4000, "a single-sample spike must survive plot decimation"
    print("PASS: Welch PSD peak/segments for every window, flat-top tone power, density Parseval, "
          "Hann ENBW, ring buffer, spike-preserving plot decimation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true", help="Check DSP helpers without GUI or hardware")
    if parser.parse_args().self_test:
        self_test()
    else:
        app, window = run_viewer()
        sys.exit(app.exec_())
