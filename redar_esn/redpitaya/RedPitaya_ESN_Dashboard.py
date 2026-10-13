"""Red Pitaya receiver dashboard: atomic vapour cell photodetector -> PW / BW.

The HackRF TX (HackRF_TX_GUI.py) illuminates the vapour cell; the photodetector
output is streamed by the Red Pitaya (REAL samples, not I/Q). Each window of
3 PRI (PRI fixed at 512 us) goes through the model saved by TrainESN_Real.py,
which has separate PW and BW readouts.

Without a trained model the dashboard still streams, plots and can RECORD
labelled windows (TX truth from UDP or the manual reference) for training.

Models: TrainPhysicalRC.py (atomic receiver as the reservoir, linear readouts
for PW, BW and the chirp rate K directly) or TrainESN_Real.py (digital ESN /
pulse features). The chirp-rate card shows the DIRECT K readout when the model
has one; BW/PW is shown next to it for comparison.

Edit CONFIG below or use the GUI, then run: python RedPitaya_ESN_Dashboard.py
Requires PyQt5, pyqtgraph, numpy, scipy and the Red Pitaya streaming
python_lib (../../python_lib by default; matching _python_lib3_XX.so).
"""

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
import csv
import importlib
import json
import math
import queue
import sys
import threading
import time

import numpy as np

from SDR_UDP_Reference import UDP_PORT, UDPListener, ReferenceComparator
from physical_rc import MODEL_FORMAT as PHYSICAL_RC_FORMAT, PhysicalRCModel
from rp_core import RealEsnModel

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

HERE = Path(__file__).resolve().parent
DECIMATIONS = (2, 4, 8, 16, 32, 64, 128, 256)   # 512 us PRI = whole samples at 125 MSPS / these


# ============================================================
# USER SETTINGS -- no argparse.
# ============================================================
@dataclass
class Settings:
    model_path: str = str(HERE / "rp_esn_model.npz")
    source: str = "Red Pitaya"           # "Red Pitaya" or "Replay file"
    replay_path: str = str(HERE / "rp_dataset")   # recorder folder or its windows_f32.bin
    host: str = ""                       # Red Pitaya IP; empty = auto-discover on the LAN
    channel: int = 1                     # IN1 = 1, IN2 = 2 (photodetector input)
    decimation: int = 32                 # 125 MSPS / 32 = 3.90625 MSPS, 2000 samples per PRI
    extra_config: str = ""               # extra "key=value" lines sent to the streaming server
    streaming_lib_dir: str = str(HERE.parent.parent / "python_lib")
    base_rate: float = 125e6             # STEMlab 125-14 ADC clock
    pri_us: float = 512.0
    pri_per_window: int = 3
    block_size: int = 65536              # streaming block size (bytes)
    adc_size: int = 1638400              # reserved ADC streaming memory (bytes)
    rx_queue_blocks: int = 256           # packs buffered between network callback and assembler
    inference_mode: str = "Live/latest"  # Or "Every window" (stops on queue overflow).
    queue_size: int = 8
    display_interval_ms: int = 100
    history_points: int = 300
    record_dir: str = str(HERE / "rp_dataset")

    @property
    def fs(self):
        return self.base_rate / self.decimation

    @property
    def samples_per_pri(self):
        return int(round(self.fs * self.pri_us * 1e-6))

    @property
    def window_samples(self):
        return self.samples_per_pri * self.pri_per_window

    @property
    def window_seconds(self):
        return self.window_samples / self.fs

    def validate(self):
        if self.source not in ("Red Pitaya", "Replay file"):
            raise ValueError("Choose Red Pitaya or Replay file.")
        if self.inference_mode not in ("Live/latest", "Every window"):
            raise ValueError("Choose Live/latest or Every window.")
        if self.channel not in (1, 2):
            raise ValueError("Select IN1 or IN2.")
        if not isinstance(self.decimation, int) or self.decimation < 1:
            raise ValueError("Decimation must be a positive integer.")
        if not math.isclose(self.fs * self.pri_us * 1e-6, self.samples_per_pri, abs_tol=1e-6):
            raise ValueError("PRI must be a whole number of samples; choose a power-of-two decimation <= 256.")
        if self.pri_per_window < 1:
            raise ValueError("A window needs at least one PRI.")
        for line in self.extra_config.splitlines():
            if line.strip() and "=" not in line:
                raise ValueError(f"Extra config line must be key=value: {line!r}")
        if any(not isinstance(v, int) or v <= 0 for v in
               (self.block_size, self.adc_size, self.rx_queue_blocks, self.queue_size,
                self.display_interval_ms, self.history_points)):
            raise ValueError("Buffer, queue and display counts must be positive integers.")


CONFIG = Settings()


class SavedModel:
    """TrainPhysicalRC.py model (PW, BW, direct K) or TrainESN_Real.py model (PW, BW)."""
    def __init__(self, path, settings):
        with np.load(path, allow_pickle=False) as archive:
            kind = str(archive["model_format"]) if "model_format" in archive.files else ""
        self.model = (PhysicalRCModel if kind == PHYSICAL_RC_FORMAT else RealEsnModel).load(path)
        if not math.isclose(self.model.fs, settings.fs, rel_tol=0, abs_tol=0.5):
            raise ValueError(f"Model Fs={self.model.fs/1e6:.6g} MSPS but RX is {settings.fs/1e6:.6g} MSPS "
                             f"(decimation {settings.decimation}). Use the decimation the model was trained with.")
        if self.model.samples_per_pri != settings.samples_per_pri:
            raise ValueError("Model samples/PRI differs from RX settings.")
        self.samples = settings.window_samples
        self.description = self.model.describe()

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.samples,) or not np.isfinite(x).all():
            raise ValueError(f"Expected {self.samples} finite samples per inference.")
        output, info = self.model.predict(x)
        if not all(np.isfinite(v) for v in output.values()):
            raise ValueError("ESN returned non-finite predictions.")
        return output, info   # PW in us, BW in kHz, K (if present) in kHz/us = 1e9 Hz/s


class WindowAssembler:
    """Produce exact, non-overlapping windows even for arbitrary read sizes."""
    def __init__(self, size):
        self.size = size
        self.pending = np.empty(0, dtype=np.float32)

    def reset(self):
        discarded = self.pending.size
        self.pending = np.empty(0, dtype=np.float32)
        return discarded

    def feed(self, x):
        combined = np.concatenate((self.pending, x)) if self.pending.size else np.asarray(x)
        count = combined.size // self.size
        windows = [combined[k*self.size:(k+1)*self.size].copy() for k in range(count)]
        self.pending = combined[count*self.size:].copy()
        return windows


@dataclass
class Frame:
    sequence: int
    x: np.ndarray
    received_at: float


class InferenceBuffer:
    """Bounded buffer: current data in live mode, explicit failure in all mode."""
    def __init__(self, mode, capacity):
        self.mode, self.capacity = mode, capacity
        self.items = deque()
        self.condition = threading.Condition()
        self.skipped = 0

    def put(self, frame):
        with self.condition:
            if self.mode == "Live/latest":
                self.skipped += len(self.items)
                self.items.clear()
            elif len(self.items) >= self.capacity:
                raise RuntimeError("ESN cannot keep up with every window. "
                                   "Stopped before silently dropping queued windows; use Live/latest.")
            self.items.append(frame)
            self.condition.notify()

    def get(self, stop):
        with self.condition:
            while not self.items and not stop.is_set():
                self.condition.wait(0.1)
            return self.items.popleft() if self.items and not stop.is_set() else None

    def counts(self):
        with self.condition:
            return len(self.items), self.skipped


# ============================================================
# SOURCES
# ============================================================
class RedPitayaSource:
    """Network ADC streaming of one Red Pitaya input via the official python_lib."""
    def __init__(self, settings):
        self.settings = settings
        self.adc = self.conf = None
        self.packets = queue.Queue(maxsize=settings.rx_queue_blocks)
        self.lock = threading.Lock()
        self.host_overflow = False
        self.events = deque(maxlen=32)
        self.last_read_report = {}
        self.first_meta = None
        lib_dir = str(Path(settings.streaming_lib_dir).resolve())
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)
        try:
            streaming = importlib.import_module("streaming")
            streaming.ADCStreamClient
        except (ImportError, OSError, AttributeError) as error:
            raise RuntimeError("Cannot import the Red Pitaya streaming python_lib from " + lib_dir +
                               f". It needs a _python_lib{sys.version_info.major}_{sys.version_info.minor}.so "
                               "matching this Python version.") from error
        source = self

        class ADCHandler(streaming.ADCCallback):
            def receivePack(self, client, pack):
                ch = pack.channel1 if source.settings.channel == 1 else pack.channel2
                raw = ch.raw                                   # Python list of int16
                if not raw:
                    return
                item = (np.array(raw, dtype=np.int16), int(ch.fpgaLost),
                        (int(ch.baseRate), int(ch.bitsPerSample), int(ch.adcBaseBits), bool(ch.attenuator_1_20)))
                try:
                    source.packets.put_nowait(item)
                except queue.Full:
                    with source.lock:
                        source.host_overflow = True            # host too slow: next data is not contiguous

            def connected(self, client, host):
                source.events.append(f"ADC stream connected to {host}")

            def disconnected(self, client, host):
                source.events.append(f"ADC stream disconnected from {host}")

            def error(self, client, host, code):
                source.events.append(f"ADC stream error {code} on {host}")

        class ConfigHandler(streaming.ConfigCallback):
            def configError(self, client, host, code):
                source.events.append(f"Config error {code} on {host}")

            def adcServerStoppedMemError(self, client, host):
                source.events.append(f"ADC server stopped on {host}: memory error")

            def adcServerStoppedNoActiveChannels(self, client, host):
                source.events.append(f"ADC server stopped on {host}: no active channels")

        try:
            self.conf = streaming.ConfigStreamClient()
            self.conf_handler = ConfigHandler()               # keep Python handlers alive
            self.conf.addCallback(self.conf_handler)
            self.adc = streaming.ADCStreamClient(self.conf)
            self.adc_handler = ADCHandler()
            self.adc.setCallback(self.adc_handler)
            host = settings.host.strip()
            if not (self.conf.connect([host]) if host else self.conf.connect()):
                raise RuntimeError("Red Pitaya streaming server not found" + (f" at {host}" if host else
                                   " on the LAN") + ". Start the streaming app on the board.")
            other = 2 if settings.channel == 1 else 1
            config = [("adc_pass_mode", "NET"), ("adc_decimation", str(settings.decimation)),
                      ("block_size", str(settings.block_size)), ("adc_size", str(settings.adc_size)),
                      (f"channel_state_{settings.channel}", "ON"), (f"channel_state_{other}", "OFF")]
            config += [tuple(part.strip() for part in line.split("=", 1))
                       for line in settings.extra_config.splitlines() if line.strip()]
            for key, value in config:
                if not self.conf.sendConfig(key, value):
                    raise RuntimeError(f"Red Pitaya rejected config {key}={value}.")
            readback = self.conf.getConfig("adc_decimation").strip()
            if readback and readback != str(settings.decimation):
                raise RuntimeError(f"Red Pitaya reports decimation {readback}, requested {settings.decimation}.")
            if not self.adc.startStreaming():
                raise RuntimeError("Red Pitaya ADC streaming did not start.")
            self.waiter = threading.Thread(target=self.adc.wait, name="rp-stream-wait", daemon=True)
            self.waiter.start()
            hosts = list(self.conf.getHosts())
            self.description = (f"Red Pitaya {', '.join(hosts) or host} | IN{settings.channel} | "
                                f"decimation {settings.decimation} = {settings.fs/1e6:.6g} MSPS | real samples")
        except Exception:
            self.close()
            raise

    def read(self, stop):
        while not stop.is_set():
            try:
                raw, lost, meta = self.packets.get(timeout=0.2)
                break
            except queue.Empty:
                continue
        else:
            return np.empty(0, dtype=np.float32), False
        base_rate, bits, base_bits, att20 = meta
        if self.first_meta is None:
            self.first_meta = meta
            if base_rate and not math.isclose(base_rate / self.settings.decimation, self.settings.fs, abs_tol=0.5):
                raise RuntimeError(f"Board ADC clock is {base_rate/1e6:g} MHz, not {self.settings.base_rate/1e6:g} MHz; "
                                   "set base_rate for this board and retrain.")
        with self.lock:
            overflow, self.host_overflow = self.host_overflow, False
        full_scale = float(2 ** ((bits or base_bits or 16) - 1))
        x = raw.astype(np.float32) / full_scale                  # fraction of ADC full scale
        reasons = []
        if lost:
            reasons.append(f"FPGA lost {lost:,} samples")
        if overflow:
            reasons.append("host receive queue overflowed")
        self.last_read_report = dict(lost=lost, overflow=overflow, attenuator_1_20=att20,
                                     detail="; ".join(reasons))
        return x, bool(reasons)

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
    """Replay windows recorded by this dashboard (float32), in real time."""
    def __init__(self, settings):
        self.settings = settings
        path = Path(settings.replay_path)
        if path.is_dir():
            meta_path, path = path / "meta.json", path / "windows_f32.bin"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if int(meta["samples_per_window"]) != settings.window_samples:
                    raise ValueError("Recorded windows were taken with a different decimation/PRI.")
        size = path.stat().st_size
        frame_bytes = settings.window_samples * 4
        if size == 0 or size % frame_bytes:
            raise ValueError(f"Replay file must hold whole {settings.window_samples}-sample float32 windows.")
        self.data = np.memmap(path, dtype="<f4", mode="r")
        self.position = 0
        self.next_read = time.monotonic()
        self.last_read_report = {}
        self.events = deque()
        self.description = f"Recorded replay | {size//frame_bytes} windows | no hardware reception"

    def read(self, stop):
        delay = self.next_read - time.monotonic()
        if delay > 0 and stop.wait(delay):
            return np.empty(0, dtype=np.float32), False
        count = self.settings.window_samples
        x = np.array(self.data[self.position:self.position+count])
        self.position = (self.position + count) % self.data.size
        self.next_read = max(self.next_read + count / self.settings.fs, time.monotonic())
        return x, False

    def close(self):
        if self.data is not None:
            self.data._mmap.close()
            self.data = None


class Recorder:
    """Append labelled windows for TrainESN_Real.py (windows_f32.bin + labels.csv + meta.json)."""
    FIELDS = ("index", "pw_us", "bw_khz", "run_id", "direction", "carrier_hz", "sequence", "unix_time")

    def __init__(self, folder, settings):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        meta = dict(fs_hz=settings.fs, decimation=settings.decimation, base_rate_hz=settings.base_rate,
                    samples_per_pri=settings.samples_per_pri, pri_per_window=settings.pri_per_window,
                    samples_per_window=settings.window_samples, pri_us=settings.pri_us,
                    channel=settings.channel, units="fraction of ADC full scale", format="float32 little-endian")
        meta_path = self.folder / "meta.json"
        if meta_path.exists():
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            if int(old["samples_per_window"]) != settings.window_samples or \
               not math.isclose(float(old["fs_hz"]), settings.fs, abs_tol=0.5):
                raise ValueError(f"{self.folder} holds windows recorded at another decimation; choose a new folder.")
        else:
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        self.bin_path, self.csv_path = self.folder / "windows_f32.bin", self.folder / "labels.csv"
        existing = self.bin_path.stat().st_size // (4 * settings.window_samples) if self.bin_path.exists() else 0
        rows = 0
        if self.csv_path.exists():
            with open(self.csv_path, newline="", encoding="utf-8") as f:
                rows = max(0, sum(1 for _ in f) - 1)
        if rows != existing:
            raise ValueError(f"{self.folder}: {existing} windows but {rows} labels. Use a new folder.")
        self.count, self.dropped = existing, 0
        self.queue = queue.Queue(maxsize=256)
        self.thread = threading.Thread(target=self._run, name="rp-recorder", daemon=True)
        self.thread.start()

    def offer(self, frame, label):
        try:
            self.queue.put_nowait((frame, label))
        except queue.Full:
            self.dropped += 1

    def _run(self):
        new_csv = not self.csv_path.exists()
        with open(self.bin_path, "ab") as bin_file, open(self.csv_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.FIELDS)
            if new_csv:
                writer.writeheader()
            while True:
                item = self.queue.get()
                if item is None:
                    break
                frame, label = item
                bin_file.write(np.asarray(frame.x, dtype="<f4").tobytes())
                writer.writerow(dict(index=self.count, sequence=frame.sequence, unix_time=f"{time.time():.3f}",
                                     **{k: label.get(k, "") for k in ("pw_us", "bw_khz", "run_id",
                                                                       "direction", "carrier_hz")}))
                self.count += 1
                if self.queue.empty():
                    bin_file.flush()
                    csv_file.flush()

    def close(self):
        self.queue.put(None)
        self.thread.join(timeout=5)


# ============================================================
# SESSION: capture thread + inference thread
# ============================================================
class ReceiverSession:
    """Independent capture/inference workers; UI reads bounded snapshots only."""
    def __init__(self, settings, source_factory=None, comparator=None):
        settings.validate()
        self.settings = settings
        self.source_factory = source_factory
        self.comparator = comparator
        self.recorder = None
        self.record_note = "Recording off"
        self.stop_event = threading.Event()
        self.messages = queue.Queue(maxsize=128)
        self.lock = threading.Lock()
        self.buffer = InferenceBuffer(settings.inference_mode, settings.queue_size)
        self.stats = dict(captured=0, predicted=0, gaps=0, received_samples=0, lost_samples=0,
                          host_overflows=0, discarded_partial_samples=0, last_gap="",
                          inference_ms=0.0, prediction_age_ms=0.0, state="Idle", error="",
                          model="", recorded=0, record_dropped=0)
        self.result = None
        self.latest_frame = None
        self.thread = None
        self.started_at = None

    @property
    def alive(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self):
        self.thread = threading.Thread(target=self._run, name="rp-capture", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        with self.lock:
            if self.stats["state"] not in ("Stopped", "Error"):
                self.stats["state"] = "Stopping"

    def set_recorder(self, recorder):
        old, self.recorder = self.recorder, recorder
        if old is not None:
            old.close()
            with self.lock:                      # keep the final counts visible after stopping
                self.stats["recorded"], self.stats["record_dropped"] = old.count, old.dropped

    def fail(self, error):
        with self.lock:
            self.stats["error"] = str(error)
            self.stats["state"] = "Error"
        self.note("ERROR: " + str(error))
        self.stop_event.set()

    def note(self, text):
        try:
            self.messages.put_nowait(text)
        except queue.Full:
            try:
                self.messages.get_nowait()
            except queue.Empty:
                pass
            try:
                self.messages.put_nowait(text)
            except queue.Full:
                pass

    def snapshot(self):
        with self.lock:
            stats, result, frame = self.stats.copy(), self.result, self.latest_frame
        stats["queued"], stats["skipped"] = self.buffer.counts()
        elapsed = max(time.monotonic() - self.started_at, 1e-9) if self.started_at else 1
        stats["rx_msps"] = stats["received_samples"] / elapsed / 1e6
        stats["prediction_hz"] = stats["predicted"] / elapsed
        recorder = self.recorder
        if recorder is not None:
            stats["recorded"], stats["record_dropped"] = recorder.count, recorder.dropped
        stats["record_note"] = self.record_note
        if result is not None:
            stats["prediction_age_ms"] = (time.monotonic() - result["frame"].received_at) * 1000
        return stats, result, frame

    def _load_model(self):
        path = Path(self.settings.model_path)
        if not self.settings.model_path.strip() or not path.exists():
            self.note(f"No model at {path}: streaming, display and recording only (train with TrainESN_Real.py).")
            return None
        model = SavedModel(path, self.settings)
        self.note("Model loaded: " + model.description + ".")
        return model

    def _record(self, frame):
        recorder = self.recorder
        if recorder is None or self.comparator is None:
            return
        label, reason = self.comparator.label_for(frame.received_at, self.settings)
        if label is None:
            self.record_note = "Recording paused: " + reason
            return
        self.record_note = f"Recording PW {label['pw_us']:.3f} µs · BW {label['bw_khz']:.1f} kHz"
        recorder.offer(frame, label)

    def _run(self):
        source = predictor = None
        limiter = threadpool_limits(limits=1) if threadpool_limits is not None else nullcontext()
        try:
            with limiter:
                with self.lock:
                    self.stats["state"] = "Loading model"
                model = self._load_model()
                if model is not None:
                    model.predict(np.zeros(self.settings.window_samples))   # compile / warm up
                with self.lock:
                    self.stats["model"] = "Loaded" if model is not None else "None (acquisition only)"
                if self.stop_event.is_set():
                    return
                factory = self.source_factory or (RedPitayaSource if self.settings.source == "Red Pitaya"
                                                  else ReplaySource)
                source = factory(self.settings)
                self.note(source.description)
                self.started_at = time.monotonic()
                with self.lock:
                    self.stats["state"] = "Receiving"
                predictor = threading.Thread(target=self._infer, args=(model,), name="rp-inference", daemon=True)
                predictor.start()
                assembler = WindowAssembler(self.settings.window_samples)
                next_gap_log_at = 0.0
                while not self.stop_event.is_set():
                    x, gap = source.read(self.stop_event)
                    while getattr(source, "events", None):
                        self.note(source.events.popleft())
                    if self.stop_event.is_set():
                        break
                    report = getattr(source, "last_read_report", {})
                    with self.lock:
                        self.stats["received_samples"] += x.size
                    if gap:
                        # Samples before this pack are missing: drop the partial window,
                        # then continue with this (internally contiguous) pack.
                        partial = assembler.reset()
                        with self.lock:
                            self.stats["gaps"] += 1
                            self.stats["lost_samples"] += report.get("lost", 0)
                            self.stats["host_overflows"] += bool(report.get("overflow"))
                            self.stats["discarded_partial_samples"] += partial
                            self.stats["last_gap"] = report.get("detail", "")
                        now = time.monotonic()
                        if self.stats["gaps"] <= 3 or now >= next_gap_log_at:
                            self.note(f"RX discontinuity #{self.stats['gaps']}: {report.get('detail', '')}; "
                                      f"discarded {partial:,} partial-window samples.")
                            next_gap_log_at = now + 1.0
                    for window in assembler.feed(x):
                        with self.lock:
                            self.stats["captured"] += 1
                            frame = Frame(self.stats["captured"], window, time.monotonic())
                            self.latest_frame = frame
                        self._record(frame)
                        self.buffer.put(frame)
        except Exception as error:
            if not self.stop_event.is_set():
                self.fail(error)
        finally:
            self.stop_event.set()
            if source is not None:
                try:
                    source.close()
                except Exception as error:
                    self.note("Receiver close error: " + str(error))
            if predictor is not None:
                predictor.join()
            self.set_recorder(None)
            with self.lock:
                if self.stats["state"] != "Error":
                    self.stats["state"] = "Stopped"

    def _infer(self, model):
        s = self.settings
        try:
            while not self.stop_event.is_set():
                frame = self.buffer.get(self.stop_event)
                if frame is None:
                    break
                started = time.monotonic()
                x = frame.x
                rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
                clipping = float(100 * np.mean(np.abs(x) >= 0.999))
                result = dict(frame=frame, rms_fs=rms, clipping_percent=clipping, completed_at=0.0)
                if model is not None:
                    output, info = model.predict(x)
                    pw_us, bw_khz = output["PW"], output["BW"]
                    valid = 0 < pw_us <= s.pri_us and 0 < bw_khz
                    derived = bw_khz / pw_us * 1e9 if valid else math.nan
                    direct = output["K"] * 1e9 if "K" in output and output["K"] > 0 else math.nan
                    quality = ("No pulse detected" if not info["detected"] else
                               "Clipping" if clipping > 0.01 else
                               "Out of range" if not valid else "Estimate available")
                    result.update(pw_us=pw_us, bw_khz=bw_khz, chirp_rate_derived=derived, chirp_rate_direct=direct,
                                  chirp_rate=direct if "K" in output else derived,
                                  snr_db=info.get("snr_db", math.nan), match=info.get("match", math.nan),
                                  rise=info.get("rise", math.nan), fall=info.get("fall", math.nan),
                                  base=info.get("base", math.nan), folded=info["folded"], quality=quality)
                else:
                    n = s.samples_per_pri
                    result.update(folded=x[:n * (x.size // n)].reshape(-1, n).mean(axis=0), rise=math.nan,
                                  fall=math.nan, base=math.nan, snr_db=math.nan, match=math.nan,
                                  quality="No model — acquisition only")
                result["inference_ms"] = (time.monotonic() - started) * 1000
                result["completed_at"] = time.monotonic()
                if self.stop_event.is_set():
                    break
                if self.comparator is not None and model is not None:
                    result["comparison"] = self.comparator.observe(result, s)
                with self.lock:
                    self.stats["predicted"] += model is not None
                    self.stats["inference_ms"] = result["inference_ms"]
                    self.result = result
        except Exception as error:
            self.fail(error)


# ============================================================
# GUI
# ============================================================
def run_dashboard():
    try:
        from PyQt5 import QtCore, QtWidgets
        import pyqtgraph as pg
    except ImportError as error:
        raise SystemExit("Install GUI packages: python -m pip install PyQt5 pyqtgraph") from error

    pg.setConfigOptions(antialias=False, background="#101d30", foreground="#b8c7db")

    class Dashboard(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.session = None
            self.comparator = ReferenceComparator()
            self.udp_listener = None
            self.seen_sequence = 0
            self.records = deque(maxlen=CONFIG.history_points)
            self.closing = False
            self.setWindowTitle("Red Pitaya · Vapour Cell ESN Receiver")
            self.resize(1400, 960)
            self.setStyleSheet("""
                QMainWindow, QWidget { background: #091221; color: #e1eaf5; font-size: 13px; }
                QGroupBox { border: 1px solid #253750; border-radius: 8px; margin-top: 14px; padding: 12px; }
                QGroupBox::title { subcontrol-origin: margin; left: 12px; color: #93aac7; }
                QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QPlainTextEdit {
                    background: #142238; border: 1px solid #304660; border-radius: 5px; padding: 6px; }
                QPushButton { background: #1b3554; border: 1px solid #335576; border-radius: 6px; padding: 9px 15px; }
                QPushButton:hover { background: #285078; }
                QPushButton:disabled { color: #637387; background: #101d30; }
                QCheckBox { spacing: 8px; }
            """)
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            outer = QtWidgets.QVBoxLayout(central)
            title = QtWidgets.QLabel("RED PITAYA  /  VAPOUR CELL  /  ESN PARAMETER ESTIMATION")
            title.setStyleSheet("font-size: 23px; font-weight: bold; color: #eaf3ff; padding: 8px;")
            outer.addWidget(title)
            self.header = QtWidgets.QLabel()
            outer.addWidget(self.header)
            split = QtWidgets.QHBoxLayout()
            outer.addLayout(split, 1)
            controls_widget = QtWidgets.QWidget()
            controls_widget.setMinimumWidth(325)
            controls = QtWidgets.QVBoxLayout(controls_widget)
            controls_scroll = QtWidgets.QScrollArea()
            controls_scroll.setWidgetResizable(True)
            controls_scroll.setFixedWidth(370)
            controls_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            controls_scroll.setWidget(controls_widget)
            split.addWidget(controls_scroll)

            setup = QtWidgets.QGroupBox("Red Pitaya and saved model")
            form = QtWidgets.QFormLayout(setup)
            self.source = QtWidgets.QComboBox()
            self.source.addItems(["Red Pitaya", "Replay file"])
            self.source.setCurrentText(CONFIG.source)
            self.model_path = QtWidgets.QLineEdit(CONFIG.model_path)
            self.replay_path = QtWidgets.QLineEdit(CONFIG.replay_path)
            self.host = QtWidgets.QLineEdit(CONFIG.host)
            self.host.setPlaceholderText("auto-discover")
            self.channel = QtWidgets.QComboBox()
            self.channel.addItems(["IN1", "IN2"])
            self.channel.setCurrentIndex(CONFIG.channel - 1)
            self.decimation = QtWidgets.QComboBox()
            for d in DECIMATIONS:
                self.decimation.addItem(f"{d}  ·  {CONFIG.base_rate/d/1e6:.6g} MSPS", d)
            self.decimation.setCurrentIndex(DECIMATIONS.index(CONFIG.decimation))
            self.decimation.currentIndexChanged.connect(self.update_header)
            self.mode = QtWidgets.QComboBox()
            self.mode.addItems(["Live/latest", "Every window"])
            self.mode.setCurrentText(CONFIG.inference_mode)
            self.extra_config = QtWidgets.QPlainTextEdit(CONFIG.extra_config)
            self.extra_config.setPlaceholderText("optional key=value per line,\ne.g. channel_attenuator_1=...")
            self.extra_config.setMaximumHeight(62)
            form.addRow("Source", self.source)
            form.addRow("Model", self.path_picker(self.model_path, "Saved model (*.npz)"))
            form.addRow("Replay", self.path_picker(self.replay_path, "Recorded windows (*.bin);;All files (*)",
                                                   allow_dir=True))
            form.addRow("Board IP", self.host)
            form.addRow("Detector input", self.channel)
            form.addRow("Decimation", self.decimation)
            form.addRow("Inference mode", self.mode)
            form.addRow("Extra config", self.extra_config)
            controls.addWidget(setup)
            buttons = QtWidgets.QHBoxLayout()
            self.start_button = QtWidgets.QPushButton("Start RX")
            self.start_button.setStyleSheet("background: #167c6b; font-weight: bold;")
            self.stop_button = QtWidgets.QPushButton("Stop")
            self.stop_button.setEnabled(False)
            buttons.addWidget(self.start_button)
            buttons.addWidget(self.stop_button)
            controls.addLayout(buttons)
            self.start_button.clicked.connect(self.start_rx)
            self.stop_button.clicked.connect(self.stop_rx)
            self.editable_setup = setup

            reference = QtWidgets.QGroupBox("TX reference over Ethernet")
            reference_form = QtWidgets.QFormLayout(reference)
            self.reference_mode = QtWidgets.QComboBox()
            self.reference_mode.addItems(["UDP", "Manual", "Off"])
            self.udp_port = QtWidgets.QSpinBox()
            self.udp_port.setRange(1, 65535)
            self.udp_port.setValue(UDP_PORT)
            self.udp_tx_ip = QtWidgets.QLineEdit()
            self.udp_tx_ip.setPlaceholderText("Optional TX PC IPv4 filter")
            self.udp_button = QtWidgets.QPushButton("Start UDP listener")
            self.udp_button.clicked.connect(self.toggle_udp)
            self.settle_ms = QtWidgets.QSpinBox()
            self.settle_ms.setRange(0, 10000)
            self.settle_ms.setValue(500)
            self.settle_ms.setSuffix(" ms")
            self.udp_label = QtWidgets.QLabel("UDP listener stopped")
            self.udp_label.setWordWrap(True)
            self.reference_pw = QtWidgets.QDoubleSpinBox()
            self.reference_pw.setRange(0.001, 512)
            self.reference_pw.setDecimals(6)
            self.reference_pw.setValue(325)
            self.reference_pw.setSuffix(" µs")
            self.reference_bw = QtWidgets.QDoubleSpinBox()
            self.reference_bw.setRange(0.001, 20000)
            self.reference_bw.setDecimals(3)
            self.reference_bw.setValue(3500)
            self.reference_bw.setSuffix(" kHz")
            reference_form.addRow("Compare", self.reference_mode)
            reference_form.addRow("UDP port", self.udp_port)
            reference_form.addRow("TX IP filter", self.udp_tx_ip)
            reference_form.addRow(self.udp_button)
            reference_form.addRow("Settling guard", self.settle_ms)
            reference_form.addRow(self.udp_label)
            reference_form.addRow("Manual PW", self.reference_pw)
            reference_form.addRow("Manual BW", self.reference_bw)
            for widget in (self.reference_pw, self.reference_bw, self.settle_ms):
                widget.valueChanged.connect(self.configure_reference)
            self.reference_mode.currentTextChanged.connect(self.configure_reference)
            self.configure_reference()
            controls.addWidget(reference)

            recording = QtWidgets.QGroupBox("Training data recorder")
            record_form = QtWidgets.QFormLayout(recording)
            self.record_dir = QtWidgets.QLineEdit(CONFIG.record_dir)
            self.record_check = QtWidgets.QCheckBox("Record labelled windows")
            self.record_check.setToolTip("Saves each settled window with the TX PW/BW (UDP or manual) "
                                         "for TrainESN_Real.py. Windows without a valid label are skipped.")
            self.record_check.toggled.connect(self.toggle_recording)
            self.record_label = QtWidgets.QLabel("Recording off")
            self.record_label.setWordWrap(True)
            record_form.addRow("Folder", self.path_picker(self.record_dir, "", directory_only=True))
            record_form.addRow(self.record_check)
            record_form.addRow(self.record_label)
            controls.addWidget(recording)

            self.stats_label = QtWidgets.QLabel("Ready. Start the Red Pitaya streaming app, then Start RX.")
            self.stats_label.setWordWrap(True)
            self.stats_label.setStyleSheet("color: #a9bbd3; padding: 10px; line-height: 140%;")
            controls.addWidget(self.stats_label)
            export = QtWidgets.QHBoxLayout()
            save_window = QtWidgets.QPushButton("Save last window")
            save_csv = QtWidgets.QPushButton("Export history")
            save_window.clicked.connect(self.save_window)
            save_csv.clicked.connect(self.export_history)
            export.addWidget(save_window)
            export.addWidget(save_csv)
            controls.addLayout(export)
            controls.addStretch()

            right = QtWidgets.QVBoxLayout()
            split.addLayout(right, 1)
            cards = QtWidgets.QHBoxLayout()
            self.pw_card = self.card(cards, "ESTIMATED PW", "µs", "#50dfc2")
            self.bw_card = self.card(cards, "ESTIMATED BW", "kHz", "#56c7ff")
            self.rate_card = self.card(cards, "CHIRP RATE MAGNITUDE", "Hz/s · direct readout if the model has one",
                                       "#ffce76")
            right.addLayout(cards)
            self.quality_label = QtWidgets.QLabel("Waiting for receiver data")
            self.quality_label.setWordWrap(True)
            self.quality_label.setStyleSheet("color: #93aac7; padding: 5px;")
            right.addWidget(self.quality_label)
            self.error_label = QtWidgets.QLabel("Waiting for UDP TX reference")
            self.error_label.setWordWrap(True)
            self.error_label.setMinimumHeight(104)
            self.error_label.setStyleSheet("background: #142238; padding: 10px; color: #b8dce8;")
            right.addWidget(self.error_label)
            plots = pg.GraphicsLayoutWidget()
            right.addWidget(plots, 1)
            self.trace_plot = plots.addPlot(row=0, col=0, colspan=2, title="Photodetector · latest three-PRI window")
            self.trace_plot.setLabel("bottom", "Time", units="s")
            self.trace_plot.setLabel("left", "Level", units="FS")
            self.trace_plot.getAxis("left").enableAutoSIPrefix(False)
            self.trace_plot.showGrid(x=True, y=True, alpha=0.15)
            self.trace_curve = self.trace_plot.plot(pen=pg.mkPen("#56c7ff", width=1))
            self.fold_plot = plots.addPlot(row=1, col=0, colspan=2,
                                           title="PRI-folded average (3 pulses) · pulse start (yellow), end (pink), baseline (dashed)")
            self.fold_plot.setLabel("bottom", "Time within PRI", units="s")
            self.fold_plot.setLabel("left", "Level", units="FS")
            self.fold_plot.getAxis("left").enableAutoSIPrefix(False)
            self.fold_plot.showGrid(x=True, y=True, alpha=0.15)
            self.fold_curve = self.fold_plot.plot(pen=pg.mkPen("#50dfc2", width=2))
            self.base_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("#93aac7", style=QtCore.Qt.DashLine))
            self.rise_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("#ffce76", width=1))
            self.fall_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("#ff819c", width=1))
            for line in (self.base_line, self.rise_line, self.fall_line):
                line.hide()
                self.fold_plot.addItem(line)
            self.pw_plot = plots.addPlot(row=2, col=0, title="PW estimate history")
            self.bw_plot = plots.addPlot(row=2, col=1, title="BW estimate history")
            self.pw_plot.setLabel("left", "PW", units="µs")
            self.bw_plot.setLabel("left", "BW", units="kHz")
            for plot in (self.pw_plot, self.bw_plot):
                plot.getAxis("left").enableAutoSIPrefix(False)
                plot.setLabel("bottom", "Elapsed", units="s")
                plot.showGrid(x=True, y=True, alpha=0.15)
            self.pw_curve = self.pw_plot.plot(pen=pg.mkPen("#50dfc2", width=2))
            self.bw_curve = self.bw_plot.plot(pen=pg.mkPen("#56c7ff", width=2))
            self.log = QtWidgets.QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumBlockCount(100)
            self.log.setMaximumHeight(90)
            outer.addWidget(self.log)
            self.update_header()
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.refresh)
            self.timer.start(CONFIG.display_interval_ms)

        def current_settings(self):
            return replace(CONFIG, model_path=self.model_path.text().strip(), source=self.source.currentText(),
                           replay_path=self.replay_path.text().strip(), host=self.host.text().strip(),
                           channel=self.channel.currentIndex() + 1, decimation=int(self.decimation.currentData()),
                           extra_config=self.extra_config.toPlainText(), inference_mode=self.mode.currentText(),
                           record_dir=self.record_dir.text().strip())

        def update_header(self, *_):
            s = self.current_settings()
            self.header.setText(f"{s.fs/1e6:.6g} MSPS real samples   ·   PRI {s.pri_us:g} µs (fixed, not estimated)   ·   "
                                f"{s.pri_per_window} PRI = {s.window_seconds*1e3:.3f} ms   ·   "
                                f"{s.window_samples:,} samples   ·   Ts {1e9/s.fs:g} ns   ·   separate PW / BW readouts")

        def path_picker(self, entry, pattern, allow_dir=False, directory_only=False):
            widget = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(entry)
            button = QtWidgets.QPushButton("…")
            button.setFixedWidth(35)
            def browse():
                if directory_only:
                    path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder", entry.text())
                else:
                    path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", entry.text(), pattern)
                    if not path and allow_dir:
                        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Or select a recording folder",
                                                                          entry.text())
                if path:
                    entry.setText(path)
            button.clicked.connect(browse)
            layout.addWidget(button)
            return widget

        def card(self, layout, title, unit, color):
            group = QtWidgets.QGroupBox(title)
            box = QtWidgets.QVBoxLayout(group)
            number = QtWidgets.QLabel("—")
            number.setStyleSheet(f"font-size: 30px; font-weight: bold; color: {color};")
            box.addWidget(number)
            box.addWidget(QtWidgets.QLabel(unit))
            layout.addWidget(group)
            return number

        def configure_reference(self, *_):
            mode = self.reference_mode.currentText()
            self.reference_pw.setEnabled(mode == "Manual")
            self.reference_bw.setEnabled(mode == "Manual")
            self.comparator.configure(mode, self.settle_ms.value(), self.reference_pw.value(), self.reference_bw.value())

        def toggle_udp(self):
            if self.udp_listener is not None:
                self.udp_listener.close()
                self.udp_listener = None
                self.udp_button.setText("Start UDP listener")
                self.udp_port.setEnabled(True)
                self.udp_tx_ip.setEnabled(True)
                self.udp_label.setText("UDP listener stopped")
                return
            try:
                self.udp_listener = UDPListener(self.comparator, port=self.udp_port.value(),
                                                expected_tx_ip=self.udp_tx_ip.text())
            except (OSError, ValueError) as error:
                QtWidgets.QMessageBox.warning(self, "UDP listener", str(error))
                return
            self.udp_button.setText("Stop UDP listener")
            self.udp_port.setEnabled(False)
            self.udp_tx_ip.setEnabled(False)
            self.log.appendPlainText(f"UDP listening on 0.0.0.0:{self.udp_listener.port}; configure this PC's LAN IP in the TX UI.")

        def toggle_recording(self, enabled):
            if self.session is None or not self.session.alive:
                self.record_label.setText("Recording starts with RX" if enabled else "Recording off")
                return
            self.apply_recording(enabled)

        def apply_recording(self, enabled):
            if not enabled:
                self.session.set_recorder(None)
                self.session.record_note = "Recording off"
                return
            try:
                self.session.set_recorder(Recorder(self.record_dir.text().strip(), self.session.settings))
                self.log.appendPlainText("Recording labelled windows to " + self.record_dir.text().strip())
            except (OSError, ValueError, KeyError) as error:
                QtWidgets.QMessageBox.warning(self, "Recorder", str(error))
                self.record_check.setChecked(False)

        def refresh_reference(self):
            snap = self.comparator.snapshot()
            if self.udp_listener is not None:
                status = self.udp_listener.snapshot()
                age = snap["age_seconds"]
                age_text = f" · age {age:.2f}s" if age is not None else ""
                self.udp_label.setText(status["error"] or f"Listening :{self.udp_listener.port} · {status['accepted']} accepted / {status['rejected']} ignored\n"
                                       f"TX {snap['source_ip'] or 'unknown'}{age_text}")
            ref = snap["reference"]
            truth = f"True TX: PW {ref['pw_us']:.6f} µs · BW {ref['bw_khz']:.3f} kHz" if ref is not None else "No active TX reference"
            if not snap["count"]:
                self.error_label.setText(f"{truth}\n{snap['reason']} · MSE: —")
                return
            last = snap["last"]
            rate = f" · Rate MSE {snap['mse_rate']:.3e} (Hz/s)²" if snap["mse_rate"] is not None else ""
            self.error_label.setText(
                f"{truth} · {snap['reason']} · n = {snap['count']:,}\n"
                f"Error: PW {last['pw_error_ns']:+.2f} ns · BW {last['bw_error_khz']:+.3f} kHz\n"
                f"MSE: PW {snap['mse_pw']:.3f} ns² · BW {snap['mse_bw']:.6f} kHz²{rate}\n"
                f"RMSE: PW {math.sqrt(snap['mse_pw']):.2f} ns · BW {math.sqrt(snap['mse_bw']):.3f} kHz · "
                f"PW within one sample: {snap['within_sample_percent']:.1f}%")

        def start_rx(self):
            if self.session is not None and self.session.alive:
                return
            settings = self.current_settings()
            try:
                self.session = ReceiverSession(settings, comparator=self.comparator)
                self.comparator.restart_rx()
            except Exception as error:
                QtWidgets.QMessageBox.warning(self, "Receiver settings", str(error))
                return
            self.seen_sequence = 0
            self.records.clear()
            for card in (self.pw_card, self.bw_card, self.rate_card):
                card.setText("—")
            self.pw_curve.setData([], [])
            self.bw_curve.setData([], [])
            self.editable_setup.setEnabled(False)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.log.appendPlainText(f"Starting {settings.source} at decimation {settings.decimation} "
                                     f"in {settings.inference_mode} mode.")
            self.session.start()
            if self.record_check.isChecked():
                self.apply_recording(True)

        def stop_rx(self):
            if self.session is not None:
                self.session.stop()
            self.stop_button.setEnabled(False)

        def refresh(self):
            self.refresh_reference()
            if self.session is None:
                return
            while True:
                try:
                    self.log.appendPlainText(self.session.messages.get_nowait())
                except queue.Empty:
                    break
            stats, result, latest = self.session.snapshot()
            fs = self.session.settings.fs
            self.stats_label.setText(
                f"{stats['state']}  ·  model: {stats['model'] or '—'}\n"
                f"Captured windows: {stats['captured']:,}\n"
                f"Predictions: {stats['predicted']:,}  ·  {stats['prediction_hz']:.1f}/s\n"
                f"Skipped inference windows: {stats['skipped']:,}\n"
                f"Queue: {stats['queued']}  ·  RX gaps: {stats['gaps']}\n"
                f"FPGA lost samples: {stats['lost_samples']:,} · host overflows: {stats['host_overflows']}\n"
                f"Read throughput: {stats['rx_msps']:.2f} MSPS (expected {fs/1e6:.3f})\n"
                f"Inference: {stats['inference_ms']:.1f} ms  ·  age {stats['prediction_age_ms']:.1f} ms")
            self.record_label.setText(f"{stats['record_note']}\nSaved windows: {stats['recorded']:,}"
                                      + (f" · dropped {stats['record_dropped']:,}" if stats['record_dropped'] else ""))
            if latest is not None:
                step = max(1, latest.x.size // 4000)
                self.trace_curve.setData(np.arange(0, latest.x.size, step) / fs, latest.x[::step])
            if result is not None:
                folded = result["folded"]
                t = np.arange(folded.size) / fs
                self.fold_curve.setData(t, folded)
                for line, value in ((self.rise_line, result["rise"]), (self.fall_line, result["fall"]),
                                    (self.base_line, result["base"])):
                    if np.isfinite(value):
                        line.setValue(value / fs if line is not self.base_line else value)
                        line.show()
                    else:
                        line.hide()
                frame = result["frame"]
                snr = f"in-pulse SNR {result['snr_db']:.1f} dB  ·  " if np.isfinite(result["snr_db"]) else ""
                if np.isfinite(result["match"]):
                    snr += f"template match {result['match']:.3f}  ·  "
                if np.isfinite(result.get("chirp_rate_direct", math.nan)):
                    snr += (f"k direct {result['chirp_rate_direct']:.4e} vs BW/PW "
                            f"{result['chirp_rate_derived']:.4e} Hz/s  ·  ")
                self.quality_label.setText(f"{result['quality']}  ·  window #{frame.sequence}  ·  {snr}"
                                          f"RMS {result['rms_fs']:.4f} FS  ·  clipping {result['clipping_percent']:.3f}%")
                if "pw_us" in result:
                    self.pw_card.setText(f"{result['pw_us']:.4f}")
                    self.bw_card.setText(f"{result['bw_khz']:.2f}")
                    self.rate_card.setText(f"{result['chirp_rate']:.3e}" if np.isfinite(result["chirp_rate"]) else "Invalid")
                    if frame.sequence != self.seen_sequence:
                        self.seen_sequence = frame.sequence
                        record = dict(elapsed_s=result["completed_at"] - self.session.started_at,
                                      sequence=frame.sequence, pw_us=result["pw_us"], bw_khz=result["bw_khz"],
                                      chirp_rate_Hz_per_s=result["chirp_rate"],
                                      chirp_rate_direct_Hz_per_s=result["chirp_rate_direct"],
                                      chirp_rate_derived_Hz_per_s=result["chirp_rate_derived"],
                                      template_match=result["match"], snr_db=result["snr_db"],
                                      rms_fs=result["rms_fs"], inference_ms=result["inference_ms"],
                                      quality=result["quality"])
                        record.update(result.get("comparison", {}))
                        self.records.append(record)
                        records = list(self.records)
                        self.pw_curve.setData([r["elapsed_s"] for r in records], [r["pw_us"] for r in records])
                        self.bw_curve.setData([r["elapsed_s"] for r in records], [r["bw_khz"] for r in records])
            if not self.session.alive:
                self.editable_setup.setEnabled(True)
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.quality_label.setText(stats["error"] or "Receiver stopped; displayed estimate is the last result.")
                if self.closing:
                    self.close()

        def save_window(self):
            if self.session is None or self.session.snapshot()[2] is None:
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save latest captured window", "rp_window.npy", "NumPy (*.npy)")
            if path:
                np.save(path, self.session.snapshot()[2].x)
                self.log.appendPlainText("Saved latest captured window: " + path)

        def export_history(self):
            if not self.records:
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export displayed prediction history", "rp_predictions.csv", "CSV (*.csv)")
            if path:
                with open(path, "w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(self.records[0]))
                    writer.writeheader()
                    writer.writerows(self.records)
                self.log.appendPlainText("Saved displayed prediction history: " + path)

        def closeEvent(self, event):
            if self.session is not None and self.session.alive:
                self.closing = True
                self.session.stop()
                self.setWindowTitle("Stopping receiver…")
                event.ignore()
            else:
                self.timer.stop()
                if self.udp_listener is not None:
                    self.udp_listener.close()
                    self.udp_listener = None
                event.accept()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = Dashboard()
    window.show()
    return app, window


if __name__ == "__main__":
    app, window = run_dashboard()
    sys.exit(app.exec_())
