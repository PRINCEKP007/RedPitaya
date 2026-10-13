"""Local bladeRF receiver dashboard for the model saved by TrainESN_1.py.

Edit CONFIG below or use the GUI, then run: python BladeRF_ESN_Dashboard.py
No training script imports, command-line arguments, or transmission are used.
"""

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
import csv
import importlib
import math
import queue
import sys
import threading
import time

import numpy as np

from SDR_UDP_Reference import UDP_PORT, UDPListener, ReferenceComparator
from esn_core import EsnModel

# These are C header macros, not shared-library symbols. CFFI bindings may
# omit them even though metadata streaming is supported. Values from Nuand:
# https://www.nuand.com/libbladeRF-doc/v2.6.0/group___s_t_r_e_a_m_i_n_g___f_o_r_m_a_t___m_e_t_a_d_a_t_a.html
BLADERF_META_FLAG_RX_NOW = 1 << 31
BLADERF_META_STATUS_OVERRUN = 1 << 0

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


# ============================================================
# USER SETTINGS -- no argparse. Frequency must match HackRF TX.
# ============================================================
@dataclass
class Settings:
    model_path: str = str(Path(__file__).resolve().parent / "esn_model.npz")
    source: str = "bladeRF"       # "bladeRF" or "Replay I/Q file"
    replay_path: str = "ESN_Train.bin"  # Raw little-endian complex128, column order F.
    frequency_mhz: float = 1000.0  # Match the actual carrier in your HackRF UI.
    gain_db: int = 30             # Match the receiver settings used for training.
    bandwidth_mhz: float = 20.0
    rx_port: int = 1              # Physical RX1=1, RX2=2; Python indices are 0/1.
    device_identifier: str = ""   # Empty selects first bladeRF.
    fs: int = 20_000_000
    pri_us: float = 512.0
    pri_per_window: int = 3
    iq_divisor: float = 2047.0     # Matches the supplied BladeRF_RX.py.
    use_metadata: bool = True     # Check hardware timestamps/overruns when supported.
    max_consecutive_rx_gaps: int = 64  # Stop a persistent gap storm instead of spinning indefinitely.
    read_windows: int = 4         # Four complete windows per sync_rx call.
    num_buffers: int = 32
    buffer_size: int = 32768      # Must be a multiple of 1024.
    num_transfers: int = 8
    timeout_ms: int = 1500
    warmup_reads: int = 4         # Discard initial acquisition transients.
    inference_mode: str = "Live/latest"  # Or "Every window" (stops on queue overflow).
    queue_size: int = 8
    display_interval_ms: int = 100
    history_points: int = 300
    minimum_power_dbfs: float = -80.0  # Quality flag only; does not alter input samples.

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
        if self.fs != 20_000_000 or not np.isclose(self.pri_us, 512.0) or self.pri_per_window != 3:
            raise ValueError("This dashboard expects Fs=20e6, PRI=512 us and three PRI per window.")
        if self.source not in ("bladeRF", "Replay I/Q file"):
            raise ValueError("Choose bladeRF or Replay I/Q file.")
        if self.inference_mode not in ("Live/latest", "Every window"):
            raise ValueError("Choose Live/latest or Every window.")
        if not np.isfinite(self.frequency_mhz) or self.frequency_mhz <= 0:
            raise ValueError("Receiver frequency must be finite and positive.")
        if not np.isfinite(self.bandwidth_mhz) or self.bandwidth_mhz <= 0:
            raise ValueError("Receiver bandwidth must be finite and positive.")
        if self.rx_port not in (1, 2) or not np.isfinite(self.iq_divisor) or self.iq_divisor <= 0:
            raise ValueError("Select RX1/RX2 and a positive I/Q divisor.")
        if any(not isinstance(v, int) or v <= 0 for v in
               (self.read_windows, self.num_buffers, self.buffer_size, self.num_transfers,
                self.max_consecutive_rx_gaps,
                self.timeout_ms, self.queue_size, self.display_interval_ms, self.history_points)):
            raise ValueError("Buffer, queue and display counts must be positive integers.")
        if self.buffer_size % 1024 or self.num_transfers >= self.num_buffers or self.warmup_reads < 0:
            raise ValueError("Buffer size must be a multiple of 1024; transfers < buffers; warmup >= 0.")


CONFIG = Settings()


class SavedESN:
    """Model saved by TrainESN_1.py: separate PW and BW readouts (PRI is fixed).

    Feature extraction and readouts live in esn_core.py, shared with training.
    Older single-W_readout models still load (both readouts use reservoir features).
    """
    def __init__(self, path, settings):
        self.model = EsnModel.load(path, samples_per_pri=settings.samples_per_pri)
        self.fs = self.model.fs
        if not np.isfinite(self.fs) or not np.isclose(self.fs, settings.fs, rtol=0, atol=0.5):
            raise ValueError(f"Saved model Fs={self.fs:g} Hz does not match RX Fs={settings.fs:g} Hz.")
        self.samples = settings.window_samples
        reservoir = self.model.reservoir
        if reservoir is not None and not 1 <= reservoir.block_size <= self.samples:
            raise ValueError("Invalid compression block size/stride in model.")
        self.description = self.model.describe()

    def predict(self, iq):
        x = np.asarray(iq, dtype=np.complex128)
        if x.shape != (self.samples,) or not np.isfinite(x).all():
            raise ValueError(f"Expected {self.samples} finite complex samples per inference.")
        output, info = self.model.predict(x)
        if not all(np.isfinite(v) for v in output.values()):
            raise ValueError("ESN returned non-finite predictions.")
        return output["PW"], output["BW"], info   # PW in us, BW in kHz


def decode_sc16_q11(buffer, divisor):
    raw = np.frombuffer(buffer, dtype="<i2")
    if raw.size % 2:
        raise ValueError("Interleaved I/Q buffer contains an odd number of int16 values.")
    # Preserve the conversion/complex64 scaling in the supplied RX reference.
    return ((raw[0::2].astype(np.float64) + 1j * raw[1::2]) / divisor).astype(np.complex64)


class WindowAssembler:
    """Produce exact, non-overlapping windows even for arbitrary read sizes."""
    def __init__(self, size):
        self.size = size
        self.pending = np.empty(0, dtype=np.complex64)

    def reset(self):
        discarded = self.pending.size
        self.pending = np.empty(0, dtype=np.complex64)
        return discarded

    def feed(self, iq):
        combined = np.concatenate((self.pending, iq)) if self.pending.size else np.asarray(iq)
        count = combined.size // self.size
        windows = [combined[k*self.size:(k+1)*self.size].copy() for k in range(count)]
        self.pending = combined[count*self.size:].copy()
        return windows


@dataclass
class Frame:
    sequence: int
    iq: np.ndarray
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
                raise RuntimeError("ESN cannot keep up with every 1.536 ms window. "
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


class BladeRFSource:
    def __init__(self, settings):
        self.settings = settings
        self.device = self.channel = None
        try:
            module = importlib.import_module("bladerf")
        except (ImportError, OSError) as error:
            raise RuntimeError("Cannot import bladeRF bindings. Use the Python environment where "
                               "your BladeRF_RX.py works, with libbladeRF installed.") from error
        self.api = getattr(module, "_bladerf", module)
        self.device = getattr(module, "BladeRF", self.api.BladeRF)(settings.device_identifier or None)
        self.ch = self.api.CHANNEL_RX(settings.rx_port - 1)
        self.metadata_enabled = settings.use_metadata and hasattr(self.api, "ffi") and hasattr(self.api, "libbladeRF")
        self.expected_timestamp = None
        self.previous_read_started_at = None
        self.last_read_report = {}
        self.usb_speed = "unknown"
        try:
            if hasattr(self.device, "Channel"):
                self.channel = self.device.Channel(self.ch)
                self.channel.frequency = int(settings.frequency_mhz * 1e6)
                self.channel.sample_rate = settings.fs
                self.channel.bandwidth = int(settings.bandwidth_mhz * 1e6)
                if self.channel.gain_mode is not None:
                    self.channel.gain_mode = self.api.GainMode.Manual
                self.channel.gain = settings.gain_db
                actual_fs = int(self.channel.sample_rate)
                actual_bw = int(self.channel.bandwidth)
            else:
                self.device.set_frequency(self.ch, int(settings.frequency_mhz * 1e6))
                self.device.set_sample_rate(self.ch, settings.fs)
                self.device.set_bandwidth(self.ch, int(settings.bandwidth_mhz * 1e6))
                if self.device.get_gain_mode(self.ch) is not None:
                    self.device.set_gain_mode(self.ch, self.api.GainMode.Manual)
                self.device.set_gain(self.ch, settings.gain_db)
                actual_fs = int(self.device.get_sample_rate(self.ch))
                actual_bw = int(self.device.get_bandwidth(self.ch))
            if actual_fs != settings.fs:
                raise ValueError(f"Hardware returned {actual_fs} Hz; model requires {settings.fs} Hz.")
            fmt = self.api.Format.SC16_Q11_META if self.metadata_enabled else self.api.Format.SC16_Q11
            self.device.sync_config(layout=self.api.ChannelLayout.RX_X1, fmt=fmt,
                                    num_buffers=settings.num_buffers, buffer_size=settings.buffer_size,
                                    num_transfers=settings.num_transfers, stream_timeout=settings.timeout_ms)
            self.meta = self.api.ffi.new("struct bladerf_metadata *") if self.metadata_enabled else None
            self.set_enabled(True)
            self.read_samples = settings.read_windows * settings.window_samples
            self.buffer = bytearray(4 * self.read_samples)
            try:
                speed = self.device.get_device_speed()
                self.usb_speed = str(getattr(speed, "name", speed))
            except Exception:
                # Optional diagnostic on bindings that do not expose this API.
                pass
            gap_text = "timestamp gap checks enabled" if self.metadata_enabled else "USB gap checks unavailable"
            self.description = (f"RX{settings.rx_port} | {actual_fs/1e6:g} MSPS | "
                                f"filter {actual_bw/1e6:g} MHz | USB {self.usb_speed} | {gap_text}")
        except Exception:
            self.close()
            raise

    def set_enabled(self, enabled):
        if self.channel is not None:
            self.channel.enable = enabled
        else:
            self.device.enable_module(self.ch, enabled)

    def read(self, stop):
        read_started_at = time.monotonic()
        interval_ms = ((read_started_at - self.previous_read_started_at) * 1000
                       if self.previous_read_started_at is not None else 0.0)
        self.previous_read_started_at = read_started_at
        if self.metadata_enabled:
            # Start each native call with fresh output fields as well as RX_NOW.
            self.meta.flags = BLADERF_META_FLAG_RX_NOW
            self.meta.status = 0
            self.meta.actual_count = 0
            self.meta.timestamp = 0
            self.device.sync_rx(self.buffer, self.read_samples, timeout_ms=self.settings.timeout_ms, meta=self.meta)
            count, stamp = int(self.meta.actual_count), int(self.meta.timestamp)
            if not 0 <= count <= self.read_samples:
                raise RuntimeError("bladeRF metadata returned an invalid sample count.")
            status = int(self.meta.status)
            overrun = bool(status & BLADERF_META_STATUS_OVERRUN)
            delta = stamp - self.expected_timestamp if self.expected_timestamp is not None else 0
            timestamp_gap = delta != 0
            gap = overrun or timestamp_gap
            self.expected_timestamp = stamp + count
            reasons = []
            if overrun:
                reasons.append("overrun flag")
            if timestamp_gap:
                reasons.append(f"timestamp delta {delta:+,} samples ({delta/self.settings.fs*1e6:+.2f} µs)")
            self.last_read_report = dict(overrun=overrun, timestamp_gap=timestamp_gap,
                                         interval_ms=interval_ms, timestamp_delta=delta,
                                         detail=(f"{'; '.join(reasons)}; status=0x{status:08x}; "
                                                 f"returned {count:,}/{self.read_samples:,} samples; "
                                                 f"read interval {interval_ms:.2f} ms") if gap else "")
            # RX sample-time jumps must be plausible relative to elapsed host
            # time. Leave generous allowance for USB buffering and timeouts.
            allowed_seconds = max(10.0, 4*(interval_ms/1000 + self.settings.timeout_ms/1000 +
                                         self.settings.num_buffers*self.settings.buffer_size/self.settings.fs))
            if timestamp_gap and abs(delta) > self.settings.fs * allowed_seconds:
                raise RuntimeError(
                    "Unreliable RX metadata: " + self.last_read_report["detail"] +
                    ". Stop RX, select plain SC16 diagnostic reception (metadata unchecked), "
                    "and restart. Collect bladeRF library/firmware/FPGA versions to diagnose "
                    "the metadata format. Plain mode cannot verify USB sample continuity.")
            iq = decode_sc16_q11(memoryview(self.buffer)[:4*count], self.settings.iq_divisor)
            return iq, gap
        self.device.sync_rx(self.buffer, self.read_samples, timeout_ms=self.settings.timeout_ms)
        self.last_read_report = dict(overrun=False, timestamp_gap=False, interval_ms=interval_ms, detail="")
        return decode_sc16_q11(self.buffer, self.settings.iq_divisor), False

    def close(self):
        if self.device is not None:
            try:
                self.set_enabled(False)
            except Exception:
                pass
            self.device.close()
            self.device = None


class ReplaySource:
    """Replay recorded complex128 columns without changing amplitude or phase."""
    def __init__(self, settings):
        self.settings = settings
        path = Path(settings.replay_path)
        size = path.stat().st_size
        frame_bytes = settings.window_samples * 16
        if size == 0 or size % frame_bytes:
            raise ValueError("Replay file must be raw little-endian complex128 and contain whole 30720-sample windows.")
        self.data = np.memmap(path, dtype="<c16", mode="r")
        self.position = 0
        self.next_read = time.monotonic()
        self.description = f"Recorded I/Q replay | {size//frame_bytes} windows | no hardware reception"

    def read(self, stop):
        delay = self.next_read - time.monotonic()
        if delay > 0 and stop.wait(delay):
            return np.empty(0, dtype=np.complex128), False
        count = self.settings.window_samples
        iq = self.data[self.position:self.position+count].copy()
        self.position = (self.position + count) % self.data.size
        self.next_read = max(self.next_read + count / self.settings.fs, time.monotonic())
        return iq, False

    def close(self):
        if self.data is not None:
            self.data._mmap.close()
            self.data = None


class ReceiverSession:
    """Independent capture/inference workers; UI reads bounded snapshots only."""
    def __init__(self, settings, source_factory=None, comparator=None):
        settings.validate()
        self.settings = settings
        self.source_factory = source_factory
        self.comparator = comparator
        self.stop_event = threading.Event()
        self.messages = queue.Queue(maxsize=128)
        self.lock = threading.Lock()
        self.buffer = InferenceBuffer(settings.inference_mode, settings.queue_size)
        self.stats = dict(captured=0, predicted=0, gaps=0, received_samples=0,
                          overruns=0, timestamp_gaps=0, rejected_samples=0, discarded_partial_samples=0,
                          last_gap="", read_interval_ms=0.0, max_read_interval_ms=0.0,
                          continuity="Not started",
                          inference_ms=0.0, prediction_age_ms=0.0, state="Idle", error="")
        self.result = None
        self.latest_frame = None
        self.thread = None
        self.started_at = None

    @property
    def alive(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self):
        self.thread = threading.Thread(target=self._run, name="bladeRF-capture", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        with self.lock:
            if self.stats["state"] not in ("Stopped", "Error"):
                self.stats["state"] = "Stopping"

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
            # Status logging must never block reception during an overrun storm.
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
        if result is not None:
            stats["prediction_age_ms"] = (time.monotonic() - result["frame"].received_at) * 1000
        return stats, result, frame

    def _run(self):
        source = predictor = None
        limiter = threadpool_limits(limits=1) if threadpool_limits is not None else nullcontext()
        try:
            with limiter:
                with self.lock:
                    self.stats["state"] = "Loading model"
                model = SavedESN(self.settings.model_path, self.settings)
                self.note("Model loaded: " + model.description + ".")
                self.note("Preparing inference before starting the receive stream...")
                model.predict(np.zeros(self.settings.window_samples, dtype=np.complex128))
                if self.stop_event.is_set():
                    return
                factory = self.source_factory or (BladeRFSource if self.settings.source == "bladeRF" else ReplaySource)
                source = factory(self.settings)
                self.note(source.description)
                with self.lock:
                    self.stats["continuity"] = ("Replay" if self.settings.source != "bladeRF" else
                                                "Metadata checks ON" if getattr(source, "metadata_enabled", False) else
                                                "UNVERIFIED — metadata OFF")
                if self.settings.source == "bladeRF" and not getattr(source, "metadata_enabled", False):
                    self.note("Plain SC16 diagnostic reception: timestamps and USB overrun metadata "
                              "are unavailable; sample continuity and accuracy remain unverified.")
                if self.settings.source == "bladeRF" and str(getattr(source, "usb_speed", "")).lower() == "high":
                    self.note("USB High = USB 2.0. Use a USB 3.0 port/cable for 20 MSPS SC16 reception.")
                for _ in range(self.settings.warmup_reads if self.settings.source == "bladeRF" else 0):
                    if self.stop_event.is_set():
                        return
                    source.read(self.stop_event)
                self.started_at = time.monotonic()
                with self.lock:
                    self.stats["state"] = "Receiving"
                predictor = threading.Thread(target=self._infer, args=(model,), name="esn-inference", daemon=True)
                predictor.start()
                assembler = WindowAssembler(self.settings.window_samples)
                next_gap_log_at = 0.0
                consecutive_gaps = 0
                while not self.stop_event.is_set():
                    iq, gap = source.read(self.stop_event)
                    if self.stop_event.is_set():
                        break
                    report = getattr(source, "last_read_report", {})
                    with self.lock:
                        # Read throughput includes returned samples subsequently rejected for gaps.
                        self.stats["received_samples"] += iq.size
                        self.stats["read_interval_ms"] = report.get("interval_ms", 0.0)
                        self.stats["max_read_interval_ms"] = max(self.stats["max_read_interval_ms"],
                                                                 self.stats["read_interval_ms"])
                    if gap:
                        consecutive_gaps += 1
                        partial_samples = assembler.reset()
                        detail = report.get("detail") or "source reported non-contiguous samples"
                        with self.lock:
                            self.stats["gaps"] += 1
                            self.stats["overruns"] += bool(report.get("overrun"))
                            self.stats["timestamp_gaps"] += bool(report.get("timestamp_gap"))
                            self.stats["rejected_samples"] += iq.size
                            self.stats["discarded_partial_samples"] += partial_samples
                            self.stats["last_gap"] = detail
                        # Discard this entire read, then resume with a fresh contiguous window.
                        now = time.monotonic()
                        if self.stats["gaps"] <= 3 or now >= next_gap_log_at:
                            self.note(f"RX discontinuity #{self.stats['gaps']}: {detail}; "
                                      f"discarded read and {partial_samples:,} partial-window samples.")
                            next_gap_log_at = now + 1.0
                        if consecutive_gaps >= self.settings.max_consecutive_rx_gaps:
                            raise RuntimeError(
                                f"RX stopped after {consecutive_gaps} consecutive discontinuous reads. "
                                "No clean windows are arriving. Check metadata/driver compatibility "
                                "and USB throughput, or select plain SC16 diagnostic reception. Last read: " + detail)
                        continue
                    consecutive_gaps = 0
                    for window in assembler.feed(iq):
                        with self.lock:
                            self.stats["captured"] += 1
                            frame = Frame(self.stats["captured"], window, time.monotonic())
                            self.latest_frame = frame
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
            with self.lock:
                if self.stats["state"] != "Error":
                    self.stats["state"] = "Stopped"

    def _infer(self, model):
        try:
            while not self.stop_event.is_set():
                frame = self.buffer.get(self.stop_event)
                if frame is None:
                    break
                started = time.monotonic()
                pw_us, bw_khz, pulse = model.predict(frame.iq)
                duration_ms = (time.monotonic() - started) * 1000
                power_dbfs = float(10*np.log10(max(float(np.mean(np.abs(frame.iq)**2)), 1e-20)))
                clip_level = 2047.0 / self.settings.iq_divisor
                clipping_percent = float(100*np.mean((np.abs(frame.iq.real) >= clip_level) |
                                                     (np.abs(frame.iq.imag) >= clip_level)))
                valid = 0 < pw_us <= self.settings.pri_us and 0 < bw_khz <= self.settings.fs / 1e3
                quality = ("Low signal" if power_dbfs < self.settings.minimum_power_dbfs else
                           "No pulse detected" if not pulse["detected"] else
                           "Clipping" if clipping_percent > 0.01 else
                           "Out of range" if not valid else "Estimate available")
                if self.stats["continuity"].startswith("UNVERIFIED"):
                    quality += " · continuity unverified"
                result = dict(frame=frame, pw_us=pw_us, bw_khz=bw_khz,
                              chirp_rate=(bw_khz/pw_us*1e9 if valid else math.nan),
                              pulse_contrast_db=pulse["contrast_db"], duty_percent=100*pulse["duty"],
                              offset_khz=pulse["f_offset_khz"],
                              power_dbfs=power_dbfs, clipping_percent=clipping_percent,
                              quality=quality, inference_ms=duration_ms,
                              completed_at=time.monotonic())
                if self.stop_event.is_set():
                    break
                if self.comparator is not None:
                    result["comparison"] = self.comparator.observe(result, self.settings)
                with self.lock:
                    self.stats["predicted"] += 1
                    self.stats["inference_ms"] = duration_ms
                    self.result = result
        except Exception as error:
            self.fail(error)


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
            self.setWindowTitle("bladeRF · ESN Receiver Dashboard")
            self.resize(1400, 930)
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
            title = QtWidgets.QLabel("bladeRF  /  ESN PARAMETER ESTIMATION")
            title.setStyleSheet("font-size: 23px; font-weight: bold; color: #eaf3ff; padding: 8px;")
            outer.addWidget(title)
            outer.addWidget(QtWidgets.QLabel("20 MSPS   ·   PRI 512 µs (fixed, not estimated)   ·   3 PRI = 1.536 ms   ·   "
                                             "30,720 complex samples   ·   Ts 50 ns   ·   separate PW / BW readouts"))
            split = QtWidgets.QHBoxLayout()
            outer.addLayout(split, 1)
            controls_widget = QtWidgets.QWidget()
            controls_widget.setMinimumWidth(325)
            controls = QtWidgets.QVBoxLayout(controls_widget)
            controls_scroll = QtWidgets.QScrollArea()
            controls_scroll.setWidgetResizable(True)
            controls_scroll.setFixedWidth(365)
            controls_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            controls_scroll.setWidget(controls_widget)
            split.addWidget(controls_scroll)
            setup = QtWidgets.QGroupBox("Receiver and saved model")
            form = QtWidgets.QFormLayout(setup)
            self.model_path = QtWidgets.QLineEdit(CONFIG.model_path)
            self.source = QtWidgets.QComboBox()
            self.source.addItems(["bladeRF", "Replay I/Q file"])
            self.source.setCurrentText(CONFIG.source)
            self.replay_path = QtWidgets.QLineEdit(CONFIG.replay_path)
            self.frequency = QtWidgets.QDoubleSpinBox()
            self.frequency.setRange(1, 6000)
            self.frequency.setDecimals(6)
            self.frequency.setValue(CONFIG.frequency_mhz)
            self.frequency.setSuffix(" MHz")
            self.gain = QtWidgets.QSpinBox()
            self.gain.setRange(-50, 100)
            self.gain.setValue(CONFIG.gain_db)
            self.gain.setSuffix(" dB")
            self.bandwidth = QtWidgets.QDoubleSpinBox()
            self.bandwidth.setRange(0.1, 60)
            self.bandwidth.setDecimals(2)
            self.bandwidth.setValue(CONFIG.bandwidth_mhz)
            self.bandwidth.setSuffix(" MHz")
            self.port = QtWidgets.QComboBox()
            self.port.addItems(["RX1", "RX2"])
            self.port.setCurrentIndex(CONFIG.rx_port - 1)
            self.mode = QtWidgets.QComboBox()
            self.mode.addItems(["Live/latest", "Every window"])
            self.mode.setCurrentText(CONFIG.inference_mode)
            self.metadata_check = QtWidgets.QCheckBox("Use metadata gap checks")
            self.metadata_check.setChecked(CONFIG.use_metadata)
            self.metadata_check.setToolTip("Uncheck to test plain SC16 Q11 like the reference RX script. "
                                          "USB sample continuity cannot be verified in plain mode.")
            form.addRow("Source", self.source)
            form.addRow("Model", self.path_picker(self.model_path, "Saved model (*.npz)"))
            form.addRow("Replay file", self.path_picker(self.replay_path, "Raw complex I/Q (*.bin);;All files (*)"))
            form.addRow("Carrier", self.frequency)
            form.addRow("RX gain", self.gain)
            form.addRow("RX filter", self.bandwidth)
            form.addRow("Physical port", self.port)
            form.addRow("Inference mode", self.mode)
            form.addRow("RX format", self.metadata_check)
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
            self.stats_label = QtWidgets.QLabel("Ready. Select the model and match the HackRF carrier.")
            self.stats_label.setWordWrap(True)
            self.stats_label.setStyleSheet("color: #a9bbd3; padding: 10px; line-height: 140%;")
            controls.addWidget(self.stats_label)
            export = QtWidgets.QHBoxLayout()
            save_iq = QtWidgets.QPushButton("Save last I/Q")
            save_csv = QtWidgets.QPushButton("Export history")
            save_iq.clicked.connect(self.save_iq)
            save_csv.clicked.connect(self.export_history)
            export.addWidget(save_iq)
            export.addWidget(save_csv)
            controls.addLayout(export)
            controls.addStretch()

            right = QtWidgets.QVBoxLayout()
            split.addLayout(right, 1)
            cards = QtWidgets.QHBoxLayout()
            self.pw_card = self.card(cards, "ESTIMATED PW", "µs", "#50dfc2")
            self.bw_card = self.card(cards, "ESTIMATED BW", "kHz", "#56c7ff")
            self.rate_card = self.card(cards, "CHIRP RATE MAGNITUDE", "Hz/s", "#ffce76")
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
            self.iq_plot = plots.addPlot(row=0, col=0, colspan=2, title="Received I/Q · latest captured three-PRI window")
            self.iq_plot.setLabel("bottom", "Time", units="s")
            self.iq_plot.setLabel("left", "Amplitude")
            self.iq_plot.getAxis("left").enableAutoSIPrefix(False)
            self.iq_plot.showGrid(x=True, y=True, alpha=0.15)
            self.iq_plot.addLegend(offset=(10, 10))
            self.curve_i = self.iq_plot.plot(pen=pg.mkPen("#56c7ff", width=1), name="I")
            self.curve_q = self.iq_plot.plot(pen=pg.mkPen("#ff819c", width=1), name="Q")
            self.envelope = self.iq_plot.plot(pen=pg.mkPen("#50dfc2", width=1), name="|I+jQ|")
            self.spectrum_plot = plots.addPlot(row=1, col=0, colspan=2, title="Relative baseband spectrum · same receive window")
            self.spectrum_plot.setLabel("bottom", "Offset from receiver carrier", units="Hz")
            self.spectrum_plot.setLabel("left", "Magnitude", units="dB relative")
            self.spectrum_plot.showGrid(x=True, y=True, alpha=0.15)
            self.spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen("#ffce76", width=1))
            self.pw_plot = plots.addPlot(row=2, col=0, title="PW estimate history")
            self.bw_plot = plots.addPlot(row=2, col=1, title="BW estimate history")
            self.pw_plot.setLabel("left", "PW", units="µs")
            self.bw_plot.setLabel("left", "BW", units="kHz")
            self.pw_plot.getAxis("left").enableAutoSIPrefix(False)
            self.bw_plot.getAxis("left").enableAutoSIPrefix(False)
            for plot in (self.pw_plot, self.bw_plot):
                plot.setLabel("bottom", "Elapsed", units="s")
                plot.showGrid(x=True, y=True, alpha=0.15)
            self.pw_curve = self.pw_plot.plot(pen=pg.mkPen("#50dfc2", width=2))
            self.bw_curve = self.bw_plot.plot(pen=pg.mkPen("#56c7ff", width=2))
            self.log = QtWidgets.QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumBlockCount(100)
            self.log.setMaximumHeight(90)
            outer.addWidget(self.log)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.refresh)
            self.timer.start(CONFIG.display_interval_ms)

        def path_picker(self, entry, pattern):
            widget = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(entry)
            button = QtWidgets.QPushButton("…")
            button.setFixedWidth(35)
            def browse():
                path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", entry.text(), pattern)
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
                f"PW within 50 ns: {snap['within_sample_percent']:.1f}%")

        def start_rx(self):
            if self.session is not None and self.session.alive:
                return
            settings = replace(CONFIG, model_path=self.model_path.text().strip(),
                               source=self.source.currentText(), replay_path=self.replay_path.text().strip(),
                               frequency_mhz=self.frequency.value(), gain_db=self.gain.value(),
                               bandwidth_mhz=self.bandwidth.value(), rx_port=self.port.currentIndex()+1,
                               use_metadata=self.metadata_check.isChecked(),
                               inference_mode=self.mode.currentText())
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
            self.log.appendPlainText(f"Starting {settings.source} in {settings.inference_mode} mode.")
            self.session.start()

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
            self.stats_label.setText(
                f"{stats['state']}\n"
                f"{stats['continuity']}\n"
                f"Captured windows: {stats['captured']:,}\n"
                f"Predictions: {stats['predicted']:,}  ·  {stats['prediction_hz']:.1f}/s\n"
                f"Skipped inference windows: {stats['skipped']:,}\n"
                f"Queue: {stats['queued']}  ·  RX gaps: {stats['gaps']}\n"
                f"Overrun flags: {stats['overruns']} · Timestamp jumps: {stats['timestamp_gaps']}\n"
                f"Rejected samples: {stats['rejected_samples']:,}\n"
                f"Read throughput: {stats['rx_msps']:.2f} MSPS\n"
                f"Read interval: {stats['read_interval_ms']:.1f} ms (max {stats['max_read_interval_ms']:.1f})\n"
                f"Inference: {stats['inference_ms']:.1f} ms\n"
                f"Prediction age: {stats['prediction_age_ms']:.1f} ms")
            if latest is not None:
                step = max(1, latest.iq.size // 3500)
                t = np.arange(0, latest.iq.size, step) / self.session.settings.fs
                plotted = latest.iq[::step]
                self.curve_i.setData(t, plotted.real)
                self.curve_q.setData(t, plotted.imag)
                self.envelope.setData(t, np.abs(plotted))
                fft_size = 8192
                spectrum = np.abs(np.fft.fftshift(np.fft.fft(latest.iq[:fft_size] * np.hanning(fft_size))))
                relative = 20*np.log10(np.maximum(spectrum, 1e-12) / max(float(spectrum.max()), 1e-12))
                frequencies = np.fft.fftshift(np.fft.fftfreq(fft_size, 1/self.session.settings.fs))
                self.spectrum_curve.setData(frequencies, relative)
            if result is not None:
                frame = result["frame"]
                self.pw_card.setText(f"{result['pw_us']:.6f}")
                self.bw_card.setText(f"{result['bw_khz']:.3f}")
                self.rate_card.setText(f"{result['chirp_rate']:.3e}" if np.isfinite(result["chirp_rate"]) else "Invalid")
                self.quality_label.setText(f"{result['quality']}  ·  prediction window #{frame.sequence}  ·  "
                                          f"power {result['power_dbfs']:.1f} dBFS  ·  clipping {result['clipping_percent']:.3f}%  ·  "
                                          f"pulse/noise {result['pulse_contrast_db']:.1f} dB  ·  duty {result['duty_percent']:.1f}%  ·  "
                                          f"offset {result['offset_khz']:+.1f} kHz")
                if frame.sequence != self.seen_sequence:
                    self.seen_sequence = frame.sequence
                    record = dict(elapsed_s=result["completed_at"] - self.session.started_at,
                                             sequence=frame.sequence, pw_us=result["pw_us"],
                                             bw_khz=result["bw_khz"], chirp_rate_Hz_per_s=result["chirp_rate"],
                                             pulse_contrast_db=result["pulse_contrast_db"],
                                             duty_percent=result["duty_percent"], offset_khz=result["offset_khz"],
                                             inference_ms=result["inference_ms"], quality=result["quality"])
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

        def save_iq(self):
            if self.session is None or self.session.snapshot()[2] is None:
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save latest captured window", "rx_window.npy", "NumPy I/Q (*.npy)")
            if path:
                frame = self.session.snapshot()[2]
                np.save(path, frame.iq)
                self.log.appendPlainText("Saved latest captured I/Q window: " + path)

        def export_history(self):
            if not self.records:
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export displayed prediction history", "rx_predictions.csv", "CSV (*.csv)")
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

