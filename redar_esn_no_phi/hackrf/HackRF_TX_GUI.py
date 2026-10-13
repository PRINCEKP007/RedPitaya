#!/usr/bin/env python3
"""Interactive, continuous pulsed-LFM transmitter for HackRF One.

Run in the same Python environment as the supplied HackRF_PW_Ch.py:
    python HackRF_TX_GUI.py
    python HackRF_TX_GUI.py --simulate
    python HackRF_TX_GUI.py --self-test

Requires numpy, matplotlib and tkinter. Hardware mode additionally requires
the python_hackrf package and libhackrf used by the reference script.
There is no dependency on the reference script's lib.Generate_Signal.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import multiprocessing as mp
import sys
import threading
import time
from pathlib import Path

import numpy as np

from SDR_UDP_Reference import UDP_PORT, UDPPublisher


MAX_PERIOD_SAMPLES = 2_000_000
FILTER_BWS = (1.75, 2.5, 3.5, 5, 5.5, 6, 7, 8, 9, 10, 12, 14, 15, 20, 24, 28)


@dataclasses.dataclass(frozen=True)
class TxSettings:
    # Defaults match the trained ESN: 1 GHz, 20 MSPS, fixed 512 us PRI,
    # PW 300-350 us and BW 3-4 MHz (mid-range shown).
    carrier_mhz: float = 1000.0
    sample_rate_msps: float = 20.0
    pri_us: float = 512.0
    pw_us: float = 325.0
    bandwidth_khz: float = 3500.0
    offset_khz: float = 0.0
    gain_db: int = 16
    amp_enabled: bool = False
    amplitude: float = 0.8
    direction: str = "Up chirp"
    filter_mhz: str = "Auto"
    mode: str = "Hardware"

    @property
    def fs(self):
        return self.sample_rate_msps * 1e6

    @property
    def period_samples(self):
        return int(round(self.pri_us * 1e-6 * self.fs))

    @property
    def pulse_samples(self):
        return int(round(self.pw_us * 1e-6 * self.fs))

    @property
    def actual_pri_us(self):
        return self.period_samples / self.fs * 1e6

    @property
    def actual_pw_us(self):
        return self.pulse_samples / self.fs * 1e6

    @property
    def filter_hz(self):
        if self.filter_mhz != "Auto":
            return int(float(self.filter_mhz) * 1e6)
        # Use the documented hackrf_transfer convention, rounded DOWN to
        # a supported filter. The GUI validates that the chirp fits it.
        target = 0.75 * self.sample_rate_msps
        return int(max(bw for bw in FILTER_BWS if bw <= target) * 1e6)

    def validate(self):
        numeric = (self.carrier_mhz, self.sample_rate_msps, self.pri_us,
                   self.pw_us, self.bandwidth_khz, self.offset_khz,
                   self.amplitude, self.gain_db)
        if not all(math.isfinite(x) for x in numeric):
            raise ValueError("Enter finite numeric values in every field.")
        if not 1 <= self.carrier_mhz <= 6000:
            raise ValueError("Carrier frequency must be between 1 and 6000 MHz.")
        if not 8 <= self.sample_rate_msps <= 20:
            raise ValueError("Use a sample rate between 8 and 20 MSPS for HackRF One.")
        if not 0 < self.pw_us < self.pri_us:
            raise ValueError("Pulse width must be positive and smaller than PRI.")
        if not 0 < self.bandwidth_khz:
            raise ValueError("Chirp bandwidth must be greater than zero.")
        if not 0 < self.amplitude <= 1:
            raise ValueError("DAC amplitude must be greater than 0 and at most 1.")
        if round(127 * self.amplitude) < 1:
            raise ValueError("DAC amplitude is too small for a nonzero 8-bit waveform. Use at least 0.004.")
        if int(self.gain_db) != self.gain_db or not 0 <= self.gain_db <= 47:
            raise ValueError("TX VGA gain must be an integer from 0 to 47 dB.")
        if self.direction not in ("Up chirp", "Down chirp"):
            raise ValueError("Choose Up chirp or Down chirp.")
        if self.mode not in ("Hardware", "Simulation"):
            raise ValueError("Choose Hardware or Simulation.")
        if self.filter_mhz != "Auto" and float(self.filter_mhz) not in FILTER_BWS:
            raise ValueError("Choose a supported baseband filter bandwidth.")
        edge = abs(self.offset_khz * 1e3) + self.bandwidth_khz * 1e3 / 2
        if edge >= self.fs / 2:
            raise ValueError("The chirp exceeds the sampled band. Reduce BW/offset or increase sample rate.")
        if edge >= self.filter_hz / 2:
            raise ValueError("The chirp exceeds the selected filter passband. Increase the filter BW or reduce BW/offset.")
        rf_low = self.carrier_mhz * 1e6 + self.offset_khz * 1e3 - self.bandwidth_khz * 500
        rf_high = rf_low + self.bandwidth_khz * 1e3
        if rf_low < 1e6 or rf_high > 6e9:
            raise ValueError("The complete RF chirp must lie between 1 MHz and 6 GHz.")
        if self.pri_us > MAX_PERIOD_SAMPLES / self.fs * 1e6:
            raise ValueError(f"PRI is too large: maximum is {MAX_PERIOD_SAMPLES / self.fs * 1e6:g} us at this sample rate.")
        if not 2 <= self.pulse_samples < self.period_samples:
            raise ValueError("PW/PRI round to invalid sample counts. Increase their separation.")
        if self.period_samples > MAX_PERIOD_SAMPLES:
            raise ValueError(f"PRI is too large: maximum is {MAX_PERIOD_SAMPLES / self.fs * 1e6:g} us at this sample rate.")
        return self

    def metadata(self):
        return {
            "requested": dataclasses.asdict(self),
            "actual": {
                "sample_rate_hz": self.fs,
                "carrier_hz": int(round(self.carrier_mhz * 1e6)),
                "pri_us": self.actual_pri_us,
                "pw_us": self.actual_pw_us,
                "period_samples": self.period_samples,
                "pulse_samples": self.pulse_samples,
                "prf_hz": self.fs / self.period_samples,
                "duty_cycle_percent": 100 * self.pulse_samples / self.period_samples,
                "chirp_rate_hz_per_s": (1 if self.direction == "Up chirp" else -1)
                * self.bandwidth_khz * 1e3 / (self.pulse_samples / self.fs),
                "baseband_filter_hz": self.filter_hz,
                "dac_peak_code": int(round(127 * self.amplitude)),
            },
            "waveform": "LFM chirp, phase reset at each pulse, followed by zero IQ",
            "timing_note": "Durations are rounded to samples, not USB buffer multiples. Host timestamps are not RF arrival timestamps.",
        }


def make_period(settings: TxSettings):
    """Build exactly one PRI as interleaved signed-int8 IQ.

    f(t) = offset - BW/2 + BW*t/PW for an up chirp.
    Phase resets to zero for every pulse; zero IQ follows until the next PRI.
    """
    settings.validate()
    n = settings.pulse_samples
    t = np.arange(n, dtype=np.float64) / settings.fs
    pw = n / settings.fs
    sign = 1 if settings.direction == "Up chirp" else -1
    bw = settings.bandwidth_khz * 1e3
    f0 = settings.offset_khz * 1e3 - sign * bw / 2
    phase = 2 * np.pi * (f0 * t + 0.5 * sign * bw / pw * t * t)
    peak = round(127 * settings.amplitude)
    iq = np.zeros(2 * settings.period_samples, dtype=np.int8)
    iq[:2*n:2] = np.rint(peak * np.cos(phase)).astype(np.int8)
    iq[1:2*n:2] = np.rint(peak * np.sin(phase)).astype(np.int8)
    return iq


class ContinuousIQ:
    """Copy a periodic IQ byte stream across arbitrary USB buffer boundaries.

    The cached tile contains whole PRIs. No file IO, plotting, waveform
    generation, locks or GUI calls occur in the hardware TX callback.
    """

    def __init__(self, period):
        period = np.asarray(period, dtype=np.int8)
        if period.ndim != 1 or not period.size or period.size % 2:
            raise ValueError("The IQ period must contain pairs of int8 samples.")
        self.period_bytes = len(period)
        self.tile = np.tile(period, max(1, math.ceil(524288 / len(period))))
        self.offset = 0
        self.total_bytes = 0

    def fill(self, buffer, length):
        pos = 0
        while pos < length:
            count = min(length - pos, len(self.tile) - self.offset)
            buffer[pos:pos+count] = self.tile[self.offset:self.offset+count]
            self.offset = (self.offset + count) % len(self.tile)
            pos += count
        self.total_bytes += length


def check_result(result, operation):
    """Bindings either raise on failure or return a libhackrf status code."""
    if isinstance(result, (int, np.integer)) and result != 0:
        raise RuntimeError(f"{operation} failed (libhackrf status {result}).")


def transmit_worker(connection, config):
    """Own hardware in a separate process; GUI remains responsive on errors."""
    settings = TxSettings(**config)
    device = None
    api = None
    initialized = False
    tx_started = False
    stop_event = threading.Event()
    callback_error = []
    cleanup_errors = []
    started = None

    def send(kind, **data):
        try:
            connection.send({"kind": kind, **data})
        except (BrokenPipeError, EOFError, OSError):
            stop_event.set()

    try:
        settings.validate()
        send("log", text="Preparing a continuous pulsed chirp stream...")
        stream = ContinuousIQ(make_period(settings))
        if connection.poll() and connection.recv() == "stop":
            return
        if settings.mode == "Hardware":
            try:
                from python_hackrf import pyhackrf as api
            except (ImportError, OSError) as exc:
                raise RuntimeError(
                    "Cannot load python_hackrf/libhackrf. Run this GUI with the same Python environment "
                    "as your working transmitter script, or choose Simulation. Details: " + str(exc)
                ) from exc
            check_result(api.pyhackrf_init(), "HackRF initialization")
            initialized = True
            device = api.pyhackrf_open()
            if device is None:
                raise RuntimeError("No HackRF opened. Check USB, permissions, and other SDR applications.")
            # No device methods are called from the GUI or the TX callback.
            for operation, value in (
                ("pyhackrf_set_sample_rate", settings.fs),
                ("pyhackrf_set_freq", int(round(settings.carrier_mhz * 1e6))),
                ("pyhackrf_set_baseband_filter_bandwidth", settings.filter_hz),
                ("pyhackrf_set_txvga_gain", settings.gain_db),
                ("pyhackrf_set_amp_enable", settings.amp_enabled),
            ):
                check_result(getattr(device, operation)(value), operation)

            def callback(_device, buffer, buffer_length, valid_length):
                try:
                    if stop_event.is_set():
                        buffer[:buffer_length] = 0
                        if hasattr(valid_length, "value"):
                            valid_length.value = 0
                        return -1
                    stream.fill(buffer, buffer_length)
                    # Current python_hackrf passes a ctypes.c_int here. It
                    # copies exactly this many bytes back to libhackrf.
                    if hasattr(valid_length, "value"):
                        valid_length.value = buffer_length
                    return 0
                except Exception as exc:
                    # The callback's failure is relayed by the worker, outside
                    # the callback. Stop must never be called inside a callback.
                    callback_error.append(str(exc))
                    stop_event.set()
                    if hasattr(valid_length, "value"):
                        valid_length.value = 0
                    return -1

            device.set_tx_callback(callback)
            # Stop clicked during initialization must prevent RF startup.
            if connection.poll() and connection.recv() == "stop":
                return
            check_result(device.pyhackrf_start_tx(), "Starting TX")
            tx_started = True

        started = time.monotonic()
        if settings.mode == "Simulation" and connection.poll() and connection.recv() == "stop":
            return
        send("running", actual=settings.metadata()["actual"], mode=settings.mode)
        send("log", text=("HackRF transmission started." if settings.mode == "Hardware"
                           else "SIMULATION started. No RF is transmitted."))
        last_report = started
        sim_buffer = np.empty(262144, dtype=np.int8)
        sim_next = started
        while not stop_event.is_set():
            # Wait on control commands, not on the GUI thread. At most 50 ms
            # until a normal stop request is noticed.
            wait_time = 0.05 if settings.mode == "Hardware" else max(0, sim_next - time.monotonic())
            if connection.poll(min(0.05, wait_time)):
                if connection.recv() == "stop":
                    stop_event.set()
                    break
            if settings.mode == "Hardware":
                if not device.pyhackrf_is_streaming():
                    raise RuntimeError("HackRF streaming stopped unexpectedly. Check USB, sample rate, and device connection.")
            elif time.monotonic() >= sim_next:
                stream.fill(sim_buffer, len(sim_buffer))
                sim_next += len(sim_buffer) / (2 * settings.fs)
                # A delayed simulation must not spin endlessly to catch up.
                sim_next = max(sim_next, time.monotonic())
            now = time.monotonic()
            if now - last_report >= 0.2:
                send("stats", elapsed=now-started, samples=stream.total_bytes // 2,
                     periods=stream.total_bytes // stream.period_bytes)
                last_report = now
        if callback_error:
            raise RuntimeError("TX callback failed: " + callback_error[0])
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        send("error", text=str(exc))
    finally:
        stop_event.set()
        if device is not None:
            if tx_started:
                try:
                    check_result(device.pyhackrf_stop_tx(), "Stopping TX")
                except Exception as exc:
                    cleanup_errors.append(str(exc))
            try:
                check_result(device.pyhackrf_close(), "Closing HackRF")
            except Exception as exc:
                cleanup_errors.append(str(exc))
        if initialized:
            try:
                check_result(api.pyhackrf_exit(), "Exiting libhackrf")
            except Exception as exc:
                cleanup_errors.append(str(exc))
        if cleanup_errors:
            send("error", text="Device cleanup failed: " + "; ".join(cleanup_errors))
        send("finished", clean=not cleanup_errors)
        connection.close()


class TransmitterGUI:
    BG = "#101720"
    PANEL = "#182330"
    FG = "#edf3fa"
    MUTED = "#94a8bd"
    BLUE = "#54b4ff"

    def __init__(self, root, simulate=False):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.tk, self.ttk = tk, ttk
        self.root = root
        self.udp_publisher = UDPPublisher()
        self.process = None
        self.connection = None
        self.state = "idle"
        self.closing = False
        self.last_error = None
        self.finished_received = False
        self.stop_requested_at = None
        self.pending_preview = None
        self.active_settings = None
        self.editors = []
        self.vars = {}
        self.metric_vars = {}
        self.log_lines = 0
        self.trace_suspended = False

        root.title("HackRF | Pulsed Chirp Transmitter")
        root.geometry("1240x960")
        root.minsize(1100, 900)
        root.configure(bg=self.BG)
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background=self.BG)
        style.configure("Panel.TFrame", background=self.PANEL)
        style.configure("TLabel", background=self.PANEL, foreground=self.FG, font=("DejaVu Sans", 10))
        style.configure("Muted.TLabel", foreground=self.MUTED)
        style.configure("TEntry", fieldbackground="#223345", foreground=self.FG,
                        insertcolor=self.FG, borderwidth=0, padding=6)
        style.map("TEntry", fieldbackground=[("disabled", "#1b2938")], foreground=[("disabled", "#7890a6")])
        style.configure("TCombobox", fieldbackground="#223345", background="#223345", foreground=self.FG, padding=5)
        style.map("TCombobox", fieldbackground=[("readonly", "#223345"), ("disabled", "#1b2938")],
                  foreground=[("readonly", self.FG), ("disabled", "#7890a6")])
        root.option_add("*TCombobox*Listbox.background", "#223345")
        root.option_add("*TCombobox*Listbox.foreground", self.FG)
        style.configure("TCheckbutton", background=self.PANEL, foreground=self.FG, font=("DejaVu Sans", 10))
        style.map("TCheckbutton", background=[("active", self.PANEL)], foreground=[("disabled", "#7890a6")])
        style.configure("TButton", padding=(10, 8), font=("DejaVu Sans", 10), foreground=self.FG, background="#2b4055")
        style.map("TButton", background=[("active", "#37546e"), ("disabled", "#253443")],
                  foreground=[("disabled", "#63798d")])
        style.configure("Start.TButton", background="#207c66", font=("DejaVu Sans", 11, "bold"))
        style.map("Start.TButton", background=[("active", "#2b987d"), ("disabled", "#253443")])
        style.configure("Stop.TButton", background="#a14e3b", font=("DejaVu Sans", 11, "bold"))
        style.map("Stop.TButton", background=[("active", "#bf5a43"), ("disabled", "#253443")])

        header = tk.Frame(root, bg=self.BG)
        header.pack(fill="x", padx=24, pady=(20, 14))
        tk.Label(header, text="HackRF Transmitter", bg=self.BG, fg=self.FG,
                 font=("DejaVu Sans", 23, "bold")).pack(side="left")
        self.status_var = tk.StringVar(value="IDLE")
        self.status_label = tk.Label(header, textvariable=self.status_var, bg="#223345", fg=self.MUTED,
                                     padx=16, pady=8, font=("DejaVu Sans", 11, "bold"))
        self.status_label.pack(side="right")
        tk.Label(root, text="Configure a pulsed LFM chirp, preview the waveform, and stream it continuously.",
                 bg=self.BG, fg=self.MUTED, font=("DejaVu Sans", 10)).pack(anchor="w", padx=25, pady=(0, 16))

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True, padx=24)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        left = ttk.Frame(body, style="Panel.TFrame", padding=18)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        left.columnconfigure(1, weight=1)
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        defaults = dataclasses.asdict(TxSettings(mode="Simulation" if simulate else "Hardware"))
        row = 0

        def section(text):
            nonlocal row
            ttk.Label(left, text=text, foreground=self.BLUE, font=("DejaVu Sans", 10, "bold")).grid(
                row=row, column=0, columnspan=2, sticky="w", pady=(8 if row else 0, 8))
            row += 1

        def field(key, label, values=None):
            nonlocal row
            ttk.Label(left, text=label).grid(row=row, column=0, sticky="w", padx=(0, 14), pady=5)
            var = tk.StringVar(value=str(defaults[key]))
            self.vars[key] = var
            if values:
                widget = ttk.Combobox(left, textvariable=var, values=values, state="readonly", width=17)
            else:
                widget = ttk.Entry(left, textvariable=var, width=19)
            widget.grid(row=row, column=1, sticky="ew", pady=5)
            self.editors.append((widget, "readonly" if values else "normal"))
            var.trace_add("write", self.schedule_preview)
            row += 1

        section("RADIO")
        field("mode", "Output mode", ("Hardware", "Simulation"))
        field("carrier_mhz", "Carrier (MHz)")
        field("sample_rate_msps", "Sample rate (MSPS)", ("8", "10", "12.5", "16", "20"))
        field("gain_db", "TX VGA gain (dB)")
        field("amplitude", "DAC amplitude (0–1)")
        self.vars["amp_enabled"] = tk.BooleanVar(value=defaults["amp_enabled"])
        amp = ttk.Checkbutton(left, text="Enable RF amplifier", variable=self.vars["amp_enabled"])
        amp.grid(row=row, column=0, columnspan=2, sticky="w", pady=(5, 0))
        self.editors.append((amp, "normal"))
        self.vars["amp_enabled"].trace_add("write", self.schedule_preview)
        row += 1
        ttk.Label(left, text="Gain setting is not output power in dBm.", style="Muted.TLabel",
                  font=("DejaVu Sans", 9)).grid(row=row, column=0, columnspan=2, sticky="w", pady=(2, 8))
        row += 1
        section("PULSE TRAIN")
        field("pri_us", "PRI (µs)")
        field("pw_us", "Pulse width (µs)")
        field("bandwidth_khz", "Chirp BW (kHz)")
        field("offset_khz", "Baseband offset (kHz)")
        field("direction", "Sweep direction", ("Up chirp", "Down chirp"))
        field("filter_mhz", "Filter BW (MHz)", ("Auto",) + tuple(str(x) for x in FILTER_BWS))
        row += 1
        buttons = ttk.Frame(left, style="Panel.TFrame")
        buttons.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(14, 8))
        buttons.columnconfigure((0, 1), weight=1)
        self.start_button = ttk.Button(buttons, text="Start TX", style="Start.TButton", command=self.start)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.stop_button = ttk.Button(buttons, text="Stop", style="Stop.TButton", command=self.stop, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        row += 1
        self.export_button = ttk.Button(left, text="Export settings / metadata", command=self.export_settings)
        self.export_button.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)
        row += 1
        ttk.Label(left, text="Stop → edit parameters → Start TX\nThe preview uses sample-rounded timing.",
                  style="Muted.TLabel", font=("DejaVu Sans", 9)).grid(row=row, column=0, columnspan=2,
                                                                       sticky="w", pady=(8, 0))

        metrics = ttk.Frame(right, style="Panel.TFrame", padding=(16, 12))
        metrics.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        for col, (key, label) in enumerate((("pw", "ACTUAL PW"), ("pri", "ACTUAL PRI"),
                                          ("prf", "PRF"), ("duty", "DUTY CYCLE"))):
            metrics.columnconfigure(col, weight=1)
            ttk.Label(metrics, text=label, style="Muted.TLabel", font=("DejaVu Sans", 9)).grid(row=0, column=col, sticky="w")
            var = tk.StringVar(value="—")
            self.metric_vars[key] = var
            ttk.Label(metrics, textvariable=var, font=("DejaVu Sans", 14, "bold")).grid(row=1, column=col, sticky="w", pady=(5, 0))

        plot_panel = ttk.Frame(right, style="Panel.TFrame", padding=12)
        plot_panel.grid(row=1, column=0, sticky="nsew")
        plot_panel.rowconfigure(1, weight=1)
        plot_panel.columnconfigure(0, weight=1)
        ttk.Label(plot_panel, text="WAVEFORM PREVIEW", foreground=self.BLUE,
                  font=("DejaVu Sans", 10, "bold")).grid(row=0, column=0, sticky="w", pady=(2, 6))
        self.figure = Figure(figsize=(6.8, 3.8), dpi=100, facecolor=self.PANEL)
        self.axes = self.figure.subplots(2, 1, sharex=True)
        self.figure.subplots_adjust(left=0.13, right=0.96, top=0.94, bottom=0.14, hspace=0.28)
        for ax in self.axes:
            ax.set_facecolor(self.PANEL)
            ax.tick_params(colors=self.MUTED, labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#35475a")
            ax.grid(color="#2e4053", alpha=0.6, linewidth=0.6)
        self.axes[0].set_ylabel("Envelope\n(normalized)", color=self.MUTED, fontsize=9)
        self.axes[1].set_ylabel("Baseband\nfrequency (kHz)", color=self.MUTED, fontsize=9)
        self.axes[1].set_xlabel("Time (µs)", color=self.MUTED, fontsize=9)
        self.envelope_line, = self.axes[0].plot([], [], color=self.BLUE, linewidth=2)
        self.frequency_line, = self.axes[1].plot([], [], color="#61d2ac", linewidth=2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_panel)
        self.canvas.get_tk_widget().configure(bg=self.PANEL, highlightthickness=0)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self.preview_note = tk.StringVar(value="")
        ttk.Label(plot_panel, textvariable=self.preview_note, style="Muted.TLabel", wraplength=610,
                  font=("DejaVu Sans", 9)).grid(row=2, column=0, sticky="w", pady=(6, 0))

        telemetry = ttk.Frame(right, style="Panel.TFrame", padding=12)
        telemetry.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self.udp_enabled_var = tk.BooleanVar(value=True)
        self.udp_host_var = tk.StringVar(value="127.0.0.1")
        self.udp_port_var = tk.StringVar(value=str(UDP_PORT))
        self.udp_status_var = tk.StringVar(value="UDP ready · enter the RX PC's Ethernet IPv4 address")
        enabled = ttk.Checkbutton(telemetry, text="Send TX reference by UDP", variable=self.udp_enabled_var)
        enabled.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))
        ttk.Label(telemetry, text="RX PC IP").grid(row=1, column=0, sticky="w")
        host = ttk.Entry(telemetry, textvariable=self.udp_host_var, width=19)
        host.grid(row=1, column=1, sticky="ew", padx=(8, 16))
        ttk.Label(telemetry, text="UDP port").grid(row=1, column=2)
        port = ttk.Entry(telemetry, textvariable=self.udp_port_var, width=7)
        port.grid(row=1, column=3, padx=(8, 0))
        telemetry.columnconfigure(1, weight=1)
        ttk.Label(telemetry, textvariable=self.udp_status_var, style="Muted.TLabel",
                  wraplength=600).grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))
        self.editors.extend([(enabled, "normal"), (host, "normal"), (port, "normal")])

        log_panel = ttk.Frame(right, style="Panel.TFrame", padding=12)
        log_panel.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        self.stats_var = tk.StringVar(value="Elapsed 00:00:00  •  IQ samples queued: 0")
        ttk.Label(log_panel, textvariable=self.stats_var, style="Muted.TLabel", font=("DejaVu Sans", 9)).pack(anchor="w", pady=(0, 8))
        self.log_widget = tk.Text(log_panel, height=5, bg="#111c28", fg=self.MUTED, relief="flat",
                                  font=("DejaVu Sans Mono", 9), padx=8, pady=8, wrap="word", state="disabled")
        self.log_widget.pack(fill="x")
        footer = tk.Frame(root, bg=self.BG)
        # Reserve footer space before giving the remaining height to the body.
        footer.pack(fill="x", padx=25, pady=(12, 15), before=body)
        self.validation_var = tk.StringVar(value="")
        self.validation_label = tk.Label(footer, textvariable=self.validation_var, bg=self.BG,
                                          fg=self.MUTED, anchor="w", wraplength=1120, font=("DejaVu Sans", 9))
        self.validation_label.pack(fill="x")
        self.log("Ready. Preview is generated locally; click Start TX to begin.")
        self.preview()
        root.after(100, self.poll_worker)

    def read_settings(self):
        raw = {key: var.get() for key, var in self.vars.items()}
        try:
            for key in ("carrier_mhz", "sample_rate_msps", "pri_us", "pw_us", "bandwidth_khz", "offset_khz", "amplitude"):
                raw[key] = float(raw[key])
            gain = float(raw["gain_db"])
            if not math.isfinite(gain) or not gain.is_integer():
                raise ValueError("TX VGA gain must be a whole number of dB.")
            raw["gain_db"] = int(gain)
            return TxSettings(**raw).validate()
        except (TypeError, OverflowError) as exc:
            raise ValueError("Check that all parameters contain valid numbers.") from exc
        except ValueError as exc:
            if "could not convert" in str(exc):
                raise ValueError("Enter a number in each numeric parameter field.") from exc
            raise

    def log(self, text):
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", f"[{stamp}] {text}\n")
        self.log_lines += 1
        if self.log_lines > 250:
            self.log_widget.delete("1.0", "2.0")
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

    def schedule_preview(self, *_):
        if self.trace_suspended:
            return
        if self.pending_preview:
            self.root.after_cancel(self.pending_preview)
        self.pending_preview = self.root.after(220, self.preview)

    def preview(self):
        self.pending_preview = None
        try:
            settings = self.read_settings()
        except ValueError as exc:
            self.validation_var.set(str(exc))
            self.validation_label.configure(fg="#ffb08c")
            self.preview_note.set("Invalid parameters — update the fields to refresh the preview.")
            if self.state == "idle":
                self.start_button.configure(state="disabled")
            return
        if self.state == "idle":
            self.start_button.configure(state="normal", text="Start simulation" if settings.mode == "Simulation" else "Start TX")
        pw, pri = settings.actual_pw_us, settings.actual_pri_us
        self.metric_vars["pw"].set(f"{pw:.4f} µs")
        self.metric_vars["pri"].set(f"{pri:.4f} µs")
        self.metric_vars["prf"].set(f"{1e6 / pri:,.2f} Hz")
        self.metric_vars["duty"].set(f"{100 * pw / pri:.2f}%")
        # Plot the exact sample-rounded timing without allocating the full
        # waveform in the GUI or overwhelming matplotlib with MHz IQ samples.
        envelope_t = [0, pw, pw, pri, pri, pri+pw, pri+pw, 2*pri]
        peak = round(127 * settings.amplitude) / 127
        self.envelope_line.set_data(envelope_t, [peak, peak, 0, 0, peak, peak, 0, 0])
        sign = 1 if settings.direction == "Up chirp" else -1
        f0 = settings.offset_khz - sign * settings.bandwidth_khz / 2
        f1 = settings.offset_khz + sign * settings.bandwidth_khz / 2
        # Last transmitted pulse sample is at PW - Ts, not at PW itself.
        last = pw - 1e6 / settings.fs
        f_last = f0 + (f1 - f0) * last / pw
        self.frequency_line.set_data([0, last, np.nan, pri, pri+last], [f0, f_last, np.nan, f0, f_last])
        self.axes[0].set_xlim(0, 2 * pri)
        self.axes[0].set_ylim(-0.08, 1.13)
        span = max(settings.bandwidth_khz, 1)
        self.axes[1].set_ylim(min(f0, f1)-0.2*span, max(f0, f1)+0.2*span)
        self.canvas.draw_idle()
        self.preview_note.set(f"Two PRIs • {settings.pulse_samples:,} pulse samples / {settings.period_samples:,} samples per PRI\n"
                              f"Filter: {settings.filter_hz/1e6:g} MHz • Phase resets at each pulse • Plot is a generated preview")
        changed = abs(pw - settings.pw_us) > 1e-9 or abs(pri - settings.pri_us) > 1e-9
        message = (f"Sample interval: {1e9/settings.fs:g} ns. "
                   + ("PW/PRI rounded to samples; actual values are shown above. " if changed else "")
                   + ("SIMULATION selected — no RF output." if settings.mode == "Simulation" else "Parameters valid. Ready for HackRF transmission."))
        self.validation_var.set(message)
        self.validation_label.configure(fg=self.MUTED)

    def set_busy(self, busy):
        for widget, normal_state in self.editors:
            widget.configure(state="disabled" if busy else normal_state)
        self.start_button.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(state="normal" if busy else "disabled")
        self.export_button.configure(state="disabled" if busy else "normal")

    def start(self):
        from tkinter import messagebox
        if self.state != "idle":
            return
        try:
            settings = self.read_settings()
            self.udp_publisher.configure(self.udp_enabled_var.get(), self.udp_host_var.get(),
                                         self.udp_port_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc), parent=self.root)
            return
        self.active_settings = settings
        self.udp_publisher.update("starting", settings, new_run=True)
        self.last_error = None
        self.finished_received = False
        self.stop_requested_at = None
        self.stats_var.set("Elapsed 00:00:00  •  IQ samples queued: 0")
        self.state = "starting"
        self.status_var.set("STARTING")
        self.status_label.configure(fg=self.BLUE)
        self.set_busy(True)
        self.log(f"Starting {settings.mode.lower()}: Fc {settings.carrier_mhz:g} MHz, PW {settings.actual_pw_us:g} µs, PRI {settings.actual_pri_us:g} µs.")
        parent, child = mp.get_context("spawn").Pipe()
        process = mp.get_context("spawn").Process(target=transmit_worker, args=(child, dataclasses.asdict(settings)), daemon=True)
        try:
            process.start()
        except Exception as exc:
            parent.close()
            child.close()
            self.state = "idle"
            self.set_busy(False)
            self.status_var.set("ERROR")
            self.status_label.configure(fg="#ffb08c")
            self.udp_publisher.update("error")
            self.log(str(exc))
            messagebox.showerror("Cannot start transmitter", str(exc), parent=self.root)
            return
        child.close()
        self.process, self.connection = process, parent

    def stop(self):
        if self.state not in ("starting", "running"):
            return
        self.state = "stopping"
        self.udp_publisher.update("stopping")
        self.status_var.set("STOPPING")
        self.status_label.configure(fg="#ffbf69")
        self.stop_button.configure(state="disabled")
        self.stop_requested_at = time.monotonic()
        self.log("Stop requested. Waiting for the transmitter to close...")
        try:
            self.connection.send("stop")
        except (BrokenPipeError, EOFError, OSError):
            pass

    def drain_messages(self):
        if self.connection:
            try:
                while self.connection.poll():
                    message = self.connection.recv()
                    kind = message["kind"]
                    if kind == "log":
                        self.log(message["text"])
                    elif kind == "running":
                        if self.state != "stopping":
                            self.state = "running"
                            self.udp_publisher.update("running")
                            simulation = message["mode"] == "Simulation"
                            self.status_var.set("SIMULATION · NO RF" if simulation else "TRANSMITTING")
                            self.status_label.configure(fg="#ffbf69" if simulation else "#61d2ac")
                    elif kind == "stats":
                        elapsed = int(message["elapsed"])
                        self.stats_var.set(f"Elapsed {elapsed//3600:02d}:{elapsed//60%60:02d}:{elapsed%60:02d}  •  IQ samples queued: {message['samples']:,}")
                    elif kind == "error":
                        self.last_error = message["text"]
                        self.udp_publisher.update("error")
                        self.log("ERROR: " + self.last_error)
                    elif kind == "finished":
                        self.finished_received = True
            except (EOFError, BrokenPipeError, OSError):
                pass

    def poll_worker(self):
        from tkinter import messagebox
        self.drain_messages()
        self.udp_status_var.set(self.udp_publisher.snapshot())
        if self.process and self.stop_requested_at and self.process.is_alive():
            if time.monotonic() - self.stop_requested_at > 8:
                # A wedged C driver must not freeze the desktop forever.
                # Do not label a forced termination as a confirmed RF stop.
                self.last_error = "The driver did not finish stopping. The worker was terminated. RF stop is unconfirmed; disconnect HackRF before restarting."
                self.log("ERROR: " + self.last_error)
                self.process.terminate()
                self.stop_requested_at = None
        if self.process and not self.process.is_alive():
            self.process.join(timeout=0)
            # The child may exit between the first pipe drain and is_alive().
            # Drain its final messages before classifying the exit.
            self.drain_messages()
            if not self.finished_received and not self.last_error:
                self.last_error = "Transmitter worker exited unexpectedly; RF stop is unconfirmed. Check the device before restarting."
            self.process.close()
            self.process = None
            self.connection.close()
            self.connection = None
            self.state = "idle"
            self.udp_publisher.update("error" if self.last_error else "stopped")
            self.set_busy(False)
            if self.last_error:
                self.status_var.set("ERROR")
                self.status_label.configure(fg="#ffb08c")
                if not self.closing:
                    messagebox.showerror("Transmitter error", self.last_error, parent=self.root)
            else:
                self.status_var.set("STOPPED")
                self.status_label.configure(fg=self.MUTED)
                self.log("Stopped. You can now edit the settings and start again.")
            if self.closing:
                self.destroy_gui()
                return
            self.preview()
        if self.closing and not self.process:
            self.destroy_gui()
            return
        self.root.after(100, self.poll_worker)

    def export_settings(self):
        from tkinter import filedialog, messagebox
        try:
            metadata = self.read_settings().metadata()
            metadata["exported_at"] = dt.datetime.now().astimezone().isoformat()
            destination = filedialog.asksaveasfilename(parent=self.root, title="Export transmitter settings",
                                                      defaultextension=".json", initialfile="hackrf_tx_settings.json",
                                                      filetypes=[("JSON", "*.json")])
            if destination:
                Path(destination).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                self.log("Settings exported to " + destination)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Export failed", str(exc), parent=self.root)

    def destroy_gui(self):
        self.udp_publisher.close()
        self.root.destroy()

    def close(self):
        self.closing = True
        if self.process:
            self.stop()
        else:
            self.destroy_gui()


def self_test():
    """Checks signal timing and callback continuity without a GUI or SDR."""
    settings = TxSettings().validate()
    assert settings.period_samples == 10240
    assert settings.pulse_samples == 6500
    period = make_period(settings)
    assert len(period) == 20480 and period.dtype == np.int8
    assert np.all(period[2*settings.pulse_samples:] == 0)
    assert np.max(np.abs(period.astype(np.int16))) <= 127
    assert period[0] == round(127 * settings.amplitude)
    # Deliberately odd byte chunks exercise IQ continuity too.
    stream = ContinuousIQ(period)
    chunks = []
    for length in (1, 13, 262144, 333333, 19, 524289):
        chunk = np.empty(length, dtype=np.int8)
        stream.fill(chunk, length)
        chunks.append(chunk)
    actual = np.concatenate(chunks)
    np.testing.assert_array_equal(actual, np.resize(period, len(actual)))
    assert stream.total_bytes == len(actual)
    # PRI must NOT be changed to satisfy a USB alignment requirement.
    unusual = dataclasses.replace(settings, pri_us=513.13, pw_us=349.14)
    assert unusual.period_samples == round(unusual.pri_us * settings.sample_rate_msps)
    assert unusual.period_samples % 1024 != 0
    assert len(make_period(unusual)) == 2*unusual.period_samples
    down = make_period(dataclasses.replace(settings, direction="Down chirp"))
    np.testing.assert_array_equal(down[::2], period[::2])
    np.testing.assert_array_equal(down[1::2], -period[1::2])
    for change in ({"sample_rate_msps": 31.25}, {"gain_db": 48}, {"pw_us": 600},
                   {"bandwidth_khz": 17000}, {"offset_khz": 9000}, {"amplitude": 1.1},
                   {"sample_rate_msps": 7},
                   {"pri_us": float("nan")}, {"pw_us": 511.999, "pri_us": 512},
                   {"filter_mhz": "1.75", "bandwidth_khz": 2000}):
        try:
            dataclasses.replace(settings, **change).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid parameters accepted: {change}")
    print("PASS: exact sample timing, zero padding, int8 bounds, up/down chirps, arbitrary callback boundaries, and invalid parameter rejection.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--simulate", action="store_true", help="Select simulation mode at launch; does not transmit RF")
    parser.add_argument("--self-test", action="store_true", help="Run signal and streaming checks without GUI/hardware")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    try:
        import tkinter as tk
        import matplotlib
        matplotlib.use("TkAgg")
        root = tk.Tk()
    except Exception as exc:
        raise SystemExit("GUI needs tkinter, matplotlib and a graphical desktop. "
                         "On Ubuntu install python3-tk and run from your desktop. Details: " + str(exc)) from exc
    TransmitterGUI(root, simulate=args.simulate)
    root.mainloop()


if __name__ == "__main__":
    mp.freeze_support()
    main()
