#!/usr/bin/env python3
"""HackRF training transmitter with CHANGING PRI: sample-grid PW and PRI, stepped BW.

Each group holds one random (PW, BW, PRI) triple for R pulses, then all three
change. PRI and PW are on the 1/Fs sample grid; BW uses its own step in Hz.
Defaults: 1 MHz carrier, 20 MSPS.

Run: python HackRF_Training_TX_PRI_GUI.py [--simulate | --self-test]
Requires numpy, matplotlib, tkinter and the working python_hackrf driver.
Hardware and simulation both save planned labels and session metadata.
"""
from __future__ import annotations
import argparse
import csv
import dataclasses
import datetime as dt
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import json
import math
import multiprocessing as mp
import threading
import time
import uuid
from pathlib import Path
import numpy as np

MAX_PERIOD_SAMPLES = 2_000_000
MAX_DATASET_BYTES = 512 * 1024 * 1024
FILTER_BWS = (1.75, 2.5, 3.5, 5, 5.5, 6, 7, 8, 9, 10, 12, 14, 15, 20, 24, 28)

@dataclasses.dataclass(frozen=True)
class TxSettings:
    carrier_mhz: float = 1.0
    sample_rate_msps: float = 20.0
    pri_us: float = 512.0
    pw_us: float = 350.0
    bandwidth_khz: float = 100.0
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
        # The HackRF LO (carrier) must be 1 MHz-6 GHz (checked above). With a 1 MHz
        # carrier the chirp may dip below 1 MHz; it must stay above 0 Hz so it does
        # not fold through DC. The GUI warns when part of it is below 1 MHz.
        if rf_low <= 0 or rf_high > 6e9:
            raise ValueError("The complete RF chirp must lie above 0 Hz and below 6 GHz. "
                             "Reduce BW or add a positive baseband offset.")
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


def check_result(result, operation):
    """Bindings either raise on failure or return a libhackrf status code."""
    if isinstance(result, (int, np.integer)) and result != 0:
        raise RuntimeError(f"{operation} failed (libhackrf status {result}).")


@dataclasses.dataclass(frozen=True)
class TrainingSettings(TxSettings):
    pri_min_us: float = 400.0
    pri_max_us: float = 600.0
    pw_min_us: float = 300.0
    pw_max_us: float = 350.0
    bw_min_khz: float = 100.0
    bw_max_khz: float = 200.0
    bw_step_hz: float = 1000.0
    pri_repeat: int = 3
    pair_count: int = 5000
    seed: int = 42
    output_directory: str = "training_runs"

    def grid(self):
        # Decimal avoids accidentally adding/removing a boundary grid point
        # because of floating-point multiplication. Bounds are inclusive.
        rate = Decimal(str(self.sample_rate_msps))
        lo = int((Decimal(str(self.pw_min_us))*rate).to_integral_value(rounding=ROUND_CEILING))
        hi = int((Decimal(str(self.pw_max_us))*rate).to_integral_value(rounding=ROUND_FLOOR))
        bw_lo = Decimal(str(self.bw_min_khz))*1000
        bw_hi = Decimal(str(self.bw_max_khz))*1000
        step = Decimal(str(self.bw_step_hz))
        levels = int(((bw_hi-bw_lo)/step).to_integral_value(rounding=ROUND_FLOOR))+1
        return lo, hi, bw_lo, step, levels

    def pri_grid(self):
        """Inclusive PRI limits in samples (1/Fs grid), rounded inward."""
        rate = Decimal(str(self.sample_rate_msps))
        lo = int((Decimal(str(self.pri_min_us))*rate).to_integral_value(rounding=ROUND_CEILING))
        hi = int((Decimal(str(self.pri_max_us))*rate).to_integral_value(rounding=ROUND_FLOOR))
        return lo, hi

    @property
    def period_samples(self):
        """Longest PRI in samples (worst case for buffers and checks)."""
        return self.pri_grid()[1]

    def as_radio(self, pulse_samples, bandwidth_hz, pri_samples=None):
        fields = {f.name: getattr(self, f.name) for f in dataclasses.fields(TxSettings)}
        fields['pw_us'] = pulse_samples/self.fs*1e6
        fields['bandwidth_khz'] = bandwidth_hz/1000
        fields['pri_us'] = (pri_samples if pri_samples is not None else self.pri_grid()[1])/self.fs*1e6
        return TxSettings(**fields)

    def validate(self):
        values = (self.sample_rate_msps, self.pri_min_us, self.pri_max_us, self.pw_min_us, self.pw_max_us,
                  self.bw_min_khz, self.bw_max_khz, self.bw_step_hz)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Enter finite numbers in the range and timing fields.")
        if not 8 <= self.sample_rate_msps <= 20:
            raise ValueError("Use a HackRF sample rate between 8 and 20 MSPS.")
        if not 0 < self.pri_min_us <= self.pri_max_us:
            raise ValueError("Require 0 < PRI min <= PRI max.")
        if not 0 < self.pw_min_us <= self.pw_max_us < self.pri_min_us:
            raise ValueError("Require 0 < PW min <= PW max < PRI min, so every pulse fits every PRI.")
        if self.pri_max_us > MAX_PERIOD_SAMPLES/self.fs*1e6:
            raise ValueError("PRI max is too large for the waveform buffer.")
        pri_lo, pri_hi = self.pri_grid()
        if pri_hi < pri_lo:
            raise ValueError("PRI limits contain no sample-grid value.")
        if not 0 < self.bw_min_khz <= self.bw_max_khz:
            raise ValueError("Require 0 < BW min <= BW max.")
        if self.bw_step_hz <= 0:
            raise ValueError("BW step must be positive, in Hz. 1/Fs is a time step for PW.")
        for name, lower, upper in (('pri_repeat', 1, 1_000_000), ('pair_count', 2, 20000), ('seed', 0, 2**32-1)):
            val = getattr(self, name)
            if not math.isfinite(val) or int(val) != val or not lower <= val <= upper:
                raise ValueError(f"{name.replace('_',' ')} must be an integer between {lower} and {upper}.")
        lo, hi, bw_lo, step, bw_levels = self.grid()
        if lo < 2 or hi < lo or hi >= pri_lo:
            raise ValueError("PW limits contain no valid sample-grid pulse, or round into the shortest PRI.")
        if not 1 <= bw_levels <= 2**31:
            raise ValueError("The BW grid is too large. Increase BW step.")
        for levels in (hi-lo+1, bw_levels, pri_hi-pri_lo+1):
            if levels == 2 and self.pair_count % 2:
                raise ValueError("Use an even pair count for a two-value range so values change at the dataset loop boundary too.")
        self.as_radio(lo, float(bw_lo), pri_lo).validate()
        self.as_radio(hi, float(bw_lo+(bw_levels-1)*step), pri_hi).validate()
        if self.pair_count*2*pri_hi > MAX_DATASET_BYTES:
            raise ValueError("Dataset may exceed 512 MiB. Reduce the number of groups or PRI max.")
        if not str(self.output_directory).strip():
            raise ValueError("Choose an output folder for training labels.")
        return self

    def summary(self):
        lo, hi, bw_lo, step, levels = self.grid()
        pri_lo, pri_hi = self.pri_grid()
        return {
            'sample_rate_hz': self.fs,
            'sample_interval_ns': 1e9/self.fs,
            'pri_min_samples': pri_lo, 'pri_max_samples': pri_hi,
            'pri_min_us': pri_lo/self.fs*1e6, 'pri_max_us': pri_hi/self.fs*1e6,
            'pri_levels': pri_hi-pri_lo+1,
            'lowest_rf_hz': self.carrier_mhz*1e6+self.offset_khz*1e3-(float(bw_lo+(levels-1)*step))/2,
            'pw_min_samples': lo, 'pw_max_samples': hi,
            'pw_min_us': lo/self.fs*1e6, 'pw_max_us': hi/self.fs*1e6,
            'pw_levels': hi-lo+1,
            'bw_min_hz': float(bw_lo), 'bw_max_hz': float(bw_lo+(levels-1)*step),
            'bw_step_hz': float(step), 'bw_levels': levels,
            'pri_repeat': self.pri_repeat,
            'pair_count': self.pair_count,
            'pair_hold_min_ms': pri_lo*self.pri_repeat/self.fs*1e3,
            'pair_hold_max_ms': pri_hi*self.pri_repeat/self.fs*1e3,
            'dataset_duration_s_expected': (pri_lo+pri_hi)/2*self.pri_repeat*self.pair_count/self.fs,
            'waveform_storage_mib_max': self.pair_count*2*pri_hi/1024**2,
            'filter_hz': self.filter_hz,
        }


def choose_other(rng, size, forbidden):
    """Uniform choice without constructing a potentially huge PW/BW grid."""
    if size == 1:
        return 0
    banned = sorted(set(forbidden))
    candidate = int(rng.integers(0, size-len(banned)))
    for value in banned:
        if candidate >= value:
            candidate += 1
    return candidate


def make_plan(settings):
    """Seeded (PW samples, BW Hz, PRI samples) per group; each changes every group."""
    settings.validate()
    lo, hi, bw_lo, step, bw_levels = settings.grid()
    pri_lo, pri_hi = settings.pri_grid()
    rng = np.random.default_rng(settings.seed)
    pw_indices, bw_indices, pri_indices = [], [], []
    for index in range(settings.pair_count):
        last = index == settings.pair_count-1
        for indices, size in ((pw_indices, hi-lo+1), (bw_indices, bw_levels), (pri_indices, pri_hi-pri_lo+1)):
            banned = ([indices[-1]] if index else []) + ([indices[0]] if last else [])
            indices.append(choose_other(rng, size, banned))
    return (
        np.asarray(pw_indices, dtype=np.int64)+lo,
        np.asarray([float(bw_lo+i*step) for i in bw_indices], dtype=np.float64),
        np.asarray(pri_indices, dtype=np.int64)+pri_lo,
    )


LABEL_FIELDS = ['pair_index', 'first_pri_index', 'first_sample_index', 'start_time_s',
                'PW_samples', 'PW_us', 'BW_Hz', 'BW_kHz', 'PRI_samples', 'PRI_us',
                'PRI_repeat', 'chirp_rate_Hz_per_s', 'sample_rate_Hz', 'carrier_Hz']


def save_plan(settings, plan, prefix='tx'):
    root = Path(settings.output_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    folder = root/(prefix+'_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    folder.mkdir()
    pw, bw, pri = plan
    with (folder/'labels.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        sign = 1 if settings.direction == 'Up chirp' else -1
        first_sample = 0                     # groups have different lengths: cumulative offsets
        for index, (n, bandwidth, n_pri) in enumerate(zip(pw, bw, pri)):
            writer.writerow(dict(zip(LABEL_FIELDS, (
                index, index*settings.pri_repeat, first_sample, first_sample/settings.fs,
                int(n), n/settings.fs*1e6, bandwidth, bandwidth/1000,
                int(n_pri), n_pri/settings.fs*1e6, settings.pri_repeat,
                sign*bandwidth/(n/settings.fs), settings.fs, round(settings.carrier_mhz*1e6),
            ))))
            first_sample += int(n_pri)*settings.pri_repeat
    requested = dataclasses.asdict(settings)
    for key in ('pw_us', 'bandwidth_khz', 'pri_us'):
        requested.pop(key, None)
    metadata = {
        'requested': requested, 'actual': settings.summary(),
        'created_at': dt.datetime.now().astimezone().isoformat(),
        'sequence': 'Seeded random (PW, BW, PRI) triples; each changing parameter differs from the previous group, including loop wrap.',
        'repetition': 'Each triple repeats for PRI_repeat consecutive PRIs of its own length. Entire dataset loops until Stop.',
        'variable_pri': 'Groups have different lengths; first_sample_index is cumulative. One pass = sum(PRI_samples)*PRI_repeat samples.',
        'label_scope': 'Planned labels for one dataset pass, not verified radiated samples or received training data.',
        'receiver_alignment': 'Detect pulse/group alignment in the received data. Host timestamps and GUI status are not RF timestamps.',
        'pass_mapping': 'global_pair_index = pass_index * pair_count + pair_index; pass indices start at 0.',
        'fixed_ranges': 'A range containing one grid value stays constant.',
    }
    (folder/'settings.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    return folder


class TrainingStream:
    """Repeat each (variable-length) period R times, then advance, across USB buffers.

    periods: list of int8 IQ arrays, one PRI each (lengths differ with PRI).
    fill() only copies from one flat buffer: no allocation or locking per callback.
    """
    def __init__(self, periods, repeats):
        self.lengths = np.asarray([len(p) for p in periods], dtype=np.int64)
        self.starts = np.r_[0, np.cumsum(self.lengths)[:-1]]
        self.flat = np.concatenate(periods)
        self.count = len(periods)
        self.repeats = repeats
        self.group_ends = np.cumsum(self.lengths*repeats)       # within one pass
        self.pass_bytes = int(self.group_ends[-1])
        self.pair = 0
        self.repeat = 0
        self.offset = 0
        self.total_bytes = 0

    def fill(self, buffer, length):
        pos = 0
        while pos < length:
            width = int(self.lengths[self.pair])
            count = min(length-pos, width-self.offset)
            start = int(self.starts[self.pair])+self.offset
            buffer[pos:pos+count] = self.flat[start:start+count]
            pos += count
            self.offset += count
            if self.offset == width:
                self.offset = 0
                self.repeat += 1
                if self.repeat == self.repeats:
                    self.repeat = 0
                    self.pair = (self.pair+1) % self.count
        self.total_bytes += length

    def position(self):
        last_byte = max(0, self.total_bytes-1)
        passes, within = divmod(last_byte, self.pass_bytes)
        group = int(np.searchsorted(self.group_ends, within, side='right'))
        group_start = int(self.group_ends[group-1]) if group else 0
        done_passes, done_within = divmod(self.total_bytes, self.pass_bytes)
        completed = done_passes*self.count+int(np.searchsorted(self.group_ends, done_within, side='right'))
        return {
            'samples': self.total_bytes//2,
            'completed_groups': completed,
            'completed_passes': done_passes,
            'last_pair': group,
            'last_pass': passes,
            'last_repeat': (within-group_start)//int(self.lengths[group]),
        }


def training_worker(connection, config):
    settings = TrainingSettings(**config)
    api = device = folder = stream = None
    initialized = started_tx = False
    stop_event = threading.Event()
    callback_errors, cleanup_errors = [], []
    result_state = 'stopped'
    rf_started_at = None
    def send(kind, **data):
        try:
            connection.send({'kind': kind, **data})
        except (EOFError, BrokenPipeError, OSError):
            stop_event.set()
    def stopped():
        if stop_event.is_set():
            return True
        if connection.poll() and connection.recv() == 'stop':
            stop_event.set()
        return stop_event.is_set()
    try:
        settings.validate()
        if stopped():
            return
        send('log', text='Generating reproducible PW/BW/PRI triples and planned labels...')
        plan = make_plan(settings)
        folder = save_plan(settings, plan)
        send('files', folder=str(folder))
        if stopped():
            return
        periods = []
        progress_step = max(1, settings.pair_count//100)
        for index, (n, bw, n_pri) in enumerate(zip(*plan)):
            if index % 16 == 0 and stopped():
                return
            periods.append(make_period(settings.as_radio(int(n), float(bw), int(n_pri))))
            if index % progress_step == 0 or index == settings.pair_count-1:
                send('preparing', percent=100*(index+1)/settings.pair_count)
        stream = TrainingStream(periods, settings.pri_repeat)
        if stopped():
            return
        if settings.mode == 'Hardware':
            try:
                from python_hackrf import pyhackrf as api
            except (ImportError, OSError) as exc:
                raise RuntimeError('Cannot load python_hackrf/libhackrf. Use the same Python environment as your working GUI. Details: '+str(exc)) from exc
            check_result(api.pyhackrf_init(), 'HackRF initialization')
            initialized = True
            device = api.pyhackrf_open()
            if device is None:
                raise RuntimeError('Could not open HackRF. Check USB and close other SDR applications.')
            for method, value in (
                ('pyhackrf_set_sample_rate', settings.fs),
                ('pyhackrf_set_freq', round(settings.carrier_mhz*1e6)),
                ('pyhackrf_set_baseband_filter_bandwidth', settings.filter_hz),
                ('pyhackrf_set_txvga_gain', settings.gain_db),
                ('pyhackrf_set_amp_enable', settings.amp_enabled),
            ):
                check_result(getattr(device, method)(value), method)
            def callback(_device, buffer, length, valid_length):
                try:
                    if stop_event.is_set():
                        buffer[:length] = 0
                        if hasattr(valid_length, 'value'):
                            valid_length.value = 0
                        return -1
                    stream.fill(buffer, length)
                    if hasattr(valid_length, 'value'):
                        valid_length.value = length
                    return 0
                except Exception as exc:
                    callback_errors.append(str(exc))
                    stop_event.set()
                    if hasattr(valid_length, 'value'):
                        valid_length.value = 0
                    return -1
            device.set_tx_callback(callback)
            if stopped():
                return
            check_result(device.pyhackrf_start_tx(), 'Starting training TX')
            started_tx = True
        if stopped():
            return
        rf_started_at = dt.datetime.now().astimezone().isoformat()
        start_clock = time.monotonic()
        send('running', mode=settings.mode)
        send('log', text=('Training transmission started; dataset loops until Stop.' if started_tx
                          else 'SIMULATION: no RF output. Dataset loops until Stop.'))
        last_report = start_clock
        sim_buffer = np.empty(262144, dtype=np.int8)
        next_sim = start_clock
        while not stopped():
            wait = .05 if started_tx else max(0, next_sim-time.monotonic())
            if connection.poll(min(.05, wait)) and connection.recv() == 'stop':
                stop_event.set()
                break
            if started_tx:
                if not device.pyhackrf_is_streaming():
                    raise RuntimeError('HackRF streaming stopped unexpectedly. Check USB and sample rate.')
            elif time.monotonic() >= next_sim:
                stream.fill(sim_buffer, len(sim_buffer))
                next_sim = max(next_sim+len(sim_buffer)/(2*settings.fs), time.monotonic())
            now = time.monotonic()
            if now-last_report >= .2:
                position = stream.position()
                index = position['last_pair']
                send('stats', elapsed=now-start_clock, **position,
                     pw_us=float(plan[0][index]/settings.fs*1e6), bw_khz=float(plan[1][index]/1000),
                     pri_us=float(plan[2][index]/settings.fs*1e6))
                last_report = now
        if callback_errors:
            raise RuntimeError('TX callback failed: '+callback_errors[0])
    except (EOFError, BrokenPipeError):
        result_state = 'control_connection_lost'
    except Exception as exc:
        result_state = 'error'
        send('error', text=str(exc))
    finally:
        stop_event.set()
        if device is not None:
            if started_tx:
                try:
                    check_result(device.pyhackrf_stop_tx(), 'Stopping TX')
                except Exception as exc:
                    cleanup_errors.append(str(exc))
            try:
                check_result(device.pyhackrf_close(), 'Closing HackRF')
            except Exception as exc:
                cleanup_errors.append(str(exc))
        if initialized:
            try:
                check_result(api.pyhackrf_exit(), 'Exiting libhackrf')
            except Exception as exc:
                cleanup_errors.append(str(exc))
        if cleanup_errors:
            result_state = 'cleanup_error'
            send('error', text='Device cleanup failed: '+'; '.join(cleanup_errors))
        if folder:
            try:
                session = {'state': result_state, 'mode': settings.mode,
                           'host_stream_started_at': rf_started_at,
                           'host_stopped_at': dt.datetime.now().astimezone().isoformat(),
                           'queue_progress': stream.position() if stream else {},
                           'note': 'Queue progress is not RF-delivery confirmation. Receiver samples are not recorded by this transmitter UI.'}
                (folder/'session.json').write_text(json.dumps(session, indent=2)+'\n', encoding='utf-8')
            except OSError as exc:
                send('error', text='Could not save session status: '+str(exc))
        send('finished', clean=not cleanup_errors)
        connection.close()


class TrainingGUI:
    BG, PANEL, FG, MUTED = '#101720', '#182330', '#edf3fa', '#94a8bd'

    def __init__(self, root, simulate=False):
        import tkinter as tk
        from tkinter import ttk, font
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.tk, self.ttk, self.root = tk, ttk, root
        self.family = font.nametofont('TkDefaultFont').actual('family')
        self.state = 'idle'
        self.process = self.connection = None
        self.pending_preview = self.stop_at = None
        self.last_error = None
        self.finished = self.closing = False
        self.active_settings = None
        self.editors, self.vars = [], {}
        self.plan = self.settings = None
        root.title('HackRF | Training Signal Generator · changing PRI')
        root.geometry('1280x900')
        root.minsize(1120, 830)
        root.configure(bg=self.BG)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)
        root.protocol('WM_DELETE_WINDOW', self.close)
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('TFrame', background=self.BG)
        style.configure('Panel.TFrame', background=self.PANEL)
        style.configure('TLabel', background=self.PANEL, foreground=self.FG, font=(self.family, 10))
        style.configure('Muted.TLabel', foreground=self.MUTED)
        style.configure('TEntry', fieldbackground='#223345', foreground=self.FG, insertcolor=self.FG, padding=5)
        style.map('TEntry', fieldbackground=[('disabled','#1b2938')], foreground=[('disabled','#7890a6')])
        style.configure('TCombobox', fieldbackground='#223345', foreground=self.FG, background='#223345', padding=4)
        style.map('TCombobox', fieldbackground=[('readonly','#223345'),('disabled','#1b2938')],
                  foreground=[('readonly',self.FG),('disabled','#7890a6')])
        root.option_add('*TCombobox*Listbox.background', '#223345')
        root.option_add('*TCombobox*Listbox.foreground', self.FG)
        style.configure('TCheckbutton', background=self.PANEL, foreground=self.FG, font=(self.family,10))
        style.map('TCheckbutton', background=[('active', self.PANEL)])
        style.configure('TNotebook', background=self.PANEL, borderwidth=0)
        style.configure('TNotebook.Tab', padding=(18,9), background='#223345', foreground=self.MUTED)
        style.map('TNotebook.Tab', background=[('selected',self.PANEL)], foreground=[('selected',self.FG)])
        style.configure('TButton', padding=(9,8), background='#2b4055', foreground=self.FG, font=(self.family,10))
        style.map('TButton', background=[('active','#37546e'),('disabled','#253443')], foreground=[('disabled','#63798d')])
        style.configure('Start.TButton', background='#207c66', font=(self.family,10,'bold'))
        style.map('Start.TButton', background=[('active','#2b987d'),('disabled','#253443')])
        style.configure('Stop.TButton', background='#a14e3b', font=(self.family,10,'bold'))
        style.map('Stop.TButton', background=[('active','#bf5a43'),('disabled','#253443')])
        style.configure('Treeview', background='#111c28', fieldbackground='#111c28', foreground=self.FG,
                        rowheight=25, font=(self.family,9))
        style.configure('Treeview.Heading', background='#223345', foreground=self.FG, font=(self.family,9,'bold'))
        header = tk.Frame(root, bg=self.BG)
        header.grid(row=0,column=0,sticky='ew',padx=22,pady=(18,8))
        tk.Label(header,text='Training Signal Generator · PRI',bg=self.BG,fg=self.FG,font=(self.family,22,'bold')).pack(side='left')
        self.status_var = tk.StringVar(value='IDLE')
        self.status_label = tk.Label(header,textvariable=self.status_var,bg='#223345',fg=self.MUTED,
                                    font=(self.family,10,'bold'),padx=14,pady=8)
        self.status_label.pack(side='right')
        tk.Label(root,text='Hold a PW/BW/PRI triple for R pulses, then change all three. The planned sequence loops until Stop.',
                 bg=self.BG,fg=self.MUTED,font=(self.family,10)).grid(row=1,column=0,sticky='w',padx=23,pady=(0,14))
        body = ttk.Frame(root)
        body.grid(row=2,column=0,sticky='nsew',padx=22)
        body.columnconfigure(1,weight=1)
        body.rowconfigure(0,weight=1)
        left = ttk.Frame(body,style='Panel.TFrame',padding=14)
        left.grid(row=0,column=0,sticky='nsew',padx=(0,16))
        left.columnconfigure(0,weight=1)
        notebook = ttk.Notebook(left)
        notebook.grid(row=0,column=0,sticky='ew')
        training = ttk.Frame(notebook,style='Panel.TFrame',padding=(8,12))
        radio = ttk.Frame(notebook,style='Panel.TFrame',padding=(8,12))
        notebook.add(training,text='Training sequence')
        notebook.add(radio,text='Radio settings')
        defaults = dataclasses.asdict(TrainingSettings(mode='Simulation' if simulate else 'Hardware',
                                    output_directory=str(Path(__file__).resolve().parent/'training_runs')))

        def add_field(parent, row, key, label, choices=None):
            ttk.Label(parent,text=label).grid(row=row,column=0,sticky='w',padx=(0,14),pady=5)
            variable = tk.StringVar(value=str(defaults[key]))
            self.vars[key] = variable
            widget = (ttk.Combobox(parent,textvariable=variable,values=choices,state='readonly',width=15)
                      if choices else ttk.Entry(parent,textvariable=variable,width=17))
            widget.grid(row=row,column=1,sticky='ew',pady=5)
            self.editors.append((widget,'readonly' if choices else 'normal'))
            variable.trace_add('write',self.schedule_preview)
            parent.columnconfigure(1,weight=1)

        fields = [('pri_min_us','PRI min (us)'),('pri_max_us','PRI max (us)'),
                  ('pw_min_us','PW min (us)'),('pw_max_us','PW max (us)'),
                  ('bw_min_khz','BW min (kHz)'),('bw_max_khz','BW max (kHz)'),
                  ('pri_repeat','PRI repeat (R)'),('bw_step_hz','BW step (Hz)'),
                  ('pair_count','Number of groups'),('seed','Random seed')]
        for row,(key,label) in enumerate(fields):
            add_field(training,row,key,label)
        ttk.Label(training,text='PW and PRI step = 1/Fs seconds.\nBW uses the separate step in Hz.\nRandom triples; same seed reproduces the plan.',
                  style='Muted.TLabel',font=(self.family,9)).grid(row=len(fields),column=0,columnspan=2,sticky='w',pady=(9,0))
        for row,(key,label,choices) in enumerate([
            ('mode','Output mode',('Hardware','Simulation')),
            ('carrier_mhz','Carrier (MHz)',None),
            ('sample_rate_msps','Sample rate (MSPS)',('8','10','12.5','16','20')),
            ('gain_db','TX VGA gain (dB)',None),
            ('amplitude','DAC amplitude (0-1)',None),
            ('offset_khz','Baseband offset (kHz)',None),
            ('direction','Sweep direction',('Up chirp','Down chirp')),
            ('filter_mhz','Filter BW (MHz)',('Auto',)+tuple(str(v) for v in FILTER_BWS)),
        ]):
            add_field(radio,row,key,label,choices)
        self.vars['amp_enabled'] = tk.BooleanVar(value=False)
        amp = ttk.Checkbutton(radio,text='Enable RF amplifier',variable=self.vars['amp_enabled'])
        amp.grid(row=8,column=0,columnspan=2,sticky='w',pady=9)
        self.editors.append((amp,'normal'))
        ttk.Label(radio,text='Gain is a hardware setting, not RF power in dBm.',style='Muted.TLabel',
                  font=(self.family,9),wraplength=330).grid(row=9,column=0,columnspan=2,sticky='w')
        ttk.Label(left,text='OUTPUT FOLDER',foreground='#54b4ff',font=(self.family,9,'bold')).grid(row=1,column=0,sticky='w',pady=(15,6))
        folder_frame = ttk.Frame(left,style='Panel.TFrame')
        folder_frame.grid(row=2,column=0,sticky='ew')
        folder_frame.columnconfigure(0,weight=1)
        self.vars['output_directory'] = tk.StringVar(value=defaults['output_directory'])
        folder_entry = ttk.Entry(folder_frame,textvariable=self.vars['output_directory'],width=25)
        folder_entry.grid(row=0,column=0,sticky='ew',padx=(0,5))
        self.editors.append((folder_entry,'normal'))
        self.browse_button = ttk.Button(folder_frame,text='Browse',command=self.browse)
        self.browse_button.grid(row=0,column=1)
        self.vars['output_directory'].trace_add('write',self.schedule_preview)
        controls = ttk.Frame(left,style='Panel.TFrame')
        controls.grid(row=3,column=0,sticky='ew',pady=(15,6))
        controls.columnconfigure((0,1),weight=1)
        self.start_button = ttk.Button(controls,text='Start training TX',style='Start.TButton',command=self.start)
        self.start_button.grid(row=0,column=0,sticky='ew',padx=(0,5))
        self.stop_button = ttk.Button(controls,text='Stop',style='Stop.TButton',command=self.stop,state='disabled')
        self.stop_button.grid(row=0,column=1,sticky='ew',padx=(5,0))
        self.preview_button = ttk.Button(controls,text='Refresh preview',command=self.preview)
        self.preview_button.grid(row=1,column=0,sticky='ew',padx=(0,5),pady=(8,0))
        self.export_button = ttk.Button(controls,text='Export plan',command=self.export_plan)
        self.export_button.grid(row=1,column=1,sticky='ew',padx=(5,0),pady=(8,0))
        ttk.Label(left,text='Start saves labels.csv, settings.json and session.json.\nThis UI transmits training signals. It does not record RX data.',
                  style='Muted.TLabel',font=(self.family,9),wraplength=365).grid(row=4,column=0,sticky='w',pady=(7,0))
        right = ttk.Frame(body)
        right.grid(row=0,column=1,sticky='nsew')
        right.columnconfigure(0,weight=1)
        right.rowconfigure(1,weight=1)
        cards = ttk.Frame(right,style='Panel.TFrame',padding=(14,10))
        cards.grid(row=0,column=0,sticky='ew',pady=(0,12))
        self.metrics = {}
        for col,(key,label) in enumerate([('grid','PW / PRI GRID'),('bw','BW STEP'),('hold','GROUP HOLD'),('duration','ONE PASS (EXPECTED)')]):
            cards.columnconfigure(col,weight=1)
            ttk.Label(cards,text=label,style='Muted.TLabel',font=(self.family,9)).grid(row=0,column=col,sticky='w')
            self.metrics[key] = tk.StringVar(value='-')
            ttk.Label(cards,textvariable=self.metrics[key],font=(self.family,13,'bold')).grid(row=1,column=col,sticky='w',pady=(4,0))
        plot_panel = ttk.Frame(right,style='Panel.TFrame',padding=12)
        plot_panel.grid(row=1,column=0,sticky='nsew',pady=(0,12))
        plot_panel.columnconfigure(0,weight=1)
        plot_panel.rowconfigure(1,weight=1)
        ttk.Label(plot_panel,text='PLANNED SEQUENCE - FIRST 12 GROUPS',foreground='#54b4ff',
                  font=(self.family,9,'bold')).grid(row=0,column=0,sticky='w')
        self.figure = Figure(figsize=(6.6,3.2),dpi=100,facecolor=self.PANEL)
        self.axes = self.figure.subplots(3,1,sharex=True)
        self.figure.subplots_adjust(left=.13,right=.97,top=.95,bottom=.14,hspace=.3)
        self.lines = []
        for ax,label,color in zip(self.axes,('PW (us)','BW (kHz)','PRI (us)'),('#54b4ff','#61d2ac','#ffbf69')):
            ax.set_facecolor(self.PANEL)
            ax.tick_params(colors=self.MUTED,labelsize=9)
            ax.set_ylabel(label,color=self.MUTED,fontsize=9)
            for spine in ax.spines.values():
                spine.set_color('#35475a')
            ax.grid(color='#2e4053',linewidth=.6)
            self.lines.append(ax.step([],[],where='post',color=color,linewidth=2)[0])
        self.axes[2].set_xlabel('Time within dataset (ms)',color=self.MUTED,fontsize=9)
        self.canvas = FigureCanvasTkAgg(self.figure,master=plot_panel)
        self.canvas.get_tk_widget().grid(row=1,column=0,sticky='nsew')
        table_frame = ttk.Frame(right,style='Panel.TFrame',padding=12)
        table_frame.grid(row=2,column=0,sticky='ew',pady=(0,12))
        ttk.Label(table_frame,text='LABEL PREVIEW - FIRST 30 GROUPS',foreground='#54b4ff',font=(self.family,9,'bold')).pack(anchor='w',pady=(0,5))
        self.table = ttk.Treeview(table_frame,columns=('pair','pw','bw','pri','repeat','sample'),show='headings',height=5)
        for key,title,width in [('pair','Group',55),('pw','PW (us)',105),('bw','BW (kHz)',95),('pri','PRI (us)',105),
                                ('repeat','PRI repeat',85),('sample','First sample',120)]:
            self.table.heading(key,text=title)
            self.table.column(key,width=width,anchor='center')
        self.table.pack(fill='x')
        stats_panel = ttk.Frame(right,style='Panel.TFrame',padding=12)
        stats_panel.grid(row=3,column=0,sticky='ew')
        self.stats_var = tk.StringVar(value='Ready. No RF is transmitted until Start.')
        ttk.Label(stats_panel,textvariable=self.stats_var,wraplength=710,font=(self.family,9)).pack(anchor='w')
        self.progress = ttk.Progressbar(stats_panel,maximum=100)
        self.progress.pack(fill='x',pady=(6,8))
        self.log_widget = tk.Text(stats_panel,height=4,bg='#111c28',fg=self.MUTED,relief='flat',
                                  font=('TkFixedFont',9),wrap='word',state='disabled',padx=6,pady=6)
        self.log_widget.pack(fill='x')
        self.validation_var = tk.StringVar(value='')
        self.validation_label = tk.Label(root,textvariable=self.validation_var,bg=self.BG,fg=self.MUTED,
                                        font=(self.family,9),anchor='w',wraplength=1200)
        self.validation_label.grid(row=3,column=0,sticky='ew',padx=23,pady=(10,14))
        self.log('Ready. PW and PRI use sample timing; bandwidth uses the BW step in Hz.')
        self.preview()
        root.after(100,self.poll_worker)

    def read_settings(self):
        values = {key: var.get() for key,var in self.vars.items()}
        ints = {'pri_repeat','pair_count','seed','gain_db'}
        strings = {'mode','direction','filter_mhz','output_directory'}
        try:
            for key,value in values.items():
                if key == 'amp_enabled' or key in strings:
                    continue
                number = float(value)
                if key in ints:
                    if not math.isfinite(number) or not number.is_integer():
                        raise ValueError(f"{key.replace('_',' ')} must be a whole number.")
                    number = int(number)
                values[key] = number
            return TrainingSettings(**values).validate()
        except (TypeError,OverflowError) as exc:
            raise ValueError('Enter valid numeric parameters.') from exc
        except ValueError as exc:
            if 'could not convert' in str(exc):
                raise ValueError('Enter a number in each numeric field.') from exc
            raise

    def log(self,text):
        self.log_widget.configure(state='normal')
        self.log_widget.insert('end',f"[{dt.datetime.now():%H:%M:%S}] {text}\n")
        if int(self.log_widget.index('end-1c').split('.')[0]) > 250:
            self.log_widget.delete('1.0','2.0')
        self.log_widget.see('end')
        self.log_widget.configure(state='disabled')

    def schedule_preview(self,*_):
        if self.pending_preview:
            self.root.after_cancel(self.pending_preview)
        self.pending_preview = self.root.after(250,self.preview)

    def preview(self):
        if self.pending_preview:
            self.root.after_cancel(self.pending_preview)
        self.pending_preview = None
        try:
            settings = self.read_settings()
            plan = make_plan(settings)
        except (ValueError,ArithmeticError) as exc:
            self.settings = self.plan = None
            self.validation_var.set(str(exc))
            self.validation_label.configure(fg='#ffb08c')
            if self.state == 'idle':
                self.start_button.configure(state='disabled')
                self.export_button.configure(state='disabled')
            return
        self.settings,self.plan = settings,plan
        summary = settings.summary()
        self.metrics['grid'].set(f"{summary['sample_interval_ns']:g} ns")
        self.metrics['bw'].set(f"{settings.bw_step_hz:g} Hz")
        self.metrics['hold'].set(f"{summary['pair_hold_min_ms']:.3f}-{summary['pair_hold_max_ms']:.3f} ms")
        self.metrics['duration'].set(f"{summary['dataset_duration_s_expected']:.3f} s")
        n = min(settings.pair_count,12)
        times = np.r_[0, np.cumsum(plan[2][:n]*settings.pri_repeat/settings.fs*1e3)]   # groups differ in length
        for ax,line,values in zip(self.axes,self.lines,(plan[0][:n]/settings.fs*1e6,plan[1][:n]/1000,
                                                       plan[2][:n]/settings.fs*1e6)):
            line.set_data(times,np.r_[values,values[-1]])
            low,high = float(min(values)),float(max(values))
            margin = max((high-low)*.2,abs(low)*.002,1e-3)
            ax.set_ylim(low-margin,high+margin)
            ax.set_xlim(0,times[-1])
        self.canvas.draw_idle()
        self.table.delete(*self.table.get_children())
        first_sample = 0
        for index,(n_pw,bw,n_pri) in enumerate(zip(*plan)):
            if index >= 30:
                break
            self.table.insert('', 'end', values=(index+1,f'{n_pw/settings.fs*1e6:.4f}',f'{bw/1000:g}',
                              f'{n_pri/settings.fs*1e6:.4f}',settings.pri_repeat,first_sample))
            first_sample += int(n_pri)*settings.pri_repeat
        fixed = []
        if summary['pw_levels'] == 1:
            fixed.append('PW')
        if summary['bw_levels'] == 1:
            fixed.append('BW')
        if summary['pri_levels'] == 1:
            fixed.append('PRI')
        note = (f"PRI {summary['pri_min_us']:.4f}-{summary['pri_max_us']:.4f} us | {summary['pri_levels']:,} PRI values | "
                f"{summary['pw_levels']:,} PW values | {summary['bw_levels']:,} BW values | "
                f"IQ memory up to {summary['waveform_storage_mib_max']:.1f} MiB")
        if summary['lowest_rf_hz'] < 1e6:
            note += f" | NOTE: chirp reaches {summary['lowest_rf_hz']/1e3:.0f} kHz RF, below HackRF's 1 MHz spec"
        if fixed:
            note += ' | '+', '.join(fixed)+' stays fixed (one grid value).'
        if settings.mode == 'Simulation':
            note += ' | SIMULATION: no RF.'
        self.validation_var.set(note)
        self.validation_label.configure(fg=self.MUTED)
        if self.state == 'idle':
            self.start_button.configure(state='normal',text='Start simulation' if settings.mode == 'Simulation' else 'Start training TX')
            self.export_button.configure(state='normal')

    def set_busy(self,busy):
        for widget,normal in self.editors:
            widget.configure(state='disabled' if busy else normal)
        for widget in (self.start_button,self.preview_button,self.export_button,self.browse_button):
            widget.configure(state='disabled' if busy else 'normal')
        self.stop_button.configure(state='normal' if busy else 'disabled')

    def browse(self):
        from tkinter import filedialog
        folder = filedialog.askdirectory(parent=self.root,title='Choose output folder')
        if folder:
            self.vars['output_directory'].set(folder)

    def export_plan(self):
        from tkinter import messagebox
        try:
            settings = self.read_settings()
            folder = save_plan(settings,make_plan(settings),prefix='plan')
            self.log('Exported planned labels and settings: '+str(folder))
        except (ValueError,OSError) as exc:
            messagebox.showerror('Export failed',str(exc),parent=self.root)

    def start(self):
        from tkinter import messagebox
        if self.state != 'idle':
            return
        try:
            settings = self.read_settings()
        except ValueError as exc:
            messagebox.showerror('Invalid settings',str(exc),parent=self.root)
            return
        self.active_settings = settings
        self.last_error,self.finished,self.stop_at = None,False,None
        self.state = 'starting'
        self.status_var.set('PREPARING')
        self.status_label.configure(fg='#54b4ff')
        self.stats_var.set('Preparing waveforms and saving planned labels...')
        self.progress['value'] = 0
        self.set_busy(True)
        self.log(f'Preparing {settings.pair_count:,} PW/BW/PRI groups, {settings.pri_repeat} pulses per group.')
        ctx = mp.get_context('spawn')
        parent,child = ctx.Pipe()
        process = ctx.Process(target=training_worker,args=(child,dataclasses.asdict(settings)),daemon=True)
        try:
            process.start()
        except Exception as exc:
            parent.close()
            child.close()
            self.state = 'idle'
            self.set_busy(False)
            self.status_var.set('ERROR')
            messagebox.showerror('Cannot start',str(exc),parent=self.root)
            return
        child.close()
        self.process,self.connection = process,parent

    def stop(self):
        if self.state not in ('starting','running'):
            return
        self.state = 'stopping'
        self.stop_at = time.monotonic()
        self.status_var.set('STOPPING')
        self.status_label.configure(fg='#ffbf69')
        self.stop_button.configure(state='disabled')
        self.log('Stop requested. Waiting for worker and device cleanup...')
        try:
            self.connection.send('stop')
        except (EOFError,BrokenPipeError,OSError):
            pass

    def drain_messages(self):
        if not self.connection:
            return
        try:
            while self.connection.poll():
                message = self.connection.recv()
                kind = message['kind']
                if kind == 'log':
                    self.log(message['text'])
                elif kind == 'files':
                    self.log('Run output folder: '+message['folder'])
                elif kind == 'preparing':
                    if self.state == 'starting':
                        self.progress['value'] = message['percent']
                elif kind == 'running':
                    if self.state != 'stopping':
                        self.state = 'running'
                        sim = message['mode'] == 'Simulation'
                        self.status_var.set('SIMULATION - NO RF' if sim else 'TRANSMITTING')
                        self.status_label.configure(fg='#ffbf69' if sim else '#61d2ac')
                        self.progress['value'] = 0
                elif kind == 'stats':
                    self.stats_var.set(f"Queued: pass {message['last_pass']+1}, group {message['last_pair']+1}/{self.active_settings.pair_count}, "
                                       f"repeat {message['last_repeat']+1}/{self.active_settings.pri_repeat}\n"
                                       f"PW {message['pw_us']:.4f} us | BW {message['bw_khz']:g} kHz | PRI {message['pri_us']:.4f} us | "
                                       f"{message['samples']:,} samples | Elapsed {message['elapsed']:.1f} s")
                    self.progress['value'] = 100*((message['completed_groups'] % self.active_settings.pair_count)/self.active_settings.pair_count)
                elif kind == 'error':
                    self.last_error = message['text']
                    self.log('ERROR: '+self.last_error)
                elif kind == 'finished':
                    self.finished = True
        except (EOFError,BrokenPipeError,OSError):
            pass

    def poll_worker(self):
        from tkinter import messagebox
        self.drain_messages()
        if self.process and self.stop_at and self.process.is_alive() and time.monotonic()-self.stop_at > 8:
            self.last_error = 'Worker did not finish stopping and was terminated. RF stop is unconfirmed; disconnect HackRF before restarting. Session metadata may be incomplete.'
            self.log('ERROR: '+self.last_error)
            self.process.terminate()
            self.stop_at = None
        if self.process and not self.process.is_alive():
            self.process.join(timeout=0)
            self.drain_messages()
            if not self.finished and not self.last_error:
                self.last_error = 'Worker exited unexpectedly. Check the device; RF stop is unconfirmed.'
            self.process.close()
            self.process = None
            self.connection.close()
            self.connection = None
            self.state = 'idle'
            self.set_busy(False)
            self.status_var.set('ERROR' if self.last_error else 'STOPPED')
            self.status_label.configure(fg='#ffb08c' if self.last_error else self.MUTED)
            if self.last_error and not self.closing:
                messagebox.showerror('Training transmitter error',self.last_error,parent=self.root)
            elif not self.last_error:
                self.log('Stopped. You can change the ranges and start a new run.')
            if self.closing:
                self.root.destroy()
                return
            self.preview()
        if self.closing and not self.process:
            self.root.destroy()
            return
        self.root.after(100,self.poll_worker)

    def close(self):
        self.closing = True
        if self.process:
            self.stop()
        else:
            self.root.destroy()


def self_test():
    settings = TrainingSettings(pair_count=20).validate()
    assert settings.carrier_mhz == 1.0 and settings.sample_rate_msps == 20.0
    plan = make_plan(settings)
    summary = settings.summary()
    assert summary['sample_interval_ns'] == 50
    assert (summary['pri_min_samples'], summary['pri_max_samples']) == (8000, 12000)
    assert summary['pri_levels'] == 4001 and summary['pw_levels'] == 1001 and summary['bw_levels'] == 101
    for a, b in zip(plan, make_plan(settings)):
        np.testing.assert_array_equal(a, b)
    assert np.all((plan[0] >= 6000) & (plan[0] <= 7000))
    assert np.all((plan[1] >= 100000) & (plan[1] <= 200000)) and np.all((plan[1]-100000) % 1000 == 0)
    assert np.all((plan[2] >= 8000) & (plan[2] <= 12000))
    assert np.all(plan[0] < plan[2]), 'Every pulse must fit inside its PRI'
    for values in plan:
        assert np.all(values != np.roll(values, 1)), 'Consecutive values must change, including loop boundary'
    frames = [make_period(settings.as_radio(int(n), float(bw), int(p))) for n, bw, p in zip(*plan)]
    for frame, n, p in zip(frames, plan[0], plan[2]):
        assert len(frame) == 2*p and np.all(frame[2*n:] == 0) and np.any(frame[:2*n] != 0)
    stream = TrainingStream(frames, 3)
    one_pass = np.concatenate([np.tile(f, 3) for f in frames])
    parts = []
    for length in (1, 17, 262144, 99999, len(one_pass)+5, 333333):
        chunk = np.empty(length, dtype=np.int8)
        stream.fill(chunk, length)
        parts.append(chunk)
    actual = np.concatenate(parts)
    np.testing.assert_array_equal(actual, np.resize(one_pass, len(actual)))
    position = stream.position()
    assert position['samples'] == len(actual)//2
    # Position bookkeeping with variable group lengths: a byte inside group 2, repeat 1.
    probe = TrainingStream(frames, 3)
    target = 3*len(frames[0])+3*len(frames[1])+len(frames[2])+10
    buf = np.empty(target, dtype=np.int8)
    probe.fill(buf, target)
    pos = probe.position()
    assert (pos['last_pair'], pos['last_repeat'], pos['completed_groups']) == (2, 1, 2), pos
    # Labels: cumulative first-sample offsets with variable PRI.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        folder = save_plan(dataclasses.replace(settings, output_directory=tmp), plan, prefix='test')
        with (folder/'labels.csv').open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        starts = [int(r['first_sample_index']) for r in rows]
        assert starts == list(np.r_[0, np.cumsum(plan[2]*3)[:-1]])
        assert [int(r['PRI_samples']) for r in rows] == list(plan[2])
    # 1 MHz carrier: chirp dips below 1 MHz but stays above 0 Hz -> allowed with a note.
    assert settings.summary()['lowest_rf_hz'] == 0.9e6
    for change in ({'pri_repeat': 0}, {'pw_max_us': 450}, {'pri_min_us': 700},
                   {'bw_min_khz': 300}, {'bw_step_hz': 0}, {'sample_rate_msps': 31.25},
                   {'bw_min_khz': 2000, 'bw_max_khz': 2100},          # chirp would cross 0 Hz
                   {'pri_min_us': 500.01, 'pri_max_us': 500.02},
                   {'pair_count': 20000, 'pri_max_us': 10000}):
        try:
            dataclasses.replace(settings, **change).validate()
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid settings accepted: '+str(change))
    print('PASS: 1 MHz / 20 MSPS defaults, PW/BW/PRI grids, every PRI holds its pulse, reproducibility, '
          'changing triples incl. loop boundary, variable-length PRI streaming, position and label offsets, '
          'and invalid settings.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulate',action='store_true',help='Select simulation mode; no RF transmission')
    parser.add_argument('--self-test',action='store_true',help='Check schedule and IQ buffering without GUI/hardware')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    try:
        import tkinter as tk
        import matplotlib
        matplotlib.use('TkAgg')
        root = tk.Tk()
    except Exception as exc:
        raise SystemExit('GUI needs tkinter, matplotlib and a graphical desktop. Details: '+str(exc)) from exc
    TrainingGUI(root,simulate=args.simulate)
    root.mainloop()


if __name__ == '__main__':
    mp.freeze_support()
    main()
