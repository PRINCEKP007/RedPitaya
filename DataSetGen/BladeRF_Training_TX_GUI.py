#!/usr/bin/env python3
"""bladeRF training transmitter: sample-grid PW, independently stepped BW.

bladeRF version of HackRF_Training_TX_GUI.py. The same signal can be sent on
TX1, TX2 or BOTH channels at the same time (TX_X2 layout: every sample goes to
both channels in one interleaved stream, so they are sample-aligned).

Run: python BladeRF_Training_TX_GUI.py [--simulate | --self-test]
Requires numpy, matplotlib, tkinter and the bladerf Python bindings (libbladeRF),
in the same environment where BladeRF_TX_RX.py works.
Hardware and simulation both save planned labels and session metadata.
"""
from __future__ import annotations
import argparse
import csv
import dataclasses
import datetime as dt
import importlib
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
MAX_DATASET_BYTES = 512 * 1024 * 1024        # int16 I/Q: 4 bytes per sample
BYTES_PER_SAMPLE = 4
BLOCK_SAMPLES = 16384                         # samples per channel per sync_tx call
DAC_FULL_SCALE = 2047                         # SC16 Q11
MIN_FS_MSPS, MAX_FS_MSPS = 0.521, 61.44       # bladeRF 2.0 sample-rate range
MAX_FS_TWO_CHANNELS_MSPS = 30.72              # 2 x SC16 at higher rates exceeds USB 3 throughput
MIN_RF_HZ, MAX_RF_HZ = 47e6, 6e9              # bladeRF 2.0 TX tuning range
GAIN_RANGE_DB = (-24, 66)                     # requested TX gain; the device may clamp (actual is logged)
SAMPLE_RATES = ('2', '4', '5', '8', '10', '12.5', '16', '20', '25', '30.72', '40', '61.44')
RF_BANDWIDTHS = ('Auto', '1.5', '2.5', '5', '10', '15', '20', '28', '40', '56')
BOOL_FIELDS = ('tx1_enabled', 'tx2_enabled')

@dataclasses.dataclass(frozen=True)
class TxSettings:
    carrier_mhz: float = 1000.0
    sample_rate_msps: float = 20.0
    pri_us: float = 512.0
    pw_us: float = 350.0
    bandwidth_khz: float = 100.0
    offset_khz: float = 0.0
    tx1_gain_db: int = 0
    tx2_gain_db: int = 0
    tx1_enabled: bool = True
    tx2_enabled: bool = True                  # same signal on both channels when both are on
    amplitude: float = 0.8
    direction: str = "Up chirp"
    rf_bandwidth_mhz: str = "Auto"            # "Auto" = sample rate (as in BladeRF_TX_RX.py), max 56 MHz
    mode: str = "Hardware"
    device_identifier: str = ""               # empty = first bladeRF, e.g. "*:serial=xxxx"
    num_buffers: int = 32
    buffer_size: int = 8192
    num_transfers: int = 8
    timeout_ms: int = 3500

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
    def tx_channels(self):
        """0-based bladeRF TX channel indices that transmit (TX1 = 0, TX2 = 1)."""
        return tuple(i for i, on in ((0, self.tx1_enabled), (1, self.tx2_enabled)) if on)

    @property
    def tx_gains(self):
        return tuple((self.tx1_gain_db, self.tx2_gain_db)[i] for i in self.tx_channels)

    @property
    def tx_label(self):
        return " + ".join(f"TX{i+1}" for i in self.tx_channels) or "no channel"

    @property
    def rf_bandwidth_hz(self):
        if str(self.rf_bandwidth_mhz).strip().lower() == "auto":
            return int(round(min(max(self.fs, 200e3), 56e6)))
        return int(round(float(self.rf_bandwidth_mhz) * 1e6))

    def validate(self):
        numeric = (self.carrier_mhz, self.sample_rate_msps, self.pri_us,
                   self.pw_us, self.bandwidth_khz, self.offset_khz,
                   self.amplitude, self.tx1_gain_db, self.tx2_gain_db)
        if not all(math.isfinite(x) for x in numeric):
            raise ValueError("Enter finite numeric values in every field.")
        if not MIN_RF_HZ <= self.carrier_mhz * 1e6 <= MAX_RF_HZ:
            raise ValueError("bladeRF 2.0 transmits between 47 and 6000 MHz.")
        if not self.tx_channels:
            raise ValueError("Enable TX1, TX2 or both.")
        if not MIN_FS_MSPS <= self.sample_rate_msps <= MAX_FS_MSPS:
            raise ValueError(f"Use a bladeRF sample rate between {MIN_FS_MSPS} and {MAX_FS_MSPS} MSPS.")
        if len(self.tx_channels) == 2 and self.sample_rate_msps > MAX_FS_TWO_CHANNELS_MSPS:
            raise ValueError(f"With TX1 + TX2 use at most {MAX_FS_TWO_CHANNELS_MSPS} MSPS "
                             "(two 16-bit channels at higher rates exceed USB 3 throughput).")
        if not 0 < self.pw_us < self.pri_us:
            raise ValueError("Pulse width must be positive and smaller than PRI.")
        if not 0 < self.bandwidth_khz:
            raise ValueError("Chirp bandwidth must be greater than zero.")
        if not 0 < self.amplitude <= 1:
            raise ValueError("DAC amplitude must be greater than 0 and at most 1.")
        if round(DAC_FULL_SCALE * self.amplitude) < 1:
            raise ValueError("DAC amplitude is too small for a nonzero 12-bit waveform. Use at least 0.0003.")
        for name in ('tx1_gain_db', 'tx2_gain_db'):
            gain = getattr(self, name)
            if int(gain) != gain or not GAIN_RANGE_DB[0] <= gain <= GAIN_RANGE_DB[1]:
                raise ValueError(f"{name[:3].upper()} gain must be a whole number of dB from "
                                 f"{GAIN_RANGE_DB[0]} to {GAIN_RANGE_DB[1]}.")
        if self.direction not in ("Up chirp", "Down chirp"):
            raise ValueError("Choose Up chirp or Down chirp.")
        if self.mode not in ("Hardware", "Simulation"):
            raise ValueError("Choose Hardware or Simulation.")
        try:
            rf_bw = self.rf_bandwidth_hz
        except ValueError as exc:
            raise ValueError("RF bandwidth must be Auto or a number of MHz.") from exc
        if not 200e3 <= rf_bw <= 56e6:
            raise ValueError("RF bandwidth must be between 0.2 and 56 MHz.")
        if self.buffer_size % 1024 or self.num_transfers >= self.num_buffers or self.timeout_ms <= 0:
            raise ValueError("Buffer size must be a multiple of 1024 and transfers fewer than buffers.")
        edge = abs(self.offset_khz * 1e3) + self.bandwidth_khz * 1e3 / 2
        if edge >= self.fs / 2:
            raise ValueError("The chirp exceeds the sampled band. Reduce BW/offset or increase sample rate.")
        if edge >= self.rf_bandwidth_hz / 2:
            raise ValueError("The chirp exceeds the RF bandwidth. Increase the RF bandwidth or reduce BW/offset.")
        rf_low = self.carrier_mhz * 1e6 + self.offset_khz * 1e3 - self.bandwidth_khz * 500
        rf_high = rf_low + self.bandwidth_khz * 1e3
        if rf_low < MIN_RF_HZ or rf_high > MAX_RF_HZ:
            raise ValueError("The complete RF chirp must lie between 47 MHz and 6 GHz for bladeRF.")
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
                "rf_bandwidth_hz": self.rf_bandwidth_hz,
                "tx_channels": self.tx_label,
                "tx_gains_db": list(self.tx_gains),
                "dac_peak_code": int(round(DAC_FULL_SCALE * self.amplitude)),
                "sample_format": "SC16 Q11 (int16 I/Q, full scale 2047)",
            },
            "waveform": "LFM chirp, phase reset at each pulse, followed by zero IQ; identical on every enabled TX channel",
            "timing_note": "Durations are rounded to samples, not USB buffer multiples. Host timestamps are not RF arrival timestamps.",
        }


def make_period(settings: TxSettings):
    """Build exactly one PRI as interleaved SC16 Q11 IQ (int16, full scale 2047).

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
    peak = round(DAC_FULL_SCALE * settings.amplitude)
    iq = np.zeros(2 * settings.period_samples, dtype=np.int16)
    iq[:2*n:2] = np.rint(peak * np.cos(phase)).astype(np.int16)
    iq[1:2*n:2] = np.rint(peak * np.sin(phase)).astype(np.int16)
    return iq


def load_bladerf():
    try:
        module = importlib.import_module('bladerf')
    except (ImportError, OSError) as exc:
        raise RuntimeError('Cannot load the bladerf Python bindings / libbladeRF. Use the Python environment '
                           'where BladeRF_TX_RX.py works, or choose Simulation. Details: '+str(exc)) from exc
    return module, getattr(module, '_bladerf', module)


def open_bladerf(settings):
    """Open, configure every enabled TX channel identically, set up the sync stream, enable.

    Returns (device, channels, report). TX_X2 is used when both channels are enabled:
    the bladeRF 2.0 shares one TX LO and streams both channels from one interleaved buffer.
    """
    module, api = load_bladerf()
    device = getattr(module, 'BladeRF', getattr(api, 'BladeRF', None))(settings.device_identifier.strip() or None)
    channels = []
    try:
        try:
            count = int(device.get_channel_count(api.Direction.TX))
        except Exception:
            count = None                       # older bindings: let channel setup report it
        if count is not None and max(settings.tx_channels) >= count:
            raise RuntimeError(f'This bladeRF has {count} TX channel(s); TX2 needs a two-channel bladeRF 2.0 micro.')
        channel_tx = getattr(module, 'CHANNEL_TX', None) or api.CHANNEL_TX
        for index, gain in zip(settings.tx_channels, settings.tx_gains):
            ch = device.Channel(channel_tx(index))
            ch.frequency = int(round(settings.carrier_mhz*1e6))
            ch.sample_rate = int(round(settings.fs))
            ch.bandwidth = int(settings.rf_bandwidth_hz)
            try:
                ch.gain_mode = type(ch.gain_mode).Manual
            except Exception:
                pass                           # TX has no AGC on some boards/bindings
            ch.gain = int(gain)
            channels.append(ch)
        actual_fs = int(channels[0].sample_rate)
        if abs(actual_fs-settings.fs) > 1:
            raise RuntimeError(f'bladeRF set {actual_fs} Hz instead of {settings.fs:.0f} Hz; the labels assume '
                               'the exact rate. Choose another sample rate.')
        layout = api.ChannelLayout.TX_X2 if len(channels) == 2 else api.ChannelLayout.TX_X1
        device.sync_config(layout=layout, fmt=api.Format.SC16_Q11, num_buffers=settings.num_buffers,
                           buffer_size=settings.buffer_size, num_transfers=settings.num_transfers,
                           stream_timeout=settings.timeout_ms)
        for ch in channels:
            ch.enable = True
        parts = [f'TX{i+1} {int(ch.frequency)/1e6:.6f} MHz, gain {ch.gain} dB, RF BW {int(ch.bandwidth)/1e6:g} MHz'
                 for i, ch in zip(settings.tx_channels, channels)]
        report = (f"bladeRF {actual_fs/1e6:g} MSPS SC16 Q11, {'TX_X2 (same signal on both)' if len(channels) == 2 else 'TX_X1'}"
                  f" | " + " | ".join(parts))
        return device, channels, report
    except Exception:
        for ch in channels:
            try:
                ch.enable = False
            except Exception:
                pass
        device.close()
        raise


def duplicate_channels(chan_buf, out_buf):
    """TX_X2 buffer: for every sample, I/Q of TX1 then I/Q of TX2 (identical signal)."""
    n = chan_buf.size//2
    pairs = chan_buf.reshape(n, 2)
    view = out_buf.reshape(n, 2, 2)
    view[:, 0, :] = pairs
    view[:, 1, :] = pairs
    return out_buf


def transmit(device, data, num_samples):
    """num_samples counts samples of all channels together (2 x per-channel for TX_X2)."""
    try:
        device.sync_tx(data, num_samples)
    except TypeError:                          # bindings that only accept bytes-like buffers
        device.sync_tx(bytearray(data.tobytes()), num_samples)


@dataclasses.dataclass(frozen=True)
class TrainingSettings(TxSettings):
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

    def as_radio(self, pulse_samples, bandwidth_hz):
        fields = {f.name: getattr(self, f.name) for f in dataclasses.fields(TxSettings)}
        fields['pw_us'] = pulse_samples/self.fs*1e6
        fields['bandwidth_khz'] = bandwidth_hz/1000
        return TxSettings(**fields)

    def validate(self):
        values = (self.sample_rate_msps, self.pri_us, self.pw_min_us, self.pw_max_us,
                  self.bw_min_khz, self.bw_max_khz, self.bw_step_hz)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Enter finite numbers in the range and timing fields.")
        if not MIN_FS_MSPS <= self.sample_rate_msps <= MAX_FS_MSPS:
            raise ValueError(f"Use a bladeRF sample rate between {MIN_FS_MSPS} and {MAX_FS_MSPS} MSPS.")
        if not 0 < self.pw_min_us <= self.pw_max_us < self.pri_us:
            raise ValueError("Require 0 < PW min <= PW max < PRI.")
        if self.pri_us > MAX_PERIOD_SAMPLES/self.fs*1e6:
            raise ValueError("PRI is too large for the waveform buffer.")
        if not 0 < self.bw_min_khz <= self.bw_max_khz:
            raise ValueError("Require 0 < BW min <= BW max.")
        if self.bw_step_hz <= 0:
            raise ValueError("BW step must be positive, in Hz. 1/Fs is a time step for PW.")
        for name, lower, upper in (('pri_repeat', 1, 1_000_000), ('pair_count', 2, 20000), ('seed', 0, 2**32-1)):
            val = getattr(self, name)
            if not math.isfinite(val) or int(val) != val or not lower <= val <= upper:
                raise ValueError(f"{name.replace('_',' ')} must be an integer between {lower} and {upper}.")
        lo, hi, bw_lo, step, bw_levels = self.grid()
        if lo < 2 or hi < lo or hi >= self.period_samples:
            raise ValueError("PW limits contain no valid sample-grid pulse, or round into the PRI boundary.")
        if not 1 <= bw_levels <= 2**31:
            raise ValueError("The BW grid is too large. Increase BW step.")
        for levels in (hi-lo+1, bw_levels):
            if levels == 2 and self.pair_count % 2:
                raise ValueError("Use an even pair count for a two-value range so values change at the dataset loop boundary too.")
        self.as_radio(lo, float(bw_lo)).validate()
        self.as_radio(hi, float(bw_lo+(bw_levels-1)*step)).validate()
        if self.pair_count*BYTES_PER_SAMPLE*self.period_samples > MAX_DATASET_BYTES:
            raise ValueError("Dataset exceeds 512 MiB. Reduce pair count or PRI.")
        if not str(self.output_directory).strip():
            raise ValueError("Choose an output folder for training labels.")
        return self

    def summary(self):
        lo, hi, bw_lo, step, levels = self.grid()
        return {
            'sample_rate_hz': self.fs,
            'sample_interval_ns': 1e9/self.fs,
            'pri_samples': self.period_samples,
            'pri_us': self.actual_pri_us,
            'pw_min_samples': lo, 'pw_max_samples': hi,
            'pw_min_us': lo/self.fs*1e6, 'pw_max_us': hi/self.fs*1e6,
            'pw_levels': hi-lo+1,
            'bw_min_hz': float(bw_lo), 'bw_max_hz': float(bw_lo+(levels-1)*step),
            'bw_step_hz': float(step), 'bw_levels': levels,
            'pri_repeat': self.pri_repeat,
            'pair_count': self.pair_count,
            'samples_per_pair_group': self.period_samples*self.pri_repeat,
            'pair_hold_ms': self.period_samples*self.pri_repeat/self.fs*1e3,
            'dataset_duration_s': self.period_samples*self.pri_repeat*self.pair_count/self.fs,
            'waveform_storage_mib': self.pair_count*BYTES_PER_SAMPLE*self.period_samples/1024**2,
            'rf_bandwidth_hz': self.rf_bandwidth_hz,
            'tx_channels': self.tx_label,
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
    settings.validate()
    lo, hi, bw_lo, step, bw_levels = settings.grid()
    rng = np.random.default_rng(settings.seed)
    pw_indices, bw_indices = [], []
    for index in range(settings.pair_count):
        last = index == settings.pair_count-1
        pw_banned = ([pw_indices[-1]] if index else []) + ([pw_indices[0]] if last else [])
        bw_banned = ([bw_indices[-1]] if index else []) + ([bw_indices[0]] if last else [])
        pw_indices.append(choose_other(rng, hi-lo+1, pw_banned))
        bw_indices.append(choose_other(rng, bw_levels, bw_banned))
    return (
        np.asarray(pw_indices, dtype=np.int64)+lo,
        np.asarray([float(bw_lo+i*step) for i in bw_indices], dtype=np.float64),
    )


LABEL_FIELDS = ['pair_index', 'first_pri_index', 'first_sample_index', 'start_time_s',
                'PW_samples', 'PW_us', 'BW_Hz', 'BW_kHz', 'PRI_samples', 'PRI_us',
                'PRI_repeat', 'chirp_rate_Hz_per_s', 'sample_rate_Hz', 'carrier_Hz']


def save_plan(settings, plan, prefix='tx'):
    root = Path(settings.output_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    folder = root/(prefix+'_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    folder.mkdir()
    pw, bw = plan
    group_samples = settings.period_samples*settings.pri_repeat
    with (folder/'labels.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        sign = 1 if settings.direction == 'Up chirp' else -1
        for index, (n, bandwidth) in enumerate(zip(pw, bw)):
            writer.writerow(dict(zip(LABEL_FIELDS, (
                index, index*settings.pri_repeat, index*group_samples, index*group_samples/settings.fs,
                int(n), n/settings.fs*1e6, bandwidth, bandwidth/1000,
                settings.period_samples, settings.actual_pri_us, settings.pri_repeat,
                sign*bandwidth/(n/settings.fs), settings.fs, round(settings.carrier_mhz*1e6),
            ))))
    requested = dataclasses.asdict(settings)
    requested.pop('pw_us', None)
    requested.pop('bandwidth_khz', None)
    metadata = {
        'requested': requested, 'actual': settings.summary(),
        'created_at': dt.datetime.now().astimezone().isoformat(),
        'sequence': 'Seeded random pairs; each changing parameter differs from the previous group, including loop wrap.',
        'repetition': 'Each pair repeats for PRI_repeat consecutive PRIs. Entire dataset loops until Stop.',
        'label_scope': 'Planned labels for one dataset pass, not verified radiated samples or received training data.',
        'receiver_alignment': 'Detect pulse/group alignment in the received data. Host timestamps and GUI status are not RF timestamps.',
        'pass_mapping': 'global_pair_index = pass_index * pair_count + pair_index; pass indices start at 0.',
        'fixed_ranges': 'A range containing one grid value stays constant.',
    }
    (folder/'settings.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    return folder


class TrainingStream:
    """Repeat each period R times, then advance the pair, across USB buffers."""
    def __init__(self, periods, repeats):
        self.periods = periods
        self.repeats = repeats
        self.pair = 0
        self.repeat = 0
        self.offset = 0
        self.total_bytes = 0

    def fill(self, buffer, length):
        pos = 0
        width = self.periods.shape[1]
        while pos < length:
            count = min(length-pos, width-self.offset)
            buffer[pos:pos+count] = self.periods[self.pair, self.offset:self.offset+count]
            pos += count
            self.offset += count
            if self.offset == width:
                self.offset = 0
                self.repeat += 1
                if self.repeat == self.repeats:
                    self.repeat = 0
                    self.pair = (self.pair+1) % len(self.periods)
        self.total_bytes += length

    def position(self):
        group_bytes = self.periods.shape[1]*self.repeats
        completed_groups = self.total_bytes//group_bytes
        last_byte = max(0, self.total_bytes-1)
        absolute_group = last_byte//group_bytes
        return {
            'samples': self.total_bytes//2,
            'completed_groups': completed_groups,
            'completed_passes': completed_groups//len(self.periods),
            'last_pair': absolute_group % len(self.periods),
            'last_pass': absolute_group//len(self.periods),
            'last_repeat': last_byte//self.periods.shape[1] % self.repeats,
        }


def training_worker(connection, config):
    settings = TrainingSettings(**config)
    device = folder = stream = None
    channels = []
    started_tx = False
    stop_event = threading.Event()
    cleanup_errors = []
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
        send('log', text='Generating reproducible PW/BW pairs and planned labels...')
        plan = make_plan(settings)
        folder = save_plan(settings, plan)
        send('files', folder=str(folder))
        if stopped():
            return
        periods = np.empty((settings.pair_count, 2*settings.period_samples), dtype=np.int16)
        progress_step = max(1, settings.pair_count//100)
        for index, (n, bw) in enumerate(zip(*plan)):
            if index % 16 == 0 and stopped():
                return
            periods[index] = make_period(settings.as_radio(int(n), float(bw)))
            if index % progress_step == 0 or index == settings.pair_count-1:
                send('preparing', percent=100*(index+1)/settings.pair_count)
        stream = TrainingStream(periods, settings.pri_repeat)
        if stopped():
            return
        n_chan = len(settings.tx_channels)
        chan_buf = np.empty(2*BLOCK_SAMPLES, dtype=np.int16)          # one channel, interleaved I/Q
        out_buf = np.empty(2*BLOCK_SAMPLES*n_chan, dtype=np.int16)    # TX_X2: TX1 and TX2 per sample
        if settings.mode == 'Hardware':
            device, channels, report = open_bladerf(settings)
            started_tx = True
            send('log', text=report)
        if stopped():
            return
        rf_started_at = dt.datetime.now().astimezone().isoformat()
        start_clock = time.monotonic()
        send('running', mode=settings.mode)
        send('log', text=(f'Training transmission started on {settings.tx_label}; dataset loops until Stop.'
                          if started_tx else f'SIMULATION ({settings.tx_label}): no RF output. Dataset loops until Stop.'))
        last_report = last_poll = next_sim = start_clock
        while True:
            if time.monotonic()-last_poll >= .05:
                last_poll = time.monotonic()
                if stopped():
                    break
            stream.fill(chan_buf, chan_buf.size)
            data = duplicate_channels(chan_buf, out_buf) if n_chan == 2 else chan_buf
            if started_tx:
                transmit(device, data, BLOCK_SAMPLES*n_chan)          # blocks while the device is busy
            else:
                next_sim += BLOCK_SAMPLES/settings.fs
                delay = next_sim-time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_sim = time.monotonic()
            now = time.monotonic()
            if now-last_report >= .2:
                position = stream.position()
                index = position['last_pair']
                send('stats', elapsed=now-start_clock, **position,
                     pw_us=float(plan[0][index]/settings.fs*1e6), bw_khz=float(plan[1][index]/1000))
                last_report = now
    except (EOFError, BrokenPipeError):
        result_state = 'control_connection_lost'
    except Exception as exc:
        result_state = 'error'
        send('error', text=str(exc))
    finally:
        stop_event.set()
        if device is not None:
            if started_tx:
                try:                           # end on silence: flush zeros through the buffers
                    zeros = np.zeros(2*BLOCK_SAMPLES*len(channels), dtype=np.int16)
                    for _ in range(max(1, settings.num_buffers*settings.buffer_size//BLOCK_SAMPLES)):
                        transmit(device, zeros, BLOCK_SAMPLES*len(channels))
                except Exception as exc:
                    cleanup_errors.append('zero flush: '+str(exc))
            for ch in channels:
                try:
                    ch.enable = False
                except Exception as exc:
                    cleanup_errors.append('disable channel: '+str(exc))
            try:
                device.close()
            except Exception as exc:
                cleanup_errors.append('close: '+str(exc))
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
        root.title('bladeRF | Training Signal Generator')
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
        tk.Label(header,text='bladeRF Training Signal Generator',bg=self.BG,fg=self.FG,font=(self.family,22,'bold')).pack(side='left')
        self.status_var = tk.StringVar(value='IDLE')
        self.status_label = tk.Label(header,textvariable=self.status_var,bg='#223345',fg=self.MUTED,
                                    font=(self.family,10,'bold'),padx=14,pady=8)
        self.status_label.pack(side='right')
        tk.Label(root,text='Hold a PW/BW pair for R pulses, then change both values. The planned sequence loops until Stop.',
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

        fields = [('pri_us','PRI (us)'),('pw_min_us','PW min (us)'),('pw_max_us','PW max (us)'),
                  ('bw_min_khz','BW min (kHz)'),('bw_max_khz','BW max (kHz)'),
                  ('pri_repeat','PRI repeat (R)'),('bw_step_hz','BW step (Hz)'),
                  ('pair_count','Number of pairs'),('seed','Random seed')]
        for row,(key,label) in enumerate(fields):
            add_field(training,row,key,label)
        ttk.Label(training,text='PW step = 1/Fs seconds.\nBW uses the separate step in Hz.\nRandom pairs; same seed reproduces the plan.',
                  style='Muted.TLabel',font=(self.family,9)).grid(row=len(fields),column=0,columnspan=2,sticky='w',pady=(9,0))
        radio_fields = [
            ('mode','Output mode',('Hardware','Simulation')),
            ('carrier_mhz','Carrier (MHz)',None),
            ('sample_rate_msps','Sample rate (MSPS)',SAMPLE_RATES),
            ('tx1_gain_db','TX1 gain (dB)',None),
            ('tx2_gain_db','TX2 gain (dB)',None),
            ('amplitude','DAC amplitude (0-1)',None),
            ('offset_khz','Baseband offset (kHz)',None),
            ('direction','Sweep direction',('Up chirp','Down chirp')),
            ('rf_bandwidth_mhz','RF bandwidth (MHz)',RF_BANDWIDTHS),
            ('device_identifier','Device (optional)',None),
        ]
        for row,(key,label,choices) in enumerate(radio_fields):
            add_field(radio,row,key,label,choices)
        channel_row = ttk.Frame(radio,style='Panel.TFrame')
        channel_row.grid(row=len(radio_fields),column=0,columnspan=2,sticky='w',pady=9)
        for key,text in (('tx1_enabled','Transmit on TX1'),('tx2_enabled','Transmit on TX2')):
            self.vars[key] = tk.BooleanVar(value=defaults[key])
            box = ttk.Checkbutton(channel_row,text=text,variable=self.vars[key])
            box.pack(side='left',padx=(0,16))
            self.editors.append((box,'normal'))
            self.vars[key].trace_add('write',self.schedule_preview)
        ttk.Label(radio,text='TX1 + TX2 send the same signal at the same time (one interleaved TX_X2 stream). '
                  'Gain is a hardware setting, not RF power in dBm.',style='Muted.TLabel',
                  font=(self.family,9),wraplength=330).grid(row=len(radio_fields)+1,column=0,columnspan=2,sticky='w')
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
        for col,(key,label) in enumerate([('grid','PW GRID'),('bw','BW STEP'),('hold','PAIR HOLD'),('duration','ONE PASS')]):
            cards.columnconfigure(col,weight=1)
            ttk.Label(cards,text=label,style='Muted.TLabel',font=(self.family,9)).grid(row=0,column=col,sticky='w')
            self.metrics[key] = tk.StringVar(value='-')
            ttk.Label(cards,textvariable=self.metrics[key],font=(self.family,13,'bold')).grid(row=1,column=col,sticky='w',pady=(4,0))
        plot_panel = ttk.Frame(right,style='Panel.TFrame',padding=12)
        plot_panel.grid(row=1,column=0,sticky='nsew',pady=(0,12))
        plot_panel.columnconfigure(0,weight=1)
        plot_panel.rowconfigure(1,weight=1)
        ttk.Label(plot_panel,text='PLANNED SEQUENCE - FIRST 12 PAIRS',foreground='#54b4ff',
                  font=(self.family,9,'bold')).grid(row=0,column=0,sticky='w')
        self.figure = Figure(figsize=(6.6,3.2),dpi=100,facecolor=self.PANEL)
        self.axes = self.figure.subplots(2,1,sharex=True)
        self.figure.subplots_adjust(left=.13,right=.97,top=.94,bottom=.16,hspace=.27)
        self.lines = []
        for ax,label,color in zip(self.axes,('PW (us)','BW (kHz)'),('#54b4ff','#61d2ac')):
            ax.set_facecolor(self.PANEL)
            ax.tick_params(colors=self.MUTED,labelsize=9)
            ax.set_ylabel(label,color=self.MUTED,fontsize=9)
            for spine in ax.spines.values():
                spine.set_color('#35475a')
            ax.grid(color='#2e4053',linewidth=.6)
            self.lines.append(ax.step([],[],where='post',color=color,linewidth=2)[0])
        self.axes[1].set_xlabel('Time within dataset (ms)',color=self.MUTED,fontsize=9)
        self.canvas = FigureCanvasTkAgg(self.figure,master=plot_panel)
        self.canvas.get_tk_widget().grid(row=1,column=0,sticky='nsew')
        table_frame = ttk.Frame(right,style='Panel.TFrame',padding=12)
        table_frame.grid(row=2,column=0,sticky='ew',pady=(0,12))
        ttk.Label(table_frame,text='LABEL PREVIEW - FIRST 30 PAIRS',foreground='#54b4ff',font=(self.family,9,'bold')).pack(anchor='w',pady=(0,5))
        self.table = ttk.Treeview(table_frame,columns=('pair','pw','bw','repeat','sample'),show='headings',height=5)
        for key,title,width in [('pair','Pair',55),('pw','PW (us)',120),('bw','BW (kHz)',110),('repeat','PRI repeat',95),('sample','First sample',135)]:
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
        self.log('Ready. PW uses sample timing; bandwidth uses the BW step in Hz.')
        self.preview()
        root.after(100,self.poll_worker)

    def read_settings(self):
        values = {key: var.get() for key,var in self.vars.items()}
        ints = {'pri_repeat','pair_count','seed','tx1_gain_db','tx2_gain_db'}
        strings = {'mode','direction','rf_bandwidth_mhz','device_identifier','output_directory'}
        try:
            for key,value in values.items():
                if key in BOOL_FIELDS or key in strings:
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
        self.metrics['hold'].set(f"{summary['pair_hold_ms']:.3f} ms")
        self.metrics['duration'].set(f"{summary['dataset_duration_s']:.3f} s")
        n = min(settings.pair_count,12)
        times = np.arange(n+1)*summary['pair_hold_ms']
        for ax,line,values in zip(self.axes,self.lines,(plan[0][:n]/settings.fs*1e6,plan[1][:n]/1000)):
            line.set_data(times,np.r_[values,values[-1]])
            low,high = float(min(values)),float(max(values))
            margin = max((high-low)*.2,abs(low)*.002,1e-3)
            ax.set_ylim(low-margin,high+margin)
            ax.set_xlim(0,times[-1])
        self.canvas.draw_idle()
        self.table.delete(*self.table.get_children())
        for index,(n_pw,bw) in enumerate(zip(*plan)):
            if index >= 30:
                break
            self.table.insert('', 'end', values=(index+1,f'{n_pw/settings.fs*1e6:.4f}',f'{bw/1000:g}',
                              settings.pri_repeat,index*settings.period_samples*settings.pri_repeat))
        fixed = []
        if summary['pw_levels'] == 1:
            fixed.append('PW')
        if summary['bw_levels'] == 1:
            fixed.append('BW')
        note = (f"Actual PRI {settings.actual_pri_us:.6f} us | {summary['pw_levels']:,} PW values | "
                f"{summary['bw_levels']:,} BW values | IQ memory {summary['waveform_storage_mib']:.1f} MiB | "
                f"{settings.tx_label} | RF BW {settings.rf_bandwidth_hz/1e6:g} MHz")
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
        self.log(f'Preparing {settings.pair_count:,} pairs, {settings.pri_repeat} pulses per pair.')
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
                    self.stats_var.set(f"Queued: pass {message['last_pass']+1}, pair {message['last_pair']+1}/{self.active_settings.pair_count}, "
                                       f"repeat {message['last_repeat']+1}/{self.active_settings.pri_repeat}\n"
                                       f"PW {message['pw_us']:.4f} us | BW {message['bw_khz']:g} kHz | "
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
            self.last_error = 'Worker did not finish stopping and was terminated. RF stop is unconfirmed; disconnect the bladeRF before restarting. Session metadata may be incomplete.'
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
    assert settings.carrier_mhz == 1000.0 and settings.fs == 20e6
    plan = make_plan(settings)
    summary = settings.summary()
    assert settings.period_samples == 10240
    assert summary['sample_interval_ns'] == 50
    assert abs(summary['pair_hold_ms']-1.536) < 1e-12
    assert summary['pw_levels'] == 1001 and summary['bw_levels'] == 101
    np.testing.assert_array_equal(plan[0], make_plan(settings)[0])
    np.testing.assert_array_equal(plan[1], make_plan(settings)[1])
    assert np.all((plan[0] >= 6000) & (plan[0] <= 7000))
    assert np.all((plan[1] >= 100000) & (plan[1] <= 200000)) and np.all((plan[1]-100000) % 1000 == 0)
    for values in plan:
        assert np.all(values != np.roll(values, 1)), 'Consecutive values must change, including loop boundary'
    frames = np.asarray([make_period(settings.as_radio(int(n), float(bw))) for n, bw in zip(*plan)])
    peak = round(DAC_FULL_SCALE*settings.amplitude)
    assert frames.dtype == np.int16 and np.abs(frames).max() <= peak
    for frame, n in zip(frames, plan[0]):
        assert np.all(frame[2*n:] == 0)
    stream = TrainingStream(frames, 3)
    one_pass = np.repeat(frames, 3, axis=0).ravel()
    parts = []
    for length in (2, 18, 262144, 99998, len(one_pass)+6):
        chunk = np.empty(length, dtype=np.int16)
        stream.fill(chunk, length)
        parts.append(chunk)
    actual = np.concatenate(parts)
    np.testing.assert_array_equal(actual, np.resize(one_pass, len(actual)))
    assert stream.position()['samples'] == len(actual)//2
    # Two-channel interleaving: TX1 and TX2 carry identical I/Q at every sample.
    chunk = np.empty(2*BLOCK_SAMPLES, dtype=np.int16)
    TrainingStream(frames, 3).fill(chunk, chunk.size)
    out = duplicate_channels(chunk, np.empty(4*BLOCK_SAMPLES, dtype=np.int16))
    for k in range(2):
        np.testing.assert_array_equal(out[2*k::4], chunk[0::2])
        np.testing.assert_array_equal(out[2*k+1::4], chunk[1::2])
    assert settings.tx_channels == (0, 1) and settings.tx_label == 'TX1 + TX2'
    single = dataclasses.replace(settings, tx2_enabled=False, sample_rate_msps=40).validate()
    assert single.tx_channels == (0,) and single.rf_bandwidth_hz == 40_000_000
    assert dataclasses.replace(settings, tx1_enabled=False).validate().tx_label == 'TX2'
    narrow = dataclasses.replace(settings, pw_min_us=300.01, pw_max_us=300.08,
                                 bw_min_khz=100.1, bw_max_khz=100.4, bw_step_hz=200)
    p, b = make_plan(narrow)
    assert np.all(p == 6001) and set(b) == {100100, 100300}
    for change in ({'pri_repeat': 0}, {'pw_max_us': 600}, {'bw_min_khz': 300}, {'bw_step_hz': 0},
                   {'sample_rate_msps': 70}, {'sample_rate_msps': 40},           # 40 MSPS needs one channel
                   {'tx1_enabled': False, 'tx2_enabled': False}, {'carrier_mhz': 30},
                   {'tx1_gain_db': 80}, {'tx2_gain_db': 1.5}, {'rf_bandwidth_mhz': '0.1'},
                   {'rf_bandwidth_mhz': 'wide'}, {'pw_min_us': 300.01, 'pw_max_us': 300.02},
                   {'pw_min_us': 300, 'pw_max_us': 300.0625, 'pair_count': 3},
                   {'pair_count': 20000, 'pri_us': 10000}):
        try:
            dataclasses.replace(settings, **change).validate()
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid settings accepted: '+str(change))
    print('PASS: 1 GHz / 20 MSPS defaults, sample-grid PW and BW grid, reproducibility, changing pairs, '
          'SC16 Q11 bounds, exact PRI repeats, buffer continuity, TX1/TX2 interleaving, channel/rate/gain '
          'limits and invalid settings.')


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
