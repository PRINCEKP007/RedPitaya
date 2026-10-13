"""UDP TX truth and running MSE. Standard library only; no RF or GUI imports.

Place beside both SDR GUIs. UDP carries programmed reference values, never I/Q.
Receiver time uses monotonic arrival times; PC clocks need not be synchronized.
"""
from collections import deque
import ipaddress
import json
import math
import socket
import threading
import time
import uuid

UDP_PORT = 5005
PROTOCOL = "sdr-tx-reference-v1"
HEARTBEAT_SECONDS = 0.25
STALE_SECONDS = 2.0
MAX_PACKET_BYTES = 4096


def ipv4(value):
    return str(ipaddress.IPv4Address(value.strip()))


def tx_parameters(settings):
    return dict(pri_us=settings.actual_pri_us, pw_us=settings.actual_pw_us,
                bw_khz=settings.bandwidth_khz, fs_hz=settings.fs,
                carrier_hz=round(settings.carrier_mhz * 1e6), offset_khz=settings.offset_khz,
                gain_db=settings.gain_db, direction=settings.direction,
                amp_enabled=settings.amp_enabled, amplitude=settings.amplitude,
                filter_hz=settings.filter_hz, pulse_samples=settings.pulse_samples,
                period_samples=settings.period_samples,
                chirp_rate_hz_per_s=(settings.bandwidth_khz * 1e9 / settings.actual_pw_us *
                                    (1 if settings.direction == "Up chirp" else -1)))


def decode_packet(data):
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError("Reference packet too large")
    packet = json.loads(data.decode("utf-8"))
    if not isinstance(packet, dict) or packet.get("protocol") != PROTOCOL:
        raise ValueError("Unknown reference protocol")
    for key in ("sender_id", "run_id"):
        if not isinstance(packet.get(key), str) or not 1 <= len(packet[key]) <= 80:
            raise ValueError("Invalid reference identity")
    if type(packet.get("seq")) is not int or packet["seq"] < 0:
        raise ValueError("Invalid reference sequence")
    if packet.get("state") not in ("starting", "running", "stopping", "stopped", "error"):
        raise ValueError("Invalid TX state")
    if packet.get("mode") not in ("Hardware", "Simulation"):
        raise ValueError("Invalid TX mode")
    p = packet.get("parameters")
    if not isinstance(p, dict):
        raise ValueError("Missing TX parameters")
    for key in ("pri_us", "pw_us", "bw_khz", "fs_hz", "carrier_hz", "offset_khz",
                "gain_db", "chirp_rate_hz_per_s"):
        value = p.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Non-finite/missing reference parameter: " + key)
    if not (0 < p["pw_us"] < p["pri_us"] <= 1e6 and
            1e6 <= p["fs_hz"] <= 20e6 and 0 < p["bw_khz"] <= p["fs_hz"] / 1000 and
            1e6 <= p["carrier_hz"] <= 6e9 and p.get("direction") in ("Up chirp", "Down chirp")):
        raise ValueError("Invalid reference range")
    if not math.isclose(abs(p["chirp_rate_hz_per_s"]), p["bw_khz"] * 1e9 / p["pw_us"], rel_tol=1e-9):
        raise ValueError("Inconsistent chirp-rate reference")
    return packet


class UDPPublisher:
    """Heartbeats while active; one-shot updates on stop/error, then silence."""
    def __init__(self):
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stop_event = threading.Event()
        self.sender_id = uuid.uuid4().hex
        self.seq = 0
        self.revision = 0
        self.sent_revision = -1
        self.endpoint = None
        self.packet = None
        self.status = "UDP disabled"
        self.thread = threading.Thread(target=self._run, name="TX-UDP", daemon=True)
        self.thread.start()

    def configure(self, enabled, host, port):
        endpoint = (ipv4(host), int(port)) if enabled else None
        if endpoint is not None and not 1 <= endpoint[1] <= 65535:
            raise ValueError("UDP port must be 1–65535")
        with self.lock:
            self.endpoint = endpoint
            self.packet = None
            self.revision += 1
            self.status = "UDP ready" if enabled else "UDP disabled"
        self.wake.set()

    def update(self, state, settings=None, new_run=False):
        with self.lock:
            if settings is None and self.packet is None:
                return
            run = uuid.uuid4().hex if new_run or self.packet is None else self.packet["run_id"]
            if settings is None:
                self.packet = {**self.packet, "state": state}
            else:
                self.packet = dict(protocol=PROTOCOL, sender_id=self.sender_id, run_id=run,
                                   state=state, mode=settings.mode, parameters=tx_parameters(settings))
            self.revision += 1
        self.wake.set()

    def snapshot(self):
        with self.lock:
            return self.status

    def _run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(0.2)
            while True:
                self.wake.wait(HEARTBEAT_SECONDS)
                self.wake.clear()
                with self.lock:
                    endpoint = self.endpoint
                    packet = self.packet
                    revision = self.revision
                    active = packet is not None and packet["state"] in ("starting", "running")
                    should_send = endpoint is not None and packet is not None and (
                        active or revision != self.sent_revision)
                    if should_send:
                        self.seq += 1
                        packet = {**packet, "seq": self.seq}
                if not should_send:
                    if self.stop_event.is_set():
                        break
                    continue
                try:
                    data = json.dumps(packet, allow_nan=False, separators=(",", ":")).encode("utf-8")
                    sock.sendto(data, endpoint)
                    status = f"UDP sent #{packet['seq']} → {endpoint[0]}:{endpoint[1]} ({packet['state']})"
                    if not active:
                        status += " · UDP paused"
                except (OSError, ValueError) as error:
                    status = "UDP error: " + str(error)
                with self.lock:
                    self.sent_revision = revision
                    self.status = status  # A send is not proof of delivery.
                if self.stop_event.is_set():
                    with self.lock:
                        pending = (self.endpoint is not None and self.packet is not None and
                                   self.revision != self.sent_revision)
                    if not pending:
                        break

    def close(self):
        self.stop_event.set()
        self.wake.set()
        self.thread.join(timeout=0.4)


class ReferenceComparator:
    """Thread-safe metrics, accumulated by inference workers, not display timer.

    Aggregates reset at a truth change, stop, stale period, source/mode change,
    or RX restart. The settling guard is conservative, not RF synchronization.
    """
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.RLock()
        self.mode = "UDP"
        self.settle_seconds = 0.5
        self.manual = (325.0, 3500.0)
        self.packet = None
        self.arrived = None
        self.changed = clock()
        self.sender = None
        self.source_ip = ""
        self.sequence = -1
        self.retired_senders = deque(maxlen=32)
        self.stale = False
        self.reason = "Waiting for UDP reference"
        self._reset()

    def _reset(self):
        self.count = self.rate_count = self.passed = 0
        self.pw_ss = self.bw_ss = self.rate_ss = 0.0
        self.last = None
        self.reason = "Waiting for eligible prediction"

    def configure(self, mode, settle_ms, pw_us=325.0, bw_khz=3500.0):
        config = (mode, float(settle_ms) / 1000, (float(pw_us), float(bw_khz)))
        with self.lock:
            if config != (self.mode, self.settle_seconds, self.manual):
                self.mode, self.settle_seconds, self.manual = config
                self.changed = self.clock()
                self._reset()

    def restart_rx(self):
        with self.lock:
            self.changed = self.clock()
            self._reset()

    def clear_udp(self):
        with self.lock:
            self.packet = None
            self.arrived = None
            self.sender = None
            self.sequence = -1
            self.retired_senders.clear()
            self._reset()
            self.reason = "Waiting for UDP reference"

    def accept(self, packet, source_ip):
        now = self.clock()
        with self.lock:
            if packet["sender_id"] in self.retired_senders:
                return False
            if self.sender == packet["sender_id"] and packet["seq"] <= self.sequence:
                return False
            # Lock to the first sender's IP until it stops or becomes stale.
            if self.packet and source_ip != self.source_ip and self.packet["state"] == "running" and now - self.arrived <= STALE_SECONDS:
                return False
            changed = (self.packet is None or self.sender != packet["sender_id"] or
                       any(self.packet[k] != packet[k] for k in ("run_id", "state", "mode", "parameters")) or self.stale)
            if self.sender and self.sender != packet["sender_id"]:
                self.retired_senders.append(self.sender)
            self.sender, self.sequence = packet["sender_id"], packet["seq"]
            self.packet, self.arrived, self.source_ip = packet, now, source_ip
            self.stale = False
            if changed and self.mode == "UDP":
                self.changed = now
                self._reset()
            return True

    def _reference(self, now):
        if self.mode == "Off":
            return None, "Reference comparison off"
        if self.mode == "Manual":
            return dict(pw_us=self.manual[0], bw_khz=self.manual[1],
                        chirp_rate_hz_per_s=self.manual[1] * 1e9 / self.manual[0]), "Manual reference"
        if self.packet is None:
            return None, "Waiting for UDP reference"
        # A confirmed inactive TX stays inactive without periodic heartbeats.
        if self.packet["state"] in ("stopping", "stopped", "error"):
            return None, "TX " + self.packet["state"] + " — MSE paused"
        if now - self.arrived > STALE_SECONDS:
            if not self.stale:
                self._reset()
                self.stale = True
            return None, "UDP stale — MSE paused"
        if self.packet["state"] != "running":
            return None, "TX " + self.packet["state"] + " — MSE paused"
        if self.packet["mode"] != "Hardware":
            return None, "TX simulation / no RF — MSE paused"
        return self.packet["parameters"], "Live UDP reference"

    def observe(self, result, settings):
        now = self.clock()
        with self.lock:
            ref, reason = self._reference(now)
            output = dict(reference_mode=self.mode, reference_run="", true_pw_us="", true_bw_khz="",
                          pw_error_ns="", bw_error_khz="", rate_error_Hz_per_s="",
                          mse_pw_ns2="", mse_bw_khz2="", mse_rate_Hz_per_s_squared="", mse_count=0,
                          comparison_status=reason)
            if ref is not None and self.mode == "UDP":
                if settings.source != "bladeRF":
                    reason = "Replay has no alignment to live TX — MSE paused"
                elif abs(ref["pri_us"] - settings.pri_us) > 0.5e6 / ref["fs_hz"] + 1e-9:
                    reason = "TX PRI differs from RX/model PRI — MSE paused"
                elif abs(ref["carrier_hz"] - round(settings.frequency_mhz * 1e6)) > 1:
                    reason = "TX/RX carriers differ — MSE paused"
                else:
                    output["reference_run"] = self.packet["run_id"]
            if ref is not None:
                output.update(true_pw_us=ref["pw_us"], true_bw_khz=ref["bw_khz"])
                if result["frame"].received_at - settings.window_seconds < self.changed + self.settle_seconds:
                    reason = "Settling / pre-change window — MSE paused"
            if not (ref is not None and reason in ("Live UDP reference", "Manual reference")):
                self.reason = reason
                output["comparison_status"] = reason
                return output
            pw_error = (result["pw_us"] - ref["pw_us"]) * 1000
            bw_error = result["bw_khz"] - ref["bw_khz"]
            if not (math.isfinite(pw_error*pw_error) and math.isfinite(bw_error*bw_error)):
                self.reason = "Non-finite estimate — MSE paused"
                output["comparison_status"] = self.reason
                return output
            self.count += 1
            self.pw_ss += pw_error**2
            self.bw_ss += bw_error**2
            self.passed += abs(pw_error) <= 1e9 / settings.fs + 1e-9
            rate_error = result["chirp_rate"] - abs(ref["chirp_rate_hz_per_s"])
            if math.isfinite(rate_error*rate_error):
                self.rate_count += 1
                self.rate_ss += rate_error**2
                output["rate_error_Hz_per_s"] = rate_error
            output.update(pw_error_ns=pw_error, bw_error_khz=bw_error,
                          mse_pw_ns2=self.pw_ss/self.count, mse_bw_khz2=self.bw_ss/self.count,
                          mse_rate_Hz_per_s_squared=self.rate_ss/self.rate_count if self.rate_count else "",
                          mse_count=self.count, comparison_status="Compared")
            self.last = output.copy()
            self.reason = "Compared"
            return output

    def snapshot(self):
        with self.lock:
            ref, reason = self._reference(self.clock())
            if ref is not None:
                reason = self.reason if self.reason not in ("Waiting for UDP reference", "Reference comparison off") else reason
            return dict(reference=ref, reason=reason, count=self.count, rate_count=self.rate_count,
                        mse_pw=self.pw_ss/self.count if self.count else None,
                        mse_bw=self.bw_ss/self.count if self.count else None,
                        mse_rate=self.rate_ss/self.rate_count if self.rate_count else None,
                        within_sample_percent=100*self.passed/self.count if self.count else None,
                        last=self.last, source_ip=self.source_ip,
                        age_seconds=self.clock()-self.arrived if self.arrived is not None else None)


class UDPListener:
    def __init__(self, comparator, bind_host="0.0.0.0", port=UDP_PORT, expected_tx_ip=""):
        self.comparator = comparator
        self.expected_tx_ip = ipv4(expected_tx_ip) if expected_tx_ip.strip() else ""
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.accepted = self.rejected = 0
        self.error = ""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind((ipv4(bind_host), int(port)))
            self.sock.settimeout(0.2)
        except Exception:
            self.sock.close()
            raise
        self.port = self.sock.getsockname()[1]
        self.comparator.clear_udp()
        self.thread = threading.Thread(target=self._run, name="RX-UDP", daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            try:
                data, address = self.sock.recvfrom(MAX_PACKET_BYTES + 1)
                packet = decode_packet(data)
                accepted = (not self.expected_tx_ip or address[0] == self.expected_tx_ip) and self.comparator.accept(packet, address[0])
                with self.lock:
                    if accepted:
                        self.accepted += 1
                    else:
                        self.rejected += 1
            except socket.timeout:
                continue
            except (ValueError, UnicodeError, KeyError, TypeError, OverflowError):
                with self.lock:
                    self.rejected += 1
            except OSError as error:
                if not self.stop_event.is_set():
                    with self.lock:
                        self.error = str(error)
                    self.comparator.clear_udp()
                break

    def snapshot(self):
        with self.lock:
            return dict(accepted=self.accepted, rejected=self.rejected, error=self.error)

    def close(self):
        self.stop_event.set()
        self.sock.close()
        self.thread.join(timeout=0.4)
        self.comparator.clear_udp()

