import numpy as np
import bladerf
import threading
import sys
from collections import deque
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from PyQt5.QtCore import QRectF

# ── SDR config ──────────────────────────────────────────────────────────────
Fs = 4e6
Fc = 90e6
tp = 1e-3

N = int(1e-3*Fs)


sdr = bladerf.BladeRF()

rx = sdr.Channel(bladerf.CHANNEL_RX(0))
tx = sdr.Channel(bladerf.CHANNEL_TX(0))

# ── RX Config ───────────────────────────────────────────────────────────────
rx.frequency = int(Fc)
rx.sample_rate = int(Fs)
rx.bandwidth = int(Fs)

rx.gain_mode = type(rx.gain_mode).Manual
rx.gain = 40
rx.enable = True

# ── TX Config ───────────────────────────────────────────────────────────────
tx.frequency = int(Fc)
tx.sample_rate = int(Fs)
tx.bandwidth = int(Fs)

# Set TX gain (adjust this if the signal is too weak or too strong/cliping)
tx.gain_mode = type(tx.gain_mode).Manual
tx.gain = 0  
tx.enable = True

print(f"RX Gain Mode: {rx.gain_mode}, RX Gain: {rx.gain}")
print(f"TX Gain Mode: {tx.gain_mode}, TX Gain: {tx.gain}")

# ── Sync Configs ────────────────────────────────────────────────────────────
# RX Sync Config
sdr.sync_config(layout=bladerf._bladerf.ChannelLayout.RX_X1,
                fmt=bladerf._bladerf.Format.SC16_Q11,
                num_buffers=16, buffer_size=8192,
                num_transfers=8, stream_timeout=3500)

# TX Sync Config
sdr.sync_config(layout=bladerf._bladerf.ChannelLayout.TX_X1,
                fmt=bladerf._bladerf.Format.SC16_Q11,
                num_buffers=16, buffer_size=8192,
                num_transfers=8, stream_timeout=3500)


# ── TX Signal Generation (500 kHz Sine Wave) ────────────────────────────────
# We use a multiple of the buffer size for smooth continuous transmission
N_tx = 8192  
f_tx = 500e3  # 500 kHz
t_tx = np.arange(N_tx) / Fs

# Generate complex exponential (e^(j*2*pi*f*t)) for a single sideband sine wave
# Amplitude is set to 0.8 to avoid clipping the DAC
tx_iq = 0.8 * np.exp(1j * 2 * np.pi * f_tx * t_tx)

# Convert to int16 Q11 format (interleaved I and Q)
tx_i = (np.real(tx_iq) * 2047).astype(np.int16)
tx_q = (np.imag(tx_iq) * 2047).astype(np.int16)
tx_data = np.empty(2 * N_tx, dtype=np.int16)
tx_data[0::2] = tx_i
tx_data[1::2] = tx_q
tx_bytes = bytearray(tx_data.tobytes())


# ── Shared buffer & Threads ──────────────────────────────────────────────────
rx_buf = deque(maxlen=1)

def rx_thread():
    while True:
        buf = bytearray(4 * N)
        sdr.sync_rx(buf, N)
        data = np.frombuffer(buf, dtype=np.int16)
        iq = (data[0::2] + 1j * data[1::2]) / 2047
        iq = iq.astype(np.complex64)
        rx_buf.append(iq)

def tx_thread():
    """Continuously transmit the 500kHz sine wave"""
    while True:
        sdr.sync_tx(tx_bytes, N_tx)

# Start both threads
threading.Thread(target=rx_thread, daemon=True).start()
threading.Thread(target=tx_thread, daemon=True).start()

# ── pyqtgraph UI ─────────────────────────────────────────────────────────────
pg.setConfigOptions(antialias=False)          # faster for SDR rates
app = QtWidgets.QApplication(sys.argv)

win = pg.GraphicsLayoutWidget(title="BladeRF RX/TX — Monitor")
win.resize(1100, 650)
win.setBackground("#01070f")

GRID_ALPHA = 0.15
PEN_I   = pg.mkPen('#00d4ff', width=1)
PEN_Q   = pg.mkPen('#ff6b6b', width=1)
PEN_PSD = pg.mkPen('#ffb347', width=1)

Nfft = 8192

t_ms   = np.arange(N) / Fs * 1e3
freqs  = np.fft.fftshift(np.fft.fftfreq(Nfft, 1/Fs))   # Hz

dummy  = np.zeros(N)
dummyF  = np.zeros(Nfft)

Nframe = 100
waterfall = np.ones((Nfft,Nframe))*.01

# ── Row 0: Time-domain I/Q ───────────────────────────────────────────────────
p_iq = win.addPlot(row=0, col=0, colspan=3, title="Time domain")
p_iq.setLabel('bottom', 'Time', units='ms')
p_iq.setLabel('left', 'Amplitude')
p_iq.setYRange(-1, 1)
p_iq.showGrid(x=True, y=True, alpha=GRID_ALPHA)
p_iq.addLegend(offset=(10, 10))
curve_i = p_iq.plot(t_ms, dummy, pen=PEN_I, name='I')
#curve_q = p_iq.plot(t_ms, dummy, pen=PEN_Q, name='Q')

# ── Row 1: Power spectrum ────────────────────────────────────────────────────
p_psd = win.addPlot(row=1, col=0, colspan=3, title="Power spectrum (dBFS)")
p_psd.setLabel('bottom', 'Frequency', units='Hz')
p_psd.setLabel('left', 'Power', units='dBFS')
p_psd.setYRange(-100, 5)
p_psd.showGrid(x=True, y=True, alpha=GRID_ALPHA)
curve_psd = p_psd.plot(freqs, dummyF, pen=PEN_PSD)

p_water = win.addPlot(row=2, col=0, colspan=3,title="Waterfall")
img = pg.ImageItem()
p_water.addItem(img)
p_water.setLabel('bottom', 'Frequency', units='Hz')
p_water.setLabel('left', 'TimeHist')
img.setImage(waterfall,autoLevels=False,levels=(-120,0))
img.setRect(QRectF(freqs[0],0,freqs[-1],99))
cmap = pg.colormap.get('inferno')
img.setColorMap(cmap)

avg_len = 1  # number of frames to average
Hist = np.zeros((avg_len,N),dtype=complex)

avg_len1 = 100
spec_buf = np.zeros((avg_len1,Nfft))
t_ms   = np.arange(N*avg_len) / Fs * 1e3

window = np.hanning(N)

def update():
    if not rx_buf:
        return
    rx = rx_buf[-1]

    Hist[0:avg_len-1,:] = Hist[1:avg_len,:]
    Hist[avg_len-1,:] = rx

    SigRxHist = Hist.reshape(N*avg_len)

    curve_i.setData(t_ms, np.real(SigRxHist))
    #curve_q.setData(t_ms, np.imag(SigRxHist))
    
    # take first Nfft samples, apply matching window
    rx_fft = rx[:N] * window
    spec = np.fft.fftshift(np.fft.fft(rx_fft, Nfft))
    power = np.abs(spec) ** 2

    spec_buf[0:avg_len1-1,:] = spec_buf[1:avg_len1,:]
    spec_buf[avg_len1-1,:] = power

    avg_power = np.mean(spec_buf, axis=0)

    psd = 10 * np.log10(avg_power / Nfft**2 + 1e-12)
    curve_psd.setData(freqs, psd)

    waterfall[:,0:Nframe-1] = waterfall[:,1:Nframe]
    waterfall[:,Nframe-1] = psd

    img.setImage(waterfall)

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10)         

win.show()
sys.exit(app.exec())