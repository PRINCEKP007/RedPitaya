# Red Pitaya live signal viewer

`RedPitaya_Live_Viewer.py` shows the live Red Pitaya signal: the time-domain trace on top and the
PSD below. It is not connected to the ESN code.

```
python RedPitaya_Live_Viewer.py              # GUI
python RedPitaya_Live_Viewer.py --self-test  # checks the PSD/buffer code, no GUI or hardware
```
Needs numpy, scipy, PyQt5, pyqtgraph and a `python_lib` build that matches your Python version.
Builds are supplied for 3.9–3.13; on 3.14, use a Python 3.13 virtual environment.

**Stream** (set when you press Start):
- Source: Red Pitaya, a replay file (`.npy`, or float32 `.bin`), or a built-in simulation.
- Board IP, or leave it empty to auto-discover.
- Inputs: IN1, IN2 or both. With both, the two channels are always shown from the same instant.
- Decimation from 1 to 8192.
- Volts per full scale: 1 for the LV jumper, about 20 for HV.
- Extra `key=value` lines sent to the streaming server.

**Time domain** (applies immediately): time span and Y auto-scale. The plot keeps the minimum and
maximum of each display bin, so short spikes are never hidden.

**PSD** (applies immediately; Welch method, one-sided):
- FFT points: 64–131072.
- Overlap: 0–95 %.
- Window: Hann, Hamming, Blackman, Blackman-Harris, Flat top, Rectangular, or Kaiser with adjustable β.
- Data per estimate: how much of the newest data each estimate averages.
- Averaging: none, exponential (N updates) or max hold.
- Units: dB/Hz (density) or dB per bin (power).
- DC removal, and a log frequency axis.

The plot title shows the resulting resolution bandwidth and the number of segments per estimate.
The readout gives mean, RMS, peak-to-peak, the PSD peak and the median noise floor for each
channel. **Save trace** writes `.npy` (volts, one column per channel) and **Save PSD** writes CSV.

The status panel shows throughput, gaps, FPGA lost samples and host overflows. Two channels
double the network load, so use decimation 8 or higher for two-channel streaming over 1 Gbit Ethernet.
