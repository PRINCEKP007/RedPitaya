# Red Pitaya RX: vapour cell photodetector → PW / BW

The HackRF TX (`../hackrf/HackRF_TX_GUI.py`) sends the pulsed chirp (PRI fixed at 512 µs,
PW 300–350 µs, BW 3–4 MHz, 1 GHz). It illuminates the atomic vapour cell, and the
photodetector output is streamed by the Red Pitaya. Those samples are **real, not I/Q**.

| File | Purpose |
|---|---|
| `RedPitaya_ESN_Dashboard.py` | Live UI (like the bladeRF one): stream, plot, estimate PW/BW, compare with the TX truth, **record labelled training windows** |
| `TrainESN_Real.py` | Trains separate PW and BW readouts on real windows |
| `rp_core.py` | Shared features, real reservoir, and readouts (used by both of the above) |
| `SDR_UDP_Reference.py` | UDP TX truth/MSE. This copy works without an RF carrier and provides labels for recording |

## Workflow
1. On the Red Pitaya, start the streaming application. The UI uses `../../python_lib`
   (pick the `.so` that matches your Python version).
2. Start `HackRF_TX_GUI.py` and send the UDP reference to this PC's IP.
3. Run `python RedPitaya_ESN_Dashboard.py` and start the UDP listener. Choose the detector
   input and the decimation (32 = 3.90625 MSPS, 2000 samples per PRI), then press **Start RX**.
   No model is needed for this step.
4. Tick **Record labelled windows**, then step the TX through PW/BW settings. Each settled
   window is appended to `rp_dataset/` (`windows_f32.bin`, `labels.csv`, `meta.json`).
   Windows recorded before the settling guard expires, or while the TX is stopped, are skipped.
5. Run `python TrainESN_Real.py` (`DATA_SOURCE = "recorded"`). This writes `rp_esn_model.npz`.
6. Restart the dashboard with that model. Keep the same decimation, because the model checks it.

## How PW and BW are estimated from a real trace
- The 3-PRI window is folded into one PRI and averaged. This removes the capture phase and averages the 3 pulses.
- **PW**: pulse start and end are found as a pair of level steps (a matched step detector),
  constrained to the PW range seen in training, then refined at the 50% crossings.
  Threshold widths and SNR are also included as features.
- **BW**: while the chirp sweeps across the atomic resonance (linewidth Γ), the response lasts
  about Γ/BW of the pulse. The width, position, spread and resampled shape of the in-pulse
  response are the BW features.
- **ESN**: a real tanh reservoir driven by the standardised trace. Each parameter's own
  readout uses it only when cross-validation says it helps.
- Pulse polarity (whether the pulse raises or lowers the detector level) is learned from the
  labels and stored in the model.
- Cross-validation and the test split are grouped by PW/BW setting, so repeated windows of one
  setting never appear in both training and validation.
