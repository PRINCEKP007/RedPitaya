# Red Pitaya RX: vapour cell photodetector → PW / BW

The HackRF TX (`../hackrf/HackRF_TX_GUI.py`) sends the pulsed chirp (PRI fixed at 512 µs,
PW 300–350 µs, BW 3–4 MHz, 1 GHz). It illuminates the atomic vapour cell, and the
photodetector output is streamed by the Red Pitaya. Those samples are **real, not I/Q**.

| File | Purpose |
|---|---|
| `RedPitaya_ESN_Dashboard.py` | Live UI (like the bladeRF one): stream, plot, estimate PW/BW, compare with the TX truth, **record labelled training windows** |
| `TrainESN_Real.py` | Trains separate PW and BW readouts on real windows |
| `TrainPhysicalRC.py` | **Atomic receiver as the reservoir**: tests whether the measured states separate the RF parameters, then trains linear readouts for PW, BW and the chirp rate K **directly**; writes `rc_report/` |
| `physical_rc.py` | Physical-reservoir states (fold, align, virtual nodes) and model used by the UI |
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

## Physical reservoir: the atomic receiver as the reservoir (`TrainPhysicalRC.py`)
The atoms already transform the RF signal nonlinearly when they convert it to an optical
signal. So the measured trace is used directly as the reservoir state, and only linear
readouts are trained. A digital transform after acquisition cannot add information that
the measurement lacks.

Input: the acquired matrix, 3·samples_per_pri × N_dataset (`X_3PRI_real.npy`), plus
`PW.csv` (µs) and `BW.csv` (kHz), either one value per window or one per setting
(repeated `REPEATS` times). Alternatively, use a recorder folder.

1. **Information tests** (before any regression), in the full node space and in the signal
   subspace (the principal directions of the setting means):
   - repeat spread compared with the distance to the nearest other setting
   - nearest-neighbour identification of the setting, and the parameter error when it misses
   - Spearman correlation between state distance and parameter distance (consistency of the mapping)
2. **Linear decodability**: grouped-CV ridge readouts. PW, BW and **K = B/T_p each have their
   own weights**. A **shuffled-label control** must fail (R² ≤ 0).
3. **Held-out settings**: RMSE, MAE, max, bias, R², MAPE, and the share within 1% and 5%,
   reported for the **direct K readout and for the derived B̂/T̂_p**. The same windows are
   also evaluated **rotated to random start phases**. A live window may hold two complete
   pulses and parts of a third; folding into one PRI and aligning to a learned template
   makes the representation independent of where the window starts.
4. Optional: a digital ESN on the same data. Where it scores better, the information was
   already in the physical states but a linear readout could not reach it.

The UI accepts the resulting `rp_physical_rc_model.npz`. Its chirp-rate card shows the
direct K readout, with B/T_p shown for comparison, and the template match warns about
unfamiliar signals.
