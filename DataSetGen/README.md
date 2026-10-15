Datasetgen

## Files
| File | Purpose |
|---|---|
| `HackRF_Training_TX_GUI.py` | Training transmitter with a fixed PRI. Random PW/BW pairs, each held for R pulses. |
| `BladeRF_Training_TX_GUI.py` | bladeRF version of the fixed-PRI transmitter. It can send the same signal on **TX1, TX2 or both at once**. |
| `BladeRF_Training_TX_PRI_GUI.py` | bladeRF version of the changing-PRI transmitter, with the same TX1/TX2 options. |
| `BladeRF_TX_RX.py` | Reference bladeRF TX/RX script (bindings usage). |
| `HackRF_Training_TX_PRI_GUI.py` | Training transmitter with a **changing PRI**. Random PW/BW/PRI triples, each held for R pulses. Defaults are a **1 MHz carrier and 20 MSPS**. |

Run `python HackRF_Training_TX_PRI_GUI.py`. Add `--simulate` to run without RF output, or `--self-test` to check it without a GUI or hardware.

### Changing-PRI version
- PRI is on the 1/Fs sample grid, like PW (50 ns at 20 MSPS). Defaults: PRI 400–600 µs, PW 300–350 µs, BW 100–200 kHz in 1 kHz steps, R = 3.
- The PW range must end below the shortest PRI, so every pulse fits in every PRI.
- PW, BW and PRI each change at every group, including the wrap at the end of a pass. The same seed reproduces the same plan.
- `labels.csv` holds `PRI_samples` and `PRI_us` for each group. Groups have different lengths, so `first_sample_index` is cumulative: one pass is `sum(PRI_samples) × R` samples.
- With a 1 MHz carrier and a BW of up to 200 kHz, the chirp reaches 900 kHz RF. The 1 MHz–6 GHz limit is therefore applied to the HackRF carrier (LO), and the chirp itself only has to stay above 0 Hz. The GUI shows a note whenever part of the chirp falls below 1 MHz.

### bladeRF versions
Run `python BladeRF_Training_TX_GUI.py` or `python BladeRF_Training_TX_PRI_GUI.py`. Add `--simulate` to run
without RF, or `--self-test` to check them without hardware. They use the same `bladerf` Python bindings as
`BladeRF_TX_RX.py`.
- **Same plans and labels as the HackRF versions.** The planned sequences, `labels.csv`, `settings.json` and the
  GUI are unchanged. Defaults are a 1 GHz carrier, 20 MSPS, and TX1 + TX2 at 0 dB gain.
- **TX1 + TX2:** with both boxes ticked the stream uses the `TX_X2` layout. Each sample goes to TX1 and then TX2
  in one interleaved buffer, so both channels send the identical signal at the same instant. The bladeRF 2.0
  shares one TX LO between them. Each channel has its own gain. If only one box is ticked, that channel
  transmits alone (`TX_X1`).
- **Samples and stream:** samples are SC16 Q11, 16-bit with full scale ±2047 (the HackRF used 8 bit).
  `sync_tx` is called in a loop, and Stop sends zeros before the channels are disabled.
- **Limits:**

  | Setting | Allowed range |
  |---|---|
  | Carrier (whole chirp) | 47 MHz – 6 GHz |
  | Sample rate | 0.521 – 61.44 MSPS (at most 30.72 MSPS with both channels on, the USB 3 limit for two 16-bit streams) |
  | RF bandwidth | Auto (= sample rate) or 0.2 – 56 MHz |
  | TX gain | −24 to 66 dB requested; the device may clamp it, and the actual gain is logged at start |
- **Exact sample rate:** if the board cannot set exactly the requested rate, the run stops, because the
  labels assume that exact rate.
