Datasetgen

## Files
| File | Purpose |
|---|---|
| `HackRF_Training_TX_GUI.py` | Training transmitter with a fixed PRI. Random PW/BW pairs, each held for R pulses. |
| `HackRF_Training_TX_PRI_GUI.py` | Training transmitter with a **changing PRI**. Random PW/BW/PRI triples, each held for R pulses. Defaults are a **1 MHz carrier and 20 MSPS**. |

Run `python HackRF_Training_TX_PRI_GUI.py`. Add `--simulate` to run without RF output, or `--self-test` to check it without a GUI or hardware.

### Changing-PRI version
- PRI is on the 1/Fs sample grid, like PW (50 ns at 20 MSPS). Defaults: PRI 400–600 µs, PW 300–350 µs, BW 100–200 kHz in 1 kHz steps, R = 3.
- The PW range must end below the shortest PRI, so every pulse fits in every PRI.
- PW, BW and PRI each change at every group, including the wrap at the end of a pass. The same seed reproduces the same plan.
- `labels.csv` holds `PRI_samples` and `PRI_us` for each group. Groups have different lengths, so `first_sample_index` is cumulative: one pass is `sum(PRI_samples) × R` samples.
- With a 1 MHz carrier and a BW of up to 200 kHz, the chirp reaches 900 kHz RF. The 1 MHz–6 GHz limit is therefore applied to the HackRF carrier (LO), and the chirp itself only has to stay above 0 Hz. The GUI shows a note whenever part of the chirp falls below 1 MHz.
