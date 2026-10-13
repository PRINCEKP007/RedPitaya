# Radar ESN, no-Φ version

This is a complete copy of `../redar_esn/` (HackRF TX, bladeRF RX, Red Pitaya RX) with the
**local Φ compression removed**. In the original, every block of `BLOCK_SIZE` samples was
reduced by a random measurement matrix `Phi_local`, and only every `STRIDE`-th result was fed
to the reservoir. Here, every raw sample drives the reservoir directly
(`z = Wres·r + Win·x[t]`, with `Win` of shape `(neurons, 1)`).

| Folder | What changed |
|---|---|
| `hackrf/` | Nothing. The TX never used Φ; it is copied so the set is complete. |
| `bladerf/` | `esn_core.Reservoir(win, wres)` runs on raw complex I/Q. `TrainESN_1.py` no longer builds `Phi_local`, `BLOCK_SIZE`, `STRIDE` or `M_local`. The dashboard loads only no-Φ models. |
| `redpitaya/` | `rp_core.RealReservoir(win, wres)` runs on the raw standardised trace. `TrainESN_Real.py` and the digital-ESN comparison in `TrainPhysicalRC.py` are updated the same way. The physical-RC path (atomic receiver as reservoir) never used Φ and is unchanged. |
| `Combined.py` | Simulation script with raw-sample reservoir input (`STRIDE = 1`). |

**Models are not interchangeable.** Models trained by the compressed version contain `Phi_local`
and are rejected here with a message, and the reverse also fails. Model formats are tagged
`*-nophi`. Physical-RC models (`rp-physical-rc-v1`) work in both versions.

**Cost.** Without compression, a window means 10× more reservoir steps on bladeRF (30,720 instead
of 3,072) and 4× more on Red Pitaya (6,000 instead of 1,500). The bladeRF default `N_res` is
lowered to 300 to compensate. Install `numba` for usable speed, or set `COMPUTE_RESERVOIR = False`
to train on the physical features alone.

**Memory.** With the same spectral radius, the reservoir now remembers about the last few dozen
raw samples. That is a much shorter span of time than when each step covered a 10-sample block.
`desired_radius` and `input_scale` may need retuning; the cross-validation printout shows whether
the reservoir features help.
