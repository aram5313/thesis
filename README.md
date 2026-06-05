# EMG Artefact Suppression — Thesis Repository

Real-time stimulation artefact suppression for surface EMG signals contaminated by transcutaneous electrical stimulation. This repository contains the benchmark pipeline, results, and visualisation tools developed for the thesis.

> **Thesis:** *Development of a Closed-Loop EMG Recording and Stimulation System with Real-Time Artefact Suppression Beyond Blanking* — Ayush Ram, University of Sydney, BEng Biomedical Honours.

---

## Key files

The following files are the core deliverables of this benchmark. Other source files, patch scripts, and exploratory notebooks exist in the repo but these four are the main outputs.

| File | What it is |
|---|---|
| `thesis_with_raw_signals_v3_2.py` | Main benchmark GUI. Loads NinaPro DB3 `.mat` files, injects synthetic stimulation artefacts with controllable dynamics (jitter, fatigue, shape morphing, tau variation), runs all 9 algorithms, and exports results. Also includes the **BATCH: ALL SUBJECTS** button for running across all 11 subjects automatically. |
| `results_visualiser__2_.py` | Standalone results viewer. Load either a single-run Excel (3-condition sweep) or a batch Excel — auto-detected. Renders bar charts, degradation lines, scatter plots, latency rankings, delta heatmap, and radar fingerprints. Batch mode adds ±std error bars and shaded bands across subjects. |
| `emg_batch_20260605_121247.xlsx` | Full benchmark results — 891 runs (11 subjects × 3 exercises × 3 conditions × 9 algorithms). Summary sheet gives grand mean ± std. Per-Subject sheet contains every individual run. |
| `full_subject_emg_analysis.pdf` | 15-page publication-quality report generated from the batch results — bar charts with error bars, degradation trajectories, RMSE × Pearson r scatter, latency rankings, delta heatmap, and radar fingerprints across all three conditions. |

---

## Algorithms benchmarked

| # | Algorithm | Category |
|---|---|---|
| 1 | Blanking | Baseline |
| 2 | Fixed template subtraction | Template |
| 3 | EWMA template subtraction | Template |
| 4 | Dual-channel DESTD | Spatial |
| 5 | Gram–Schmidt Orthogonalisation (GSO) | Spatial |
| 6 | LMS adaptive filter | Adaptive |
| 7 | ε-NLMS adaptive filter | Adaptive |
| 8 | RLS adaptive filter | Adaptive |
| 9 | CEEMDAN (AA-IF) | Decomposition |

**Metrics:** SNR (dB) · RMSE (µV) · Pearson r · Latency (ms/s)

**Key finding:** GSO was the only algorithm whose performance was statistically invariant across all three conditions (ΔSNR = +0.02 dB, Static → Stress), confirmed across all 11 NinaPro DB3 subjects and three exercise types.

---

## Dataset

**NinaPro DB3** — 11 subjects, 12-channel surface EMG at 2000 Hz (Delsys Trigno electrodes).

Download all 11 subject zip files from the official NinaPro site:

> **https://ninapro.hevs.ch/instructions/DB3.html**

Once downloaded, place all zips into a single folder (do **not** unzip them — the pipeline handles extraction automatically):

```
ninapro_db3/
├── s1_0.zip
├── s2_0.zip
├── s3_0.zip
    ...
└── s11_0.zip
```

Then open `thesis_with_raw_signals_v3_2.py`, click **BATCH: ALL SUBJECTS**, and point it at that folder.

> NinaPro data is not redistributed in this repo. Download from the official site and place under your local data folder before running.

---

## Requirements

- Python 3.11+
- numpy · scipy · pandas · matplotlib · openpyxl · tkinter

```bash
pip install numpy scipy pandas matplotlib openpyxl
```

tkinter is included with most Python installations. If missing on Linux: `sudo apt install python3-tk`.

---

## Running the benchmark

**Single subject (interactive):**
```bash
python thesis_with_raw_signals_v3_2.py
```
Load any NinaPro DB3 `.mat` file via the GUI, adjust sliders, and run individual algorithms or the full 3-condition sweep.

**All 11 subjects (batch):**
1. Download all 11 NinaPro DB3 zips into one folder
2. Open the GUI, click **BATCH: ALL SUBJECTS**
3. Select the folder containing the zips
4. Choose a save path for the output Excel
5. Wait ~10–30 min — progress shows in the status bar

**Viewing results:**
```bash
python results_visualiser__2_.py
```
Click **Load Excel File** and select either the batch Excel or any single-run export.

---

## Three-condition benchmark

Results are reported across three escalating artefact conditions, anchored to literature:

| Condition | Jitter | Fatigue | Shape morph | Tau jitter |
|---|---|---|---|---|
| Static | 0 ms | 0% | 0% | 0% |
| Moderate | ±1 ms | 15% | 5% | 15% |
| Stress | ±2 ms | 30% | 15% | 30% |

Moderate and Stress parameters are anchored to Sennels 1997 Fig. 9 (worst-case amplitude/tau variation) and Sensors 2021 (DESTD realistic FES protocol).

---

## References

| Paper | Role in benchmark |
|---|---|
| Wang et al. 2021 | GS-APEF / LMS-AF — GSO and LMS metric targets |
| Chen et al. 2023 | DESTD — spatial subtraction and GSO FPGA validation |
| Sennels et al. 1997 | Adaptive M-stage filtering — Stress condition anchoring |
| Mandrile et al. 2003 | ARV-based contamination metric — Blanking characterisation |
| Limnuson et al. 2014 | IIR template subtraction FPGA — EWMA parameter justification |
| Liu et al. 2025 | Randles-model ASIC — artefact tail model (τ = 5 ms) |
| Andrews et al. 2023 | AA-IF / EMD-BF — CEEMDAN implementation and FFT metric |

---

## License

Thesis research code. Not licensed for redistribution. Contact the author before reuse.
