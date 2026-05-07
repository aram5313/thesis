# thesis

# EMG Artefact Suppression — Thesis Repository

Real-time stimulus artefact rejection and suppression for surface EMG signals contaminated by transcutaneous electrical stimulation. This repo contains all working code, benchmark results, datasets, and ongoing change plans for the thesis.

> **Status:** active development. Results and methodology are evolving — see [`CHANGELOG`](#changelog) and [`Planned changes`](#planned-changes) for what's stable and what's in flux.

---

## Overview

This project benchmarks a range of artefact-suppression algorithms for EMG signals contaminated by stimulation artefacts, evaluated against literature targets from Wang 2021, Chen 2023, Sennels 1997, Mandrile 2003, Limnuson 2014, Liu 2014/2025, and Andrews 2023.

Algorithms currently implemented:

- Blanking (baseline)
- Fixed-template subtraction
- EWMA (exponentially weighted moving average) template subtraction
- Dual-channel DESTD
- Gram–Schmidt orthogonalisation (GSO)
- LMS, ε-NLMS, RLS adaptive filters
- CEEMDAN

Each algorithm is evaluated under multiple metric "modes" corresponding to different evaluation conventions in the literature (Wang/Chen, Sennels, Liu ASR, Mandrile, Limnuson, Andrews).

---

## Requirements

- **Python 3.11.9**
- See [`requirements.txt`](requirements.txt) for the full pin list.

Recommended setup:

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Repository structure

```
.
├── src/                     # Algorithm implementations
├── benchmarks/              # Benchmark runner + metric definitions
├── data/                    # Ninapro segments and synthetic test signals
│   ├── ninapro/             # Raw Ninapro .mat files (rest segments used as noise)
│   └── synthetic/           # Generated test signals
├── results/                 # Exported benchmark spreadsheets (.xlsx)
├── notebooks/               # Exploratory analysis and figure generation
├── docs/                    # Thesis-related notes, planned changes, references
├── versions/                # Frozen historical versions of the pipeline
└── README.md
```

---

## Datasets

Primary dataset: **Ninapro DB** (subject `S1_E1_A1` and others, 12-channel sEMG @ 2000 Hz).

The pipeline uses two noise regimes:

1. **Static synthetic noise** — Gaussian noise added to clean templates. Used for early validation only; results in `results/2026-04-27_static/`.
2. **Dynamic noise from Ninapro rest segments** — rest-period segments from Ninapro are used directly as the noise source, on the basis that these segments contain no active contraction or stimulation and therefore approximate realistic background activity. Current default; results in `results/2026-05-07_dynamic/`.

> Ninapro raw data is **not redistributed** in this repo. Download from the official Ninapro site and place under `data/ninapro/`.

---

## Running the benchmark

From the repo root:

```bash
# Run the full benchmark suite on the default dataset
python -m benchmarks.run

# Specify a Ninapro file
python -m benchmarks.run --data data/ninapro/S1_E1_A1.mat

# Override key parameters
python -m benchmarks.run --fs 2000 --stim 50 --blank-ms 5 --n-avg 10
```

Results are exported as a timestamped `.xlsx` into `results/`, with sheets:

- `All Metrics` — algorithm × metric matrix vs literature targets
- `Raw Data` — long-format per-mode results
- `Literature Targets` — reference values from the cited papers
- `Params` — run configuration

---

## Reproducing reported results

| Run | Date | Data | Notes |
|-----|------|------|-------|
| Static-noise baseline | 2026-04-27 | Synthetic, 4 kHz | Early validation, near-zero Pearson r across most methods |
| Dynamic-noise (current) | 2026-05-07 | Ninapro `S1_E1_A1.mat`, 2 kHz | Substantial improvement; some results still under verification |

To reproduce the dynamic-noise run:

```bash
python -m benchmarks.run --data data/ninapro/S1_E1_A1.mat --noise dynamic
```

---

## Planned changes

Tracked in [`docs/CHANGES.md`](docs/CHANGES.md). Headline items:

- [ ] Verify dynamic-noise import and alignment — some metrics in the 2026-05-07 run still look off and may reflect a pipeline issue rather than algorithmic behaviour.
- [ ] Sweep across additional Ninapro subjects, not just `S1_E1_A1`.
- [ ] Add statistical significance testing (paired comparisons across subjects).
- [ ] Tighten parameter defaults per algorithm (LMS μ, RLS λ) based on per-subject behaviour.
- [ ] Add real-time latency profiling on representative hardware.
- [ ] Script finalisation (due Monday).

---

## Versions

Historical, frozen pipeline versions live in `versions/` for reference and reproducibility. The active version is at the repo root. Each frozen version has its own README noting what changed and why it was retired.

---

## Changelog

- **2026-05-07** — Switched default noise source to Ninapro rest segments (dynamic noise). Results improved substantially on real data; verification of import pipeline ongoing.
- **2026-04-27** — Static synthetic-noise benchmark established as baseline.
- *Earlier* — Initial implementations of blanking, template subtraction, and adaptive filter families.

---

## References

Key papers driving the metric targets and methodology:

- Wang et al., 2021 — GS-APEF / LMS-AF
- Chen et al., 2023 — DESTD
- Sennels et al., 1997 — Adaptive M-stage filtering
- Mandrile et al., 2003 — ARV-based contamination metric
- Limnuson et al., 2014 — IIR template subtraction (FPGA)
- Liu et al., 2014 — Savitzky–Golay M-wave recovery
- Liu et al., 2025 — Pole-shifting / fixed-template ASIC
- Andrews et al., 2023 — AA-IF / EMD-BF FFT preservation

Full reference list: [`docs/references.bib`](docs/references.bib).

---

## License & use

Thesis research code. Not currently licensed for redistribution. Contact the author before reuse.
