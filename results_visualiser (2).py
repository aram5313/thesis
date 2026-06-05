"""
EMG Artefact Suppression — Results Visualiser
Ayush Ram · Thesis A&B · University of Sydney

Standalone companion to emg_gui_v3.py.

Reads the 3-condition sweep Excel export produced by the main GUI
(the file from "EXPORT 3-COND SWEEP") and generates seven
publication-quality plot panels:

  1. Bar chart — SNR, RMSE, Pearson r, Latency across all 9 algorithms
                 for a single selected condition (selectable via sidebar)
  2. Grouped bar — same 4 metrics side-by-side for Static vs Stress
                   (degradation pattern at a glance)
  3. Line chart — condition degradation curves per algorithm
                  (one line per algo, x-axis = Static / Moderate / Stress)
  4. Radar / spider — per-algorithm multi-metric fingerprint (normalised)
  5. RMSE × r scatter — quality trade-off space, all 9 algos × 3 conditions
  6. Latency bar — compute cost only, sorted ascending
  7. Delta heatmap — like the main heatmap but showing % CHANGE from Static
                     (highlights which algos are robust vs brittle)

Usage:
    python results_visualiser.py

Then click "Load Excel File" and select the 3-condition sweep .xlsx.

Dependencies: pandas  openpyxl  numpy  matplotlib  tkinter
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import warnings
warnings.filterwarnings("ignore")

# ─── palette (matches main GUI) ───────────────────────────────────────────────
BG      = "#0f1117"
SURFACE = "#222633"
CARD    = "#20232f"
BORDER  = "#2e3347"
TEXT    = "#e8eaf0"
MUTED   = "#6b7280"
ACCENT  = "#6c8fff"
ACCENT2 = "#34d399"

ALGO_COLORS = {
    "Blanking"       : "#6b7280",
    "Fixed template" : "#94a3b8",
    "EWMA template"  : "#38bdf8",
    "Dual-ch (DESTD)": "#6c8fff",
    "GSO"            : "#34d399",
    "LMS"            : "#a78bfa",
    "ε-NLMS"         : "#c084fc",
    "RLS"            : "#f97316",
    "CEEMDAN"        : "#fb7185",
}

# Canonical order for display
ALGO_ORDER = [
    "Blanking", "Fixed template", "EWMA template",
    "Dual-ch (DESTD)", "GSO", "LMS", "ε-NLMS", "RLS", "CEEMDAN"
]

# ALGO_NAMES_HOTFIX_APPLIED
# Reverse map: display name -> display name (identity, for batch loader)
ALGO_NAMES = {
    "Blanking"        : "Blanking",
    "Fixed template"  : "Fixed template",
    "EWMA template"   : "EWMA template",
    "Dual-ch (DESTD)" : "Dual-ch (DESTD)",
    "GSO"             : "GSO",
    "LMS"             : "LMS",
    "ε-NLMS"          : "ε-NLMS",
    "RLS"             : "RLS",
    "CEEMDAN"         : "CEEMDAN",
}

CONDITIONS = ["Static", "Moderate", "Stress"]
COND_COLORS = {
    "Static"  : "#34d399",
    "Moderate": "#f59e0b",
    "Stress"  : "#ef4444",
}

METRICS = {
    "snr"    : ("SNR (dB)",       True,   "{:+.1f}"),
    "rmse"   : ("RMSE (µV)",      False,  "{:.1f}"),
    "r"      : ("Pearson r",      True,   "{:.3f}"),
    "latency": ("Latency (ms/s)", False,  "{:.4f}"),
}

matplotlib.rcParams.update({
    "figure.facecolor" : BG,
    "axes.facecolor"   : SURFACE,
    "axes.edgecolor"   : BORDER,
    "axes.labelcolor"  : MUTED,
    "axes.titlecolor"  : TEXT,
    "axes.grid"        : True,
    "grid.color"       : BORDER,
    "grid.linewidth"   : 0.5,
    "xtick.color"      : MUTED,
    "ytick.color"      : MUTED,
    "xtick.labelsize"  : 7,
    "ytick.labelsize"  : 7,
    "axes.titlesize"   : 9,
    "axes.labelsize"   : 8,
    "legend.fontsize"  : 7.5,
    "legend.framealpha": 0.25,
    "legend.edgecolor" : BORDER,
    "font.family"      : "monospace",
    "text.color"       : TEXT,
})


# ─── data loading ─────────────────────────────────────────────────────────────

def load_condition_sheet(path: str) -> dict[str, pd.DataFrame]:
    """
    Read the 3-condition sweep workbook.
    Returns dict {condition_name: DataFrame with inter-mode rows only}.
    """
    xl = pd.ExcelFile(path)
    result = {}
    for sheet in xl.sheet_names:
        cond = None
        sl = sheet.lower()
        if "static" in sl:
            cond = "Static"
        elif "moderate" in sl:
            cond = "Moderate"
        elif "stress" in sl:
            cond = "Stress"
        if cond is None:
            continue
        raw = xl.parse(sheet, header=None)
        raw = raw.fillna("")
        # Find header row
        hr = None
        for i in range(min(10, len(raw))):
            row = [str(x).strip().lower() for x in raw.iloc[i]]
            if "algorithm" in row and "mode" in row and "snr" in row:
                hr = i
                break
        if hr is None:
            continue
        df = raw.iloc[hr:].copy()
        df.columns = [str(x).strip().lower() for x in raw.iloc[hr]]
        df = df.iloc[1:].reset_index(drop=True)
        df["mode"] = df["mode"].astype(str).str.strip().str.lower()
        df = df[df["mode"] == "inter"].copy()
        # Normalise algorithm name column
        df["algorithm"] = df["algorithm"].astype(str).str.strip()
        # Only keep rows with a recognisable algorithm name
        df = df[df["algorithm"].isin(ALGO_ORDER)].copy()
        for col in ["snr", "rmse", "r", "latency"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        result[cond] = df
    return result


def build_pivot(data: dict[str, pd.DataFrame], metric: str) -> pd.DataFrame:
    """
    Returns a DataFrame: rows = algorithms (ALGO_ORDER), cols = conditions.
    """
    rows = {cond: {} for cond in CONDITIONS}
    for cond, df in data.items():
        for _, row in df.iterrows():
            algo = row["algorithm"]
            if metric in df.columns:
                val = row[metric]
                if pd.notna(val):
                    rows[cond][algo] = float(val)
    return pd.DataFrame(rows).reindex(index=ALGO_ORDER)


# ─── individual plot functions ─────────────────────────────────────────────────

def _style_ax(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, pad=5)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BORDER)


def plot_bar_single_condition(ax, data, condition, metric):
    """Bar chart: all 9 algorithms for one condition × one metric."""
    df = data.get(condition)
    if df is None or df.empty:
        ax.text(0.5, 0.5, f"No data for {condition}", ha="center",
                va="center", transform=ax.transAxes, color=MUTED)
        return
    metric_label, higher, fmt = METRICS[metric]
    vals = []
    algos = []
    cols = []
    for algo in ALGO_ORDER:
        sub = df[df["algorithm"] == algo]
        if sub.empty or metric not in sub.columns:
            continue
        v = sub[metric].iloc[0]
        if pd.notna(v):
            vals.append(float(v))
            algos.append(algo)
            cols.append(ALGO_COLORS.get(algo, ACCENT))
    if not vals:
        return
    x = np.arange(len(algos))
    bars = ax.bar(x, vals, color=cols, edgecolor=BORDER, linewidth=0.5, width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels([a[:9] for a in algos], rotation=35, ha="right", fontsize=7)
    vmax = max(abs(v) for v in vals) if vals else 1
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + vmax * 0.01,
                fmt.format(v), ha="center", va="bottom",
                fontsize=6, color=MUTED)
    _style_ax(ax, f"{condition}  —  {metric_label}", ylabel=metric_label)
    # Highlight best bar
    best_idx = vals.index(max(vals) if higher else min(vals))
    bars[best_idx].set_edgecolor("white")
    bars[best_idx].set_linewidth(1.8)


def plot_grouped_bar(ax, data, metric, conditions=("Static", "Stress")):
    """
    Side-by-side bars: Static vs Stress (or any two conditions).
    One group per algorithm.
    """
    metric_label, higher, fmt = METRICS[metric]
    n_conds = len(conditions)
    width = 0.35
    algos_used = ALGO_ORDER

    x = np.arange(len(algos_used))
    for ci, cond in enumerate(conditions):
        df = data.get(cond, pd.DataFrame())
        vals = []
        for algo in algos_used:
            sub = df[df["algorithm"] == algo] if not df.empty else pd.DataFrame()
            if sub.empty or metric not in (sub.columns if not sub.empty else []):
                vals.append(0.0)
            else:
                v = sub[metric].iloc[0]
                vals.append(float(v) if pd.notna(v) else 0.0)
        offset = (ci - (n_conds - 1) / 2) * width
        ax.bar(x + offset, vals, width=width,
               color=COND_COLORS[cond], alpha=0.85,
               edgecolor=BORDER, linewidth=0.4,
               label=cond)
    ax.set_xticks(x)
    ax.set_xticklabels([a[:7] for a in algos_used], rotation=40, ha="right", fontsize=7)
    ax.legend(loc="upper right")
    _style_ax(ax, f"Static vs Stress — {metric_label}", ylabel=metric_label)


def plot_degradation_lines(ax, data, metric):
    """
    Line chart: x = condition (Static / Moderate / Stress),
    one line per algorithm, shows degradation trajectory.
    """
    metric_label, higher, fmt = METRICS[metric]
    x = np.arange(len(CONDITIONS))
    for algo in ALGO_ORDER:
        ys = []
        for cond in CONDITIONS:
            df = data.get(cond, pd.DataFrame())
            sub = df[df["algorithm"] == algo] if not df.empty else pd.DataFrame()
            if sub.empty or metric not in (sub.columns if not sub.empty else []):
                ys.append(np.nan)
            else:
                v = sub[metric].iloc[0]
                ys.append(float(v) if pd.notna(v) else np.nan)
        col = ALGO_COLORS.get(algo, ACCENT)
        ax.plot(x, ys, "o-", color=col, lw=1.5, ms=5, label=algo[:10])
        # Label endpoint
        last_valid = [(i, y) for i, y in enumerate(ys) if not np.isnan(y)]
        if last_valid:
            ix, iy = last_valid[-1]
            ax.text(ix + 0.05, iy, algo[:6], fontsize=6,
                    color=col, va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, fontsize=8)
    ax.legend(loc="upper left", ncol=2, fontsize=6)
    _style_ax(ax, f"Degradation trajectory — {metric_label}",
              xlabel="Condition", ylabel=metric_label)


def plot_scatter(ax, data):
    """
    RMSE × Pearson r scatter for all algos × all conditions.
    Each algorithm has a fixed colour; condition is coded by marker.
    """
    markers = {"Static": "o", "Moderate": "s", "Stress": "^"}
    for cond in CONDITIONS:
        df = data.get(cond, pd.DataFrame())
        if df.empty:
            continue
        for algo in ALGO_ORDER:
            sub = df[df["algorithm"] == algo]
            if sub.empty:
                continue
            r_val  = sub["r"].iloc[0]   if "r"    in sub.columns else np.nan
            rm_val = sub["rmse"].iloc[0] if "rmse" in sub.columns else np.nan
            if pd.isna(r_val) or pd.isna(rm_val):
                continue
            ax.scatter(float(rm_val), float(r_val),
                       color=ALGO_COLORS.get(algo, ACCENT),
                       marker=markers[cond], s=55, alpha=0.85,
                       edgecolors="white", linewidths=0.4, zorder=3)
            if cond == "Stress":
                ax.text(float(rm_val) + 0.5, float(r_val), algo[:5],
                        fontsize=5.5, color=ALGO_COLORS.get(algo, ACCENT))
    # legend for markers
    for cond, mk in markers.items():
        ax.scatter([], [], marker=mk, color="#888", label=cond, s=40)
    # legend for algos
    for algo, col in ALGO_COLORS.items():
        ax.scatter([], [], color=col, marker="o", s=30, label=algo[:9])
    ax.legend(loc="lower right", ncol=2, fontsize=5.5)
    _style_ax(ax, "Quality trade-off space  (RMSE × Pearson r)",
              xlabel="RMSE (µV)", ylabel="Pearson r")
    # Ideal corner annotation
    ax.annotate("← lower RMSE\nbetter →",
                xy=(0.02, 0.97), xycoords="axes fraction",
                fontsize=6, color=MUTED, va="top")


def plot_latency_bar(ax, data, condition="Static"):
    """
    Horizontal bar chart sorted by latency (ascending = fastest first).
    """
    df = data.get(condition, pd.DataFrame())
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color=MUTED)
        return
    algos, lats = [], []
    for algo in ALGO_ORDER:
        sub = df[df["algorithm"] == algo]
        if not sub.empty and "latency" in sub.columns:
            v = sub["latency"].iloc[0]
            if pd.notna(v):
                algos.append(algo)
                lats.append(float(v))
    if not algos:
        return
    # Sort by latency
    pairs = sorted(zip(lats, algos))
    lats_s = [p[0] for p in pairs]
    algos_s = [p[1] for p in pairs]
    cols = [ALGO_COLORS.get(a, ACCENT) for a in algos_s]
    y = np.arange(len(algos_s))
    bars = ax.barh(y, lats_s, color=cols, edgecolor=BORDER, linewidth=0.4, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(algos_s, fontsize=7)
    for bar, v in zip(bars, lats_s):
        ax.text(v + max(lats_s) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=6, color=MUTED)
    _style_ax(ax, f"Latency — {condition}  (ms per sample · lower = faster)",
              xlabel="Latency (ms/sample)")
    # Real-time threshold line — 1 ms/sample at 2 kHz = 2 s of compute per second
    # For the per-sample value: the threshold is approximately 0.5 ms on a 2 kHz system
    ax.axvline(0.5, color="#ef4444", lw=1.0, ls="--", alpha=0.6)
    ax.text(0.5, len(algos_s) * 0.95, "RT threshold", fontsize=6,
            color="#ef4444", ha="left")


def plot_delta_heatmap(ax, data):
    """
    Heatmap of percentage change from Static to Stress per (algo × metric).
    Red = degradation, green = improvement.
    """
    metrics_order = ["snr", "rmse", "r", "latency"]
    metric_labels = [METRICS[m][0] for m in metrics_order]
    higher_is_better = [METRICS[m][1] for m in metrics_order]

    delta_matrix = np.full((len(ALGO_ORDER), len(metrics_order)), np.nan)

    for ai, algo in enumerate(ALGO_ORDER):
        for mi, metric in enumerate(metrics_order):
            static_df = data.get("Static", pd.DataFrame())
            stress_df = data.get("Stress", pd.DataFrame())
            s_sub = static_df[static_df["algorithm"] == algo] if not static_df.empty else pd.DataFrame()
            str_sub = stress_df[stress_df["algorithm"] == algo] if not stress_df.empty else pd.DataFrame()
            if s_sub.empty or str_sub.empty:
                continue
            if metric not in s_sub.columns or metric not in str_sub.columns:
                continue
            sv = s_sub[metric].iloc[0]
            sv2 = str_sub[metric].iloc[0]
            if pd.isna(sv) or pd.isna(sv2) or abs(float(sv)) < 1e-9:
                continue
            pct_change = (float(sv2) - float(sv)) / abs(float(sv)) * 100.0
            # Flip sign: if higher is better and value went DOWN, that's negative (bad)
            # if lower is better and value went UP, that's also negative (bad)
            signed = pct_change if higher_is_better[mi] else -pct_change
            delta_matrix[ai, mi] = signed

    # Clip for colour scale
    clipped = np.clip(delta_matrix, -100, 100)
    masked = np.ma.masked_invalid(clipped)

    im = ax.imshow(masked, cmap="RdYlGn", aspect="auto", vmin=-80, vmax=80)

    # Labels
    ax.set_xticks(np.arange(len(metrics_order)))
    ax.set_xticklabels(metric_labels, fontsize=8, fontweight="bold")
    ax.set_yticks(np.arange(len(ALGO_ORDER)))
    ax.set_yticklabels(ALGO_ORDER, fontsize=8)

    # Cell text
    for ai in range(len(ALGO_ORDER)):
        for mi in range(len(metrics_order)):
            v = delta_matrix[ai, mi]
            if np.isnan(v):
                continue
            col = "white" if abs(clipped[ai, mi]) > 50 else TEXT
            ax.text(mi, ai, f"{v:+.0f}%",
                    ha="center", va="center", fontsize=7,
                    fontweight="bold", color=col)

    # Grid
    ax.set_xticks(np.arange(-0.5, len(metrics_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ALGO_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                 label="% change Static→Stress\n(green = improved, red = degraded)")
    _style_ax(ax, "Robustness: % change from Static → Stress\n"
                  "(green = improved / held, red = degraded)")


def plot_radar(fig, axes_row, data, condition):
    """
    Draw one radar (spider) chart per algorithm in a row of small axes.
    Metrics normalised to [0, 1] across all algorithms within this condition.
    """
    metrics_r = ["snr", "rmse", "r", "latency"]
    metric_labels_r = ["SNR", "RMSE\n(inv)", "r", "Lat\n(inv)"]
    higher = [True, False, True, False]

    df = data.get(condition, pd.DataFrame())
    if df.empty:
        return

    # Build raw matrix
    raw = np.full((len(ALGO_ORDER), len(metrics_r)), np.nan)
    for ai, algo in enumerate(ALGO_ORDER):
        sub = df[df["algorithm"] == algo]
        if sub.empty:
            continue
        for mi, m in enumerate(metrics_r):
            if m in sub.columns:
                v = sub[m].iloc[0]
                if pd.notna(v):
                    raw[ai, mi] = float(v)

    # Normalise per metric
    norm = np.full_like(raw, 0.5)
    for mi in range(len(metrics_r)):
        col = raw[:, mi]
        valid = col[~np.isnan(col)]
        if len(valid) < 2:
            continue
        mn, mx = valid.min(), valid.max()
        if np.isclose(mn, mx):
            continue
        scaled = (col - mn) / (mx - mn)
        if not higher[mi]:
            scaled = 1.0 - scaled
        norm[:, mi] = np.where(np.isnan(col), 0.5, scaled)

    N = len(metrics_r)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for ai, (algo, ax) in enumerate(zip(ALGO_ORDER, axes_row)):
        ax.set_facecolor(CARD)
        col = ALGO_COLORS.get(algo, ACCENT)
        vals = norm[ai].tolist() + [norm[ai, 0]]

        ax.plot(angles, vals, color=col, lw=1.8)
        ax.fill(angles, vals, color=col, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels_r, size=6, color=TEXT)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([], size=0)
        ax.set_ylim(0, 1)
        ax.spines["polar"].set_color(BORDER)
        ax.grid(color=BORDER, linewidth=0.5)
        ax.set_title(algo[:11], size=7, color=col, pad=4)


# ─── main GUI ─────────────────────────────────────────────────────────────────

VIEWS = [
    ("bars",      "📊  Bar charts",        "Per-condition · 4 metrics"),
    ("grouped",   "⬛  Static vs Stress",  "Grouped comparison"),
    ("lines",     "📈  Degradation lines", "Condition trajectory per algo"),
    ("scatter",   "⭕  RMSE × r scatter",  "Quality trade-off space"),
    ("latency",   "⏱  Latency",           "Compute cost ranking"),
    ("delta",     "🔥  Delta heatmap",     "% change Static → Stress"),
    ("radar",     "🕸  Radar fingerprints","Per-algo multi-metric (static)"),
]




# ── Batch Excel detection and loading ─────────────────────────────────────────

def _is_batch_file(path: str) -> bool:
    """Return True if this Excel has a Per-Subject sheet (batch output)."""
    try:
        xl = __import__("pandas").ExcelFile(path)
        return "Per-Subject" in xl.sheet_names
    except Exception:
        return False


def load_batch_file(path: str) -> dict:
    """
    Load a batch Excel (produced by BATCH: ALL SUBJECTS).
    Reads the Per-Subject sheet and computes mean ± std per
    (algorithm, condition, metric) across all subjects and exercises.

    Returns a dict with two keys:
      "grand"   : {condition: DataFrame}  — pooled across all exercises
      "by_ex"   : {exercise: {condition: DataFrame}}  — per-exercise split
      "n_subj"  : int
      "n_ex"    : int

    Each DataFrame has columns: algorithm, snr, rmse, r, latency,
    snr_std, rmse_std, r_std, latency_std  (same names as single-run mode
    plus _std variants), with one row per algorithm.
    """
    import pandas as pd
    import numpy as np

    ps = pd.read_excel(path, sheet_name="Per-Subject")

    # Normalise column names
    col_map = {}
    for c in ps.columns:
        cl = c.lower()
        if "snr"     in cl: col_map[c] = "snr"
        elif "rmse"  in cl: col_map[c] = "rmse"
        elif "pearson" in cl or cl == "r": col_map[c] = "r"
        elif "latency" in cl: col_map[c] = "latency"
        elif "subject" in cl: col_map[c] = "subject"
        elif "exercise" in cl: col_map[c] = "exercise"
        elif "condition" in cl: col_map[c] = "condition"
        elif "algorithm" in cl: col_map[c] = "algorithm"
    ps = ps.rename(columns=col_map)

    # Capitalise condition for consistency
    ps["condition"] = ps["condition"].str.strip().str.capitalize()
    ps["condition"] = ps["condition"].replace(
        {"Moderate": "Moderate", "Stress": "Stress", "Static": "Static"})

    metrics = ["snr", "rmse", "r", "latency"]
    conditions = ["Static", "Moderate", "Stress"]
    exercises  = sorted(ps["exercise"].dropna().unique())
    n_subj     = ps["subject"].nunique()

    def _agg_df(subset):
        """Turn a filtered slice of ps into a summary DataFrame."""
        rows = []
        for algo in ALGO_ORDER:
            row = {"algorithm": algo, "mode": "inter"}
            for m in metrics:
                col = ps[(ps["algorithm"] == algo)].index  # just to check
                vals = subset[subset["algorithm"] == algo][m].dropna()
                row[m]          = float(vals.mean()) if len(vals) > 0 else float("nan")
                row[m + "_std"] = float(vals.std())  if len(vals) > 1 else 0.0
            rows.append(row)
        df = __import__("pandas").DataFrame(rows)
        df["algorithm"] = df["algorithm"].map(ALGO_NAMES).fillna(df["algorithm"])
        # rename algorithm col back to match existing code expectations
        df = df.rename(columns={"algorithm": "algorithm"})
        return df

    # Grand (all exercises pooled)
    grand = {}
    for cond in conditions:
        sub = ps[ps["condition"] == cond]
        grand[cond] = _agg_df(sub)

    # Per-exercise
    by_ex = {}
    for ex in exercises:
        by_ex[ex] = {}
        for cond in conditions:
            sub = ps[(ps["exercise"] == ex) & (ps["condition"] == cond)]
            by_ex[ex][cond] = _agg_df(sub)

    return {
        "grand" : grand,
        "by_ex" : by_ex,
        "n_subj": n_subj,
        "n_ex"  : len(exercises),
        "path"  : path,
    }

class VisualizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EMG Results Visualiser  —  Ayush Ram · USYD")
        self.configure(bg=BG)
        self.minsize(1200, 720)

        self.data_file  = None
        self.data       = {}   # {condition: DataFrame}
        self.batch_data = None  # populated when batch Excel loaded
        self.view_var   = tk.StringVar(value="bars")
        self.cond_var   = tk.StringVar(value="Static")
        self.metric_var = tk.StringVar(value="snr")
        self.file_var   = tk.StringVar(value="No file loaded.")

        self._build()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Sidebar
        sb_outer = tk.Frame(self, bg=SURFACE, width=256)
        sb_outer.grid(row=0, column=0, sticky="nsew")
        sb_outer.grid_propagate(False)
        sb_outer.rowconfigure(0, weight=1)
        sb_outer.columnconfigure(0, weight=1)

        sb_canvas = tk.Canvas(sb_outer, bg=SURFACE, highlightthickness=0, width=240)
        sb_canvas.grid(row=0, column=0, sticky="nsew")
        sb_scroll = ttk.Scrollbar(sb_outer, orient="vertical", command=sb_canvas.yview)
        sb_scroll.grid(row=0, column=1, sticky="ns")
        sb_canvas.configure(yscrollcommand=sb_scroll.set)
        sb = tk.Frame(sb_canvas, bg=SURFACE)
        sb_canvas.create_window((0, 0), window=sb, anchor="nw", width=240)
        sb.bind("<Configure>", lambda e: sb_canvas.configure(
            scrollregion=sb_canvas.bbox("all")))
        sb_canvas.bind_all("<MouseWheel>",
            lambda e: sb_canvas.yview_scroll(int(-1 * e.delta / 120), "units"))

        self._build_sidebar(sb)

        # Plot area
        rp = tk.Frame(self, bg=BG)
        rp.grid(row=0, column=1, sticky="nsew")
        rp.rowconfigure(0, weight=1)
        rp.columnconfigure(0, weight=1)

        self.fig    = plt.figure(figsize=(11, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=rp)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.status_var = tk.StringVar(value="Load a 3-condition sweep Excel file to begin.")
        tk.Label(rp, textvariable=self.status_var, font=("Courier", 7),
                 bg=SURFACE, fg=MUTED, anchor="w", padx=10
                 ).grid(row=1, column=0, sticky="ew")

    def _build_sidebar(self, sb):
        def sec(t):
            tk.Frame(sb, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=7)
            tk.Label(sb, text=t, font=("Courier", 7, "bold"),
                     bg=SURFACE, fg=MUTED).pack(anchor="w", padx=14, pady=(0, 3))

        tk.Label(sb, text="RESULTS VIEWER", font=("Courier", 12, "bold"),
                 bg=SURFACE, fg=TEXT).pack(anchor="w", padx=14, pady=(16, 0))
        tk.Label(sb, text="Thesis A&B  ·  Ayush Ram  ·  USYD",
                 font=("Courier", 8), bg=SURFACE, fg=MUTED
                 ).pack(anchor="w", padx=14, pady=(1, 4))

        sec("DATA SOURCE")
        tk.Label(sb, textvariable=self.file_var, font=("Courier", 8),
                 bg=SURFACE, fg=ACCENT, wraplength=220, justify="left"
                 ).pack(anchor="w", padx=14, pady=(0, 6))
        tk.Button(sb, text="Load Excel File",
                  font=("Courier", 9, "bold"),
                  bg=ACCENT, fg=BG, relief="flat", padx=10, pady=6,
                  cursor="hand2", command=self._load
                  ).pack(fill=tk.X, padx=14, pady=(0, 4))

        sec("VIEW")
        for vid, lbl, sub in VIEWS:
            f = tk.Frame(sb, bg=SURFACE); f.pack(fill=tk.X, padx=10, pady=1)
            tk.Radiobutton(f, text=lbl, variable=self.view_var, value=vid,
                           font=("Courier", 9, "bold"), bg=SURFACE, fg=TEXT,
                           selectcolor=CARD, activebackground=SURFACE,
                           activeforeground=ACCENT, cursor="hand2",
                           command=self._draw).pack(anchor="w")
            tk.Label(f, text=f"  {sub}", font=("Courier", 7),
                     bg=SURFACE, fg=MUTED).pack(anchor="w")

        sec("CONDITION (for bar / latency views)")
        for cond in CONDITIONS:
            f = tk.Frame(sb, bg=SURFACE); f.pack(fill=tk.X, padx=10, pady=1)
            tk.Radiobutton(f, text=cond, variable=self.cond_var, value=cond,
                           font=("Courier", 9), bg=SURFACE, fg=COND_COLORS[cond],
                           selectcolor=CARD, activebackground=SURFACE,
                           activeforeground=COND_COLORS[cond], cursor="hand2",
                           command=self._draw).pack(anchor="w")

        sec("METRIC (for bars / degradation lines)")
        for key, (label, _, _) in METRICS.items():
            f = tk.Frame(sb, bg=SURFACE); f.pack(fill=tk.X, padx=10, pady=1)
            tk.Radiobutton(f, text=label, variable=self.metric_var, value=key,
                           font=("Courier", 8), bg=SURFACE, fg=TEXT,
                           selectcolor=CARD, activebackground=SURFACE,
                           activeforeground=ACCENT, cursor="hand2",
                           command=self._draw).pack(anchor="w")

        sec("EXPORT")
        tk.Button(sb, text="  Save PNG (300 dpi)  ",
                  font=("Courier", 9, "bold"),
                  bg="#34d399", fg="#0f1117", relief="flat", padx=8, pady=6,
                  cursor="hand2", command=self._save_png
                  ).pack(fill=tk.X, padx=14, pady=(0, 4))

        tk.Button(sb, text="  Save PDF  ",
                  font=("Courier", 9, "bold"),
                  bg="#f59e0b", fg="#0f1117", relief="flat", padx=8, pady=6,
                  cursor="hand2", command=self._save_pdf
                  ).pack(fill=tk.X, padx=14, pady=(0, 4))

        tk.Button(sb, text="  Export Full Report PDF  ",
                  font=("Courier", 9, "bold"),
                  bg="#a78bfa", fg="#0f1117", relief="flat", padx=8, pady=6,
                  cursor="hand2", command=self._export_full_report
                  ).pack(fill=tk.X, padx=14, pady=(0, 4))
        tk.Label(sb, text="  All views · cover page · 300 dpi",
                 font=("Courier", 7), bg=SURFACE, fg=MUTED
                 ).pack(anchor="w", padx=14, pady=(0, 16))

    # ── file loading ──────────────────────────────────────────────────────────

    def _load(self):
        path = filedialog.askopenfilename(
            title="Select Excel results file (single-run or batch)",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")])
        if not path:
            return

        # ── Detect file type ──────────────────────────────────────────────────
        if _is_batch_file(path):
            try:
                bd = load_batch_file(path)
            except Exception as e:
                messagebox.showerror("Batch load error", str(e))
                return
            self.batch_data = bd
            self.data       = {}   # clear any single-run data
            n  = bd["n_subj"]
            ex = bd["n_ex"]
            self.file_var.set(
                f"{os.path.basename(path)}\n"
                f"BATCH MODE — {n} subjects  ·  {ex} exercises")
            self.status_var.set(
                f"Batch loaded: {os.path.basename(path)}  ·  "
                f"{n} subjects  ·  {ex} exercises")
            self._draw()
            return

        # ── Single-run Excel (original behaviour) ─────────────────────────────
        self.batch_data = None
        try:
            self.data = load_condition_sheet(path)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return
        if not self.data:
            messagebox.showerror(
                "No usable data",
                "Could not find Static / Moderate / Stress sheets with\n"
                "Algorithm, Mode, snr, rmse, r, latency columns.\n\n"
                "Make sure you are loading the 3-condition sweep export,\n"
                "not the batch export.  For batch files, the Per-Subject\n"
                "sheet is detected automatically.")
            return
        found = list(self.data.keys())
        algos = sorted({a for df in self.data.values()
                        for a in df["algorithm"].unique()})
        self.file_var.set(
            f"{os.path.basename(path)}\n"
            f"Conditions: {found}\nAlgos: {len(algos)}")
        self.status_var.set(
            f"Loaded {os.path.basename(path)}  ·  "
            f"{len(found)} conditions  ·  {len(algos)} algorithms")
        self._draw()



    # ── batch-aware _draw wrapper ─────────────────────────────────────────────

    def _draw(self, *_):
        if self.batch_data is not None:
            self._draw_batch()
            return
        if not self.data:
            return
        # ── original single-run draw logic (unchanged) ────────────────────────
        self.fig.clf()
        view   = self.view_var.get()
        cond   = self.cond_var.get()
        metric = self.metric_var.get()
        metric_label = METRICS[metric][0]

        if view == "bars":
            gs = gridspec.GridSpec(2, 2, figure=self.fig, hspace=0.6, wspace=0.38)
            for idx, (m_key, (m_lbl, _, _)) in enumerate(METRICS.items()):
                ax = self.fig.add_subplot(gs[idx // 2, idx % 2])
                plot_bar_single_condition(ax, self.data, cond, m_key)
            self.fig.suptitle(
                f"Algorithm performance — {cond} condition",
                fontsize=13, fontweight="bold", color=TEXT)

        elif view == "grouped":
            gs = gridspec.GridSpec(2, 2, figure=self.fig, hspace=0.6, wspace=0.38)
            for idx, m_key in enumerate(METRICS):
                ax = self.fig.add_subplot(gs[idx // 2, idx % 2])
                plot_grouped_bar(ax, self.data, m_key)
            self.fig.suptitle("Static vs Stress — all metrics",
                               fontsize=13, fontweight="bold", color=TEXT)

        elif view == "lines":
            ax = self.fig.add_subplot(111)
            plot_degradation_lines(ax, self.data, metric)
            self.fig.suptitle(
                f"Degradation trajectory — {metric_label}",
                fontsize=13, fontweight="bold", color=TEXT)

        elif view == "scatter":
            ax = self.fig.add_subplot(111)
            plot_scatter(ax, self.data)
            self.fig.suptitle("Quality trade-off space: RMSE × Pearson r",
                               fontsize=13, fontweight="bold", color=TEXT)

        elif view == "latency":
            ax = self.fig.add_subplot(111)
            plot_latency_bar(ax, self.data, cond)
            self.fig.suptitle(f"Latency ranking — {cond}",
                               fontsize=13, fontweight="bold", color=TEXT)

        elif view == "delta":
            ax = self.fig.add_subplot(111)
            plot_delta_heatmap(ax, self.data)
            self.fig.suptitle("Robustness: percentage change Static → Stress",
                               fontsize=13, fontweight="bold", color=TEXT)

        elif view == "radar":
            gs = gridspec.GridSpec(3, 3, figure=self.fig,
                                   hspace=0.55, wspace=0.42)
            axes_row = []
            for ai in range(9):
                r, c = divmod(ai, 3)
                ax = self.fig.add_subplot(gs[r, c], polar=True)
                axes_row.append(ax)
            plot_radar(self.fig, axes_row, self.data, cond)
            self.fig.suptitle(
                f"Multi-metric radar fingerprints — {cond}",
                fontsize=13, fontweight="bold", color=TEXT)

        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
        self.canvas.draw()
        self.status_var.set(
            f"View: {view}  ·  Condition: {cond}  ·  Metric: {metric_label}")

    # ── batch drawing ─────────────────────────────────────────────────────────

    def _draw_batch(self, *_):
        """Main draw dispatcher for batch mode."""
        import numpy as np
        self.fig.clf()
        view   = self.view_var.get()
        cond   = self.cond_var.get()
        metric = self.metric_var.get()
        bd     = self.batch_data
        n      = bd["n_subj"]

        if view == "bars":
            self._batch_bars(cond)
        elif view == "grouped":
            self._batch_grouped()
        elif view == "lines":
            self._batch_lines(metric)
        elif view == "scatter":
            self._batch_scatter()
        elif view == "latency":
            self._batch_latency(cond)
        elif view == "delta":
            self._batch_delta_heatmap()
        elif view == "radar":
            self._batch_radar(cond)
        else:
            self._batch_bars(cond)

        self.fig.suptitle(
            f"[BATCH — {n} subjects]  " +
            self.fig.texts[0].get_text() if self.fig.texts else "",
            fontsize=11, fontweight="bold", color=TEXT)

        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
        self.canvas.draw()
        self.status_var.set(
            f"BATCH MODE  ·  {n} subjects  ·  View: {view}  ·  Cond: {cond}"
            f"  ·  Metric: {METRICS[metric][0]}")

    def _batch_df(self, cond, exercise=None):
        """Return the right DataFrame for the given condition."""
        bd = self.batch_data
        if exercise:
            return bd["by_ex"].get(exercise, {}).get(cond,
                   bd["grand"].get(cond))
        return bd["grand"].get(cond)

    # ── bars with error bars ──────────────────────────────────────────────────
    def _batch_bars(self, cond):
        import numpy as np
        gs = gridspec.GridSpec(2, 2, figure=self.fig, hspace=0.62, wspace=0.4)
        for idx, (m_key, (m_lbl, higher, fmt)) in enumerate(METRICS.items()):
            ax   = self.fig.add_subplot(gs[idx // 2, idx % 2])
            df   = self._batch_df(cond)
            if df is None or df.empty:
                continue
            algos = [a for a in ALGO_ORDER if a in df["algorithm"].values]
            vals  = [float(df[df["algorithm"]==a][m_key].iloc[0])
                     for a in algos]
            errs  = [float(df[df["algorithm"]==a][m_key+"_std"].iloc[0])
                     if m_key+"_std" in df.columns else 0.0
                     for a in algos]
            cols  = [ALGO_COLORS.get(a, ACCENT) for a in algos]

            x    = np.arange(len(algos))
            bars = ax.bar(x, vals, color=cols, edgecolor=BORDER,
                          linewidth=0.5, width=0.62, alpha=0.85)
            ax.errorbar(x, vals, yerr=errs, fmt="none",
                        color="white", capsize=4, capthick=1.2,
                        elinewidth=1.2, alpha=0.8)

            ax.set_xticks(x)
            ax.set_xticklabels([a[:9] for a in algos],
                               rotation=35, ha="right", fontsize=7)
            vmax = max(abs(v) for v in vals) if vals else 1
            for bar, v, e in zip(bars, vals, errs):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + vmax*0.015,
                        fmt.format(v), ha="center", va="bottom",
                        fontsize=5.5, color=MUTED)

            ax.set_title(f"{cond}  —  {m_lbl}  (mean±std, n={self.batch_data['n_subj']})",
                         pad=5, fontsize=8.5)
            ax.set_ylabel(m_lbl)
            ax.spines[["top","right"]].set_visible(False)
            ax.spines[["left","bottom"]].set_color(BORDER)

            best_idx = vals.index(max(vals) if higher else min(vals))
            bars[best_idx].set_edgecolor("white")
            bars[best_idx].set_linewidth(1.8)

        self.fig.texts.clear()
        self.fig.suptitle(
            f"[BATCH — {self.batch_data['n_subj']} subjects]  "
            f"Algorithm performance — {cond} condition  (mean ± std)",
            fontsize=11, fontweight="bold", color=TEXT)

    # ── grouped Static vs Stress with error bars ──────────────────────────────
    def _batch_grouped(self):
        import numpy as np
        gs = gridspec.GridSpec(2, 2, figure=self.fig, hspace=0.62, wspace=0.4)
        conditions = ("Static", "Stress")
        n = self.batch_data["n_subj"]

        for idx, (m_key, (m_lbl, higher, fmt)) in enumerate(METRICS.items()):
            ax    = self.fig.add_subplot(gs[idx // 2, idx % 2])
            width = 0.32
            x     = np.arange(len(ALGO_ORDER))

            for ci, cond in enumerate(conditions):
                df   = self._batch_df(cond)
                vals = [float(df[df["algorithm"]==a][m_key].iloc[0])
                        if not df[df["algorithm"]==a].empty else 0.0
                        for a in ALGO_ORDER]
                errs = [float(df[df["algorithm"]==a][m_key+"_std"].iloc[0])
                        if (not df[df["algorithm"]==a].empty and
                            m_key+"_std" in df.columns) else 0.0
                        for a in ALGO_ORDER]
                offset = (ci - 0.5) * width
                bars = ax.bar(x + offset, vals, width=width,
                              color=COND_COLORS[cond], alpha=0.85,
                              edgecolor=BORDER, linewidth=0.4, label=cond)
                ax.errorbar(x + offset, vals, yerr=errs,
                            fmt="none", color="white",
                            capsize=3, capthick=1.0, elinewidth=1.0, alpha=0.8)

            ax.set_xticks(x)
            ax.set_xticklabels([a[:7] for a in ALGO_ORDER],
                               rotation=40, ha="right", fontsize=7)
            ax.legend(loc="upper right", fontsize=7)
            ax.set_title(f"Static vs Stress — {m_lbl}  (n={n})", fontsize=8.5)
            ax.set_ylabel(m_lbl)
            ax.spines[["top","right"]].set_visible(False)
            ax.spines[["left","bottom"]].set_color(BORDER)

        self.fig.texts.clear()
        self.fig.suptitle(
            f"[BATCH — {n} subjects]  Static vs Stress — all metrics",
            fontsize=11, fontweight="bold", color=TEXT)

    # ── degradation lines with ±std shaded band ───────────────────────────────
    def _batch_lines(self, metric):
        import numpy as np
        ax           = self.fig.add_subplot(111)
        metric_label = METRICS[metric][0]
        cond_list    = ["Static", "Moderate", "Stress"]
        x            = np.arange(len(cond_list))
        n            = self.batch_data["n_subj"]

        for algo in ALGO_ORDER:
            ys   = []
            errs = []
            for cond in cond_list:
                df = self._batch_df(cond)
                if df is None or df[df["algorithm"]==algo].empty:
                    ys.append(np.nan); errs.append(0.0); continue
                row = df[df["algorithm"]==algo].iloc[0]
                ys.append(float(row[metric]) if metric in row else np.nan)
                errs.append(float(row[metric+"_std"])
                            if metric+"_std" in row else 0.0)

            col = ALGO_COLORS.get(algo, ACCENT)
            ax.plot(x, ys, "o-", color=col, lw=1.8, ms=5, label=algo[:10])

            # Shaded ±std band
            ys_a   = np.array(ys,   dtype=float)
            errs_a = np.array(errs, dtype=float)
            valid  = ~np.isnan(ys_a)
            if valid.any():
                ax.fill_between(x[valid],
                                (ys_a - errs_a)[valid],
                                (ys_a + errs_a)[valid],
                                color=col, alpha=0.13)

            last = [(i, y) for i, y in enumerate(ys) if not np.isnan(y)]
            if last:
                ix, iy = last[-1]
                ax.text(ix+0.05, iy, algo[:6], fontsize=6,
                        color=col, va="center")

        ax.set_xticks(x)
        ax.set_xticklabels(cond_list, fontsize=9)
        ax.legend(loc="upper left", ncol=2, fontsize=6)
        ax.set_title(
            f"Degradation trajectory — {metric_label}  "
            f"(mean ± std, n={n} subjects)",
            fontsize=9)
        ax.set_xlabel("Condition")
        ax.set_ylabel(metric_label)
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color(BORDER)

        self.fig.texts.clear()
        self.fig.suptitle(
            f"[BATCH — {n} subjects]  Degradation — {metric_label}",
            fontsize=11, fontweight="bold", color=TEXT)

    # ── scatter (mean position, ±std as cross hairs) ──────────────────────────
    def _batch_scatter(self):
        import numpy as np
        ax      = self.fig.add_subplot(111)
        markers = {"Static": "o", "Moderate": "s", "Stress": "^"}
        n       = self.batch_data["n_subj"]

        for cond in ["Static", "Moderate", "Stress"]:
            df = self._batch_df(cond)
            if df is None:
                continue
            for algo in ALGO_ORDER:
                row = df[df["algorithm"]==algo]
                if row.empty:
                    continue
                r_val  = float(row["r"].iloc[0])
                rm_val = float(row["rmse"].iloc[0])
                r_std  = float(row["r_std"].iloc[0])   if "r_std"    in row.columns else 0.0
                rm_std = float(row["rmse_std"].iloc[0]) if "rmse_std" in row.columns else 0.0

                ax.scatter(rm_val, r_val,
                           color=ALGO_COLORS.get(algo, ACCENT),
                           marker=markers[cond], s=60, alpha=0.85,
                           edgecolors="white", linewidths=0.4, zorder=3)
                # cross-hair error bars
                ax.errorbar(rm_val, r_val,
                            xerr=rm_std, yerr=r_std,
                            fmt="none",
                            color=ALGO_COLORS.get(algo, ACCENT),
                            alpha=0.35, capsize=2, elinewidth=0.8)
                if cond == "Stress":
                    ax.text(rm_val + rm_std + 0.5, r_val,
                            algo[:5], fontsize=5.5,
                            color=ALGO_COLORS.get(algo, ACCENT))

        for cond, mk in markers.items():
            ax.scatter([], [], marker=mk, color="#888", label=cond, s=40)
        for algo, col in ALGO_COLORS.items():
            ax.scatter([], [], color=col, marker="o", s=30, label=algo[:9])
        ax.legend(loc="lower right", ncol=2, fontsize=5.5)
        ax.annotate("← lower RMSE\nbetter →",
                    xy=(0.02, 0.97), xycoords="axes fraction",
                    fontsize=6, color=MUTED, va="top")
        ax.set_xlabel("RMSE (µV)")
        ax.set_ylabel("Pearson r")
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color(BORDER)

        self.fig.texts.clear()
        self.fig.suptitle(
            f"[BATCH — {n} subjects]  "
            "Quality trade-off: RMSE × Pearson r  (mean ± std cross-hairs)",
            fontsize=11, fontweight="bold", color=TEXT)

    # ── latency bar with error bars ───────────────────────────────────────────
    def _batch_latency(self, cond):
        import numpy as np
        ax  = self.fig.add_subplot(111)
        df  = self._batch_df(cond)
        n   = self.batch_data["n_subj"]
        if df is None or df.empty:
            return

        pairs = sorted([
            (float(df[df["algorithm"]==a]["latency"].iloc[0]), a)
            for a in ALGO_ORDER
            if not df[df["algorithm"]==a].empty
        ])
        lats  = [p[0] for p in pairs]
        algos = [p[1] for p in pairs]
        errs  = [float(df[df["algorithm"]==a]["latency_std"].iloc[0])
                 if "latency_std" in df.columns else 0.0
                 for a in algos]
        cols  = [ALGO_COLORS.get(a, ACCENT) for a in algos]

        y    = np.arange(len(algos))
        bars = ax.barh(y, lats, color=cols, edgecolor=BORDER,
                       linewidth=0.4, height=0.6)
        ax.errorbar(lats, y, xerr=errs, fmt="none",
                    color="white", capsize=3, capthick=1.0,
                    elinewidth=1.0, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(algos, fontsize=7)
        for bar, v, e in zip(bars, lats, errs):
            ax.text(v + e + max(lats)*0.01,
                    bar.get_y() + bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=6, color=MUTED)
        ax.axvline(0.5, color="#ef4444", lw=1.0, ls="--", alpha=0.6)
        ax.text(0.5, len(algos)*0.95, "RT threshold",
                fontsize=6, color="#ef4444", ha="left")
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color(BORDER)

        self.fig.texts.clear()
        self.fig.suptitle(
            f"[BATCH — {n} subjects]  Latency — {cond}  "
            f"(mean ± std, ms/s, n={n})",
            fontsize=11, fontweight="bold", color=TEXT)

    # ── delta heatmap (same logic, from grand means) ──────────────────────────
    def _batch_delta_heatmap(self):
        import numpy as np
        ax  = self.fig.add_subplot(111)
        n   = self.batch_data["n_subj"]

        metrics_order  = ["snr", "rmse", "r", "latency"]
        metric_labels  = [METRICS[m][0] for m in metrics_order]
        higher_is_better = [METRICS[m][1] for m in metrics_order]

        delta_matrix = np.full((len(ALGO_ORDER), len(metrics_order)), np.nan)

        for ai, algo in enumerate(ALGO_ORDER):
            for mi, metric in enumerate(metrics_order):
                sv  = self.batch_data["grand"].get("Static")
                stv = self.batch_data["grand"].get("Stress")
                if sv is None or stv is None:
                    continue
                sv_row  = sv[sv["algorithm"] == algo]
                stv_row = stv[stv["algorithm"] == algo]
                if sv_row.empty or stv_row.empty:
                    continue
                s_val  = float(sv_row[metric].iloc[0])
                st_val = float(stv_row[metric].iloc[0])
                if np.isnan(s_val) or np.isnan(st_val) or abs(s_val) < 1e-9:
                    continue
                pct    = (st_val - s_val) / abs(s_val) * 100.0
                signed = pct if higher_is_better[mi] else -pct
                delta_matrix[ai, mi] = signed

        clipped = np.clip(delta_matrix, -100, 100)
        masked  = np.ma.masked_invalid(clipped)
        im = ax.imshow(masked, cmap="RdYlGn", aspect="auto",
                       vmin=-80, vmax=80)

        ax.set_xticks(np.arange(len(metrics_order)))
        ax.set_xticklabels(metric_labels, fontsize=8, fontweight="bold")
        ax.set_yticks(np.arange(len(ALGO_ORDER)))
        ax.set_yticklabels(ALGO_ORDER, fontsize=8)

        for ai in range(len(ALGO_ORDER)):
            for mi in range(len(metrics_order)):
                v = delta_matrix[ai, mi]
                if np.isnan(v):
                    continue
                col = "white" if abs(clipped[ai, mi]) > 50 else TEXT
                ax.text(mi, ai, f"{v:+.0f}%",
                        ha="center", va="center", fontsize=7,
                        fontweight="bold", color=col)

        ax.set_xticks(np.arange(-0.5, len(metrics_order), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ALGO_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        import matplotlib.pyplot as plt
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                     label="% change Static→Stress  (green=better, red=worse)")

        self.fig.texts.clear()
        self.fig.suptitle(
            f"[BATCH — {n} subjects]  "
            "Robustness: % change Static → Stress  (grand mean across all subjects)",
            fontsize=11, fontweight="bold", color=TEXT)

    # ── radar fingerprints (from grand means) ─────────────────────────────────
    def _batch_radar(self, cond):
        import numpy as np
        n  = self.batch_data["n_subj"]
        gs = gridspec.GridSpec(3, 3, figure=self.fig,
                               hspace=0.55, wspace=0.42)
        axes_row = []
        for ai in range(9):
            r, c = divmod(ai, 3)
            ax = self.fig.add_subplot(gs[r, c], polar=True)
            axes_row.append(ax)

        # Build a fake data dict that plot_radar can consume
        df = self._batch_df(cond)
        if df is not None:
            fake_data = {cond: df.rename(
                columns={"snr":"snr","rmse":"rmse","r":"r","latency":"latency"})}
            plot_radar(self.fig, axes_row, fake_data, cond)

        self.fig.texts.clear()
        self.fig.suptitle(
            f"[BATCH — {n} subjects]  "
            f"Radar fingerprints — {cond}  (grand mean)",
            fontsize=11, fontweight="bold", color=TEXT)


    def _save(self, ext):  # EXPORT_BATCH_HOTFIX_APPLIED
        if not self.data and not getattr(self, "batch_data", None):
            messagebox.showwarning("Nothing to save", "Load data first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=f".{ext}",
            filetypes=[(f"{ext.upper()} file", f"*.{ext}"), ("All files", "*.*")],
            initialfile=f"emg_results_{self.view_var.get()}.{ext}")
        if not path:
            return
        self.fig.savefig(path, dpi=300, bbox_inches="tight",
                         facecolor=BG)
        messagebox.showinfo("Saved", f"Saved to:\n{path}")

    def _save_png(self): self._save("png")
    def _save_pdf(self): self._save("pdf")

    def _export_full_report(self):  # EXPORT_FULL_BATCH_HOTFIX_APPLIED
        """
        Render ALL views into a multi-page PDF.
        Works in both single-run mode (self.data) and batch mode (self.batch_data).
        """
        has_data  = bool(self.data)
        has_batch = bool(getattr(self, "batch_data", None))

        if not has_data and not has_batch:
            messagebox.showwarning("Nothing to export", "Load data first.")
            return

        from matplotlib.backends.backend_pdf import PdfPages
        import datetime

        mode_tag = "BATCH" if has_batch else "single-run"
        path = filedialog.asksaveasfilename(
            title=f"Save full results report ({mode_tag})",
            defaultextension=".pdf",
            filetypes=[("PDF file", "*.pdf"), ("All files", "*.*")],
            initialfile=(
                f"emg_full_report_"
                f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"))
        if not path:
            return

        self.status_var.set("Generating full report PDF — please wait…")
        self.update_idletasks()

        try:
            with PdfPages(path) as pdf:

                def new_fig(w=14, h=9):
                    return plt.figure(figsize=(w, h), facecolor=BG)

                def save_close(fig, title_str=""):
                    if title_str:
                        fig.suptitle(title_str, fontsize=12,
                                     fontweight="bold", color=TEXT)
                    fig.tight_layout(rect=[0, 0, 1, 0.95])
                    pdf.savefig(fig, facecolor=BG,
                                bbox_inches="tight", dpi=200)
                    plt.close(fig)

                ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                src = self.file_var.get().replace("\n", "  |  ")
                n_subj = (self.batch_data["n_subj"]
                          if has_batch else 1)

                # ── Cover page ────────────────────────────────────────────
                fig = new_fig(14, 9)
                ax  = fig.add_subplot(111)
                ax.set_facecolor(BG)
                ax.axis("off")
                ax.text(0.5, 0.82,
                    "EMG Stimulation Artefact Suppression",
                    ha="center", va="center", fontsize=22,
                    fontweight="bold", color=TEXT,
                    fontfamily="monospace", transform=ax.transAxes)
                ax.text(0.5, 0.74,
                    f"Benchmark Results Report  [{mode_tag.upper()}]",
                    ha="center", va="center", fontsize=15,
                    color=ACCENT, fontfamily="monospace",
                    transform=ax.transAxes)
                ax.text(0.5, 0.66,
                    f"Thesis A&B  ·  Ayush Ram  ·  University of Sydney"
                    + (f"  ·  n={n_subj} subjects" if has_batch else ""),
                    ha="center", va="center", fontsize=11,
                    color=MUTED, fontfamily="monospace",
                    transform=ax.transAxes)
                ax.plot([0.1, 0.9], [0.60, 0.60],
                    color=BORDER, linewidth=1.0,
                    transform=ax.transAxes)
                ax.text(0.5, 0.54,
                    f"Generated: {ts}",
                    ha="center", va="center", fontsize=9,
                    color=MUTED, fontfamily="monospace",
                    transform=ax.transAxes)
                ax.text(0.5, 0.48,
                    f"Source: {src[:110]}",
                    ha="center", va="center", fontsize=8,
                    color=MUTED, fontfamily="monospace",
                    transform=ax.transAxes)
                # Algorithm colour key
                ax.text(0.5, 0.40, "Algorithm colour key",
                    ha="center", va="center", fontsize=9,
                    color=ACCENT2, fontfamily="monospace",
                    transform=ax.transAxes)
                for i, algo in enumerate(ALGO_ORDER):
                    col = ALGO_COLORS.get(algo, ACCENT)
                    x   = 0.08 + (i % 5) * 0.18
                    y   = 0.32 - (i // 5) * 0.07
                    ax.add_patch(plt.Rectangle(
                        (x - 0.01, y - 0.015), 0.025, 0.03,
                        facecolor=col, edgecolor="none",
                        transform=ax.transAxes))
                    ax.text(x + 0.02, y, algo, ha="left", va="center",
                        fontsize=8, color=col,
                        fontfamily="monospace",
                        transform=ax.transAxes)
                pdf.savefig(fig, facecolor=BG,
                            bbox_inches="tight", dpi=200)
                plt.close(fig)

                # ── Pages 2-4: Bar charts per condition ───────────────────
                for cond in CONDITIONS:
                    fig = new_fig()
                    self.fig = fig          # point draw methods at this fig
                    if has_batch:
                        self.cond_var.set(cond)
                        self._batch_bars(cond)
                    else:
                        gs = gridspec.GridSpec(2, 2, figure=fig,
                                              hspace=0.6, wspace=0.38)
                        for idx, m_key in enumerate(METRICS):
                            ax = fig.add_subplot(gs[idx // 2, idx % 2])
                            plot_bar_single_condition(
                                ax, self.data, cond, m_key)
                    save_close(fig,
                        f"Algorithm performance — {cond} condition"
                        + (f"  (mean±std, n={n_subj})" if has_batch else ""))

                # ── Page 5: Static vs Stress ──────────────────────────────
                fig = new_fig()
                self.fig = fig
                if has_batch:
                    self._batch_grouped()
                else:
                    gs = gridspec.GridSpec(2, 2, figure=fig,
                                          hspace=0.6, wspace=0.38)
                    for idx, m_key in enumerate(METRICS):
                        ax = fig.add_subplot(gs[idx // 2, idx % 2])
                        plot_grouped_bar(ax, self.data, m_key)
                save_close(fig, "Static vs Stress — all metrics"
                    + (f"  (mean±std, n={n_subj})" if has_batch else ""))

                # ── Pages 6-9: Degradation lines ──────────────────────────
                for m_key, (m_label, _, _) in METRICS.items():
                    fig = new_fig(12, 7)
                    self.fig = fig
                    if has_batch:
                        self.metric_var.set(m_key)
                        self._batch_lines(m_key)
                    else:
                        ax = fig.add_subplot(111)
                        plot_degradation_lines(ax, self.data, m_key)
                    save_close(fig,
                        f"Degradation trajectory — {m_label}"
                        + (f"  (mean±std, n={n_subj})" if has_batch else ""))

                # ── Page 10: Scatter ──────────────────────────────────────
                fig = new_fig(12, 8)
                self.fig = fig
                if has_batch:
                    self._batch_scatter()
                else:
                    ax = fig.add_subplot(111)
                    plot_scatter(ax, self.data)
                save_close(fig, "Quality trade-off space: RMSE × Pearson r")

                # ── Pages 11-12: Latency ──────────────────────────────────
                for cond in ("Static", "Stress"):
                    fig = new_fig(12, 7)
                    self.fig = fig
                    if has_batch:
                        self.cond_var.set(cond)
                        self._batch_latency(cond)
                    else:
                        ax = fig.add_subplot(111)
                        plot_latency_bar(ax, self.data, cond)
                    save_close(fig, f"Latency ranking — {cond}")

                # ── Page 13: Delta heatmap ────────────────────────────────
                fig = new_fig(13, 8)
                self.fig = fig
                if has_batch:
                    self._batch_delta_heatmap()
                else:
                    ax = fig.add_subplot(111)
                    plot_delta_heatmap(ax, self.data)
                save_close(fig, "Robustness: % change Static → Stress")

                # ── Pages 14-15: Radar fingerprints ──────────────────────
                for cond in ("Static", "Stress"):
                    fig = new_fig(14, 10)
                    self.fig = fig
                    if has_batch:
                        self.cond_var.set(cond)
                        self._batch_radar(cond)
                    else:
                        gs = gridspec.GridSpec(3, 3, figure=fig,
                                             hspace=0.55, wspace=0.42)
                        axes_row = []
                        for ai in range(9):
                            r, c = divmod(ai, 3)
                            axi = fig.add_subplot(
                                gs[r, c], polar=True)
                            axes_row.append(axi)
                        plot_radar(fig, axes_row, self.data, cond)
                    save_close(fig,
                        f"Multi-metric radar fingerprints — {cond}")

            # Restore the interactive canvas figure
            self.fig = plt.figure(figsize=(11, 7))
            self.canvas.figure = self.fig

            self.status_var.set(
                f"Full report saved → {os.path.basename(path)}")
            messagebox.showinfo(
                "Report saved",
                f"15-page PDF saved to:\n{path}\n\n"
                f"Mode: {mode_tag}"
                + (f"  ·  {n_subj} subjects" if has_batch else ""))

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Export failed", str(e))
            self.status_var.set("Report export failed — see error dialog")


if __name__ == "__main__":
    np.random.seed(0)
    app = VisualizerApp()
    app.mainloop()
