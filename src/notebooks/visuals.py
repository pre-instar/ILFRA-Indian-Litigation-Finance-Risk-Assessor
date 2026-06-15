
"""
scripts/generate_images.py
--------------------------
Generates all presentation images from your real IBBI data and model outputs.
Run from project root: python scripts/generate_images.py

Outputs saved to: scripts/ppt_images/
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

OUTPUT_DIR = Path("scripts/ppt_images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 180,
    "figure.facecolor": "white",
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

NAVY  = "#1E2761"
TEAL  = "#028090"
GOLD  = "#F4A835"
RED   = "#E24B4A"
GREEN = "#1A9E75"
AMBER = "#C9982A"
GREY  = "#6B7280"
ICE   = "#CADCFC"

df = pd.read_csv("data/raw/ibbi_real.csv")
df["cirp_start_date"] = pd.to_datetime(df["cirp_start_date"], errors="coerce")

print(f"Loaded {len(df)} cases")


# ── 1. Dataset overview bar chart ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("ILFRA — Dataset Overview (1,932 Real CIRP Cases)", 
             fontsize=14, fontweight="bold", color=NAVY, y=1.02)

# 1a. Resolution vs Liquidation
counts = df["resolution_status"].value_counts()
bars = axes[0].bar(["Resolution\nPlan Approved", "Liquidation\nOrder"],
                   [counts.get("Resolution Plan Approved", 0),
                    counts.get("Liquidation Order", 0)],
                   color=[GREEN, RED], width=0.5, edgecolor="white")
for bar in bars:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f"{int(bar.get_height())}", ha="center", fontweight="bold",
                 fontsize=13, color=NAVY)
axes[0].set_title("Case Outcomes", fontweight="bold", color=NAVY)
axes[0].set_ylabel("Number of Cases")
axes[0].set_ylim(0, max(counts.values) * 1.15)

# 1b. Cases by admission year
yearly = df.groupby("admission_year").size()
axes[1].bar(yearly.index, yearly.values, color=TEAL, edgecolor="white")
axes[1].set_title("Cases by Admission Year", fontweight="bold", color=NAVY)
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Number of Cases")
axes[1].tick_params(axis="x", rotation=45)

# 1c. Realisation % distribution
fav  = df[df["resolution_status"] == "Resolution Plan Approved"]["realisation_pct"].dropna()
unf  = df[df["resolution_status"] == "Liquidation Order"]["realisation_pct"].dropna()
axes[2].hist(fav, bins=25, alpha=0.7, color=GREEN, label="Resolution", edgecolor="white")
axes[2].hist(unf, bins=25, alpha=0.7, color=RED,   label="Liquidation", edgecolor="white")
axes[2].set_title("Realisation % Distribution", fontweight="bold", color=NAVY)
axes[2].set_xlabel("Realisation %")
axes[2].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_dataset_overview.png", bbox_inches="tight")
plt.close()
print("✅ 01_dataset_overview.png")


# ── 2. Feature distributions ──────────────────────────────────────────────────
from src.training.feature_engineering import build_ibc_features

df_feat, _ = build_ibc_features(df, fit=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Key Feature Distributions", fontsize=14, fontweight="bold", color=NAVY)

features = [
    ("log_admitted_claim",         "Log(Admitted Claim)", TEAL),
    ("log_liquidation_value",      "Log(Liquidation Value)", NAVY),
    ("claim_to_liquidation_ratio", "Claim / Liquidation Ratio", AMBER),
    ("admission_year",             "Admission Year", GREEN),
]

for ax, (feat, label, col) in zip(axes.flatten(), features):
    data = df_feat[feat].dropna()
    ax.hist(data, bins=30, color=col, alpha=0.85, edgecolor="white")
    ax.axvline(data.median(), color=RED, linestyle="--", linewidth=1.5,
               label=f"Median: {data.median():.2f}")
    ax.set_title(label, fontweight="bold", color=NAVY)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_feature_distributions.png", bbox_inches="tight")
plt.close()
print("✅ 02_feature_distributions.png")


# ── 3. Correlation heatmap ────────────────────────────────────────────────────
feat_cols = ["log_admitted_claim", "log_liquidation_value",
             "claim_to_liquidation_ratio", "is_large_case",
             "admission_year", "favourable_outcome",
             "realisation_pct", "duration_days"]
feat_cols = [c for c in feat_cols if c in df_feat.columns]

fig, ax = plt.subplots(figsize=(8, 6))
corr = df_feat[feat_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax,
            annot_kws={"size": 9}, linewidths=0.5)
ax.set_title("Feature Correlation Matrix", fontsize=13,
             fontweight="bold", color=NAVY, pad=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("✅ 03_correlation_heatmap.png")


# ── 4. Model metrics bar chart ────────────────────────────────────────────────
# ── 4. Model metrics bar chart ────────────────────────────────────────────────
from pathlib import Path
import pandas as pd

MODELS_DIR = Path("models")

# Load from training_metrics.csv if available, else fall back to your known values
metrics_path = MODELS_DIR / "training_metrics.csv"
if metrics_path.exists():
    m = pd.read_csv(metrics_path, index_col=0)
    # training_metrics.csv columns vary — probe defensively
    def _get(row, *keys):
        for k in keys:
            if k in m.columns and row in m.index:
                v = m.loc[row, k]
                if pd.notna(v):
                    return float(v)
        return None

    dur_mae  = _get("ibc_duration",  "mae_days")  
    dur_r2   = _get("ibc_duration",  "r2")      
    out_auc  = _get("ibc_outcome",   "auc")        
    real_mae = _get("realisation",   "mae_pct")     
    real_r2  = _get("realisation",   "r2")         
else:
    # ← replace these with your actual numbers from train.py output
    dur_mae, dur_r2  = 321,  0.29
    out_auc          = 0.84
    real_mae, real_r2 = 14, 0.615

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("Model Performance on Real IBBI Data",
             fontsize=14, fontweight="bold", color=NAVY, y=1.02)

# Duration
axes[0].bar(["MAE (days)", "R²"],
            [dur_mae, dur_r2],
            color=[TEAL, NAVY], width=0.4)
axes[0].set_title("Duration Regressor", fontweight="bold", color=NAVY)
for bar, val in zip(axes[0].patches, [dur_mae, dur_r2]):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + bar.get_height()*0.03,
                 f"{val}", ha="center", fontweight="bold", color=NAVY, fontsize=11)
axes[0].set_ylim(0, max(dur_mae, dur_r2) * 1.2)

# Outcome — only AUC, no F1 since you didn't mention it
axes[1].bar(["AUC"],
            [out_auc],
            color=[GREEN], width=0.35)
axes[1].set_title("Outcome Classifier", fontweight="bold", color=NAVY)
axes[1].set_ylim(0, 1.15)
for bar, val in zip(axes[1].patches, [out_auc]):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", fontweight="bold", color=NAVY, fontsize=11)

# Realisation
axes[2].bar(["MAE (%)", "R²"],
            [real_mae, real_r2],
            color=[AMBER, GREEN], width=0.4)
axes[2].set_title("Realisation Regressor", fontweight="bold", color=NAVY)
for bar, val in zip(axes[2].patches, [real_mae, real_r2]):
    axes[2].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + bar.get_height()*0.03,
                 f"{val}", ha="center", fontweight="bold", color=NAVY, fontsize=11)
axes[2].set_ylim(0, max(real_mae, real_r2) * 1.2)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_model_metrics.png", bbox_inches="tight")
plt.close()
print(f"✅ 04_model_metrics.png  (dur_mae={dur_mae}, dur_r2={dur_r2}, auc={out_auc}, real_mae={real_mae}, real_r2={real_r2})")


# ── 5. SHAP global importance (loads from models/ if available) ───────────────
import joblib
from pathlib import Path

MODELS_DIR = Path("models")
shap_files = {
    "Outcome":     MODELS_DIR / "ibc_outcome_shap_values.csv",
    "Duration":    MODELS_DIR / "ibc_duration_shap_values.csv",
    "Realisation": MODELS_DIR / "realisation_shap_values.csv",
}

available = {k: v for k, v in shap_files.items() if v.exists()}

if available:
    fig, axes = plt.subplots(1, len(available), figsize=(5*len(available), 5))
    if len(available) == 1:
        axes = [axes]
    fig.suptitle("Global SHAP Feature Importance", 
                 fontsize=14, fontweight="bold", color=NAVY)
    colors = [GREEN, TEAL, AMBER]
    for ax, (name, path), col in zip(axes, available.items(), colors):
        shap_df = pd.read_csv(path, index_col=0)
        shap_df.columns = ["mean_abs_shap"]
        top = shap_df.sort_values("mean_abs_shap").tail(6)
        ax.barh(top.index, top["mean_abs_shap"], color=col, edgecolor="white")
        ax.set_title(f"{name} Model", fontweight="bold", color=NAVY)
        ax.set_xlabel("Mean |SHAP Value|")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_shap_importance.png", bbox_inches="tight")
    plt.close()
    print("✅ 05_shap_importance.png")
else:
    print("⚠  05_shap_importance.png skipped — run train.py first to generate SHAP files")


# ── 6. Calibration curve ──────────────────────────────────────────────────────
raw_path = MODELS_DIR / "calibration_curve_raw.csv"
cal_path = MODELS_DIR / "calibration_curve_cal.csv"

if raw_path.exists() and cal_path.exists():
    raw_df = pd.read_csv(raw_path)
    cal_df = pd.read_csv(cal_path)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0,1],[0,1], "k--", linewidth=1, label="Perfect calibration", alpha=0.5)
    ax.plot(raw_df["mean_predicted"], raw_df["fraction_positive"],
            "o-", color=RED, linewidth=2, label="Before calibration")
    ax.plot(cal_df["mean_predicted"], cal_df["fraction_positive"],
            "o-", color=GREEN, linewidth=2, label="After calibration (Platt)")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positive Outcomes")
    ax.set_title("Reliability Diagram — Outcome Model", 
                 fontweight="bold", color=NAVY)
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_calibration_curve.png", bbox_inches="tight")
    plt.close()
    print("✅ 06_calibration_curve.png")
else:
    print("⚠  06_calibration_curve.png skipped — run train.py first")


# ── 7. Duration distribution by outcome ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
fav_dur  = df[df["favourable_outcome"]==1]["duration_days"].dropna()
unfav_dur = df[df["favourable_outcome"]==0]["duration_days"].dropna()
ax.hist(unfav_dur, bins=35, alpha=0.7, color=RED,   label=f"Liquidation (n={len(unfav_dur)})", edgecolor="white")
ax.hist(fav_dur,   bins=35, alpha=0.7, color=GREEN, label=f"Resolution (n={len(fav_dur)})", edgecolor="white")
ax.axvline(fav_dur.median(),   color=GREEN, linestyle="--", linewidth=2,
           label=f"Resolution median: {fav_dur.median():.0f}d")
ax.axvline(unfav_dur.median(), color=RED,   linestyle="--", linewidth=2,
           label=f"Liquidation median: {unfav_dur.median():.0f}d")
ax.set_xlabel("Duration (days)")
ax.set_ylabel("Count")
ax.set_title("CIRP Duration by Outcome", fontsize=13, fontweight="bold", color=NAVY)
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_duration_by_outcome.png", bbox_inches="tight")
plt.close()
print("✅ 07_duration_by_outcome.png")


# ── 8. Claim-to-liquidation ratio vs realisation scatter ─────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
scatter_df = df_feat[df_feat["realisation_pct"].notna() & 
                     df_feat["claim_to_liquidation_ratio"].notna()].copy()
scatter_df = scatter_df[scatter_df["claim_to_liquidation_ratio"] <= 20]

colors_scatter = [GREEN if v==1 else RED for v in scatter_df["favourable_outcome"]]
ax.scatter(scatter_df["claim_to_liquidation_ratio"],
           scatter_df["realisation_pct"],
           c=colors_scatter, alpha=0.45, s=20, edgecolors="none")
ax.set_xlabel("Claim-to-Liquidation Ratio")
ax.set_ylabel("Realisation %")
ax.set_title("Claim Ratio vs Realisation (key feature)", 
             fontsize=13, fontweight="bold", color=NAVY)
green_patch = mpatches.Patch(color=GREEN, label="Resolution")
red_patch   = mpatches.Patch(color=RED,   label="Liquidation")
ax.legend(handles=[green_patch, red_patch])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_ratio_vs_realisation.png", bbox_inches="tight")
plt.close()
print("✅ 08_ratio_vs_realisation.png")


# ── 9. Pipeline architecture diagram (pure matplotlib) ───────────────────────
fig, ax = plt.subplots(figsize=(14, 3.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 3.5)
ax.axis("off")
fig.patch.set_facecolor(NAVY)
ax.set_facecolor(NAVY)

stages = [
    ("IBBI\nIngestion",    TEAL,    "ibbi_channel.py"),
    ("Feature\nEng.",      "#4F46E5","feature_engineering.py"),
    ("LightGBM\nTraining", "#7C3AED","train.py"),
    ("SHAP\nExplain.",     AMBER,   "TreeExplainer"),
    ("CBR\nEngine",        GREEN,   "cbr_engine.py"),
    ("GenAI\nNarrative",   "#EC4899","Mistral-7B"),
    ("Streamlit\nDashboard",TEAL,   "streamlit_app.py"),
]

box_w, box_h = 1.6, 1.4
gap = 0.4
for i, (label, col, sub) in enumerate(stages):
    x = 0.3 + i*(box_w + gap)
    rect = mpatches.FancyBboxPatch((x, 0.8), box_w, box_h,
        boxstyle="round,pad=0.1", facecolor=col, edgecolor="none", alpha=0.95)
    ax.add_patch(rect)
    ax.text(x + box_w/2, 0.8 + box_h*0.62, label,
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color="white", linespacing=1.4)
    ax.text(x + box_w/2, 0.8 + box_h*0.18, sub,
            ha="center", va="center", fontsize=7, color="white", alpha=0.8)
    # arrow
    if i < len(stages)-1:
        ax.annotate("", xy=(x + box_w + gap, 0.8 + box_h/2),
                    xytext=(x + box_w, 0.8 + box_h/2),
                    arrowprops=dict(arrowstyle="->", color=ICE, lw=1.5))

ax.text(7, 0.3, "Data Flow →", ha="center", fontsize=9,
        color=ICE, alpha=0.6, style="italic")
ax.text(7, 3.2, "ILFRA — End-to-End Pipeline", ha="center",
        fontsize=12, fontweight="bold", color="white")

plt.tight_layout(pad=0)
plt.savefig(OUTPUT_DIR / "09_pipeline_diagram.png", bbox_inches="tight",
            facecolor=NAVY)
plt.close()
print("✅ 09_pipeline_diagram.png")


# ── 10. Admission year survival bias ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Admission Year Analysis (Survival Bias)", 
             fontsize=13, fontweight="bold", color=NAVY)

yearly = df.groupby("admission_year").agg(
    cases=("favourable_outcome","count"),
    resolution_rate=("favourable_outcome","mean"),
).reset_index()

axes[0].bar(yearly["admission_year"], yearly["cases"], color=TEAL, edgecolor="white")
axes[0].set_title("Cases per Admission Year", fontweight="bold", color=NAVY)
axes[0].set_xlabel("Year"); axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=45)

axes[1].plot(yearly["admission_year"], yearly["resolution_rate"],
             "o-", color=GREEN, linewidth=2.5, markersize=7)
axes[1].axhline(0.5, color="grey", linestyle="--", alpha=0.5, label="50% baseline")
axes[1].fill_between(yearly["admission_year"], yearly["resolution_rate"],
                     0.5, where=yearly["resolution_rate"]>0.5,
                     alpha=0.15, color=GREEN)
axes[1].set_title("Resolution Rate by Year\n(post-2021 inflated by survival bias)",
                  fontweight="bold", color=NAVY)
axes[1].set_xlabel("Year"); axes[1].set_ylabel("Resolution Rate")
axes[1].set_ylim(0, 1); axes[1].legend()
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "10_survival_bias.png", bbox_inches="tight")
plt.close()
print("✅ 10_survival_bias.png")


print(f"\n✅ Done. All images saved to {OUTPUT_DIR.resolve()}")
print("Images that need models/ (run train.py first): 05, 06")