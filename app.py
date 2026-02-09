from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(
    page_title="UrbanPulse",
    page_icon="🌆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal, readable CSS (works with your dark theme config.toml)
st.markdown(
    """
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; }
hr { margin: 1.2rem 0; }
.smallcap { color: rgba(230,237,243,0.78); font-size: 0.95rem; line-height: 1.35; }
.card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 0.95rem 1.05rem;
  background: rgba(255,255,255,0.03);
}
.badge {
  display:inline-block;
  padding: 0.16rem 0.55rem;
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 999px;
  margin-right: .35rem;
  margin-top: .25rem;
  background: rgba(255,255,255,0.04);
  font-size: 0.85rem;
}
.h2tight { margin: .25rem 0 .35rem 0; }
.note {
  padding: .65rem .8rem;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.02);
}
</style>
""",
    unsafe_allow_html=True,
)

PRED_DIR = Path("outputs/preds")
MANIFEST_PATH = PRED_DIR / "manifest.csv"
RESULTS_DIR = Path("outputs/results")

TARGET_LABELS = {
    "median_income": "Median household income (USD)",
    "median_rent": "Median rent (USD)",
    "median_home_value": "Median home value (USD)",
    "poverty_rate": "Poverty rate (0–1)",
    "unemployment_rate": "Unemployment rate (0–1)",
    "bachelors_plus_rate": "Bachelor’s+ rate (0–1)",
}

MODEL_LABELS = {
    "xgboost": "XGBoost (GBDT)",
    "extra_trees": "ExtraTrees (Ensemble)",
    "hist_gb": "HistGradientBoosting",
    "ridge": "Ridge Regression",
    "mlp": "MLP (Neural Net)",
}

FEATURESET_LABELS = {
    "osm_only": "OSM-only",
    "osm_311": "OSM + 311 (NYC)",
}


def fmt_target(t: str) -> str:
    return TARGET_LABELS.get(t, t)


def fmt_model(m: str) -> str:
    return MODEL_LABELS.get(m, m)


def fmt_featureset(fs: str) -> str:
    return FEATURESET_LABELS.get(fs, fs)


@st.cache_data(show_spinner=False)
def load_manifest() -> pd.DataFrame:
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    if MANIFEST_PATH.exists():
        m = pd.read_csv(MANIFEST_PATH)
        required = ["city", "model", "feature_set", "target", "path"]
        for c in required:
            if c not in m.columns:
                raise ValueError(f"manifest.csv missing column: {c}")
        # drop stale rows if parquet removed
        m = m[m["path"].astype(str).apply(lambda p: Path(p).exists())].copy()
        return m.reset_index(drop=True)

    # fallback scan
    rows = []
    for p in PRED_DIR.glob("*.parquet"):
        parts = p.stem.split("__")
        if len(parts) < 4:
            continue
        city, model, feature_set = parts[0], parts[1], parts[2]
        target = "__".join(parts[3:]).replace("__", "_")
        rows.append({"city": city, "model": model, "feature_set": feature_set, "target": target, "path": str(p)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_preds(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for c in ["lat", "lon", "y_true", "y_pred", "error"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy()
    if "error" not in df.columns and {"y_true", "y_pred"}.issubset(df.columns):
        df["error"] = df["y_pred"] - df["y_true"]
    return df


def robust_quantiles(x: pd.Series) -> Tuple[float, float]:
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return 0.0, 1.0
    lo = float(x.quantile(0.02))
    hi = float(x.quantile(0.98))
    if math.isclose(lo, hi):
        lo, hi = float(x.min()), float(x.max())
    if math.isclose(lo, hi):
        lo, hi = lo - 1.0, hi + 1.0
    return lo, hi


def compute_metrics(df: pd.DataFrame) -> dict:
    if not {"y_true", "y_pred"}.issubset(df.columns):
        return {"r2": float("nan"), "mae": float("nan"), "rmse": float("nan"), "n": int(len(df))}

    y = pd.to_numeric(df["y_true"], errors="coerce").to_numpy()
    p = pd.to_numeric(df["y_pred"], errors="coerce").to_numpy()
    e = p - y
    mae = float(np.nanmean(np.abs(e)))
    rmse = float(np.sqrt(np.nanmean(e**2)))

    y_mean = float(np.nanmean(y))
    ss_res = float(np.nansum((y - p) ** 2))
    ss_tot = float(np.nansum((y - y_mean) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    return {"r2": r2, "mae": mae, "rmse": rmse, "n": int(np.isfinite(y).sum())}


def download_csv_button(df: pd.DataFrame, filename: str, label: str = "⬇️ Download CSV"):
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def make_folium_map(df: pd.DataFrame, value_col: str, mode: str, title: str) -> folium.Map:
    center = [float(df["lat"].median()), float(df["lon"].median())]
    m = folium.Map(location=center, zoom_start=10, tiles="CartoDB Positron", control_scale=True)

    vals = pd.to_numeric(df[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    vmin, vmax = robust_quantiles(vals)

    def colorize(v: float) -> str:
        a = (v - vmin) / (vmax - vmin + 1e-9)
        a = float(np.clip(a, 0, 1))
        r = int(255 * a)
        b = int(255 * (1 - a))
        g = 110
        return f"#{r:02x}{g:02x}{b:02x}"

    folium.map.Marker(
        center,
        icon=folium.DivIcon(
            html=f"""
            <div style="font-size:14px; font-weight:700; background:white;
                        padding:6px 10px; border-radius:12px; border:1px solid #ddd;">
              {title}
            </div>
            """
        ),
    ).add_to(m)

    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["lat", "lon", value_col]).copy()

    if len(d) == 0:
        return m

    if mode == "Heatmap":
        heat_data = []
        for _, r in d.iterrows():
            v = float(r[value_col])
            w = float(np.clip((v - vmin) / (vmax - vmin + 1e-9), 0, 1))
            heat_data.append([float(r["lat"]), float(r["lon"]), w])
        HeatMap(heat_data, radius=16, blur=18, max_zoom=12).add_to(m)
        return m

    cluster = MarkerCluster().add_to(m) if mode == "Cluster" else None

    for _, r in d.iterrows():
        v = float(r[value_col])
        tip = (
            f"<b>Tract</b>: {r.get('tract_id','—')}<br>"
            f"<b>True</b>: {r.get('y_true',np.nan):,.2f}<br>"
            f"<b>Pred</b>: {r.get('y_pred',np.nan):,.2f}<br>"
            f"<b>Error</b>: {r.get('error',np.nan):,.2f}"
        )
        cm = folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=6,
            color=colorize(v),
            fill=True,
            fill_opacity=0.85,
            popup=folium.Popup(tip, max_width=330),
        )
        if cluster is not None:
            cm.add_to(cluster)
        else:
            cm.add_to(m)

    return m


def quartile_report(df: pd.DataFrame) -> pd.DataFrame:
    """Error by truth quartiles (quick robustness diagnostic)."""
    if not {"y_true", "y_pred"}.issubset(df.columns):
        return pd.DataFrame()

    d = df[["y_true", "y_pred"]].copy()
    d["y_true"] = pd.to_numeric(d["y_true"], errors="coerce")
    d["y_pred"] = pd.to_numeric(d["y_pred"], errors="coerce")
    d = d.dropna().copy()
    if len(d) < 20:
        return pd.DataFrame()

    d["error"] = d["y_pred"] - d["y_true"]
    d["abs_error"] = np.abs(d["error"])
    d["q"] = pd.qcut(d["y_true"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])

    out = (
        d.groupby("q", observed=True)
        .agg(n=("error", "size"), mae=("abs_error", "mean"), bias=("error", "mean"))
        .reset_index()
    )
    return out


def local_explain(city: str, target: str, feature_set: str, model: str, df: pd.DataFrame) -> str:
    """Strong, offline explanation that reads well for layman + reviewer."""
    met = compute_metrics(df)
    r2 = met["r2"]
    mae = met["mae"]
    rmse = met["rmse"]
    n = met["n"]

    # Top errors (if available)
    lines = []
    lines.append(f"### You are viewing")
    lines.append(f"- **City:** {city.upper()}")
    lines.append(f"- **Target:** {fmt_target(target)}")
    lines.append(f"- **Feature set:** {fmt_featureset(feature_set)}")
    lines.append(f"- **Model:** {fmt_model(model)}")
    lines.append("")
    lines.append("### What this dashboard does (in one line)")
    lines.append("It predicts tract-level socioeconomic indicators from open urban signals (e.g., OSM-derived features) and visualizes where predictions are accurate vs. biased.")
    lines.append("")
    lines.append("### How to interpret the metrics")
    lines.append(f"- **R² = {r2:.3f}**: fraction of target variation explained on this cached test split (higher is better).")
    lines.append(f"- **MAE = {mae:,.0f}**, **RMSE = {rmse:,.0f}**: average error magnitude (RMSE penalizes big misses more).")
    lines.append(f"- **N = {n:,}**: number of tracts in this split.")
    lines.append("")
    lines.append("### How to read the map layers")
    lines.append("- **Prediction**: the model’s estimate for each tract.")
    lines.append("- **Ground truth**: the observed value from your dataset (e.g., ACS-derived indicators).")
    lines.append("- **Error (pred − true)**: where the model over/under-estimates (positive = over-predict).")
    lines.append("")
    lines.append("### What you should check first (quick workflow)")
    lines.append("1) Open **Leaderboard** → pick the highest R² with reasonable MAE/RMSE.")
    lines.append("2) Switch to **Error layer** → look for systematic spatial bias (whole regions consistently red/blue).")
    lines.append("3) Use **Diagnostics** → confirm scatter tightness and error distribution symmetry.")
    lines.append("4) Use **Quartiles** → ensure errors don’t explode for the lowest/highest-income tracts.")
    lines.append("")
    lines.append("### Common failure modes (what reviewers expect you to know)")
    lines.append("- **Label noise / time mismatch**: targets (ACS) reflect different periods than OSM snapshot.")
    lines.append("- **Spatial heterogeneity**: city-specific urban form changes feature–target relationships.")
    lines.append("- **Nonlinearities**: tree ensembles often outperform linear models when features interact.")
    lines.append("- **Outliers**: a few tracts can dominate RMSE; clip outliers on the map to see patterns.")
    lines.append("")
    lines.append("### What makes this project technically strong")
    lines.append("- **Reproducible caching**: fixed prediction artifacts per city/model/feature/target.")
    lines.append("- **Model comparison**: consistent evaluation across multiple algorithms.")
    lines.append("- **Error analytics**: tract-level error inspection + quartile-based robustness check.")
    lines.append("- **Scalable design**: add cities/feature sets without rewriting the UI.")
    return "\n".join(lines)


# ----------------------------
# Load manifest
# ----------------------------
m = load_manifest()
if m.empty:
    st.error("No cached predictions found. Run: python scripts/cache_predictions.py ...")
    st.stop()

# ----------------------------
# Header
# ----------------------------
st.markdown(
    """
<div class="card">
  <h1 style="margin:0;">🌆 UrbanPulse</h1>
  <div class="smallcap">
    Explore tract-level predictions from open urban signals. Compare cities, targets, and models with interactive maps and diagnostics.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Selections")

    cities = sorted(m["city"].dropna().unique().tolist())
    city = st.selectbox("City", cities, index=0)

    m_city = m[m["city"] == city].copy()
    feature_sets = sorted(m_city["feature_set"].dropna().unique().tolist())
    feature_set = st.selectbox("Feature set", feature_sets, index=0, format_func=fmt_featureset)

    m_fs = m_city[m_city["feature_set"] == feature_set].copy()
    targets = sorted(m_fs["target"].dropna().unique().tolist())
    target = st.selectbox("Target", targets, index=0, format_func=fmt_target)

    m_t = m_fs[m_fs["target"] == target].copy()
    models = sorted(m_t["model"].dropna().unique().tolist())
    model = st.selectbox("Model", models, index=0, format_func=fmt_model)

    st.divider()
    st.subheader("Map settings")
    map_mode = st.radio("Map mode", ["Cluster", "Points", "Heatmap"], index=0)
    layer = st.radio("Color by", ["Prediction", "Error", "Ground truth"], index=0)
    clip_outliers = st.toggle("Clip outliers (2–98%)", value=True)

# ----------------------------
# Load selected preds
# ----------------------------
sel = m[(m.city == city) & (m.feature_set == feature_set) & (m.target == target) & (m.model == model)]
if sel.empty:
    st.error("No row found for selection. Check outputs/preds/manifest.csv")
    st.stop()

path = str(sel.iloc[0]["path"])
df = load_preds(path)

# ----------------------------
# Selection summary row
# ----------------------------
met = compute_metrics(df)
st.markdown(
    f"""
<div class="card">
  <div class="badge">{city.upper()}</div>
  <div class="badge">{fmt_target(target)}</div>
  <div class="badge">{fmt_featureset(feature_set)}</div>
  <div class="badge">{fmt_model(model)}</div>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.25])
c1.metric("R²", f"{met['r2']:.3f}" if np.isfinite(met["r2"]) else "—")
c2.metric("MAE", f"{met['mae']:,.0f}" if np.isfinite(met["mae"]) and met["mae"] >= 1 else (f"{met['mae']:.4f}" if np.isfinite(met["mae"]) else "—"))
c3.metric("RMSE", f"{met['rmse']:,.0f}" if np.isfinite(met["rmse"]) and met["rmse"] >= 1 else (f"{met['rmse']:.4f}" if np.isfinite(met["rmse"]) else "—"))
c4.metric("N", f"{met['n']:,}")
with c5:
    st.write("")
    download_csv_button(df, f"{city}_{model}_{feature_set}_{target}.csv", label="⬇️ Download current view (CSV)")

# ----------------------------
# Tabs
# ----------------------------
tab_map, tab_diag, tab_lead, tab_compare, tab_explain = st.tabs(
    ["🗺️ Map", "📈 Diagnostics", "🏆 Leaderboard", "🌍 Compare", "💡 Explain"]
)

with tab_map:
    if layer == "Prediction":
        value_col = "y_pred"
        title = "Predicted"
    elif layer == "Error":
        value_col = "error"
        title = "Error (pred − true)"
    else:
        value_col = "y_true"
        title = "Ground truth"

    dmap = df.copy()
    if value_col not in dmap.columns:
        st.error(f"Missing column: {value_col}")
    else:
        dmap[value_col] = pd.to_numeric(dmap[value_col], errors="coerce")
        dmap = dmap.dropna(subset=[value_col, "lat", "lon"]).copy()

        if clip_outliers and len(dmap) > 10:
            lo, hi = robust_quantiles(dmap[value_col])
            dmap[value_col] = dmap[value_col].clip(lo, hi)

        fmap = make_folium_map(
            dmap,
            value_col=value_col,
            mode=map_mode,
            title=f"{title} — {fmt_model(model)}",
        )
        st_folium(fmap, height=650, use_container_width=True)
        st.caption("Tip: Cluster is best for tract-level inspection; Heatmap is best for overall patterns.")

with tab_diag:
    st.markdown("### Quick diagnostics")
    need = {"y_true", "y_pred"}
    if not need.issubset(df.columns):
        st.warning("This cached file is missing y_true/y_pred.")
    else:
        d = df[["y_true", "y_pred", "error", "tract_id", "lat", "lon"]].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_true", "y_pred"]).copy()
        if "error" not in d.columns or d["error"].isna().all():
            d["error"] = d["y_pred"] - d["y_true"]
        d["abs_error"] = np.abs(d["error"])

        cc1, cc2 = st.columns([1.15, 0.85])
        with cc1:
            st.markdown("#### Predicted vs True")
            st.scatter_chart(d, x="y_true", y="y_pred", use_container_width=True)
            st.caption("Tighter around the diagonal = better calibration.")
        with cc2:
            st.markdown("#### Error distribution")
            st.bar_chart(d["error"].dropna(), use_container_width=True)
            st.caption("Centered near 0 = less bias; long tails = outliers.")

        st.markdown("#### Error by truth quartile (robustness check)")
        qr = quartile_report(df)
        if qr.empty:
            st.info("Not enough rows to compute quartiles (need ~20+).")
        else:
            st.dataframe(qr, use_container_width=True)
            st.caption("MAE by quartile helps spot whether the model fails on low vs high-value tracts.")

        st.markdown("#### Biggest misses (top 20 tracts)")
        st.dataframe(d.sort_values("abs_error", ascending=False).head(20), use_container_width=True)

with tab_lead:
    st.markdown("### Best models for this city/feature/target")
    sub = m[(m.city == city) & (m.feature_set == feature_set) & (m.target == target)].copy()

    rows = []
    for _, r in sub.iterrows():
        try:
            dd = load_preds(str(r["path"]))
            if not {"y_true", "y_pred"}.issubset(dd.columns):
                continue
            mm = compute_metrics(dd)
            rows.append(
                {
                    "Model": fmt_model(str(r["model"])),
                    "R²": mm["r2"],
                    "MAE": mm["mae"],
                    "RMSE": mm["rmse"],
                    "N": mm["n"],
                }
            )
        except Exception:
            continue

    lb = pd.DataFrame(rows)
    if lb.empty:
        st.info("Leaderboard unavailable. Ensure cached prediction files exist.")
    else:
        lb = lb.sort_values(["R²", "MAE"], ascending=[False, True])
        st.dataframe(lb, use_container_width=True)
        download_csv_button(lb, f"{city}_{feature_set}_{target}_leaderboard.csv", label="⬇️ Download leaderboard (CSV)")

with tab_compare:
    st.markdown("### Cross-city robustness (if you ran experiments)")
    cfile = RESULTS_DIR / "cross_city_summary.csv"
    if not cfile.exists():
        st.info("cross_city_summary.csv not found. Run: python experiments/run_all.py")
    else:
        cc = pd.read_csv(cfile)
        # filter by target if column exists
        if "target" in cc.columns:
            cc = cc[cc["target"] == target].copy()
        st.dataframe(cc.sort_values(["R2"], ascending=False), use_container_width=True)
        download_csv_button(cc, "cross_city_summary_filtered.csv", label="⬇️ Download cross-city (CSV)")

with tab_explain:
    st.markdown(local_explain(city, target, feature_set, model, df))

    st.markdown("---")
    st.markdown("## FAQ (Layman-friendly)")
    with st.expander("What am I looking at?"):
        st.write(
            "Each dot is a **census tract**. The dashboard shows either the **true value**, the **model’s prediction**, "
            "or the **error** for that tract. Clusters help you click tracts; heatmaps help you see broad patterns."
        )
    with st.expander("What does Error (pred − true) mean?"):
        st.write(
            "**Positive error** means the model predicted **too high**. **Negative error** means it predicted **too low**. "
            "If one region is consistently positive/negative, the model may have a **systematic spatial bias**."
        )
    with st.expander("Which model should I trust?"):
        st.write(
            "Use the **Leaderboard** tab. Prefer higher **R²** with lower **MAE/RMSE**. Then confirm on the map that "
            "errors aren’t concentrated in a single neighborhood type."
        )
    with st.expander("Why do I only see dots and not polygons?"):
        st.write(
            "This version visualizes tract centroids (lat/lon). Polygon boundaries require a tract shapefile/GeoJSON. "
            "If you want, we can add optional GeoJSON support later without breaking this app."
        )

    st.markdown("---")
    st.markdown("## FAQ (Technical / Reviewer)")
    with st.expander("What is the modeling setup (features → target)?"):
        st.write(
            "For each tract, the feature vector comes from **open urban signals** (e.g., OSM-derived counts/densities). "
            "Targets are socioeconomic indicators (income, rent, home value, poverty, unemployment, education). "
            "Models are trained per **(city, target, feature_set)** and cached as tract-level predictions."
        )
    with st.expander("How should I interpret R² in this context?"):
        st.write(
            "R² measures how much tract-level variation is explained on the cached test split. "
            "Moderate R² can still be valuable because socioeconomic variables are noisy and influenced by unobserved factors."
        )
    with st.expander("What are the key risks / limitations (and how do you address them)?"):
        st.write(
            "- **Temporal mismatch** (OSM vs ACS period) → mitigate via robust metrics and cross-city checks.\n"
            "- **Spatial heterogeneity** → compare across cities and inspect spatial error maps.\n"
            "- **Outliers** → visualize error distribution + allow outlier clipping for map readability.\n"
            "- **Label noise** → rely on MAE/RMSE and error diagnostics, not just R²."
        )
    with st.expander("What makes this project ‘MSCS-ready’ and not a toy?"):
        st.write(
            "It has a reproducible evaluation loop (manifest + cached predictions), multi-model benchmarking, "
            "tract-level error analytics, and a UI that supports both interpretability and comparison."
        )

    st.markdown("---")
    st.markdown("## Reviewer cheat-sheet (what to look at in 60 seconds)")
    st.markdown(
        """
- **Leaderboard**: pick top model for a target.
- **Map → Error**: check if bias clusters geographically.
- **Diagnostics**: check scatter tightness and error distribution.
- **Quartiles**: ensure errors don’t explode for low/high tracts.
"""
    )

st.caption("If options look wrong: re-generate outputs/preds/manifest.csv via scripts/cache_predictions.py")
