from __future__ import annotations

import math
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

import altair as alt
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(page_title="UrbanPulse — Socioeconomic Sensing", layout="wide")

PRED_DIR = Path("outputs/preds")
MANIFEST = PRED_DIR / "manifest.csv"
LEADERBOARD = Path("outputs/results/model_leaderboard.csv")

TARGETS_ORDER = [
    "median_income",
    "median_rent",
    "median_home_value",
    "poverty_rate",
    "unemployment_rate",
    "bachelors_plus_rate",
]

TARGET_LABELS = {
    "median_income": "Median income (USD)",
    "median_rent": "Median rent (USD)",
    "median_home_value": "Median home value (USD)",
    "poverty_rate": "Poverty rate",
    "unemployment_rate": "Unemployment rate",
    "bachelors_plus_rate": "Bachelor’s+ rate",
}

RATE_TARGETS = {"poverty_rate", "unemployment_rate", "bachelors_plus_rate"}

MODEL_LABELS = {
    "xgboost": "XGBoost (strong baseline)",
    "extra_trees": "ExtraTrees (robust ensemble)",
    "hist_gb": "HistGradientBoosting",
    "ridge": "Ridge regression",
    "mlp": "MLP (advanced/experimental)",
}

DEFAULT_MODELS = ["xgboost", "extra_trees", "hist_gb", "ridge"]  # keep MLP behind Advanced


@st.cache_data(show_spinner=False)
def load_manifest() -> pd.DataFrame:
    if not MANIFEST.exists():
        raise FileNotFoundError("outputs/preds/manifest.csv not found. Run scripts/cache_predictions.py first.")
    m = pd.read_csv(MANIFEST)

    # types
    for c in ["r2", "mae", "rmse", "spearman"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")
    # standardize
    m["city"] = m["city"].astype(str)
    m["model"] = m["model"].astype(str)
    m["feature_set"] = m["feature_set"].astype(str)
    m["target"] = m["target"].astype(str)
    m["path"] = m["path"].astype(str)
    return m


@st.cache_data(show_spinner=False)
def load_leaderboard() -> pd.DataFrame | None:
    if LEADERBOARD.exists():
        lb = pd.read_csv(LEADERBOARD)
        for c in ["r2", "mae", "rmse", "spearman"]:
            if c in lb.columns:
                lb[c] = pd.to_numeric(lb[c], errors="coerce")
        return lb
    return None


@st.cache_data(show_spinner=False)
def load_preds(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # coerce numeric columns
    for c in ["lat", "lon", "y_true", "y_pred", "error"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "y_true", "y_pred"]).copy()
    if "error" not in df.columns:
        df["error"] = df["y_pred"] - df["y_true"]
    return df


def fmt_target(t: str) -> str:
    return TARGET_LABELS.get(t, t)


def fmt_model(m: str) -> str:
    return MODEL_LABELS.get(m, m)


def fmt_value(target: str, v: float) -> str:
    if pd.isna(v):
        return "—"
    if target in RATE_TARGETS:
        return f"{v:.3f}"
    if target in {"median_income", "median_rent"}:
        return f"${v:,.0f}"
    if target == "median_home_value":
        return f"${v:,.0f}"
    return f"{v:,.3f}"


def safe_download_csv(df: pd.DataFrame, filename: str, label: str = "Download CSV"):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def metric_cards(target: str, r2: float, mae: float, rmse: float, sp: float):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{r2:.3f}" if pd.notna(r2) else "—")
    c2.metric("MAE", fmt_value(target, mae))
    c3.metric("RMSE", fmt_value(target, rmse))
    c4.metric("Spearman", f"{sp:.3f}" if pd.notna(sp) else "—")


def _center_zoom(df: pd.DataFrame):
    lat0 = float(df["lat"].median())
    lon0 = float(df["lon"].median())
    # heuristic zoom based on span
    lat_span = float(df["lat"].quantile(0.99) - df["lat"].quantile(0.01))
    lon_span = float(df["lon"].quantile(0.99) - df["lon"].quantile(0.01))
    span = max(lat_span, lon_span)
    if span < 0.08:
        zoom = 12
    elif span < 0.2:
        zoom = 11
    elif span < 0.6:
        zoom = 10
    else:
        zoom = 9
    return lat0, lon0, zoom


def folium_points_map(df: pd.DataFrame, value_col: str, target: str, title: str):
    d = df[["lat", "lon", value_col, "tract_id", "y_true", "y_pred", "error"]].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])

    if len(d) == 0:
        st.warning("No values to plot for this selection.")
        return

    lat0, lon0, zoom = _center_zoom(d)

    m = folium.Map(location=[lat0, lon0], zoom_start=zoom, tiles="CartoDB positron", control_scale=True)

    # robust min/max for color scaling
    vmin = float(d[value_col].quantile(0.02))
    vmax = float(d[value_col].quantile(0.98))
    if math.isclose(vmin, vmax):
        vmin, vmax = float(d[value_col].min()), float(d[value_col].max())
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-9

    def norm(v):
        return max(0.0, min(1.0, (float(v) - vmin) / (vmax - vmin)))

    # marker size
    radius = 6 if len(d) < 400 else 4

    cluster = MarkerCluster(name="Tracts").add_to(m)

    for _, r in d.iterrows():
        a = norm(r[value_col])
        # simple perceptual ramp: blue->red
        col = folium.utilities.color_brewer("YlOrRd", n=9)[min(8, int(a * 8))]

        tt = (
            f"Tract: {r['tract_id']}<br>"
            f"True: {fmt_value(target, float(r['y_true']))}<br>"
            f"Pred: {fmt_value(target, float(r['y_pred']))}<br>"
            f"Err: {fmt_value(target, float(r['error']))}"
        )
        folium.CircleMarker(
            location=(float(r["lat"]), float(r["lon"])),
            radius=radius,
            color=col,
            fill=True,
            fill_color=col,
            fill_opacity=0.75,
            opacity=0.8,
            tooltip=folium.Tooltip(tt, sticky=True),
        ).add_to(cluster)

    folium.LayerControl(collapsed=True).add_to(m)
    st.markdown(f"**{title}**")
    st_folium(m, height=560, use_container_width=True)


def scatter_chart(df: pd.DataFrame, target: str):
    base = df[["y_true", "y_pred"]].copy().dropna()
    base["abs_error"] = (base["y_pred"] - base["y_true"]).abs()

    chart = (
        alt.Chart(base)
        .mark_circle(opacity=0.55)
        .encode(
            x=alt.X("y_true:Q", title="True"),
            y=alt.Y("y_pred:Q", title="Predicted"),
            tooltip=[
                alt.Tooltip("y_true:Q", title="True"),
                alt.Tooltip("y_pred:Q", title="Pred"),
                alt.Tooltip("abs_error:Q", title="|Err|"),
            ],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)


def residual_hist(df: pd.DataFrame, target: str):
    base = df[["error"]].copy().dropna()
    chart = (
        alt.Chart(base)
        .mark_bar()
        .encode(
            x=alt.X("error:Q", bin=alt.Bin(maxbins=50), title="Residual (pred − true)"),
            y=alt.Y("count()", title="Count"),
            tooltip=[alt.Tooltip("count()", title="Count")],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def pick_best_model(m_sub: pd.DataFrame) -> str | None:
    if len(m_sub) == 0:
        return None
    # prefer R2, tie-break by MAE
    tmp = m_sub.copy()
    tmp["r2"] = pd.to_numeric(tmp.get("r2", np.nan), errors="coerce")
    tmp["mae"] = pd.to_numeric(tmp.get("mae", np.nan), errors="coerce")
    tmp = tmp.sort_values(["r2", "mae"], ascending=[False, True])
    return str(tmp.iloc[0]["model"])


st.title("UrbanPulse — Multicity Socioeconomic Sensing")
st.caption("Interactive evaluation + maps over tract-level predictions from open urban data signals.")

m = load_manifest()
lb = load_leaderboard()

with st.sidebar:
    st.header("Controls")

    cities = sorted(m["city"].unique().tolist())
    city = st.selectbox("City", cities, index=0)

    # feature sets by city
    feats = sorted(m.loc[m["city"] == city, "feature_set"].unique().tolist())
    feature_set = st.selectbox("Feature set", feats, index=0)

    # targets
    targets_here = sorted(m.loc[(m["city"] == city) & (m["feature_set"] == feature_set), "target"].unique().tolist())
    # stable ordering
    targets_here = sorted(targets_here, key=lambda x: TARGETS_ORDER.index(x) if x in TARGETS_ORDER else 999)
    target = st.selectbox("Target", targets_here, format_func=fmt_target)

    st.divider()
    advanced = st.toggle("Advanced models (show MLP)", value=False)

    models_here = sorted(m.loc[(m["city"] == city) & (m["feature_set"] == feature_set) & (m["target"] == target), "model"].unique().tolist())

    # default selection: leaderboard if available, else best in manifest subset, else first
    default_model = None
    if lb is not None:
        hit = lb[(lb.city == city) & (lb.feature_set == feature_set) & (lb.target == target)]
        if len(hit) > 0:
            default_model = str(hit.iloc[0]["model"])

    if default_model is None:
        default_model = pick_best_model(m[(m.city == city) & (m.feature_set == feature_set) & (m.target == target)])

    # filter models for default UI
    if not advanced:
        filtered = [x for x in models_here if x in DEFAULT_MODELS]
        if filtered:
            models_here = filtered
        # ensure default exists
        if default_model not in models_here and len(models_here) > 0:
            default_model = models_here[0]

    model = st.selectbox("Model", models_here, index=models_here.index(default_model) if default_model in models_here else 0, format_func=fmt_model)

    st.divider()
    view_mode = st.radio("Map value", ["Predicted", "True", "Error"], horizontal=True)


# locate parquet path
row = m[(m.city == city) & (m.feature_set == feature_set) & (m.target == target) & (m.model == model)]
if len(row) == 0:
    st.error("No cached prediction file for this selection. Re-run scripts/cache_predictions.py.")
    st.stop()

path = str(row.iloc[0]["path"])
df = load_preds(path)

# Metrics (from manifest)
r2 = float(row.iloc[0].get("r2", np.nan))
mae = float(row.iloc[0].get("mae", np.nan))
rmse = float(row.iloc[0].get("rmse", np.nan))
sp = float(row.iloc[0].get("spearman", np.nan))

top = st.container()
with top:
    st.subheader(f"{city.upper()} · {fmt_target(target)} · {fmt_model(model)} · {feature_set}")
    metric_cards(target, r2, mae, rmse, sp)

tab1, tab2, tab3, tab4 = st.tabs(["Map", "Charts", "Compare models", "Compare cities"])

with tab1:
    c1, c2 = st.columns([2, 1])

    with c2:
        st.markdown("#### Downloads")
        safe_download_csv(df, f"{city}__{model}__{feature_set}__{target}.csv", label="Download selected predictions (CSV)")
        safe_download_csv(m, "manifest.csv", label="Download manifest (CSV)")
        if LEADERBOARD.exists():
            lb2 = pd.read_csv(LEADERBOARD)
            safe_download_csv(lb2, "model_leaderboard.csv", label="Download leaderboard (CSV)")

        st.divider()
        st.markdown("#### Map settings")
        show_cluster = st.toggle("Cluster markers", value=True)
        st.caption("Tip: Zoom in to see tract tooltips clearly.")

    with c1:
        if view_mode == "Predicted":
            col = "y_pred"
        elif view_mode == "True":
            col = "y_true"
        else:
            col = "error"

        title = f"{fmt_target(target)} — {view_mode}"
        # If not clustering, render without MarkerCluster by duplicating quickly
        if show_cluster:
            folium_points_map(df, col, target, title)
        else:
            # minimal non-cluster version
            d = df[["lat","lon",col,"tract_id","y_true","y_pred","error"]].copy()
            d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[col, "lat", "lon"])
            lat0, lon0, zoom = _center_zoom(d)
            m0 = folium.Map(location=[lat0, lon0], zoom_start=zoom, tiles="CartoDB positron", control_scale=True)

            vmin = float(d[col].quantile(0.02)); vmax = float(d[col].quantile(0.98))
            if math.isclose(vmin, vmax): vmin, vmax = float(d[col].min()), float(d[col].max())
            if math.isclose(vmin, vmax): vmax = vmin + 1e-9

            def norm(v): return max(0.0, min(1.0, (float(v) - vmin) / (vmax - vmin)))

            radius = 6 if len(d) < 400 else 4
            for _, r in d.iterrows():
                a = norm(r[col])
                colr = folium.utilities.color_brewer("YlOrRd", n=9)[min(8, int(a * 8))]
                tt = (
                    f"Tract: {r['tract_id']}<br>"
                    f"True: {fmt_value(target, float(r['y_true']))}<br>"
                    f"Pred: {fmt_value(target, float(r['y_pred']))}<br>"
                    f"Err: {fmt_value(target, float(r['error']))}"
                )
                folium.CircleMarker(
                    location=(float(r["lat"]), float(r["lon"])),
                    radius=radius,
                    color=colr,
                    fill=True,
                    fill_color=colr,
                    fill_opacity=0.75,
                    opacity=0.8,
                    tooltip=folium.Tooltip(tt, sticky=True),
                ).add_to(m0)

            st.markdown(f"**{title}**")
            st_folium(m0, height=560, use_container_width=True)

with tab2:
    left, right = st.columns(2)
    with left:
        st.markdown("#### Predicted vs True")
        scatter_chart(df, target)
    with right:
        st.markdown("#### Residual distribution")
        residual_hist(df, target)

with tab3:
    st.markdown("### Compare models (same city / feature set / target)")
    m_sub = m[(m.city == city) & (m.feature_set == feature_set) & (m.target == target)].copy()
    m_sub = m_sub.sort_values(["r2","mae"], ascending=[False, True])

    st.dataframe(m_sub[["model","r2","mae","rmse","spearman"]].rename(columns={"model":"model"}), use_container_width=True)

    # choose 2 models to compare
    model_choices = m_sub["model"].tolist()
    if len(model_choices) >= 2:
        a = st.selectbox("Model A", model_choices, index=0, format_func=fmt_model)
        b = st.selectbox("Model B", model_choices, index=1, format_func=fmt_model)

        ra = m_sub[m_sub.model == a].iloc[0]
        rb = m_sub[m_sub.model == b].iloc[0]

        dfa = load_preds(str(ra["path"]))
        dfb = load_preds(str(rb["path"]))

        # align by tract_id (safe)
        merged = dfa.merge(dfb[["tract_id","y_pred"]].rename(columns={"y_pred":"y_pred_b"}), on="tract_id", how="inner")
        merged["pred_diff"] = merged["y_pred"] - merged["y_pred_b"]

        c1, c2 = st.columns(2)
        with c1:
            folium_points_map(merged.rename(columns={"y_pred":"y_pred"}), "y_pred", target, f"Model A: {fmt_model(a)} — Predicted")
        with c2:
            folium_points_map(merged.rename(columns={"y_pred_b":"y_pred"}), "y_pred", target, f"Model B: {fmt_model(b)} — Predicted")

        st.divider()
        folium_points_map(merged, "pred_diff", target, f"Difference map (A − B): {fmt_model(a)} minus {fmt_model(b)}")
        safe_download_csv(merged[["tract_id","lat","lon","y_true","y_pred","y_pred_b","pred_diff"]], f"{city}__{feature_set}__{target}__compare_{a}_vs_{b}.csv", label="Download compare table (CSV)")

    else:
        st.info("Not enough models cached for this target to compare.")

with tab4:
    st.markdown("### Compare cities (same model / feature set / target)")
    # for compare cities, feature set must exist; target must exist
    cities_ok = []
    for c in sorted(m["city"].unique().tolist()):
        if len(m[(m.city==c) & (m.feature_set==feature_set) & (m.target==target) & (m.model==model)]) > 0:
            cities_ok.append(c)

    st.caption("This shows how performance changes across cities with the same model + feature set + target.")
    rows = []
    for c in cities_ok:
        r = m[(m.city==c) & (m.feature_set==feature_set) & (m.target==target) & (m.model==model)].iloc[0]
        rows.append({
            "city": c,
            "r2": float(r.get("r2", np.nan)),
            "mae": float(r.get("mae", np.nan)),
            "rmse": float(r.get("rmse", np.nan)),
            "spearman": float(r.get("spearman", np.nan)),
            "path": str(r["path"]),
        })
    t = pd.DataFrame(rows).sort_values(["r2","mae"], ascending=[False,True])
    st.dataframe(t.drop(columns=["path"]), use_container_width=True)
    safe_download_csv(t.drop(columns=["path"]), f"compare_cities__{model}__{feature_set}__{target}.csv", label="Download city comparison (CSV)")
