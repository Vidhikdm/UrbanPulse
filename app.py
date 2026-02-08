from __future__ import annotations

import math
import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster

st.set_page_config(page_title="UrbanPulse Dashboard", page_icon="🌆", layout="wide")

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

TILESETS = {
    "CartoDB Positron": "CartoDB positron",
    "CartoDB DarkMatter": "CartoDB dark_matter",
    "OpenStreetMap": "OpenStreetMap",
    "Stamen Terrain": "Stamen Terrain",
}

# -----------------------------
# Utilities
# -----------------------------
@st.cache_data(show_spinner=False)
def load_manifest() -> pd.DataFrame:
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    if MANIFEST_PATH.exists():
        m = pd.read_csv(MANIFEST_PATH)
        need = ["city", "model", "feature_set", "target", "path"]
        for c in need:
            if c not in m.columns:
                raise ValueError(f"manifest.csv missing column: {c}")
        # keep only rows that exist on disk
        m = m[m["path"].astype(str).map(lambda p: Path(p).exists())].copy()
        return m

    # fallback: scan parquet files (best-effort)
    rows = []
    for p in PRED_DIR.glob("*.parquet"):
        name = p.stem
        parts = name.split("__")
        if len(parts) < 4:
            continue
        city, model, feature_set = parts[0], parts[1], parts[2]
        target = "__".join(parts[3:]).replace("__", "_")
        rows.append({"city": city, "model": model, "feature_set": feature_set, "target": target, "path": str(p)})
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def load_preds(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # enforce numeric types robustly
    for c in ["lat", "lon", "y_true", "y_pred", "error"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy()
    if "error" not in df.columns and {"y_true", "y_pred"}.issubset(df.columns):
        df["error"] = df["y_pred"] - df["y_true"]
    return df

def robust_quantiles(x: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(x, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return 0.0, 1.0
    lo = float(x.quantile(0.02))
    hi = float(x.quantile(0.98))
    if math.isclose(lo, hi):
        lo, hi = float(x.min()), float(x.max())
    if math.isclose(lo, hi):
        lo, hi = lo - 1.0, hi + 1.0
    return lo, hi

def fmt_target(t: str) -> str:
    return TARGET_LABELS.get(t, t)

def fmt_model(m: str) -> str:
    return MODEL_LABELS.get(m, m)

def fmt_featureset(fs: str) -> str:
    return FEATURESET_LABELS.get(fs, fs)

def metric_row(df: pd.DataFrame) -> dict:
    y = pd.to_numeric(df["y_true"], errors="coerce").to_numpy()
    p = pd.to_numeric(df["y_pred"], errors="coerce").to_numpy()
    e = p - y

    mae = float(np.nanmean(np.abs(e)))
    rmse = float(np.sqrt(np.nanmean(e**2)))

    y_mean = float(np.nanmean(y))
    ss_res = float(np.nansum((y - p) ** 2))
    ss_tot = float(np.nansum((y - y_mean) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    # spearman (robust; avoid scipy dependency)
    # approximate via rank correlation using pandas
    try:
        sp = float(pd.Series(y).corr(pd.Series(p), method="spearman"))
    except Exception:
        sp = float("nan")

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Spearman": sp}

def metric_cards(df: pd.DataFrame):
    m = metric_row(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{m['R2']:.3f}" if np.isfinite(m["R2"]) else "—")
    c2.metric("Spearman", f"{m['Spearman']:.3f}" if np.isfinite(m["Spearman"]) else "—")
    c3.metric("MAE", f"{m['MAE']:,.0f}" if m["MAE"] >= 1 else f"{m['MAE']:.4f}")
    c4.metric("RMSE", f"{m['RMSE']:,.0f}" if m["RMSE"] >= 1 else f"{m['RMSE']:.4f}")

def safe_download_csv(df: pd.DataFrame, name: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇️ Download CSV: {name}",
        data=csv,
        file_name=name,
        mime="text/csv",
        use_container_width=True,
    )

def _colorize(v: float, vmin: float, vmax: float) -> str:
    a = (v - vmin) / (vmax - vmin + 1e-9)
    a = float(np.clip(a, 0, 1))
    # blue -> red
    r = int(255 * a)
    b = int(255 * (1 - a))
    g = int(85)
    return f"#{r:02x}{g:02x}{b:02x}"

def make_folium_map(df: pd.DataFrame, value_col: str, mode: str, title: str, tiles: str) -> folium.Map:
    d = df.copy()

    # critical: coerce value column to numeric (prevents your quantile crash)
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=["lat", "lon", value_col]).copy()

    center = [float(d["lat"].median()), float(d["lon"].median())]
    m = folium.Map(location=center, zoom_start=10, tiles=tiles, control_scale=True)

    vmin, vmax = robust_quantiles(d[value_col])

    # title badge
    folium.map.Marker(
        center,
        icon=folium.DivIcon(
            html=f"""
            <div style="
                font-size:16px; font-weight:700; background:white;
                padding:6px 10px; border-radius:10px; border:1px solid #ddd;
                box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            ">
              {title}
            </div>
            """
        ),
    ).add_to(m)

    # legend (simple)
    legend_html = f"""
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 9999;
        background: white; padding: 10px 12px; border-radius: 10px;
        border: 1px solid #ddd; box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        font-size: 12px;
    ">
      <div style="font-weight:700; margin-bottom:6px;">Legend</div>
      <div>Color by: <b>{value_col}</b></div>
      <div style="margin-top:6px;">Min: {vmin:,.3f}</div>
      <div>Max: {vmax:,.3f}</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    if mode == "Cluster":
        cluster = MarkerCluster(name="Tracts").add_to(m)
        for _, r in d.iterrows():
            v = float(r[value_col])
            tip = (
                f"Tract: {r.get('tract_id','—')}<br>"
                f"True: {r.get('y_true',np.nan):,.2f}<br>"
                f"Pred: {r.get('y_pred',np.nan):,.2f}<br>"
                f"Error: {r.get('error',np.nan):,.2f}"
            )
            folium.CircleMarker(
                location=[float(r["lat"]), float(r["lon"])],
                radius=5,
                color=_colorize(v, vmin, vmax),
                fill=True,
                fill_opacity=0.85,
                popup=folium.Popup(tip, max_width=350),
            ).add_to(cluster)
        return m

    # default: Points
    for _, r in d.iterrows():
        v = float(r[value_col])
        tip = (
            f"Tract: {r.get('tract_id','—')}<br>"
            f"True: {r.get('y_true',np.nan):,.2f}<br>"
            f"Pred: {r.get('y_pred',np.nan):,.2f}<br>"
            f"Error: {r.get('error',np.nan):,.2f}"
        )
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=6,
            color=_colorize(v, vmin, vmax),
            fill=True,
            fill_opacity=0.85,
            popup=folium.Popup(tip, max_width=350),
        ).add_to(m)
    return m

def explain_offline(city: str, target: str, model: str, feature_set: str) -> str:
    return f"""
**What you’re seeing**
- Each dot is a Census tract (small neighborhood-sized area).
- The map shows **{fmt_target(target)}** for **{city.upper()}**.
- **Model**: {fmt_model(model)}  
- **Signals**: {fmt_featureset(feature_set)}

**How to interpret**
- *Prediction*: what the model estimates for each tract.
- *Error*: prediction − ground truth.  
  Positive = model overestimates; negative = underestimates.

**Why different cities behave differently**
- Cities have different urban form + reporting patterns (OSM coverage, 311 behavior, density).
- Cross-city robustness is a key goal of UrbanPulse: do models generalize across geography?
""".strip()

def try_llm_answer(question: str, context: str) -> str | None:
    """
    Optional LLM copilot:
    - Only runs if OPENAI_API_KEY is present.
    - If not present (or library missing), returns None (no crash).
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    # If you want this feature, install the official OpenAI python package:
    #   pip install openai
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You explain urban ML maps to non-experts. Be concise, accurate, and avoid making up facts."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None

# -----------------------------
# UI
# -----------------------------
st.title("🌆 UrbanPulse — Urban Socioeconomic Sensing Dashboard")
st.caption("Compare models, targets, and cities. Explore predictions with robust maps, downloads, and model comparisons.")

m = load_manifest()
if m.empty:
    st.error("No cached predictions found. Run: python scripts/cache_predictions.py ...")
    st.stop()

# Sidebar selectors – always valid (derived from manifest)
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

    # REMOVE heatmap; keep only the two modes
    view_mode = st.radio("Map mode", ["Cluster", "Points"], index=0)
    layer = st.radio("Color by", ["Prediction", "Error", "Ground truth"], index=0)
    tiles_label = st.selectbox("Basemap", list(TILESETS.keys()), index=0)

sel = m[(m.city == city) & (m.feature_set == feature_set) & (m.target == target) & (m.model == model)]
if sel.empty:
    st.error("No row found for selection. Check outputs/preds/manifest.csv generation.")
    st.stop()

path = str(sel.iloc[0]["path"])
df = load_preds(path)

# Header + metrics + export
top = st.columns([1.6, 1.2, 1.0])
with top[0]:
    st.subheader(f"{city.upper()} · {fmt_target(target)}")
    st.write(f"**Model:** {fmt_model(model)}  ·  **Signals:** {fmt_featureset(feature_set)}")
    st.caption(f"Source: {path}")
with top[1]:
    st.subheader("Metrics (this cached split)")
    metric_cards(df)
with top[2]:
    st.subheader("Export")
    safe_download_csv(df, f"{city}_{model}_{feature_set}_{target}.csv")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Map",
    "📈 Diagnostics",
    "🏆 Leaderboard",
    "🌍 City Compare",
    "💬 Explain / Copilot",
])

with tab1:
    st.markdown("### Spatial view")
    if layer == "Prediction":
        value_col = "y_pred"
        title = "Predicted"
    elif layer == "Error":
        value_col = "error"
        title = "Error (pred − true)"
    else:
        value_col = "y_true"
        title = "Ground truth"

    try:
        fmap = make_folium_map(
            df,
            value_col=value_col,
            mode=view_mode,
            title=f"{title} — {fmt_model(model)}",
            tiles=TILESETS[tiles_label],
        )
        st_folium(fmap, height=650, use_container_width=True)
    except Exception as ex:
        st.exception(ex)
        st.warning("Map rendering failed. Verify the cached parquet has numeric columns and try switching Cluster/Points.")

with tab2:
    st.markdown("### Model diagnostics")
    if {"y_true", "y_pred"}.issubset(df.columns):
        chart_df = df[["y_true", "y_pred", "error"]].dropna().copy()
        chart_df["abs_error"] = np.abs(chart_df["error"])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Pred vs True (quick check)**")
            st.scatter_chart(chart_df, x="y_true", y="y_pred", use_container_width=True)
        with c2:
            st.markdown("**Absolute error distribution**")
            st.bar_chart(chart_df["abs_error"].replace([np.inf, -np.inf], np.nan).dropna(), use_container_width=True)

        st.markdown("**Top errors (tracts)**")
        tdf = df[["tract_id", "y_true", "y_pred", "error", "lat", "lon"]].copy()
        tdf["abs_error"] = np.abs(pd.to_numeric(tdf["error"], errors="coerce"))
        st.dataframe(tdf.sort_values("abs_error", ascending=False).head(25), use_container_width=True)
    else:
        st.warning("This cached file is missing y_true/y_pred columns.")

with tab3:
    st.markdown("### Best models for this City / Feature set / Target")
    sub = m[(m.city == city) & (m.feature_set == feature_set) & (m.target == target)].copy()
    if sub.empty:
        st.warning("No leaderboard rows available.")
    else:
        rows = []
        for _, r in sub.iterrows():
            try:
                d = load_preds(str(r["path"]))
                mm = metric_row(d)
                rows.append({
                    "Model": fmt_model(str(r["model"])),
                    "R2": mm["R2"],
                    "Spearman": mm["Spearman"],
                    "MAE": mm["MAE"],
                    "RMSE": mm["RMSE"],
                    "n": int(len(d)),
                })
            except Exception:
                continue
        lb = pd.DataFrame(rows)
        if lb.empty:
            st.warning("Could not compute leaderboard from cached files.")
        else:
            st.dataframe(lb.sort_values(["R2", "MAE"], ascending=[False, True]), use_container_width=True)
            safe_download_csv(lb, f"leaderboard_{city}_{feature_set}_{target}.csv")

with tab4:
    st.markdown("### City Compare (same target, models across cities)")
    cities_all = sorted(m["city"].dropna().unique().tolist())
    compare_cities = st.multiselect("Choose cities", cities_all, default=cities_all[:3] if len(cities_all) >= 3 else cities_all)

    # choose a feature set that exists for each city; simplest: show per-city available
    rows = []
    for c in compare_cities:
        subc = m[(m.city == c) & (m.target == target)].copy()
        if subc.empty:
            continue
        for _, r in subc.iterrows():
            try:
                d = load_preds(str(r["path"]))
                mm = metric_row(d)
                rows.append({
                    "City": c,
                    "Feature set": fmt_featureset(str(r["feature_set"])),
                    "Model": fmt_model(str(r["model"])),
                    "R2": mm["R2"],
                    "Spearman": mm["Spearman"],
                    "MAE": mm["MAE"],
                    "RMSE": mm["RMSE"],
                    "n": int(len(d)),
                })
            except Exception:
                continue

    comp = pd.DataFrame(rows)
    if comp.empty:
        st.warning("No comparable rows found. Make sure you cached predictions for these cities/targets.")
    else:
        st.dataframe(comp.sort_values(["R2", "MAE"], ascending=[False, True]), use_container_width=True)
        safe_download_csv(comp, f"city_compare_{target}.csv")

    st.markdown("### Cross-city robustness (from experiments, if available)")
    cfile = RESULTS_DIR / "cross_city_summary.csv"
    if cfile.exists():
        cc = pd.read_csv(cfile)
        st.dataframe(cc.sort_values(["R2"], ascending=False), use_container_width=True)
        safe_download_csv(cc, "cross_city_summary.csv")
    else:
        st.info("cross_city_summary.csv not found. Run: python experiments/run_all.py")

with tab5:
    st.markdown("### Explain (for non-experts)")
    expl = explain_offline(city, target, model, feature_set)
    st.info(expl)

    st.markdown("### Optional Copilot (LLM) — if you set `OPENAI_API_KEY`")
    question = st.text_input("Ask a question about this map / result (optional):", "")
    if question.strip():
        context = (
            f"City={city}, target={target}, model={model}, feature_set={feature_set}. "
            f"Dataset rows={len(df)}. "
            "The map shows tract points with y_true, y_pred, error. "
        )
        ans = try_llm_answer(question.strip(), context=context)
        if ans is None:
            st.warning("Copilot is off (no OPENAI_API_KEY set, or openai package not installed).")
            st.write("Offline answer:")
            st.write("• This dashboard is a tract-level prediction viewer. If you want an AI copilot, set an API key and install `openai`.")
        else:
            st.success(ans)

st.divider()
st.caption("Tip: If something looks empty, confirm cached predictions exist: outputs/preds/*.parquet and outputs/preds/manifest.csv")
