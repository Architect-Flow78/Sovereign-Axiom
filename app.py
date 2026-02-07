# ============================================================
# STREAMLIT — NASA CMAPSS FD001 VERIFICATION (NO matplotlib)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import gzip
import tempfile
import os

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="NASA CMAPSS Degradation Verifier",
    layout="wide"
)

st.title("🚀 NASA CMAPSS FD001 — Verification Dashboard")

# ============================================================
# UTILS
# ============================================================

def open_file(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")

# ============================================================
# DEGRADATION DETECTOR
# ============================================================

class DegradationDetector:
    def __init__(self, late_frac=0.3, mean_sigma=1.5):
        self.late_frac = late_frac
        self.mean_sigma = mean_sigma

    def detect(self, df):
        mask = pd.Series(False, index=df.index)

        sensors = [c for c in df.columns if c.startswith("sensor_")]

        for eng_id, g in df.groupby("engine_id"):
            g = g.sort_values("cycle")
            n = len(g)
            if n < 50:
                continue

            split = int(n * (1 - self.late_frac))
            base = g.iloc[:split]
            late = g.iloc[split:]

            degraded = False
            for c in sensors:
                b = base[c].dropna()
                l = late[c].dropna()
                if len(b) < 10 or len(l) < 10:
                    continue
                if b.std() > 0 and abs(l.mean() - b.mean()) > self.mean_sigma * b.std():
                    degraded = True

            if degraded:
                mask.loc[late.index] = True

        return mask

# ============================================================
# ENGINE
# ============================================================

def run_engine(path):
    columns = (
        ["engine_id", "cycle"]
        + [f"op_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    df = pd.read_csv(open_file(path), sep=r"\s+", names=columns)
    df = df.apply(pd.to_numeric, errors="coerce")

    detector = DegradationDetector()
    df["is_degraded"] = detector.detect(df)

    return df

# ============================================================
# STREAMLIT UI
# ============================================================

uploaded_file = st.file_uploader(
    "Upload NASA CMAPSS train_FD001.txt",
    type=["txt", "gz"]
)

if uploaded_file and st.button("🚀 Run Engine"):

    with st.spinner("Processing..."):
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(uploaded_file.read())
        tmp.close()

        df = run_engine(tmp.name)

    st.success("Done")

    # ---------------- Metrics ----------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Clean", int((~df["is_degraded"]).sum()))
    c3.metric("Degraded", int(df["is_degraded"].sum()))

    # ---------------- Downloads ----------------
    st.subheader("📥 Download data")

    st.download_button(
        "Download CLEAN rows (CSV)",
        df[~df["is_degraded"]].to_csv(index=False),
        "clean_rows.csv"
    )

    st.download_button(
        "Download DEGRADED rows (CSV)",
        df[df["is_degraded"]].to_csv(index=False),
        "degraded_rows.csv"
    )

    # ---------------- Graph ----------------
    st.subheader("📈 Degradation visualization")

    engine_id = st.selectbox(
        "Select engine_id",
        sorted(df["engine_id"].unique())
    )

    sensor = st.selectbox(
        "Select sensor",
        [c for c in df.columns if c.startswith("sensor_")]
    )

    g = df[df["engine_id"] == engine_id][["cycle", sensor, "is_degraded"]]

    st.line_chart(
        g.set_index("cycle")[sensor],
        height=250
    )

    st.scatter_chart(
        g[g["is_degraded"]].set_index("cycle")[sensor],
        height=250
    )

    st.info(
        "🔴 Точки на scatter-графике — деградация.\n"
        "📸 Сделай скриншот и пришли сюда."
    )

else:
    st.info("Upload file and press Run Engine")
