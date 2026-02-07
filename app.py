# ============================================================
# STREAMLIT — NASA CMAPSS FD001 VERIFICATION VERSION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import gzip
import hashlib
import time
import tempfile
import os
from datetime import datetime
import matplotlib.pyplot as plt

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
    def __init__(self, late_frac=0.3, mean_sigma=1.5, slope_thresh=0.0005):
        self.late_frac = late_frac
        self.mean_sigma = mean_sigma
        self.slope_thresh = slope_thresh

    def detect(self, df):
        mask = pd.Series(False, index=df.index)
        reason = pd.Series("", index=df.index)

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
                if abs(l.mean() - b.mean()) > self.mean_sigma * b.std():
                    degraded = True

            if degraded:
                mask.loc[late.index] = True
                reason.loc[late.index] = "degradation"

        return mask, reason

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
    mask, reason = detector.detect(df)

    df["is_degraded"] = mask
    df["reason"] = reason

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

    engine_ids = sorted(df["engine_id"].unique())
    engine_id = st.selectbox("Select engine_id", engine_ids)

    sensor = st.selectbox(
        "Select sensor",
        [c for c in df.columns if c.startswith("sensor_")]
    )

    g = df[df["engine_id"] == engine_id]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(g["cycle"], g[sensor], label="sensor", alpha=0.6)
    ax.scatter(
        g[g["is_degraded"]]["cycle"],
        g[g["is_degraded"]][sensor],
        color="red",
        s=10,
        label="degraded"
    )
    ax.set_xlabel("cycle")
    ax.set_ylabel(sensor)
    ax.legend()

    st.pyplot(fig)

    st.info(
        "👉 Сделай скриншот этого графика и пришли сюда в чат.\n"
        "Мы посмотрим, совпадает ли деградация с физикой отказа."
    )

else:
    st.info("Upload file and press Run Engine")
