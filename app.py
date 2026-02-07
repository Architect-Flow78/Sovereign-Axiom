# ============================================================
# STREAMLIT — NASA CMAPSS FD001 DEGRADATION ENGINE (FINAL)
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

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="NASA CMAPSS Degradation Engine",
    layout="wide"
)

st.title("🚀 NASA CMAPSS FD001 – Degradation Detection Engine")

# ============================================================
# UTILS
# ============================================================

def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()

def open_file(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")

# ============================================================
# DEGRADATION DETECTOR (NASA-OPTIMIZED)
# ============================================================

class DegradationDetector:
    def __init__(
        self,
        baseline_frac=0.2,
        late_frac=0.3,
        mean_sigma=1.5,
        slope_thresh=0.0005
    ):
        self.baseline_frac = baseline_frac
        self.late_frac = late_frac
        self.mean_sigma = mean_sigma
        self.slope_thresh = slope_thresh

    def detect(self, df):
        mask = pd.Series(False, index=df.index)
        reason = pd.Series("", index=df.index)

        numeric_cols = [
            c for c in df.columns
            if c.startswith("sensor_")
        ]

        for eng_id, g in df.groupby("engine_id"):
            g = g.sort_values("cycle")
            n = len(g)
            if n < 50:
                continue

            b_end = int(n * self.baseline_frac)
            l_start = int(n * (1 - self.late_frac))

            base = g.iloc[:b_end]
            late = g.iloc[l_start:]

            degraded = False
            degraded_sensors = []

            for c in numeric_cols:
                b = base[c].dropna()
                l = late[c].dropna()
                if len(b) < 10 or len(l) < 10:
                    continue

                sigma = b.std()
                mean_shift = abs(l.mean() - b.mean())
                mean_anom = sigma > 0 and mean_shift > self.mean_sigma * sigma

                x = g["cycle"].values
                y = g[c].values
                if np.std(y) == 0:
                    continue
                slope = np.polyfit(x, y, 1)[0]
                slope_anom = abs(slope) > self.slope_thresh

                if mean_anom or slope_anom:
                    degraded = True
                    degraded_sensors.append(c)

            if degraded:
                late_idx = g.index[g["cycle"] >= g["cycle"].quantile(1 - self.late_frac)]
                mask.loc[late_idx] = True
                for c in degraded_sensors:
                    reason.loc[late_idx] += f"degradation_{c};"

        return mask, reason

# ============================================================
# PROFILER
# ============================================================

class Profiler:
    def __init__(self):
        self.stats = {}

    def update(self, df):
        for c in df.columns:
            stt = self.stats.setdefault(c, {
                "rows": 0,
                "nulls": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None
            })
            s = df[c]
            stt["rows"] += len(s)
            stt["nulls"] += int(s.isna().sum())
            if pd.api.types.is_numeric_dtype(s):
                stt["mean"] = float(s.mean())
                stt["std"] = float(s.std())
                stt["min"] = float(s.min())
                stt["max"] = float(s.max())

    def export(self):
        return self.stats

# ============================================================
# ENGINE
# ============================================================

class Engine:
    def __init__(self):
        self.detector = DegradationDetector()
        self.profiler = Profiler()
        self.samples = []

    def run(self, path):
        columns = (
            ["engine_id", "cycle"]
            + [f"op_{i}" for i in range(1, 4)]
            + [f"sensor_{i}" for i in range(1, 22)]
        )

        reader = pd.read_csv(
            open_file(path),
            sep=r"\s+",
            names=columns,
            chunksize=50000
        )

        total = clean = bad = 0
        t0 = time.time()

        for chunk in reader:
            chunk = chunk.apply(pd.to_numeric, errors="coerce")

            # RUL proxy
            max_cycle = chunk.groupby("engine_id")["cycle"].transform("max")
            chunk["rul_proxy"] = max_cycle - chunk["cycle"]

            sm, sr = self.detector.detect(chunk)
            chunk["__reason"] = sr

            good = chunk[~sm]
            bad_rows = chunk[sm]

            if len(self.samples) < 100:
                self.samples.extend(
                    bad_rows.head(100 - len(self.samples)).to_dict("records")
                )

            self.profiler.update(chunk.drop(columns="__reason"))

            total += len(chunk)
            clean += len(good)
            bad += len(bad_rows)

        return {
            "rows_total": total,
            "rows_clean": clean,
            "rows_anomalies": bad,
            "rows_per_sec": int(total / max(time.time() - t0, 1)),
            "profile": self.profiler.export(),
            "sample_anomalies": self.samples,
            "hash": sha256_file(path),
            "finished": datetime.utcnow().isoformat()
        }

# ============================================================
# STREAMLIT UI
# ============================================================

uploaded_file = st.file_uploader(
    "Upload NASA CMAPSS file (train_FD001.txt)",
    type=["txt", "gz"]
)

if st.button("🚀 Run Engine") and uploaded_file:

    with st.spinner("Processing..."):
        tmp_dir = tempfile.mkdtemp()
        path = os.path.join(tmp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.read())

        engine = Engine()
        report = engine.run(path)

    st.success("Done")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", report["rows_total"])
    c2.metric("Clean", report["rows_clean"])
    c3.metric("Anomalies", report["rows_anomalies"])
    c4.metric("Rows/sec", report["rows_per_sec"])

    st.subheader("Column Profile")
    st.dataframe(pd.DataFrame(report["profile"]).T, width="stretch")

    if report["sample_anomalies"]:
        st.subheader("Sample Degradation Rows (Late Cycles)")
        st.dataframe(pd.DataFrame(report["sample_anomalies"]), width="stretch")

    st.subheader("Raw Report")
    st.json(report)

else:
    st.info("Upload train_FD001.txt and click Run")
