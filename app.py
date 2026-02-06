# ================================
# STREAMLIT + ENGINE (ALL-IN-ONE)
# ================================

import streamlit as st
import argparse
import json
import os
import gzip
import hashlib
import logging
import time
from datetime import datetime
from typing import Dict, Callable
import pandas as pd
import numpy as np
import tempfile

# ================================
# STREAMLIT CONFIG
# ================================

st.set_page_config(
    page_title="Streaming Data Quality Engine",
    layout="wide"
)

# ================================
# UTILS
# ================================

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

def infer_type(series):
    s = series.dropna().head(200)
    try:
        pd.to_numeric(s, downcast="integer")
        return "int"
    except:
        pass
    try:
        pd.to_numeric(s)
        return "float"
    except:
        pass
    try:
        pd.to_datetime(s, errors="raise")
        return "datetime"
    except:
        return "string"

# ================================
# OPERATORS
# ================================

OPERATORS: Dict[str, Callable] = {}

def register_operator(name, fn):
    OPERATORS[name] = fn

register_operator(">", lambda s,v: s > v)
register_operator("<", lambda s,v: s < v)
register_operator(">=", lambda s,v: s >= v)
register_operator("<=", lambda s,v: s <= v)
register_operator("==", lambda s,v: s == v)
register_operator("!=", lambda s,v: s != v)
register_operator("contains", lambda s,v: s.astype(str).str.contains(v, na=False))
register_operator("regex", lambda s,v: s.astype(str).str.match(v, na=False))
register_operator("between", lambda s,v: s.between(v[0], v[1]))
register_operator("is_null", lambda s,v: s.isna())
register_operator("not_null", lambda s,v: s.notna())

# ================================
# SCHEMA
# ================================

class Schema:
    def __init__(self, cfg):
        self.separator = cfg.get("separator", ",")
        self.columns = cfg.get("columns")
        self.types = cfg.get("types", {})
        self.drop = set(cfg.get("drop", []))
        self.chunk_size = int(cfg.get("chunk_size", 50000))
        self.output = cfg.get("output", "csv")

# ================================
# RULES
# ================================

class Rule:
    def __init__(self, field=None, operator=None, value=None,
                 expr=None, severity="error", name=None):
        self.field = field
        self.operator = operator
        self.value = value
        self.expr = expr
        self.severity = severity
        self.name = name or field or expr

    def apply(self, df):
        if self.expr:
            return df.eval(self.expr)
        return OPERATORS[self.operator](df[self.field], self.value)

class RuleEngine:
    def __init__(self, rules, mode="any"):
        self.rules = [Rule(**r) for r in rules]
        self.mode = mode

    def validate(self, columns):
        for r in self.rules:
            if r.field and r.field not in columns:
                raise ValueError(r.field)

    def evaluate(self, df):
        err = pd.Series(False, index=df.index)
        warn = pd.Series(False, index=df.index)
        reason = pd.Series("", index=df.index)

        for r in self.rules:
            m = r.apply(df)
            if r.severity == "error":
                err |= m
            else:
                warn |= m
            reason[m] += f"{r.name};"

        return err, warn, reason

# ================================
# APPROX DISTINCT
# ================================

class HLL:
    def __init__(self, buckets=256):
        self.buckets = buckets
        self.reg = [0]*buckets

    def add(self, v):
        h = hash(v)
        b = h & (self.buckets-1)
        w = h >> 8
        rank = len(bin(w)) - len(bin(w).rstrip("0"))
        self.reg[b] = max(self.reg[b], rank)

    def count(self):
        return int(self.buckets / sum(2**-r for r in self.reg))

# ================================
# RUNNING STATS
# ================================

class RunningStats:
    def __init__(self):
        self.n=0
        self.mean=0
        self.M2=0
        self.min=None
        self.max=None

    def update(self, x):
        for v in x.dropna():
            self.n+=1
            d=v-self.mean
            self.mean+=d/self.n
            self.M2+=d*(v-self.mean)
            self.min=v if self.min is None else min(self.min,v)
            self.max=v if self.max is None else max(self.max,v)

    def std(self):
        return (self.M2/(self.n-1))**0.5 if self.n>1 else 0

# ================================
# STATS DETECTOR
# ================================

class StatsDetector:
    def __init__(self, z=3.5):
        self.z=z

    def detect(self, df):
        mask=pd.Series(False,index=df.index)
        reason=pd.Series("",index=df.index)

        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            med=df[c].median()
            mad=np.median(np.abs(df[c]-med))
            if mad>0:
                z=0.6745*(df[c]-med)/mad
            else:
                std=df[c].std()
                if std==0 or np.isnan(std):
                    continue
                z=(df[c]-df[c].mean())/std
            m=np.abs(z)>self.z
            mask|=m
            reason[m]+=f"stat_{c};"
        return mask,reason

# ================================
# PROFILER
# ================================

class Profiler:
    def __init__(self):
        self.stats={}

    def update(self,df):
        for c in df.columns:
            stt=self.stats.setdefault(c,{
                "rows":0,
                "nulls":0,
                "unique":HLL(),
                "numeric":RunningStats(),
                "top":{}
            })
            s=df[c]
            stt["rows"]+=len(s)
            stt["nulls"]+=int(s.isna().sum())
            for v in s.dropna():
                stt["unique"].add(v)
            if pd.api.types.is_numeric_dtype(s):
                stt["numeric"].update(s)
            stt["top"]=s.value_counts().head(5).to_dict()

    def export(self):
        out={}
        for c,stt in self.stats.items():
            o={
                "rows":stt["rows"],
                "nulls":stt["nulls"],
                "unique_estimate":stt["unique"].count(),
                "top":stt["top"]
            }
            rs=stt["numeric"]
            if rs.n>0:
                o.update({
                    "min":rs.min,
                    "max":rs.max,
                    "mean":rs.mean,
                    "std":rs.std()
                })
            out[c]=o
        return out

# ================================
# ENGINE
# ================================

class Engine:
    def __init__(self,schema=None,rules=None):
        self.schema=Schema(json.load(open(schema))) if schema else None
        self.rules=RuleEngine(json.load(open(rules)) if rules else [])
        self.stats=StatsDetector()
        self.profiler=Profiler()
        self.samples=[]

    def auto_schema(self,path):
        with open_file(path) as f:
            s=pd.read_csv(f,nrows=200)
        return Schema({
            "columns":list(s.columns),
            "types":{c:infer_type(s[c]) for c in s.columns}
        })

    def cast(self,df):
        if not self.schema:
            return df
        for c,t in self.schema.types.items():
            if t=="int":
                df[c]=pd.to_numeric(df[c],errors="coerce",downcast="integer")
            elif t=="float":
                df[c]=pd.to_numeric(df[c],errors="coerce")
            elif t=="datetime":
                df[c]=pd.to_datetime(df[c],errors="coerce")
        return df

    def run(self,path):
        if not self.schema:
            self.schema=self.auto_schema(path)

        self.rules.validate(self.schema.columns)

        reader=pd.read_csv(
            open_file(path),
            sep=self.schema.separator,
            chunksize=self.schema.chunk_size,
            dtype=str
        )

        total=clean=bad=0
        t0=time.time()

        for chunk in reader:
            chunk=self.cast(chunk)

            em,wm,rr=self.rules.evaluate(chunk)
            sm,sr=self.stats.detect(chunk)

            mask=em|sm
            chunk["__error_reason"]=rr+sr

            good=chunk[~mask]
            bad_rows=chunk[mask]

            if len(self.samples)<100:
                self.samples.extend(
                    bad_rows.head(100-len(self.samples)).to_dict("records")
                )

            self.profiler.update(chunk.drop(columns="__error_reason"))

            total+=len(chunk)
            clean+=len(good)
            bad+=len(bad_rows)

        report={
            "input":path,
            "hash":sha256_file(path),
            "finished":datetime.utcnow().isoformat(),
            "rows_total":total,
            "rows_clean":clean,
            "rows_anomalies":bad,
            "rows_per_sec":int(total/(time.time()-t0)),
            "profile":self.profiler.export(),
            "sample_anomalies":self.samples
        }

        return report

# ================================
# STREAMLIT UI
# ================================

st.title("🚀 Streaming Data Quality Engine")

file = st.file_uploader("Upload CSV / TXT / GZ", type=["csv","txt","gz"])

if st.button("Run") and file:

    with st.spinner("Processing..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            path = tmp.name

        engine = Engine()
        report = engine.run(path)

    st.success("Done")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows",report["rows_total"])
    c2.metric("Clean",report["rows_clean"])
    c3.metric("Anomalies",report["rows_anomalies"])
    c4.metric("Rows/sec",report["rows_per_sec"])

    st.subheader("Column Profile")
    st.dataframe(pd.DataFrame(report["profile"]).T)

    if report["sample_anomalies"]:
        st.subheader("Sample Anomalies")
        st.dataframe(pd.DataFrame(report["sample_anomalies"]))

    st.subheader("Raw JSON")
    st.json(report)

else:
    st.info("Upload file and click Run")
