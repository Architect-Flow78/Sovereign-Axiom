import streamlit as st
import pandas as pd
import numpy as np
import math
import hashlib
import time
from datetime import datetime

# ============================================================
# CORE 1: ТВОЙ ПРОМЫШЛЕННЫЙ БЛОК (UTILS & STATS)
# ============================================================

class HLL:
    def __init__(self, buckets=256):
        self.buckets = buckets
        self.reg = [0]*buckets
    def add(self, v):
        h = hash(str(v))
        b = h & (self.buckets-1)
        w = h >> 8
        rank = len(bin(w)) - len(bin(w).rstrip("0"))
        self.reg[b] = max(self.reg[b], rank)
    def count(self):
        return int(self.buckets / (sum(2**-r for r in self.reg) + 1e-9))

class RunningStats:
    def __init__(self):
        self.n, self.mean, self.M2 = 0, 0, 0
        self.min, self.max = None, None
    def update(self, x_series):
        for v in x_series.dropna():
            self.n += 1
            d = v - self.mean
            self.mean += d / self.n
            self.M2 += d * (v - self.mean)
            self.min = v if self.min is None else min(self.min, v)
            self.max = v if self.max is None else max(self.max, v)
    def std(self):
        return (self.M2 / (self.n - 1))**0.5 if self.n > 1 else 0

# ============================================================
# CORE 2: НАШ РЕЗОНАНСНЫЙ БЛОК (TORUS / GOLDEN RATIO)
# ============================================================

GOLDEN_K = 1.61803398875

def get_coherence_score(signal_slice):
    if len(signal_slice) < 2: return 1.0
    phases = [(v * GOLDEN_K) % 1.0 for v in signal_slice]
    x = np.mean([math.cos(2 * math.pi * p) for p in phases])
    y = np.mean([math.sin(2 * math.pi * p) for p in phases])
    return math.sqrt(x**2 + y**2)

# ============================================================
# INTERFACE: STREAMLIT LAB STAND
# ============================================================

st.set_page_config(page_title="Axioma Flow: Renazzo-X", layout="wide")
st.title("💠 Axioma Flow | Renazzo-X Engine")
st.write("Индустриальный анализатор потоков телеметрии (L0-Flow Protocol)")

uploaded_file = st.file_uploader("Загрузи train_FD001.txt", type=['txt'])

if uploaded_file:
    # 1. Загрузка данных (Имитируем твой Engine.run)
    df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
    
    # Настройки прибора
    st.sidebar.header("Калибровка системы")
    engine_id = st.sidebar.selectbox("ID Двигателя", df[0].unique())
    sensor_idx = st.sidebar.slider("Сенсор (11 - Давление)", 2, 25, 11)
    sensitivity = st.sidebar.slider("Чувствительность", 0.1, 3.0, 1.2)
    noise_floor = st.sidebar.slider("Порог шума (%)", 0, 20, 8)
    
    # Выборка данных
    engine_data = df[df[0] == engine_id].copy()
    raw_values = engine_data[sensor_idx].values
    cycles = engine_data[1].values
    
    # 2. Профилирование (Твой Profiler)
    hll = HLL()
    rs = RunningStats()
    for v in raw_values: hll.add(v)
    rs.update(pd.Series(raw_values))
    
    # 3. Анализ Резонанса (IGA)
    norm = (raw_values - raw_values.min()) / (raw_values.max() - raw_values.min() + 1e-9)
    chaos_map = []
    
    # Калибровка по первым 25 циклам
    ref_window = 10
    baseline_scores = [get_coherence_score(norm[max(0, i-ref_window):i+1]) for i in range(25)]
    health_ref = np.mean(baseline_scores)
    
    log_entries = []
    
    for i in range(len(norm)):
        chunk = norm[max(0, i-ref_window):i+1]
        score = get_coherence_score(chunk)
        # Формула Хаоса
        chaos_idx = max(0, (health_ref - score) * 100 * sensitivity)
        if chaos_idx < noise_floor: chaos_idx = 0
        
        chaos_map.append(chaos_idx)
        
        if i > 30 and chaos_idx > 15:
            log_entries.append({
                "Cycle": int(cycles[i]),
                "Value": round(raw_values[i], 2),
                "Chaos_Index": round(chaos_idx, 2),
                "Status": "🛑 CRITICAL" if chaos_idx > 35 else "⚠️ WARNING"
            })

    # 4. Визуализация
    col1, col2, col3 = st.columns([1, 1, 1])
    col1.metric("Unique Vals (HLL)", hll.count())
    col2.metric("Mean Value", round(rs.mean, 2))
    col3.metric("Std Dev", round(rs.std(), 2))

    tab1, tab2 = st.tabs(["📉 Графики", "📋 Технический отчет"])
    
    with tab1:
        c1, c2 = st.columns(2)
        c1.subheader("Сенсор (Телеметрия)")
        c1.line_chart(raw_values)
        c2.subheader("Индекс Хаоса (L0-Flow)")
        c2.area_chart(chaos_map)

    with tab2:
        if log_entries:
            st.dataframe(pd.DataFrame(log_entries), use_container_width=True)
        else:
            st.success("Аномалий не зафиксировано. Система в резонансе.")

    # Хеш файла (как в твоем коде)
    file_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    st.caption(f"File SHA-256: {file_hash} | Engine: Renazzo-X v2.1")

else:
    st.info("Ожидание потока данных...")
