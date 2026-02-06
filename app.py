import streamlit as st
import pandas as pd
import numpy as np
import math
import random
from collections import deque

# --- CORE ENGINE ---
def ema(o, n, a): return a * o + (1 - a) * n
def phase(x, K): return (x * K) % 1.0

def circular_coherence(ph):
    if len(ph) < 1: return 0.5
    sc = sum(math.cos(2*math.pi*p) for p in ph) / len(ph)
    ss = sum(math.sin(2*math.pi*p) for p in ph) / len(ph)
    return math.sqrt(sc*sc + ss*ss)

class InvariantCell:
    def __init__(self, K):
        self.K = K
        self.fast = 0.5
        self.last_C = 0.5
    def update(self, values, alpha=0.9):
        phases = [phase(v, self.K) for v in values]
        C = circular_coherence(phases)
        self.fast = ema(self.fast, C, alpha)
        self.last_C = C
        return C

# --- UI ---
st.set_page_config(page_title="L0 Sovereign Diagnostic", layout="wide")
st.title("🛡️ L0-Flow: Sovereign Resonance Lab")
st.write("Место действия: Ренаццо. Объект: Динамика Invariant.")

uploaded_file = st.file_uploader("Загрузи файл (NASA FD001.txt)", type=['txt', 'csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep="\s+", header=None)
    
    # Настройки
    engine_id = st.sidebar.selectbox("ID Двигателя", df[0].unique(), index=0)
    # Датчик 11 (индекс 11) - один из самых чувствительных к износу
    sensor_idx = st.sidebar.slider("Датчик (NASA: 11-Pressure, 4-Temp)", 2, 25, 11)
    k_factor = st.sidebar.slider("Резонанс K", 0.5, 4.0, 1.618)
    
    # Сигнал
    raw_signal = df[df[0] == engine_id][sensor_idx].values
    norm_signal = (raw_signal - raw_signal.min()) / (raw_signal.max() - raw_signal.min() + 1e-9)
    
    # Прогон через "Организм"
    cell = InvariantCell(K=k_factor)
    history_c = []
    resistance = []
    
    # Эталон (первые 20 циклов)
    baseline = 0
    
    for i, v in enumerate(norm_signal):
        c = cell.update([v], alpha=0.8) # Ускорили реакцию
        history_c.append(c)
        
        if i == 20: baseline = np.mean(history_c)
        
        # Считаем Сопротивление (отклонение от нормы)
        if i > 20:
            res = abs(c - baseline) * 10 # Усиливаем для наглядности
            resistance.append(res)
        else:
            resistance.append(0)

    # Вывод графиков
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Сырой сигнал датчика")
        st.line_chart(raw_signal)
    with col2:
        st.subheader("Потеря когерентности (Аномалия)")
        st.area_chart(resistance)

    # Вердикт
    if np.mean(resistance[-10:]) > 1.5:
        st.error(f"🛑 ОБНАРУЖЕН ПРЕДЕЛЬНЫЙ ИЗНОС. Рой фиксирует разрушение структуры данных.")
    else:
        st.success(f"💎 Система стабильна. Резонанс в норме.")
else:
    st.info("Загрузи файл NASA, чтобы увидеть работу Роя.")
