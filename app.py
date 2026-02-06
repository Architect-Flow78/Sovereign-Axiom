import streamlit as st
import pandas as pd
import numpy as np
import math
import random
from collections import deque

# ============================================================
# CORE ENGINE: SOVEREIGN RESONANCE NETWORK (v4.2-Full)
# ============================================================

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
        self.threshold = 0.01
        self.last_C = 0.5
    
    def update(self, values):
        phases = [phase(v, self.K) for v in values]
        C = circular_coherence(phases)
        D = abs(C - self.fast)
        self.threshold = math.exp(0.99 * math.log(self.threshold + 1e-9) + 0.01 * math.log(D + 1e-9))
        self.fast = ema(self.fast, C, 0.9)
        breach = D > self.threshold * 3.0
        self.last_C = C
        return C, breach

class SovereignOrganism:
    def __init__(self, id_tag):
        self.id = id_tag
        self.cell = InvariantCell(random.uniform(1.2, 2.8))
        self.need = 0.0
        self.shield = 1.0
        self.history = deque(maxlen=100)

    def update(self, signal_chunk, best_K, field_strength):
        C, breach = self.cell.update(signal_chunk)
        if self.need > 0.4: self.cell.K = ema(self.cell.K, best_K, 0.92)
        target_need = max(0.0, 0.7 - C) - 0.25 * field_strength
        self.need = ema(self.need, target_need, 0.9)
        self.shield = ema(self.shield, C, 0.995)
        self.cell.K += random.uniform(-0.001, 0.001)
        self.cell.K = max(0.5, min(3.5, self.cell.K))
        self.history.append(C)
        return C

# ============================================================
# UI & DATA ADAPTER
# ============================================================

st.set_page_config(page_title="L0-Flow Sovereign Tool", layout="wide")

st.title("🛡️ Sovereign Mind Diagnostic System")
st.subheader("Renazzo Engine Analysis: Collective Morphodynamic Organism v4.2")

uploaded_file = st.file_uploader("Загрузите данные (NASA Turbofan .txt / CSV)", type=['txt', 'csv'])

if uploaded_file is not None:
    try:
        # 1. Загрузка
        sep = "\s+" if uploaded_file.name.endswith('.txt') else ","
        df = pd.read_csv(uploaded_file, sep=sep, header=None)
        
        # 2. Настройки
        st.sidebar.header("Параметры Роя")
        engine_id = st.sidebar.selectbox("ID Двигателя", df[0].unique())
        sensor_idx = st.sidebar.slider("Сенсор (NASA 7=Temp, 12=RPM)", 2, 25, 7)
        swarm_size = st.sidebar.slider("Размер Роя (Кол-во клеток)", 5, 50, 10)
        
        # 3. Подготовка сигнала
        raw_signal = df[df[0] == engine_id][sensor_idx].values
        norm_signal = (raw_signal - raw_signal.min()) / (raw_signal.max() - raw_signal.min() + 1e-9)
        
        # 4. Инициализация Роя
        if 'swarm' not in st.session_state:
            st.session_state.swarm = [SovereignOrganism(f"S{i}") for i in range(swarm_size)]
        
        # 5. Процессинг (Симуляция жизни через весь файл)
        best_K = 1.618
        field_strength = 0.0
        swarm_results = []
        global_coherence = []

        for val in norm_signal:
            step_coherence = []
            for agent in st.session_state.swarm:
                c = agent.update([val], best_K, field_strength)
                step_coherence.append(c)
            
            # Обновляем поле для следующего шага
            high_flow = [c for c in step_coherence if c > 0.75]
            field_strength = len(high_flow) / swarm_size
            if high_flow:
                best_K = sum(a.cell.K for a in st.session_state.swarm if a.history[-1] > 0.75) / (len(high_flow) + 1e-9)
            
            global_coherence.append(np.mean(step_coherence))

        # 6. ВИЗУАЛИЗАЦИЯ (То, что уничтожит скепсис)
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Сырой сигнал датчика**")
            st.line_chart(raw_signal)
            
        with col2:
            st.write("**Поле когерентности Роя (Здоровье системы)**")
            st.area_chart(global_coherence)

        # 7. ДИАГНОСТИКА
        anomaly_threshold = 0.65
        st.divider()
        if global_coherence[-1] < anomaly_threshold:
            st.error(f"🛑 ОБНАРУЖЕНА ДЕСТРУКЦИЯ: Когерентность роя упала до {global_coherence[-1]:.3f}. Система разрушается.")
        else:
            st.success(f"💎 ИНВАРИАНТ СТАБИЛЕН: Когерентность {global_coherence[-1]:.3f}. Система в потоке.")

    except Exception as e:
        st.error(f"Ошибка системы: {e}")
else:
    st.info("Ожидание данных для инициализации Роя...")
