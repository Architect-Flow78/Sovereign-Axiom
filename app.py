import streamlit as st
import pandas as pd
import numpy as np
import math

# --- L0-Flow: ГЕОМЕТРИЯ ЗОЛОТОГО СЕЧЕНИЯ ---
GOLDEN_K = 1.61803398875

def get_coherence_score(signal_slice):
    if len(signal_slice) < 2: 
        return 1.0
    # Проецируем каждое число на фазу Тора
    phases = [(v * GOLDEN_K) % 1.0 for v in signal_slice]
    # Считаем векторную сумму (Когерентность)
    x = np.mean([math.cos(2 * math.pi * p) for p in phases])
    y = np.mean([math.sin(2 * math.pi * p) for p in phases])
    return math.sqrt(x**2 + y**2)

# --- ИНТЕРФЕЙС ---
st.set_page_config(page_title="L0-Flow Test", layout="wide")
st.title("🛡️ Тест Двигателя: Резонанс vs Хаос")

uploaded_file = st.file_uploader("Загрузи train_FD001.txt", type=['txt'])

if uploaded_file:
    # 1. Читаем NASA данные
    df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
    engine_id = st.sidebar.selectbox("Выбери Мотор", df[0].unique(), index=0)
    # Датчик 11 — это "сердце" турбины
    sensor_idx = 11 
    
    raw_values = df[df[0] == engine_id][sensor_idx].values
    # Нормализация
    norm = (raw_values - raw_values.min()) / (raw_values.max() - raw_values.min() + 1e-9)
    
    # 2. АНАЛИЗ РОЕМ
    anomaly_map = []
    window = 5 
    
    for i in range(len(norm)):
        chunk = norm[max(0, i-window):i+1]
        score = get_coherence_score(chunk)
        anomaly_map.append((1.0 - score) * 100)

    # 3. ВИЗУАЛИЗАЦИЯ
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📡 Сигнал датчика (Вход)")
        st.line_chart(raw_values)
    with col2:
        st.subheader("🔥 Аномалия по Золотому Сечению (Выход)")
        st.area_chart(anomaly_map)

    # ВЕРДИКТ
    final_risk = np.mean(anomaly_map[-10:])
    if final_risk > 10:
        st.error(f"ТЕСТ: ПРОВАЛ. Мотор разрушается. Индекс Хаоса: {final_risk:.2f}%")
    else:
        st.success(f"ТЕСТ: УСПЕХ. Поток в резонансе. Индекс Хаоса: {final_risk:.2f}%")
else:
    st.info("Жду файл NASA для проведения теста...")
