import streamlit as st
import pandas as pd
import numpy as np
import math

# --- ГЕОМЕТРИЯ ЗОЛОТОГО СЕЧЕНИЯ ---
GOLDEN_K = 1.61803398875

def get_coherence(values):
    if len(values) < 2: return 1.0
    # Проекция фазы на круг
    phases = [(v * GOLDEN_K) % 1.0 for v in values]
    x = np.mean([math.cos(2 * math.pi * p) for p in phases])
    y = np.mean([math.sin(2 * math.pi * p) for p in phases])
    return math.sqrt(x**2 + y**2)

# --- ИНТЕРФЕЙС ---
st.set_page_config(page_title="L0-Flow Diagnostic", layout="wide")
st.title("💠 Sovereign Torus: NASA FD001 Analysis")

file = st.file_uploader("Загрузи файл train_FD001.txt", type=['txt'])

if file:
    # Загрузка и выбор данных
    data = pd.read_csv(file, sep=r"\s+", header=None)
    engine_id = st.sidebar.selectbox("Двигатель №", data[0].unique())
    sensor_id = st.sidebar.slider("Сенсор (11 - Давление, 4 - Темп)", 2, 25, 11)
    
    # Подготовка сигнала
    subset = data[data[0] == engine_id][sensor_id].values
    norm = (subset - subset.min()) / (subset.max() - subset.min() + 1e-9)
    
    # Рой анализирует поток
    anomalies = []
    window = 10
    for i in range(len(norm)):
        chunk = norm[max(0, i-window):i+1]
        coh = get_coherence(chunk)
        # Аномалия - это потеря когерентности (1.0 - coh)
        anomalies.append((1.0 - coh) * 100)

    # ВЫВОД РЕЗУЛЬТАТА
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📈 Состояние Сенсора (NASA)")
        st.line_chart(subset)
    with c2:
        st.subheader("🔥 Индекс Разрушения (L0-Flow)")
        st.area_chart(anomalies)

    # ВЕРДИКТ: Понятный даже ребенку
    score = np.mean(anomalies[-10:])
    if score > 15:
        st.error(f"⚠️ КРИТИЧЕСКИЙ ИЗНОС: {score:.1f}% — Двигатель на пределе!")
    elif score > 5:
        st.warning(f"⚡ ПРЕДУПРЕЖДЕНИЕ: {score:.1f}% — Появление усталости металла.")
    else:
        st.success(f"💎 ПОТОК ЧИСТ: {score:.1f}% — Система в резонансе.")

    st.markdown("---")
    st.write("**Как читать это:** Слева — просто цифры датчика. Справа — то, как Рой видит 'хрипы' в этих цифрах через Золотое Сечение. Если правый график растет — значит, Тор системы искривляется.")
