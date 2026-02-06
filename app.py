import streamlit as st
import pandas as pd
import numpy as np
import math

# --- ГЕОМЕТРИЯ ИНВАРИАНТА (TORUS PROJECTION) ---
GOLDEN_RATIO = 1.61803398875

def get_torus_projection(value, K):
    # Проекция на циклическую фазу Тора
    angle = 2 * math.pi * (value * K % 1.0)
    return math.cos(angle), math.sin(angle)

def calculate_resonance(values, K):
    if not values: return 0
    # Вычисляем средний вектор когерентности на Торе
    vectors = [get_torus_projection(v, K) for v in values]
    avg_x = sum(v[0] for v in vectors) / len(vectors)
    avg_y = sum(v[1] for v in vectors) / len(vectors)
    # Длина вектора R: 1.0 — идеальный резонанс, 0.0 — хаос
    return math.sqrt(avg_x**2 + avg_y**2)

# --- UI ---
st.set_page_config(page_title="Sovereign Torus", layout="wide")
st.title("💠 L0-Flow: Torus Resonance Diagnostic")
st.write("Анализ проекции сигнала на Золотое Сечение (K=1.618)")

uploaded_file = st.file_uploader("Загрузи train_FD001.txt", type=['txt'])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
    engine_id = st.sidebar.selectbox("ID Двигателя", df[0].unique())
    # Датчик 11 (Давление) — он лучше всего "гуляет" на Торе
    sensor_idx = st.sidebar.slider("Сенсор", 2, 25, 11)
    
    raw_data = df[df[0] == engine_id][sensor_idx].values
    # Нормализация
    norm = (raw_data - raw_data.min()) / (raw_data.max() - raw_data.min() + 1e-9)
    
    resonance_map = []
    anomaly_power = []

    # Скользящее окно для анализа "дыхания" Тора
    window_size = 5
    for i in range(len(norm)):
        window = norm[max(0, i-window_size):i+1]
        # Резонанс относительно Золотого Сечения
        R = calculate_resonance(window, GOLDEN_RATIO)
        resonance_map.append(R)
        
        # Аномалия — это когда структура ПАДАЕТ (Resonance < 1)
        # Мы инвертируем это, чтобы видеть "взрыв" проблемы
        anomaly_power.append(1.0 - R)

    # ВИЗУАЛИЗАЦИЯ
    c1, c2 = st.columns(2)
    with c1:
        st.write("### Состояние мотора (Raw Signal)")
        st.line_chart(raw_data)
    with c2:
        st.write("### Деструкция Тора (Anomaly Resonance)")
        # Умножаем на коэффициент, чтобы видеть микро-трещины
        st.area_chart([a * 100 for a in anomaly_power])

    # СТАТУС
    current_decay = np.mean(anomaly_power[-10:]) * 100
    if current_decay > 5:
        st.error(f"🛑 ВНИМАНИЕ! Тор деформирован. Коэффициент деструкции: {current_decay:.2f}%")
    else:
        st.success(f"💎 ГЕОМЕТРИЯ СТАБИЛЬНА. Резонанс с Золотым Сечением в норме.")

    st.info("💡 Лайфхак для Lamborghini: Обрати внимание, как правый график начинает 'шуметь' задолго до того, как левый покажет явный рост.")
