import streamlit as st
import pandas as pd
import numpy as np
import math

# --- CORE: ГЕОМЕТРИЯ ИНВАРИАНТА ---
GOLDEN_RATIO = 1.61803398875

def get_torus_coords(value, K):
    angle = 2 * math.pi * (value * K % 1.0)
    return math.cos(angle), math.sin(angle)

def calculate_resonance(window_data, K):
    if len(window_data) == 0: return 1.0
    vectors = [get_torus_coords(v, K) for v in window_data]
    avg_x = sum(v[0] for v in vectors) / len(vectors)
    avg_y = sum(v[1] for v in vectors) / len(vectors)
    return math.sqrt(avg_x**2 + avg_y**2)

# --- UI ---
st.set_page_config(page_title="Sovereign Axiom v1.0", layout="wide")
st.title("🛡️ L0-Flow: Sovereign Mind Diagnostic")
st.write("Объект: Анализ термодинамики через Золотое Сечение.")

uploaded_file = st.file_uploader("Загрузи train_FD001.txt", type=['txt'])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
    engine_id = st.sidebar.selectbox("ID Двигателя", df[0].unique(), index=0)
    # Датчик 11 (Давление) — самый информативный
    sensor_idx = st.sidebar.slider("Сенсор", 2, 25, 11)
    
    raw_data = df[df[0] == engine_id][sensor_idx].values
    # Нормализация для проекции
    norm = (raw_data - raw_data.min()) / (raw_data.max() - raw_data.min() + 1e-9)
    
    anomaly_power = []
    torus_points = []
    
    window_size = 7
    for i in range(len(norm)):
        window = norm[max(0, i-window_size):i+1]
        R = calculate_resonance(window, GOLDEN_RATIO)
        anomaly_power.append((1.0 - R) * 100)
        
        # Собираем точки для визуализации проекции
        tx, ty = get_torus_coords(norm[i], GOLDEN_RATIO)
        torus_points.append({'x': tx, 'y': ty, 'cycle': i})

    # ВИЗУАЛИЗАЦИЯ
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Сырой сигнал")
        st.line_chart(raw_data)
        
        st.subheader("Деформация Тора (%)")
        st.area_chart(anomaly_power)

    with col2:
        st.subheader("Проекция на плоскость Тора")
        points_df = pd.DataFrame(torus_points)
        # Показываем последние 50 точек — если они разбросаны, значит системе конец
        st.scatter_chart(points_df.tail(50), x='x', y='y', color='#ff4b4b')
        st.info("💡 Когда мотор здоров, красные точки образуют четкую дугу. Когда мотор умирает — они превращаются в хаотичное облако.")

    # ВЕРДИКТ
    last_anomaly = np.mean(anomaly_power[-10:])
    if last_anomaly > 15:
        st.error(f"🛑 КРИТИЧЕСКИЙ ВЫЛЕТ ИЗ РЕЗОНАНСА: {last_anomaly:.2f}%")
    else:
        st.success(f"💎 СТРУКТУРА СОХРАНЕНА. Аномалия: {last_anomaly:.2f}%")
