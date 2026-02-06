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

def calculate_resonance(window_data, K):
    # Проверка на пустоту для NumPy
    if len(window_data) == 0: 
        return 1.0
    
    # Вычисляем средний вектор когерентности на Торе
    vectors = [get_torus_projection(v, K) for v in window_data]
    avg_x = sum(v[0] for v in vectors) / len(vectors)
    avg_y = sum(v[1] for v in vectors) / len(vectors)
    
    # Длина вектора R: 1.0 — идеальный резонанс, 0.0 — полный хаос
    return math.sqrt(avg_x**2 + avg_y**2)

# --- UI ---
st.set_page_config(page_title="Sovereign Torus Lab", layout="wide")
st.title("💠 L0-Flow: Torus Resonance Diagnostic")
st.write("Место действия: Ренаццо. Проекция на Тор через Золотое Сечение.")

uploaded_file = st.file_uploader("Загрузи train_FD001.txt", type=['txt'])

if uploaded_file:
    # Используем r"\s+" чтобы избежать предупреждений в логах
    df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
    
    engine_id = st.sidebar.selectbox("ID Двигателя", df[0].unique(), index=0)
    # Датчик 11 — давление, Датчик 4 — температура
    sensor_idx = st.sidebar.slider("Сенсор (11 - лучший для аномалий)", 2, 25, 11)
    
    raw_data = df[df[0] == engine_id][sensor_idx].values
    # Нормализация (Сигнал в диапазон 0-1 для Тора)
    norm = (raw_data - raw_data.min()) / (raw_data.max() - raw_data.min() + 1e-9)
    
    anomaly_power = []
    
    # Скользящее окно: Рой смотрит на 5 шагов сразу
    window_size = 5
    for i in range(len(norm)):
        window = norm[max(0, i-window_size):i+1]
        
        # Считаем резонанс текущего момента с Золотым Сечением
        R = calculate_resonance(window, GOLDEN_RATIO)
        
        # Аномалия — это "деформация" Тора (1.0 - R)
        # Усиливаем микро-колебания в 100 раз
        anomaly_power.append((1.0 - R) * 100)

    # ВИЗУАЛИЗАЦИЯ
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Сырой сигнал (NASA Sensor)")
        st.line_chart(raw_data)
    with col2:
        st.subheader("Деформация Тора (Resonance Anomaly)")
        # Это график того, как мотор "вылетает" из Золотого Сечения
        st.area_chart(anomaly_power)

    # ВЕРДИКТ
    current_anomaly = np.mean(anomaly_power[-10:])
    if current_anomaly > 10:
        st.error(f"⚠️ КРИТИЧЕСКАЯ ДЕФОРМАЦИЯ: {current_anomaly:.2f}%. Структура Тора разрушена.")
    elif current_anomaly > 3:
        st.warning(f"⚡ ПРЕД-АНОМАЛИЯ: {current_anomaly:.2f}%. Появление 'шума' в резонансе.")
    else:
        st.success(f"💎 ИДЕАЛЬНЫЙ РЕЗОНАНС: {current_anomaly:.2f}%. Тор стабилен.")

else:
    st.info("Загрузи файл, чтобы запустить проекцию на Тор.")
