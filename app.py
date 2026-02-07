import streamlit as st
import pandas as pd
import numpy as np
import math

# --- CORE: ИНВАРИАНТ ГЕОМЕТРИИ (IGA) ---
GOLDEN_K = 1.61803398875

def get_coherence_score(signal_slice):
    if len(signal_slice) < 2: return 1.0
    phases = [(v * GOLDEN_K) % 1.0 for v in signal_slice]
    x = np.mean([math.cos(2 * math.pi * p) for p in phases])
    y = np.mean([math.sin(2 * math.pi * p) for p in phases])
    return math.sqrt(x**2 + y**2)

# --- UI ---
st.set_page_config(page_title="Axioma Lab Stand", layout="wide")
st.title("🔬 Axioma Flow: Лабораторный Тест")

uploaded_file = st.file_uploader("Загрузи train_FD001.txt для теста", type=['txt'])

if uploaded_file:
    # --- НАСТРОЙКИ "НЕЖНОСТИ" (Calibration) ---
    st.sidebar.header("Настройка прибора")
    sensitivity = st.sidebar.slider("Чувствительность (Sensitivity)", 0.1, 2.0, 1.0, help="Чем выше, тем раньше бьем тревогу")
    window_size = st.sidebar.slider("Окно анализа (Window)", 3, 20, 7, help="Размер выборки для поиска резонанса")
    noise_threshold = st.sidebar.slider("Порог шума (Noise Floor %)", 0, 20, 5)

    df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
    engine_id = st.sidebar.selectbox("ID Двигателя", df[0].unique(), index=0)
    
    # Датчик 11 (Давление на выходе из ЛПЦ)
    raw_values = df[df[0] == engine_id][11].values
    cycles = df[df[0] == engine_id][1].values
    
    # Нормализация
    norm = (raw_values - raw_values.min()) / (raw_values.max() - raw_values.min() + 1e-9)
    
    # --- АНАЛИЗ ---
    results = []
    # Калибровка по первым 20 циклам (эталон здоровья)
    baseline_scores = [get_coherence_score(norm[max(0, i-window_size):i+1]) for i in range(20)]
    health_ref = np.mean(baseline_scores)

    for i in range(len(norm)):
        chunk = norm[max(0, i-window_size):i+1]
        score = get_coherence_score(chunk)
        
        # Вычисляем Хаос с учетом чувствительности
        chaos = max(0, (health_ref - score) * 100 * sensitivity)
        
        # Фильтруем фоновый шум
        if chaos < noise_threshold: chaos = 0
            
        results.append({
            "Cycle": int(cycles[i]),
            "Value": raw_values[i],
            "Chaos": round(chaos, 2),
            "Coherence": round(score, 4)
        })

    res_df = pd.DataFrame(results)

    # --- ВИЗУАЛИЗАЦИЯ ТЕСТА ---
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Показания датчика")
        st.line_chart(res_df.set_index("Cycle")["Value"])
    with c2:
        st.subheader("Индекс Хаоса (Твое 'Золотое Сечение')")
        st.area_chart(res_df.set_index("Cycle")["Chaos"])

    # --- ПРОВЕРКА ТОЧНОСТИ ---
    st.subheader("📊 Протокол испытаний")
    
    # Ищем точку первого обнаружения
    detection_point = res_df[res_df["Chaos"] > 15].head(1)
    
    if not detection_point.empty:
        st.warning(f"🎯 Прибор зафиксировал аномалию на цикле: **{detection_point.iloc[0]['Cycle']}**")
        st.info(f"Фактическая смерть мотора: **{res_df.iloc[-1]['Cycle']}** цикл. Запас времени: **{int(res_df.iloc[-1]['Cycle'] - detection_point.iloc[0]['Cycle'])}** циклов.")
    
    st.dataframe(res_df[res_df["Chaos"] > 0], use_container_width=True)
