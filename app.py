import streamlit as st
import pandas as pd
import numpy as np
import math

# --- L0-Flow: ГЕОМЕТРИЯ ЗОЛОТОГО СЕЧЕНИЯ ---
GOLDEN_K = 1.61803398875

def get_coherence_score(signal_slice):
    if len(signal_slice) < 2: return 1.0
    phases = [(v * GOLDEN_K) % 1.0 for v in signal_slice]
    x = np.mean([math.cos(2 * math.pi * p) for p in phases])
    y = np.mean([math.sin(2 * math.pi * p) for p in phases])
    return math.sqrt(x**2 + y**2)

# --- ИНТЕРФЕЙС ---
st.set_page_config(page_title="L0-Flow Professional", layout="wide")
st.title("🛡️ Протокол Диагностики: Калиброванный Резонанс")

uploaded_file = st.file_uploader("Загрузи train_FD001.txt", type=['txt'])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
    engine_id = st.sidebar.selectbox("Выбери Мотор", df[0].unique(), index=0)
    sensor_idx = 11 
    
    engine_data = df[df[0] == engine_id]
    cycles = engine_data[1].values
    raw_values = engine_data[sensor_idx].values
    norm = (raw_values - raw_values.min()) / (raw_values.max() - raw_values.min() + 1e-9)
    
    anomaly_map = []
    log_data = []
    
    # --- ЭТАП 1: КАЛИБРОВКА (Первые 30 циклов - обучение) ---
    baseline_scores = []
    for i in range(min(30, len(norm))):
        chunk = norm[max(0, i-5):i+1]
        baseline_scores.append(get_coherence_score(chunk))
    
    avg_baseline = np.mean(baseline_scores) # Уровень "здорового" шума

    # --- ЭТАП 2: АНАЛИЗ ---
    for i in range(len(norm)):
        chunk = norm[max(0, i-5):i+1]
        score = get_coherence_score(chunk)
        
        # Считаем отклонение именно от ЭТАЛОНА здоровья
        # Если score стал сильно ниже эталона - это износ
        chaos_idx = max(0, (avg_baseline - score) * 100)
        anomaly_map.append(chaos_idx)
        
        # В таблицу пишем только реальные проблемы (когда хаос выше 10%)
        if i > 30 and chaos_idx > 10:
            log_data.append({
                "Цикл": int(cycles[i]),
                "Резонанс (0-1)": round(score, 3),
                "Индекс Хаоса (%)": round(chaos_idx, 2),
                "Прогноз": "⚠️ УСТАЛОСТЬ" if chaos_idx < 25 else "🛑 КРИТИЧЕСКИЙ ИЗНОС"
            })

    # ВИЗУАЛИЗАЦИЯ
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Линия датчика (Сырые данные)")
        st.line_chart(raw_values)
    with c2:
        st.subheader("Детектор Разрушения (L0-Flow)")
        st.area_chart(anomaly_map)

    # ОТЧЕТ
    st.subheader("📋 Таблица аномалий (после калибровки)")
    if log_data:
        report_df = pd.DataFrame(log_data)
        st.dataframe(report_df, use_container_width=True)
        st.download_button("Скачать отчет", report_df.to_csv(index=False).encode('utf-8'), "engine_report_calibrated.csv")
    else:
        st.success("Система в идеальном резонансе. Аномалий выше порога не обнаружено.")

else:
    st.info("Загрузи данные для запуска калиброванного теста.")
