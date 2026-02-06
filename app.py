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
st.set_page_config(page_title="L0-Flow Table Report", layout="wide")
st.title("🛡️ Протокол Диагностики: Резонансный Износ")

uploaded_file = st.file_uploader("Загрузи train_FD001.txt", type=['txt'])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
    engine_id = st.sidebar.selectbox("Выбери Мотор", df[0].unique(), index=0)
    sensor_idx = 11 # Наш основной датчик давления
    
    # Данные конкретного мотора
    engine_data = df[df[0] == engine_id]
    cycles = engine_data[1].values
    raw_values = engine_data[sensor_idx].values
    norm = (raw_values - raw_values.min()) / (raw_values.max() - raw_values.min() + 1e-9)
    
    anomaly_map = []
    log_data = [] # Сюда пишем таблицу
    
    for i in range(len(norm)):
        chunk = norm[max(0, i-5):i+1]
        score = get_coherence_score(chunk)
        chaos_idx = (1.0 - score) * 100
        anomaly_map.append(chaos_idx)
        
        # Записываем в таблицу только если хаос выше нормы (3% - порог шума)
        if chaos_idx > 3:
            log_data.append({
                "ID Мотора": int(engine_id),
                "Цикл (Время)": int(cycles[i]),
                "Датчик №": sensor_idx,
                "Значение": round(raw_values[i], 2),
                "Индекс Хаоса (%)": round(chaos_idx, 2),
                "Статус": "⚠️ ПРЕД-АНОМАЛИЯ" if chaos_idx < 10 else "🛑 РАЗРУШЕНИЕ"
            })

    # ВИЗУАЛИЗАЦИЯ (Графики оставляем для контроля)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Линия датчика")
        st.line_chart(raw_values)
    with c2:
        st.subheader("Пульс Хаоса")
        st.area_chart(anomaly_map)

    # --- ТАБЛИЧНЫЙ ОТЧЕТ ---
    st.subheader("📋 Таблица критических состояний")
    if log_data:
        report_df = pd.DataFrame(log_data)
        st.dataframe(report_df, use_container_width=True)
        
        # Кнопка скачивания
        csv = report_df.to_csv(index=False).encode('utf-8')
        st.download_button("Скачать отчет (CSV)", csv, "engine_report.csv", "text/csv")
    else:
        st.success("В этом моторе аномалий не обнаружено. Резонанс чист.")

else:
    st.info("Загрузи данные для формирования таблицы...")
