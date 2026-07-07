import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# --- 웹 페이지 설정 ---
st.set_page_config(
    page_title="청소년 정신 건강 데이터 EDA",
    page_icon="📊",
    layout="wide"
)

# --- 커스텀 스타일 ---
st.markdown("""
    <style>
        .main-title { font-size: 28px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
        .sub-title { font-size: 15px; color: #4B5563; margin-bottom: 25px; }
        .section-box { padding: 18px; background-color: #F3F4F6; border-radius: 8px; margin-bottom: 20px; }
        .success-box { padding: 12px; background-color: #DEF7EC; color: #03543F; border-radius: 6px; font-size: 14px; margin-bottom: 10px; }
        .info-box { padding: 12px; background-color: #EBF5FF; color: #1E429F; border-radius: 6px; font-size: 14px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 데이터 전처리 및 정제 (EDA) 현황판</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">데이터셋의 결측치와 이상치를 정제하고, 전/후 데이터를 실시간으로 비교 분석합니다.</div>', unsafe_allow_html=True)

# 파일 읽기 경로 안전화
data_path = "Teen_Mental_Health_Dataset.csv"

if os.path.exists(data_path):
    df_raw = pd.read_csv(data_path)
    df = df_raw.copy()
    
    # [1] 결측치 정제 처리
    replaced_null_count = 0
    raw_null_total = df_raw.isnull().sum().sum()
    
    for col in df.columns:
        null_cnt = df[col].isnull().sum()
        if null_cnt > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].mean())
            else:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_val)
            replaced_null_count += null_cnt

    # [2] IQR 방식 이상치 정제 처리
    replaced_outlier_count = 0
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_cnt = outlier_mask.sum()
        
        if outlier_cnt > 0:
            normal_mean = df.loc[~outlier_mask, col].mean()
            df.loc[outlier_mask, col] = normal_mean
            replaced_outlier_count += outlier_cnt

    # --- 화면 배치 ---
    st.markdown("### 🔍 1. 데이터 정제 리포트 & 전/후 비교")
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="총 샘플 수 (행/열)", value=f"{df.shape[0]}행 × {df.shape[1]}열")
    with m2:
        st.metric(label="원본 결측치 수", value=f"{raw_null_total} 개", delta=f"-{replaced_null_count}" if replaced_null_count > 0 else None)
    with m3:
        st.metric(label="탐지된 이상치 수", value=f"{replaced_outlier_count} 개", delta="대체 완료" if replaced_outlier_count > 0 else None)
    with m4:
        st.metric(label="데이터 상태", value="정상(Clean)" if df.isnull().sum().sum() == 0 else "미완료")

    tab1, tab2, tab3 = st.tabs(["📄 원본 vs 정제본 데이터", "📊 결측치 분포 비교", "📈 통계량 변화"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("🚨 [전처리 전] 원본 데이터")
            st.dataframe(df_raw.head(5), use_container_width=True)
        with c2:
            st.caption("✅ [전처리 후] 정제 데이터")
            st.dataframe(df.head(5), use_container_width=True)

    with tab2:
        null_compare_df = pd.DataFrame({"전처리 전": df_raw.isnull().sum(), "전처리 후": df.isnull().sum()})
        st.dataframe(null_compare_df.T, use_container_width=True)

    with tab3:
        if len(numeric_cols) > 0:
            selected_num_col = st.selectbox("변수를 선택하세요:", numeric_cols)
            stat_compare = pd.DataFrame({"전처리 전": df_raw[selected_num_col].describe(), "전처리 후": df[selected_num_col].describe()})
            st.dataframe(stat_compare, use_container_width=True)

    # --- 시각화 패널 ---
    st.markdown("---")
    st.markdown("### 🎛️ 2. 시각화 제어 패널")
    
    columns = df.columns.tolist()
    available_colors = ["None"] + [col for col in ["depression_label", "gender", "platform_usage", "social_interaction_level"] if col in columns]
    
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox("📌 X축 변수 선택", options=columns, index=0)
    with col2:
        y_axis = st.selectbox("📌 Y축 변수 선택", options=columns, index=min(1, len(columns)-1))
    with col3:
        color_target = st.selectbox("🎨 색상 그룹", options=available_colors, index=0)
    st.markdown('</div>', unsafe_allow_html=True)

    color_var = None if color_target == "None" else color_target
    chart_type = st.radio("📊 차트 형태", ["산점도 (Scatter)", "바 차트 (Bar)", "박스 플롯 (Box)", "히스토그램 (Histogram)"], horizontal=True)
    
    try:
        if chart_type == "산점도 (Scatter)":
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_var, title="✨ 산점도 분석", template="plotly_white")
        elif chart_type == "바 차트 (Bar)":
            df_grouped = df.groupby([x_axis] + ([color_var] if color_var else [])).mean(numeric_only=True).reset_index()
            fig = px.bar(df_grouped, x=x_axis, y=y_axis, color=color_var, title="📊 평균값 바 차트", barmode="group", template="plotly_white")
        elif chart_type == "박스 플롯 (Box)":
            fig = px.box(df, x=x_axis, y=y_axis, color=color_var, title="📦 박스 플롯", points="all", template="plotly_white")
        elif chart_type == "히스토그램 (Histogram)":
            fig = px.histogram(df, x=x_axis, color=color_var, title="📐 히스토그램", barmode="overlay", template="plotly_white")
            
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"차트 생성 중 오류가 발생했습니다. 변수 조합을 확인해주세요. ({e})")
else:
    st.error(f"❌ '{data_path}' 파일을 찾을 수 없습니다. GitHub 저장소 올리실 때 파일 이름 대소문자가 정확한지, 루트 폴더에 잘 들어가 있는지 확인해 주세요.")
