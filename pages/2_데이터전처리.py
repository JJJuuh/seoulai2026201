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

# --- 커스텀 스타일 (CSS 기법 적용) ---
st.markdown("""
    <style>
        .main-title { font-size: 28px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
        .sub-title { font-size: 15px; color: #4B5563; margin-bottom: 25px; }
        .section-box { padding: 18px; background-color: #F3F4F6; border-radius: 8px; margin-bottom: 20px; }
        .success-box { padding: 12px; background-color: #DEF7EC; color: #03543F; border-radius: 6px; font-size: 14px; margin-bottom: 10px; }
        .info-box { padding: 12px; background-color: #EBF5FF; color: #1E429F; border-radius: 6px; font-size: 14px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- 헤더 섹션 ---
st.markdown('<div class="main-title">📊 데이터 전처리 및 정제 (EDA) 현황판</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">데이터셋의 결측치와 이상치를 정제하고, 전/후 데이터를 실시간으로 비교 분석합니다.</div>', unsafe_allow_html=True)

# GitHub/Streamlit Cloud 경로 호환을 위한 절대 경로 계산
base_path = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
data_path = os.path.join(base_path, "Teen_Mental_Health_Dataset.csv")

if os.path.exists(data_path):
    # 1. 최초 데이터 로드 (원본 보존)
    df_raw = pd.read_csv(data_path)
    df = df_raw.copy()
    
    # ---------------------------------------------------------
    # 🛠️ 데이터 정제 및 수행 (결측치 & 이상치)
    # ---------------------------------------------------------
    
    # [1] 결측치 정제 처리
    replaced_null_count = 0
    raw_null_total = df_raw.isnull().sum().sum()
    
    for col in df.columns:
        null_cnt = df[col].isnull().sum()
        if null_cnt > 0:
            if df[col].dtype in ['int64', 'float64']:
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
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

    # ---------------------------------------------------------
    # 🔄 전처리 전/후 대시보드 화면 배치
    # ---------------------------------------------------------
    st.markdown("### 🔍 1. 데이터 정제 리포트 & 전/후 비교")
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="총 샘플 수 (행/열)", value=f"{df.shape[0]}행 × {df.shape[1]}열")
    with m2:
        st.metric(label="원본 결측치(NaN) 수", value=f"{raw_null_total} 개", delta=f"-{replaced_null_count} 처리 완료" if replaced_null_count > 0 else "정상")
    with m3:
        st.metric(label="탐지된 이상치(Outlier) 수", value=f"{replaced_outlier_count} 개", delta="평균값 대체 완료" if replaced_outlier_count > 0 else "정상", delta_color="normal")
    with m4:
        st.metric(label="데이터 무결성(정상율)", value="100%" if df.isnull().sum().sum() == 0 else "미완료")

    if replaced_null_count > 0 or replaced_outlier_count > 0:
        st.markdown(f'<div class="success-box">🎉 <b>정제 완료:</b> 결측치 {replaced_null_count}개 및 이상치 {replaced_outlier_count}개가 정상 데이터 범주의 평균값(Mean)/최빈값으로 안전하게 조정되었습니다.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">ℹ️ 데이터가 아주 깨끗합니다! 탐지된 결측치나 극단적 이상치가 없어 원본 그대로 유지됩니다.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📄 데이터프레임 원본 vs 정제본 비교", "📊 결측치 분포 비교", "📈 수치 통계량 변화 비교"])
    
    with tab1:
        st.markdown("##### 💡 상위 5개 행 데이터 비교")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("🚨 [전처리 전] 원본 데이터 샘플")
            st.dataframe(df_raw.head(5), use_container_width=True)
        with c2:
            st.caption("✅ [전처리 후] 정제 완료 데이터 샘플")
            st.dataframe(df.head(5), use_container_width=True)

    with tab2:
        st.markdown("##### 💡 변수별 결측치(NaN) 잔존 개수 비교")
        null_compare_df = pd.DataFrame({
            "전처리 전 (Original)": df_raw.isnull().sum(),
            "전처리 후 (Cleaned)": df.isnull().sum()
        })
        st.dataframe(null_compare_df.T, use_container_width=True)

    with tab3:
        st.markdown("##### 💡 수치형 변수 기술 통계 변화")
        if len(numeric_cols) > 0:
            selected_num_col = st.selectbox("변화량을 확인할 수치형 변수를 선택하세요:", numeric_cols)
            stat_compare = pd.DataFrame({
                "전처리 전 (Original)": df_raw[selected_num_col].describe(),
                "전처리 후 (Cleaned)": df[selected_num_col].describe()
            })
            st.dataframe(stat_compare, use_container_width=True)
        else:
            st.info("수치형 변수가 존재하지 않습니다.")

    # ---------------------------------------------------------
    # 📈 데이터 시각화 제어판 설정
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🎛️ 2. 고도화된 시각화 제어 패널")
    
    columns = df.columns.tolist()
    available_colors = ["None"] + [col for col in ["depression_label", "gender", "platform_usage", "social_interaction_level"] if col in columns]
    
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        default_x = columns.index("daily_social_media_hours") if "daily_social_media_hours" in columns else 0
        x_axis = st.selectbox("📌 X축 변수 선택 (독립변수)", options=columns, index=default_x)
    with col2:
        default_y = columns.index("stress_level") if "stress_level" in columns else min(1, len(columns)-1)
        y_axis = st.selectbox("📌 Y축 변수 선택 (수치형 권장)", options=columns, index=default_y)
    with col3:
        color_target = st.selectbox("🎨 색상 구분 그룹 (Group Color)", options=available_colors, index=1 if len(available_colors) > 1 else 0)
    st.markdown('</div>', unsafe_allow_html=True)

    color_var = None if color_target == "None" else color_target
    
    chart_type = st.radio(
        "📊 시각화 할 차트 형태 선택", 
        ["산점도 (Scatter)", "바 차트 (Bar)", "박스 플롯 (Box)", "히스토그램 (Histogram)"],
        horizontal=True
    )
    
    st.markdown("#### 📈 분석 차트 시각화 결과")
    
    try:
        is_numeric_x = df[x_axis].dtype in ['int64', 'float64']
        is_numeric_y = df[y_axis].dtype in ['int64', 'float64']
        
        if chart_type == "산점도 (Scatter)":
            # statsmodels 패키지가 깔려있어야만 OLS 트렌드선이 안정적으로 작동합니다.
            use_trend = "ols" if (is_numeric_x and is_numeric_y and color_var is None) else None
            fig = px.scatter(
                df, x=x_axis, y=y_axis, color=color_var,
                title=f"✨ {x_axis}와 {y_axis}의 상호 연관성 분석 (정제 데이터)",
                trendline=use_trend,
                opacity=0.7,
                marginal_x="box" if is_numeric_x else None
            )
            
        elif chart_type == "바 차트 (Bar)":
            df_grouped = df.groupby([x_axis] + ([color_var] if color_var else [])).mean(numeric_only=True).reset_index()
            fig = px.bar(
                df_grouped, x=x_axis, y=y_axis, color=color_var,
                title=f"📊 {x_axis} 그룹별 {y_axis}의 평균값 분석 (정제 데이터)",
                barmode="group"
            )
            
        elif chart_type == "박스 플롯 (Box)":
            fig = px.box(
                df, x=x_axis, y=y_axis, color=color_var,
                title=f"📦 {x_axis} 범주에 따른 {y_axis} 수치 분포 (이상치 정제 완료 상태)",
                points="all" 
            )
            
        elif chart_type == "히스토그램 (Histogram)":
            fig = px.histogram(
                df, x=x_axis, color=color_var,
                title=f"📐 {x_axis} 변수의 빈도 및 누적 분포 분석",
                barmode="overlay",
                marginal="rug" 
            )
            
        fig.update_layout(
            title_font=dict(size=18, family="Malgun Gothic, sans-serif", color="#1E3A8A"),
            hovermode="closest",
            template="plotly_white",
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"⚠️ 선택하신 변수 조합은 해당 차트 유형으로 렌더링할 수 없습니다. 변수 형태를 재조정해주세요.")
        st.info(f"💡 디버깅 에러 메시지: {e}")

else:
    st.error(f"❌ '{os.path.basename(data_path)}' 데이터셋을 찾을 수 없습니다. GitHub 저장소 루트 경로에 파일이 올바르게 업로드되었는지 확인해 주세요.")
