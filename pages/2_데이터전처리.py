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

# --- 커스텀 스타일 (CSS 기법 적용 - 오류 원인 수정 완료) ---
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
st.markdown('<div class="main-title">📊 3. 데이터 시각화 & 전처리 (EDA)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">결측치와 이상치를 평균값으로 정제하고, 고도화된 Plotly 시각화를 통해 핵심 패턴을 탐색합니다.</div>', unsafe_allow_html=True)

data_path = "Teen_Mental_Health_Dataset.csv"

if os.path.exists(data_path):
    # 최초 데이터 로드 (원본 보존)
    df_raw = pd.read_csv(data_path)
    df = df_raw.copy()
    
    # ---------------------------------------------------------
    # 🛠️ 데이터 전처리 파트 (결측치 & 이상치 평균값 대체)
    # ---------------------------------------------------------
    st.markdown("### 🛠️ 1. 고급 데이터 정제 (결측치 & 이상치 처리)")
    
    with st.expander("🔍 결측치 및 이상치 처리 현황 열기", expanded=True):
        col_info1, col_info2 = st.columns([1, 2])
        
        with col_info1:
            st.markdown("##### 📊 원본 데이터 크기")
            st.metric(label="총 샘플(행) 수", value=f"{df.shape[0]}개")
            st.metric(label="총 변수(열) 수", value=f"{df.shape[1]}개")
            
        with col_info2:
            st.markdown("##### 🧼 평균값 기반 자동 정제 진행 상황")
            
            # 1. 결측치(NaN)를 평균값으로 대체
            replaced_null_count = 0
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    null_cnt = df[col].isnull().sum()
                    if null_cnt > 0:
                        mean_val = df[col].mean()
                        df[col] = df[col].fillna(mean_val)
                        replaced_null_count += null_cnt
                else:
                    null_cnt = df[col].isnull().sum()
                    if null_cnt > 0:
                        mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                        df[col] = df[col].fillna(mode_val)
            
            if replaced_null_count > 0:
                st.markdown(f'<div class="success-box">✅ 수치형 변수의 결측치 <b>{replaced_null_count}개</b>를 각 변수의 <b>평균값(Mean)</b>으로 대체했습니다.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">ℹ️ 데이터셋에 결측치(NaN)가 존재하지 않습니다.</div>', unsafe_allow_html=True)
            
            # 2. IQR 방식을 활용한 이상치(Outlier) 탐지 및 평균값 대체
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
                    normal_mean = df_raw.loc[~outlier_mask, col].mean()
                    df.loc[outlier_mask, col] = normal_mean
                    replaced_outlier_count += outlier_cnt
                    
            if replaced_outlier_count > 0:
                st.markdown(f'<div class="success-box">✅ IQR 기준 범위를 벗어난 이상치(Outlier) 데이터 <b>{replaced_outlier_count}개</b>를 탐지하여 정상범위 <b>평균값</b>으로 대체 완료했습니다.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">ℹ️ 탐지된 뚜렷한 극단적 이상치(Outlier)가 없습니다. 모두 정상 범위 내에 있습니다.</div>', unsafe_allow_html=True)
                
        st.markdown("##### 📄 전처리가 완료된 데이터셋 미리보기 (상위 5개 행)")
        st.dataframe(df.head(5), use_container_width=True)

    # ---------------------------------------------------------
    # 📈 데이터 시각화 제어판 설정
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🎛️ 2. 시각화 제어 패널 설정")
    
    columns = df.columns.tolist()
    available_colors = ["None"] + [col for col in ["depression_label", "gender", "platform_usage", "social_interaction_level"] if col in columns]
    
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        default_x = columns.index("daily_social_media_hours") if "daily_social_media_hours" in columns else 0
        x_axis = st.selectbox("📌 X축 변수 선택 (독립변수)", options=columns, index=default_x)
    with col2:
        default_y = columns.index("stress_level") if "stress_level" in columns else min(1, len(columns)-1)
        y_axis = st.selectbox("📌 Y축 변수 Choice (수치형 권장)", options=columns, index=default_y)
    with col3:
        color_target = st.selectbox("🎨 색상 구분 그룹 (Group Color)", options=available_colors, index=1 if len(available_colors) > 1 else 0)
    st.markdown('</div>', unsafe_allow_html=True)

    color_var = None if color_target == "None" else color_target
    
    chart_type = st.radio(
        "📊 시각화 할 차트 형태 선택", 
        ["산점도 (Scatter)", "바 차트 (Bar)", "박스 플롯 (Box)", "히스토그램 (Histogram)"],
        horizontal=True
    )
    
    # --- 차트 렌더링 영역 ---
    st.markdown("#### 📈 분석 차트 시각화 결과")
    
    try:
        is_numeric_x = df[x_axis].dtype in ['int64', 'float64']
        is_numeric_y = df[y_axis].dtype in ['int64', 'float64']
        
        if chart_type == "산점도 (Scatter)":
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
    
    # --- 정보 가이드 ---
    with st.expander("💡 데이터 시각화 및 인사이트 도출 가이드"):
        st.markdown(f"""
        * 현재 정제 데이터셋 크기: **{df.shape[0]}행**
        * 이상치와 결측치가 원본 데이터의 평균값으로 부드럽게 조정되어 누락 없이 안정적인 통계 분석이 가능합니다.
        """)

else:
    st.error("❌ 'Teen_Mental_Health_Dataset.csv' 데이터셋을 찾을 수 없습니다. 경로를 다시 한 번 확인해 주세요.")
