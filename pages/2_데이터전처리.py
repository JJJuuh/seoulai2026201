import streamlit as st
import pandas as pd
import numpy as np
import os

# --- 1. 앱 페이지 설정 ---
st.set_page_config(
    page_title="데이터 전처리 시스템",
    page_icon="🧼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 직관적인 화이트 모드 레이아웃 스타일 설정 (CSS) ---
st.markdown("""
    <style>
        .stApp { background-color: #FAFAFA; color: #111827; }
        
        .app-top-bar {
            background-color: #FFFFFF;
            padding: 20px 24px;
            border-bottom: 1px solid #E5E7EB;
            margin-bottom: 25px;
        }
        .app-title { font-size: 24px; font-weight: 700; color: #111827; }
        .app-title span { color: #2563EB; }
        .app-desc { font-size: 13px; color: #6B7280; margin-top: 4px; }
        
        .app-section-card {
            background-color: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
            margin-bottom: 25px;
        }
        .section-heading { font-size: 18px; font-weight: 700; color: #111827; margin-bottom: 15px; }
        
        .lbl-before { background-color: #FEF2F2; color: #DC2626; font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 4px; display: inline-block; margin-bottom: 8px; }
        .lbl-after { background-color: #ECFDF5; color: #059669; font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 4px; display: inline-block; margin-bottom: 8px; }
    </style>
""", unsafe_allow_html=True)

# 상단 헤더 출력
st.markdown("""
    <div class="app-top-bar">
        <div class="app-title">🧼 데이터 정제 및 전/후 비교 시스템</div>
        <div class="app-desc">결측치 및 이상치를 실시간으로 탐색하고 정제를 수행하여, 전처리 전/후의 DataFrame을 대조 검증합니다.</div>
    </div>
""", unsafe_allow_html=True)


# --- 3. 사이드바 제어 패널 ---
with st.sidebar:
    st.markdown("<p style='font-size: 16px; font-weight: 700; color: #111827;'>🎛️ 정제 옵션 설정</p>", unsafe_allow_html=True)
    st.markdown("---")
    selected_strategy = st.radio("⚡ 수치형 결측치 대체 기준", ["전체 평균값 (Mean)", "전체 중앙값 (Median)"])
    iqr_multiplier = st.slider("📐 이상치 탐색 임계 지수 (IQR)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)


# --- 4. 데이터 로드 및 데모 데이터 생성 (예외 처리) ---
data_file_path = "Teen_Mental_Health_Dataset.csv"

if os.path.exists(data_file_path):
    df_raw = pd.read_csv(data_file_path)
else:
    # 파일 유실 시 정상 구동을 보장하기 위한 데모 데이터셋 자동 생성
    np.random.seed(42)
    records = 200
    mock_data = {
        "daily_social_media_hours": np.random.uniform(0.5, 8.0, records),
        "stress_level": np.random.randint(1, 11, records).astype(float),
        "social_interaction_level": np.random.randint(1, 6, records).astype(float),
        "gender": np.random.choice(["Male", "Female"], records),
        "depression_label": np.random.choice(["Mild", "Moderate", "Severe"], records)
    }
    df_raw = pd.DataFrame(mock_data)
    
    # 임의로 결측치(NaN) 및 이상치(Outlier) 주입
    df_raw.loc[np.random.choice(records, 20), "daily_social_media_hours"] = np.nan
    df_raw.loc[np.random.choice(records, 12), "stress_level"] = np.nan
    df_raw.loc[np.random.choice(records, 6), "stress_level"] = 999.0
    df_raw.loc[np.random.choice(records, 6), "daily_social_media_hours"] = -50.0
    st.sidebar.caption("💡 *Notice: 데모 데이터로 가동 중입니다.*")

df = df_raw.copy()


# --- 5. 백엔드 연산: 결측치 및 이상치 탐색/정제 파이프라인 ---
# [1] 결측치 탐색 및 정제
total_null_detected = df_raw.isnull().sum().sum()
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(exclude=['int64', 'float64']).columns

if len(numeric_features) > 0:
    if "평균값" in selected_strategy:
        df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())
    else:
        df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

if len(categorical_features) > 0:
    for cat_col in categorical_features:
        df[cat_col] = df[cat_col].fillna(df[cat_col].mode()[0] if not df[cat_col].mode().empty else "Unknown")

# [2] IQR 기반 이상치 탐색 및 정제
total_outliers_detected = 0
for num_col in numeric_features:
    q1 = df[num_col].quantile(0.25)
    q3 = df[num_col].quantile(0.75)
    iqr = q3 - q1
    
    lower_limit = q1 - (iqr_multiplier * iqr)
    upper_limit = q3 + (iqr_multiplier * iqr)
    
    outlier_condition = (df[num_col] < lower_limit) | (df[num_col] > upper_limit)
    outlier_count = outlier_condition.sum()
    
    if outlier_count > 0:
        pure_mean = df.loc[~outlier_condition, num_col].mean()
        df.loc[outlier_condition, num_col] = pure_mean
        total_outliers_detected += outlier_count


# --- 6. 프론트엔드 UI 레이어 배치 ---

# [모듈 1] 결측치 및 이상치 탐색 보고 (지표 요약)
with st.container():
    st.markdown('<div class="app-section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">🎯 1. 결측치 및 이상치 탐색 결과</div>', unsafe_allow_html=True)
    
    stat_1, stat_2, stat_3, stat_4 = st.columns(4)
    with stat_1:
        st.metric(label="📊 총 데이터 구조", value=f"{df.shape[0]} Rows", delta=f"{df.shape[1]} Columns")
    with stat_2:
        st.metric(label="🧼 탐색 및 정제된 결측치 (NaN)", value=f"{total_null_detected} 건")
    with stat_3:
        st.metric(label="🚨 탐색 및 정제된 이상치 (IQR)", value=f"{total_outliers_detected} 건", delta_color="inverse")
    with stat_4:
        st.metric(label="🔋 최종 정제 상태", value="100% 완료" if df.isnull().sum().sum() == 0 else "진행 중")
    st.markdown('</div>', unsafe_allow_html=True)


# [모듈 2] 전처리 전 / 후 DataFrame 1:1 비교
with st.container():
    st.markdown('<div class="app-section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">🔄 2. 전처리 전 (Original) vs 전처리 후 (Cleaned) DataFrame 비교</div>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<span class="lbl-before">🚨 [전처리 전] 오염 데이터프레임 원본</span>', unsafe_allow_html=True)
        st.dataframe(df_raw.head(15), use_container_width=True, height=450)
    with col_right:
        st.markdown('<span class="lbl-after">✅ [전처리 후] 정제 완료 데이터프레임 원본</span>', unsafe_allow_html=True)
        st.dataframe(df.head(15), use_container_width=True, height=450)
    st.markdown('</div>', unsafe_allow_html=True)


# [모듈 3] 데이터 정제 수행 상세 보고서 (탭 구조)
with st.container():
    st.markdown('<div class="app-section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">📑 3. 데이터 정제 수행 세부 통계 리포트</div>', unsafe_allow_html=True)
    
    report_tab1, report_tab2 = st.tabs([
        "📋 변수별 결측치 제거 상태 확인",
        "📊 이상치 정제에 따른 기술 통계량 변동 대조"
    ])
    
    with report_tab1:
        # 가독성 표기 버그 수정 완료 파트 (전치 코드 .T 제거 및 딕셔너리 구조 개편)
        null_audit_table = pd.DataFrame({
            "정제 전 결측치 수 (Original)": df_raw.isnull().sum(),
            "정제 후 결측치 수 (Cleaned)": df.isnull().sum(),
            "정제 수행 보정 수량": df_raw.isnull().sum() - df.isnull().sum()
        })
        st.dataframe(null_audit_table, use_container_width=True)
        
    with report_tab2:
        if len(numeric_features) > 0:
            target_metric_col = st.selectbox("분석 대상 변수 선택:", numeric_features)
            
            summary_audit_df = pd.DataFrame({
                "정제 전 기초통계량 (Original)": df_raw[target_metric_col].describe(),
                "정제 후 기초통계량 (Cleaned)": df[target_metric_col].describe()
            })
            st.dataframe(summary_audit_df.style.format("{:.3f}"), use_container_width=True)
        else:
            st.info("비교 가능한 수치형 변수가 없습니다.")
            
    st.markdown('</div>', unsafe_allow_html=True)
