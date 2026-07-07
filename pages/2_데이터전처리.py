import streamlit as st
import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 하이엔드 어플리케이션 환경 정의
# ==========================================
st.set_page_config(
    page_title="DataClean Studio Ultra",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 글래스모피즘 및 다크 럭셔리 스킨 커스텀 마이징
st.markdown("""
    <style>
        .stApp { background-color: #0F172A; color: #F8FAFC; }
        
        /* 초격차 엔터프라이즈 탑바 */
        .premium-topbar {
            background: linear-gradient(135deg, #1E1B4B 0%, #2E1065 100%);
            padding: 28px 35px;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            margin-bottom: 30px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }
        .topbar-title { font-size: 30px; font-weight: 900; letter-spacing: -1px; background: linear-gradient(to right, #38BDF8, #A78BFA); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .topbar-subtitle { font-size: 13px; color: #94A3B8; margin-top: 6px; font-weight: 400; }
        
        /* 상용 데이터 뷰어 카드 구조 */
        .app-window-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(16px);
            border-radius: 16px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 30px;
        }
        .window-badge { display: inline-block; background: rgba(167, 139, 250, 0.15); color: #A78BFA; font-size: 11px; font-weight: 700; padding: 4px 12px; border-radius: 9999px; margin-bottom: 12px; }
        .window-title { font-size: 18px; font-weight: 700; color: #F8FAFC; margin-bottom: 15px; }
        
        /* 인터페이스 가이드 탭 태그 */
        .tag-raw { background: rgba(239, 68, 68, 0.12); color: #F87171; font-size: 12px; font-weight: 700; padding: 6px 14px; border-radius: 8px; display: inline-block; margin-bottom: 12px; border: 1px solid rgba(239, 68, 68, 0.2); }
        .tag-clean { background: rgba(16, 185, 129, 0.12); color: #34D399; font-size: 12px; font-weight: 700; padding: 6px 14px; border-radius: 8px; display: inline-block; margin-bottom: 12px; border: 1px solid rgba(16, 185, 129, 0.2); }
    </style>
""", unsafe_allow_html=True)

# 메인 탑바 렌더링
st.markdown("""
    <div class="premium-topbar">
        <div class="topbar-title">💎 DataClean Studio Ultra</div>
        <div class="topbar-subtitle">상용 엔터프라이즈급 데이터 클렌징 콤포넌트 및 클라우드 동기화 모듈 인터페이스</div>
    </div>
""", unsafe_allow_html=True)


# ==========================================
# 2. 고성능 인프라 데이터 제어 패널 (사이드바)
# ==========================================
with st.sidebar:
    st.markdown("### 🎛️ ENGINE CONTROLS")
    st.caption("실시간 파이프라인 매개변수 제어")
    st.markdown("---")
    
    selected_strategy = st.radio("⚡ 수치형 임퓨테이션 전략", ["Mean (전체 평균)", "Median (중앙값)"], index=0)
    iqr_multiplier = st.slider("📐 이상치 경계 임계 지수 (IQR)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
    
    st.markdown("---")
    st.markdown("📥 **EXPORTS MANAGEMENT**")
    st.caption("정제 완료 데이터 원장 로컬 추출 엔진 가동 가능")


# ==========================================
# 3. 데이터 로딩 허브 (서버리스 인프라 예외 처리)
# ==========================================
data_file_path = "Teen_Mental_Health_Dataset.csv"

if os.path.exists(data_file_path):
    df_raw = pd.read_csv(data_file_path)
else:
    # 깃허브 가동용 시뮬레이션 고밀도 데이터셋 빌드 (이탈 방지)
    np.random.seed(777)
    records = 300
    mock_schema = {
        "daily_social_media_hours": np.random.uniform(0.5, 9.0, records),
        "stress_level": np.random.randint(1, 11, records).astype(float),
        "social_interaction_level": np.random.randint(1, 6, records).astype(float),
        "gender": np.random.choice(["Male", "Female"], records),
        "depression_label": np.random.choice(["Mild", "Moderate", "Severe"], records)
    }
    df_raw = pd.DataFrame(mock_schema)
    
    # 정제 시연을 위한 인위적 오염원 주입
    df_raw.loc[np.random.choice(records, 25), "daily_social_media_hours"] = np.nan
    df_raw.loc[np.random.choice(records, 15), "stress_level"] = np.nan
    df_raw.loc[np.random.choice(records, 7), "stress_level"] = 999.0  
    df_raw.loc[np.random.choice(records, 7), "daily_social_media_hours"] = -99.0 
    st.sidebar.caption("⚡ *Sandbox: 가상 데이터 그리드 인젝션 완결*")

df = df_raw.copy()


# ==========================================
# ⚙️ 4. GOAT급 초고속 정제 파이프라인 엔진 (No-Loop Vectorization)
# ==========================================
total_null_detected = df_raw.isnull().sum().sum()
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(exclude=['int64', 'float64']).columns

# 벡터라이징 가속 연산 패치
if len(numeric_features) > 0:
    if "Mean" in selected_strategy:
        df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())
    else:
        df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

if len(categorical_features) > 0:
    for cat_col in categorical_features:
        df[cat_col] = df[cat_col].fillna(df[cat_col].mode()[0] if not df[cat_col].mode().empty else "Unknown")

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


# ==========================================
# 📱 5. 상용급 인터랙티브 어플리케이션 레이아웃
# ==========================================

# 모듈 1: 대시보드 스코어 정보 보드
with st.container():
    st.markdown('<div class="app-window-card">', unsafe_allow_html=True)
    st.markdown('<span class="window-badge">SYSTEM METRICS</span>', unsafe_allow_html=True)
    st.markdown('<div class="window-title">📊 실시간 파이프라인 정제 메트릭</div>', unsafe_allow_html=True)
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    with stat_col1:
        st.metric(label="🛡️ 스캔된 로 데이터", value=f"{df.shape[0]} Rows", delta=f"{df.shape[1]} Cols")
    with stat_col2:
        st.metric(label="🧼 보정된 결측치 수", value=f"{total_null_detected} EA", delta="Imputation 완결")
    with stat_col3:
        st.metric(label="🚨 격리된 이상치 수", value=f"{total_outliers_detected} EA", delta="정상 범위 수렴", delta_color="inverse")
    with stat_col4:
        st.metric(label="🔋 데이터 파이프라인 상태", value="100.00 %" if df.isnull().sum().sum() == 0 else "프로세싱 지연")
    st.markdown('</div>', unsafe_allow_html=True)


# 모듈 2: [신기술 적용] 인터랙티브 스플릿 스크린 (사용자가 직접 정제본 수정 가능)
with st.container():
    st.markdown('<div class="app-window-card">', unsafe_allow_html=True)
    st.markdown('<span class="window-badge">INTERACTIVE DATA GRID</span>', unsafe_allow_html=True)
    st.markdown('<div class="window-title">🔄 전처리 전(Raw) vs 후(Cleaned) 데이터 에디터 스크리닝</div>', unsafe_allow_html=True)
    st.caption("오른쪽 정제 완결 데이터는 실제 상용 그리드 앱처럼 셀을 더블 클릭해 값을 직접 커스텀 수정할 수 있습니다.")
    
    layout_left, layout_right = st.columns(2)
    with layout_left:
        st.markdown('<span class="tag-raw">🚨 [BEFORE] 인입 오염 데이터 원본 (수정 불가)</span>', unsafe_allow_html=True)
        st.dataframe(df_raw.head(12), use_container_width=True, height=400)
    with layout_right:
        st.markdown('<span class="tag-clean">✅ [AFTER] 엔진 정제 완료 데이터 (실시간 수정 가능)</span>', unsafe_allow_html=True)
        # 상용 앱 핵심 컴포넌트 데이터 에디터 바인딩
        edited_df = st.data_editor(df.head(12), use_container_width=True, height=400, key="goat_editor")
    st.markdown('</div>', unsafe_allow_html=True)


# 모듈 3: 실시간 데이터 내보내기 및 정밀 심사 리포트
with st.container():
    st.markdown('<div class="app-window-card">', unsafe_allow_html=True)
    st.markdown('<span class="window-badge">DATA PIPELINE OUTPUT</span>', unsafe_allow_html=True)
    st.markdown('<div class="window-title">💾 정제 완료 데이터 다운로드 및 품질 검증 보고서</div>', unsafe_allow_html=True)
    
    # 실제 상용 앱처럼 가공 원장을 파일로 즉시 다운로드받는 컨트롤러
    csv_buffer = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 정제 완료된 전체 데이터셋 (CSV) 로컬로 즉시 내보내기",
        data=csv_buffer,
        file_name="Cleaned_Teen_Mental_Health_Dataset.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    
    report_ui_tab1, report_ui_tab2 = st.tabs([
        "📋 데이터 변수별 소실 통계 검증 일람", 
        "📊 이상치 보정에 따른 기초 통계 변동 추이 요약"
    ])
    
    with report_ui_tab1:
        null_audit_df = pd.DataFrame({
            "정제 전 결측치 총합 (Original)": df_raw.isnull().sum(),
            "정제 후 결측치 총합 (Cleaned)": df.isnull().sum(),
            "자동 보정 복구 수량": df_raw.isnull().sum() - df.isnull().sum()
        })
        st.dataframe(null_audit_df, use_container_width=True)
        
    with report_ui_tab2:
        if len(numeric_features) > 0:
            target_metric_col = st.selectbox("분석 타겟 변수 선택 실시간 동기화:", numeric_features)
            summary_audit_df = pd.DataFrame({
                "정제 전 기초통계량 (Original)": df_raw[target_metric_col].describe(),
                "정제 후 기초통계량 (Cleaned)": df[target_metric_col].describe()
            })
            st.dataframe(summary_audit_df.style.format("{:.3f}"), use_container_width=True)
            
    st.markdown('</div>', unsafe_allow_html=True)
