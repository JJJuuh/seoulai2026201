import streamlit as st
import pandas as pd
import numpy as np
import os

# --- [앱 기본 설정 & 테마화] ---
st.set_page_config(
    page_title="CleanMetric Pro — Data Imputation Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- [고급 앱 인터페이스용 CSS] ---
st.markdown("""
    <style>
        /* 메인 프레임워크 배경 및 폰트 클렌징 */
        .stApp { background-color: #F8FAFC; }
        
        /* 상용 SaaS 스타일 상단 탑바 */
        .app-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 15px 25px;
            background: #FFFFFF;
            border-bottom: 1px solid #E2E8F0;
            border-radius: 12px;
            margin-bottom: 25px;
        }
        .brand-logo { font-size: 20px; font-weight: 800; color: #0F172A; letter-spacing: -0.5px; }
        .brand-logo span { color: #2563EB; }
        .app-status-badge { background: #E0F2FE; color: #0369A1; padding: 4px 12px; border-radius: 9999px; font-size: 12px; font-weight: 600; }
        
        /* 메인 데이터 대시보드 카드 */
        .metric-card-container {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
            margin-bottom: 20px;
        }
        .section-title { font-size: 16px; font-weight: 700; color: #1E293B; margin-bottom: 15px; display: flex; align-items: center; gap: 8px; }
        
        /* 데이터프레임 헤더 라벨 */
        .table-label-raw { background: #FEF2F2; color: #991B1B; padding: 6px 12px; border-radius: 6px; font-size: 13px; font-weight: 600; margin-bottom: 10px; display: inline-block; }
        .table-label-clean { background: #ECFDF5; color: #065F46; padding: 6px 12px; border-radius: 6px; font-size: 13px; font-weight: 600; margin-bottom: 10px; display: inline-block; }
    </style>
""", unsafe_allow_html=True)

# --- [App 상단 커스텀 탑바 인프라] ---
st.markdown("""
    <div class="app-topbar">
        <div class="brand-logo">🛡️ CleanMetric <span>Pro</span></div>
        <div class="app-status-badge">Engine v2.4 Active</div>
    </div>
""", unsafe_allow_html=True)

# --- [사이드바: 프리미엄 환경설정 패널] ---
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    st.write("품질 진단 및 파이프라인 가동 규칙")
    st.markdown("---")
    
    # 실제 프로덕션 앱처럼 옵션을 주는 구성
    imputation_strategy = st.radio("수치형 결측치 대체 방법", ["Mean (평균값)", "Median (중앙값)"], index=0)
    outlier_threshold = st.slider("IQR 가중치 임계값", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
    
    st.markdown("---")
    st.markdown("💡 **파이프라인 요약**")
    st.caption("본 애플리케이션은 원본 데이터 원격 검싱 후 결측치 및 통계적 극단치를 탐지하여 격리/보정하는 자동화 대시보드입니다.")

# --- [데이터 로드 및 무결성 예외 처리] ---
data_path = "Teen_Mental_Health_Dataset.csv"

if os.path.exists(data_path):
    df_raw = pd.read_csv(data_path)
else:
    # 깃허브 등 파일 유실 환경을 대비한 앱 구동 전용 엔지니어링 데이터셋
    np.random.seed(42)
    n_samples = 200
    mock_data = {
        "daily_social_media_hours": np.random.uniform(1.0, 7.5, n_samples),
        "stress_level": np.random.randint(1, 10, n_samples).astype(float),
        "social_interaction_level": np.random.randint(1, 5, n_samples).astype(float),
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "depression_label": np.random.choice(["Mild", "Moderate", "Severe"], n_samples)
    }
    df_raw = pd.DataFrame(mock_data)
    # 인위적 에러 주입
    df_raw.loc[np.random.choice(n_samples, 15), "daily_social_media_hours"] = np.nan
    df_raw.loc[np.random.choice(n_samples, 10), "stress_level"] = np.nan
    df_raw.loc[np.random.choice(n_samples, 5), "stress_level"] = 99.0 
    df_raw.loc[np.random.choice(n_samples, 5), "daily_social_media_hours"] = -8.0
    st.sidebar.caption("⚡ *Notice: 임시 검싱 데모 스트림 운용 중*")

df = df_raw.copy()

# ---------------------------------------------------------
# 🛠️ [정제 파이프라인 연산]
# ---------------------------------------------------------
# 1. 결측치 정제
raw_null_total = df_raw.isnull().sum().sum()
replaced_null_count = 0

for col in df.columns:
    null_cnt = df[col].isnull().sum()
    if null_cnt > 0:
        if df[col].dtype in ['int64', 'float64']:
            fill_val = df[col].mean() if "Mean" in imputation_strategy else df[col].median()
            df[col] = df[col].fillna(fill_val)
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
        replaced_null_count += null_cnt

# 2. IQR 이상치 정제
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
replaced_outlier_count = 0

for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - outlier_threshold * iqr
    upper_bound = q3 + outlier_threshold * iqr
    
    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
    outlier_cnt = outlier_mask.sum()
    
    if outlier_cnt > 0:
        df.loc[outlier_mask, col] = df.loc[~outlier_mask, col].mean()
        replaced_outlier_count += outlier_cnt


# ---------------------------------------------------------
# 📊 [인터페이스 레이어 1: 메인 KPI 스코어바]
# ---------------------------------------------------------
st.markdown('<div class="section-title">📊 데이터 품질 매트릭 모니터링</div>', unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(label="총 분석 레코드 크기", value=f"{df.shape[0]} Rows", delta=f"{df.shape[1]} Columns")
with kpi2:
    st.metric(label="탐지된 총 결측치 (NaN)", value=f"{raw_null_total} 건", delta="보정 완결", delta_color="normal")
with kpi3:
    st.metric(label="통계적 이상치 (Outlier)", value=f"{replaced_outlier_count} 건", delta="평균 대체 완료", delta_color="inverse")
with kpi4:
    st.metric(label="데이터 파이프라인 헬스", value="100 %" if df.isnull().sum().sum() == 0 else "오류 보정 필요")


# ---------------------------------------------------------
# 🔄 [인터페이스 레이어 2: 데이터 무결성 검증 및 전/후 전면 비교]
# ---------------------------------------------------------
st.markdown("---")
st.markdown('<div class="section-title">🔄 전처리 전/후 데이터 다차원 대조 인터페이스</div>', unsafe_allow_html=True)

# 복잡한 탭 레이아웃을 하나의 유기적 화면으로 제어하기 위해 탭 UI 생성
app_tab1, app_tab2, app_tab3 = st.tabs([
    "📂 Raw vs Cleaned 원장 데이터 스크리닝", 
    "📊 변수별 데이터 소실률(결측치) 집계", 
    "📈 정제 변수 통계 변동 추이 요약"
])

# [탭 1: 로 데이터프레임 전/후 대조]
with app_tab1:
    col_raw, col_clean = st.columns(2)
    with col_raw:
        st.markdown('<span class="table-label-raw">🚨 [INCOMING] 오염 원본 데이터 스냅샷</span>', unsafe_allow_html=True)
        st.dataframe(df_raw.head(12), use_container_width=True, height=450)
    with col_clean:
        st.markdown('<span class="table-label-clean">✅ [PROCESSED] 정제 완결 데이터 원장</span>', unsafe_allow_html=True)
        st.dataframe(df.head(12), use_container_width=True, height=450)

# [탭 2: 결측치 집계 리포팅]
with app_tab2:
    st.markdown("##### 📝 변수별 결측치 제거 결과 매트릭스")
    null_report = pd.DataFrame({
        "정제 전 결측치 (Original)": df_raw.isnull().sum(),
        "정제 후 결측치 (Cleaned)": df.isnull().sum(),
        "클렌징 수량": df_raw.isnull().sum() - df.isnull().sum()
    })
    st.dataframe(null_report, use_container_width=True)

# [탭 3: 수치 통계량 보정 추이 확인]
with app_tab3:
    st.markdown("##### 📝 극단치 제거에 따른 주요 통계 메트릭 변동표")
    if len(numeric_cols) > 0:
        target_view_col = st.selectbox("추이를 모니터링할 변수(컬럼)를 선택하세요:", numeric_cols)
        
        comparison_summary = pd.DataFrame({
            "정제 전 통계량 (Original)": df_raw[target_view_col].describe(),
            "정제 후 통계량 (Cleaned)": df[target_view_col].describe()
        })
        # 소수점 둘째자리까지 정돈하여 시인성 최적화
        st.dataframe(comparison_summary.style.format("{:.2f}"), use_container_width=True)
    else:
        st.info("데이터셋 내에 분석 가능한 수치형 인프라가 존재하지 않습니다.")
