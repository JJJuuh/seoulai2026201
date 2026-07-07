import streamlit as st
import pandas as pd
import numpy as np
import os

# --- [1. 앱 시스템 및 글로벌 테마 정의] ---
st.set_page_config(
    page_title="DataClean Studio Pro",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- [2. 앱 스킨 완벽 커스터마이징 (고급 CSS)] ---
st.markdown("""
    <style>
        /* 글로벌 배경 및 앱 컨테이너 핏 조정 */
        .stApp { background-color: #F1F5F9; }
        
        /* 모던 모바일/웹앱 감성의 헤더 바 */
        .app-header-container {
            background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
            padding: 24px;
            border-radius: 16px;
            color: #FFFFFF;
            margin-bottom: 25px;
            box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.15);
        }
        .app-main-title { font-size: 26px; font-weight: 800; letter-spacing: -0.5px; }
        .app-main-title span { color: #38BDF8; }
        .app-sub-title { font-size: 13px; color: #94A3B8; margin-top: 6px; }
        
        /* 모듈형 카드 디자인 */
        .app-card {
            background-color: #FFFFFF;
            border-radius: 14px;
            padding: 20px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
            margin-bottom: 20px;
        }
        .card-header-badge {
            display: inline-block;
            background: #EFF6FF;
            color: #1D4ED8;
            font-size: 11px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 6px;
            margin-bottom: 12px;
            text-transform: uppercase;
        }
        .card-heading { font-size: 18px; font-weight: 700; color: #0F172A; margin-bottom: 15px; }
        
        /* 테이블 상단 상태 표식 */
        .status-tag-red { background: #FEE2E2; color: #991B1B; font-size: 12px; font-weight: 700; padding: 6px 12px; border-radius: 6px; display: inline-block; margin-bottom: 10px; }
        .status-tag-green { background: #D1FAE5; color: #065F46; font-size: 12px; font-weight: 700; padding: 6px 12px; border-radius: 6px; display: inline-block; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- [3. 상단 앱 헤더 영역] ---
st.markdown("""
    <div class="app-header-container">
        <div class="app-main-title">💎 DataClean <span>Studio Pro</span></div>
        <div class="app-sub-title">데이터셋 결측치 보정 및 IQR 이상치 제거 프로세스를 관리하는 전문 데이터 클렌징 어플리케이션입니다.</div>
    </div>
""", unsafe_allow_html=True)

# --- [4. 가상 데이터 안전 제어 센서 (파일 없어도 무조건 구동)] ---
data_path = "Teen_Mental_Health_Dataset.csv"

if os.path.exists(data_path):
    df_raw = pd.read_csv(data_path)
else:
    # 깃허브 등 원격 서버 배포 시 터짐 방지용 정밀 모크 데이터
    np.random.seed(42)
    n_samples = 250
    mock_data = {
        "daily_social_media_hours": np.random.uniform(1.0, 7.0, n_samples),
        "stress_level": np.random.randint(1, 10, n_samples).astype(float),
        "social_interaction_level": np.random.randint(1, 5, n_samples).astype(float),
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "depression_label": np.random.choice(["Mild", "Moderate", "Severe"], n_samples)
    }
    df_raw = pd.DataFrame(mock_data)
    # 인위적인 결측치 & 극단적 이상치 주입 (탐색 기능 시연용)
    df_raw.loc[np.random.choice(n_samples, 18), "daily_social_media_hours"] = np.nan
    df_raw.loc[np.random.choice(n_samples, 12), "stress_level"] = np.nan
    df_raw.loc[np.random.choice(n_samples, 6), "stress_level"] = 150.0  # 아웃라이어 상한선 돌파
    df_raw.loc[np.random.choice(n_samples, 6), "daily_social_media_hours"] = -25.0  # 아웃라이어 하한선 돌파
    st.sidebar.caption("⚡ *Live: 엔지니어링 데모 샌드박스 가동 중*")

df = df_raw.copy()

# --- [5. 앱 우측 사이드바: 고성능 제어 패널] ---
with st.sidebar:
    st.markdown("### 🎛️ ENGINE 제어 센터")
    st.write("실시간 정제 알고리즘 스위치")
    st.markdown("---")
    
    num_strategy = st.selectbox("수치형 데이터 정제 기법", ["컬럼 평균값 (Mean)", "컬럼 중앙값 (Median)"])
    iqr_weight = st.slider("이상치 판단 가중치 (IQR 지수)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
    
    st.markdown("---")
    st.markdown("🔒 **보안 및 규정**")
    st.caption("모든 데이터 연산은 브라우저 세션 내에서 독립적으로 처리되며 외부로 유출되지 않습니다.")

# ---------------------------------------------------------
# ⚙️ [백엔드 처리 아키텍처: 데이터 정제 가동]
# ---------------------------------------------------------
# 1. 결측치 연산
raw_null_total = df_raw.isnull().sum().sum()
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['int64', 'float64']:
            fill_val = df[col].mean() if "평균값" in num_strategy else df[col].median()
            df[col] = df[col].fillna(fill_val)
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

# 2. 이상치 연산
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
replaced_outlier_count = 0

for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - iqr_weight * iqr
    upper_bound = q3 + iqr_weight * iqr
    
    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
    outlier_cnt = outlier_mask.sum()
    
    if outlier_cnt > 0:
        df.loc[outlier_mask, col] = df.loc[~outlier_mask, col].mean()
        replaced_outlier_count += outlier_cnt


# ---------------------------------------------------------
# 📱 [프론트엔드 레이어: 메인 앱 위젯 인터페이스]
# ---------------------------------------------------------

# 모듈 1: 실시간 대시보드 스코어 카드
with st.container():
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<span class="card-header-badge">Status Metric</span>', unsafe_allow_html=True)
    st.markdown('<div class="card-heading">🎯 실시간 품질 진단 스코어보드</div>', unsafe_allow_html=True)
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(label="📊 스캔된 데이터 구조", value=f"{df.shape[0]} Rows", delta=f"{df.shape[1]} Columns")
    with kpi2:
        st.metric(label="🧼 복구된 결측치 수", value=f"{raw_null_total} 개", delta="보정 완결", delta_color="normal")
    with kpi3:
        st.metric(label="🚨 조정된 이상치 수", value=f"{replaced_outlier_count} 개", delta="평균 대체 완료", delta_color="inverse")
    with kpi4:
        st.metric(label="🛡️ 데이터 파이프라인 무결성", value="100.0 %" if df.isnull().sum().sum() == 0 else "프로세싱 오류")
    st.markdown('</div>', unsafe_allow_html=True)


# 모듈 2: 전/후 원장 데이터 전면 비교 격자 (Split-Screen View)
with st.container():
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<span class="card-header-badge">Data Split View</span>', unsafe_allow_html=True)
    st.markdown('<div class="card-heading">🔄 원본 로우 데이터 vs 정제 가공 데이터 실시간 대조</div>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<span class="status-tag-red">🚨 [BEFORE] 인입 원본 데이터 원장 (수정 전)</span>', unsafe_allow_html=True)
        st.dataframe(df_raw.head(11), use_container_width=True, height=380)
    with col_right:
        st.markdown('<span class="status-tag-green">✅ [AFTER] 엔진 정제 완료 데이터 원장 (수정 후)</span>', unsafe_allow_html=True)
        st.dataframe(df.head(11), use_container_width=True, height=380)
    st.markdown('</div>', unsafe_allow_html=True)


# 모듈 3: 탭 구조의 세부 통계/소실 리포트 인터페이스
with st.container():
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<span class="card-header-badge">Analysis Report</span>', unsafe_allow_html=True)
    st.markdown('<div class="card-heading">📑 품질 개선 분석 결과 보고서</div>', unsafe_allow_html=True)
    
    report_tab1, report_tab2 = st.tabs(["📋 변수별 결측치(NaN) 소실 복구율 테이블", "📊 이상치 보정에 따른 기술 통계량 변동 지표"])
    
    with report_tab1:
        st.markdown("<p style='font-size:14px; color:#475569;'>정제 전 데이터셋의 변수별 누락 상태가 완벽하게 복구되었는지 진단합니다.</p>", unsafe_allow_html=True)
        null_report = pd.DataFrame({
            "정제 전 결측치 총량 (Original)": df_raw.isnull().sum(),
            "정제 후 결측치 총량 (Cleaned)": df.isnull().sum(),
            "엔진 보정 수량": df_raw.isnull().sum() - df.isnull().sum()
        })
        st.dataframe(null_report, use_container_width=True)
        
    with report_tab2:
        st.markdown("<p style='font-size:14px; color:#475569;'>IQR 임계 범위를 벗어났던 극단치가 평균값으로 매끄럽게 흡수되면서 안정화된 기초 통계 매트릭스입니다.</p>", unsafe_allow_html=True)
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("실시간 변동 분석 타겟 변수 선택:", numeric_cols)
            
            comparison_summary = pd.DataFrame({
                "정제 전 통계량 (Original)": df_raw[selected_col].describe(),
                "정제 후 통계량 (Cleaned)": df[selected_col].describe()
            })
            st.dataframe(comparison_summary.style.format("{:.2f}"), use_container_width=True)
        else:
            st.info("비교 분석할 수 있는 수치형 컬럼 인프라가 없습니다.")
            
    st.markdown('</div>', unsafe_allow_html=True)
