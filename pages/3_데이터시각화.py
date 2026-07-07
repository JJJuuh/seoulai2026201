import streamlit as st
import pandas as pd
import os

# 페이지 기본 설정
st.set_page_config(
    page_title="청소년 정신 건강 데이터 분석",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------------------------------------------
# 🔥 1. 메인 헤더 타이틀 영역
# ----------------------------------------------------------------
st.title("📱 소셜미디어가 청소년 정신건강에 미치는 영향")
st.caption("2026학년도 인공지능과 미래사회 나만의 AI 데이터 분석 웹/앱 제작 프로젝트 🚀")
st.markdown("---")

data_path = "Teen_Mental_Health_Dataset.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    
    # ----------------------------------------------------------------
    # 📊 2. 프로젝트 개요 및 수집 데이터 메트릭 (상단 카드 정돈)
    # ----------------------------------------------------------------
    st.subheader("📌 1. 수집 데이터 개요")
    st.markdown("본 프로젝트는 디지털 매체 사용과 청소년 심리 상태 간의 연관성을 증명하기 위해 실제 1,000명의 학생 데이터를 수집 및 정제하여 분석을 진행합니다.")
    
    # 주요 수치 카드 배치
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="총 조사 청소년 수", value=f"{len(df)} 명")
    with m2:
        st.metric(label="평균 SNS 사용 시간", value=f"{df['daily_social_media_hours'].mean():.1f} 시간")
    with m3:
        st.metric(label="평균 수면 시간", value=f"{df['sleep_hours'].mean():.1f} 시간")
    with m4:
        st.metric(label="평균 스트레스 지수", value=f"{df['stress_level'].mean():.1f} / 10점")

    st.markdown("---")

    # ----------------------------------------------------------------
    # 🗺️ 3. 대시보드 전체 목차 및 구성 항목 안내 (2단 레이아웃)
    # ----------------------------------------------------------------
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🗺️ 대시보드 핵심 구조 안내")
        st.markdown("""
        좌측 사이드바 메뉴를 이동하며 우리 연구팀이 설계한 데이터 파이프라인의 실시간 연동 과정을 확인하실 수 있습니다.
        
        * **1️⃣ 데이터 개요 및 실태 조사 (현재 페이지)**: 전체 수집 데이터의 특징 파악 및 문제 정의
        * **2️⃣ 데이터 전처리 및 정제**: 결측치 정제 및 학습 데이터셋 가공 과정
        * **3️⃣ 데이터 자유 시각화 존 (EDA)**: 5대 핵심 차트(박스, 바이올린, 히스토그램, 히트맵, 산점도) 탐색
        * **4️⃣ 인공지능 모델링 및 위험 예측기**: 머신러닝 알고리즘 기반 위험도 시뮬레이터 운영
        """)

    with col2:
        st.subheader("🎯 분석 대상 고유 지표 (Features)")
        st.markdown("수집된 데이터셋의 모든 항목을 독립적인 요소로 분류하여 추적합니다.")
        
        # 가독성을 극대화한 2단 컬럼 항목 나열
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.info("**📱 매체 사용 지표**\n* SNS 사용량 (`daily_social_media_hours`)\n* 선호 플랫폼 (`platform_usage`)\n* 취침 전 화면 시청 (`screen_time_before_sleep`)")
            st.success("**🏃‍♂️ 신체 및 일상 지표**\n* 일일 운동 시간 (`physical_activity`)\n* 오프라인 소통 (`social_interaction_level`)\n* 평균 수면 시간 (`sleep_hours`)")
        with f_col2:
            st.warning("**🧠 정신 건강 결과 지표**\n* 체감 스트레스 (`stress_level`)\n* 정서적 불안감 (`anxiety_level`)\n* 스마트폰 과몰입 중독도 (`addiction_level`)")
            st.error("**🏫 기본 인적 및 학업 지표**\n* 대상자 연령 및 성별 (`age` / `gender`)\n* 학업 성취도 GPA (`academic_performance`)")

    st.markdown("---")

    # ----------------------------------------------------------------
    # 📰 4. 사회적 배경 실태 조사 데이터 가이드
    # ----------------------------------------------------------------
    st.subheader("📰 디지털 미디어 과의존 사회적 배경 및 실태")
    
    # HTML 카드 스타일을 차분하고 눈이 안 아픈 회색/블루 조합으로 전면 수정
    st.markdown("""
    <div style="background-color:#f8fafc; padding:20px; border-left:6px solid #3b82f6; border-radius:8px; margin-bottom:15px;">
        <h4 style="margin-top:0; color:#1e293b; font-size:18px;">📊 연령대별 청소년 인터넷·스마트폰 과의존 고위험군 비중 (전수조사 통계)</h4>
        <p style="color:#475569; margin-bottom:8px;">매체 사용 시간이 증가함에 따라 고등 학년으로 올라갈수록 정서적 과몰입 위험이 심화되는 경향을 보입니다.</p>
        <ul style="color:#334155; font-size:15px; line-height: 2;">
            <li>초등학교 4학년 군: <b>18.5%</b></li>
            <li>중학교 1학년 군: <b>16.3%</b></li>
            <li>고등학교 1학년 군: <span style="color:#ef4444; font-weight:bold;">20.6% (역대 최고 위험치 도달)</span></li>
        </ul>
        <small style="color:#64748b;">🔗 관련 자료 확인: <a href="https://www.yna.co.kr/view/AKR20260611086300017" target="_blank" style="color:#3b82f6; text-decoration:none; font-weight:bold;">[연합뉴스 관련 실태조사 기사 원문]</a></small>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("❌ 'Teen_Mental_Health_Dataset.csv' 파일을 불러올 수 없습니다. 경로를 확인해 주세요.")
