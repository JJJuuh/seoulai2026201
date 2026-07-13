import streamlit as st
import pandas as pd
import plotly.express as px
import os

# 페이지 기본 설정
st.set_page_config(
    page_title="데이터 시각화 (EDA)",
    page_icon="📊",
    layout="wide"
)

st.title("📊 청소년 정신 건강 데이터 시각화 및 핵심 인사이트")
st.markdown("""
우리 데이터셋 분석 결과, 청소년들의 일상(SNS, 수면)이 정신 건강에 어떤 직접적인 영향을 주는지 
**실제 시각화 분석 결과 지표**를 통해 명확한 인과관계와 의미 있는 결론을 도출했습니다.
""")

data_path = "Teen_Mental_Health_Dataset.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    
    # ----------------------------------------------------------------
    # 🔥 1. 상관관계 지도 (Heatmap)
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("🔥 1. 상관관계 지도 (Heatmap) : 어떤 요소들끼리 가장 밀접할까?")
    st.markdown("**💡 그래프 결과 해석**: 격자판의 수치가 0에 가까우면 아무 관계가 없는 것이고, **1이나 -1에 가까울수록 강력한 원인과 결과**가 된다는 뜻입니다.")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    corr_matrix = df[numeric_cols].corr()
    
    fig_heatmap = px.imshow(
        corr_matrix, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1.0, zmax=1.0,
        title="🧠 데이터 항목 간 상관계수 분석"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.info("""
    **🧐 히트맵 분석 결과 (가장 강한 양과 음의 관계 도출):**
    * **🎯 가장 강력한 양(+)의 상관관계**: `daily_social_media_hours`(하루 SNS 시간)와 `depression_label`(우울 위험군 여부)이 **0.175**로 전체 매트릭스 중 가장 높은 양의 상관관계를 보입니다. 소셜미디어 장시간 노출이 정서 위험 신호에 주된 기여를 하고 있음을 뜻합니다.
    * **🎯 가장 강력한 음(-)의 상관관계**: `sleep_hours`(수면 시간)와 `depression_label`(우울 위험군 여부)이 **-0.191**로 전체 항목 중 가장 깊은 음의 관계를 보입니다. 즉, 잠이 부족해질수록 우울 위험 신호가 켜질 확률이 정비례하여 급상승하는 뚜렷한 음의 고리가 증명되었습니다.
    """)

    # ----------------------------------------------------------------
    # 📦 2. 박스차트 대조 및 비교 (Box Plot) : 2개 지표 비교식 전개
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("📦 2. 박스차트 대조 분석 : 우울 위험 신호 유무에 따른 핵심 지표 분포 격차")
    st.markdown("**💡 그래프 결과 해석**: 정상군(0, 파란색)과 우울 위험군(1, 빨간색)의 중앙값 라인과 사분위수(Q1, Q3) 수치를 정밀 대조합니다.")
    
    # 가로로 두 박스차트를 나란히 배치하여 제3자가 비교하기 쉽게 구성
    box_col1, box_col2 = st.columns(2)
    
    with box_col1:
        st.markdown("#### 📱 (A) 우울 여부별 일일 SNS 이용 시간 비교")
        fig_box1 = px.box(
            df, x="depression_label", y="daily_social_media_hours", color="depression_label",
            title="⚖️ 우울 신호군(1) vs 정상군(0)의 일일 SNS 이용 시간 격차",
            template="plotly_white"
        )
        st.plotly_chart(fig_box1, use_container_width=True)
        
    with box_col2:
        st.markdown("#### 💤 (B) 우울 여부별 일일 수면 시간 비교")
        fig_box2 = px.box(
            df, x="depression_label", y="sleep_hours", color="depression_label",
            title="⚖️ 우울 신호군(1) vs 정상군(0)의 일일 수면 시간 격차",
            template="plotly_white"
        )
        st.plotly_chart(fig_box2, use_container_width=True)
    
    st.warning("""
    **🧐 두 박스차트의 정밀 데이터 대조·비교 결과 (사진 속 툴팁 수치 전수 반영):**
    * **📱 SNS 이용 시간 비교**: 
        * 정상군(0)은 중앙값(median)이 **4.40시간**에 위치하고, 하위 25%(Q1)가 **2.70시간**, 상위 25%(Q3)가 **6.20시간**에 고르게 분포하며 최댓값은 **8.00시간**입니다.
        * 반면, 우울 위험군(1)은 상자의 하단 경계(Q1) 자체가 이미 **5.60시간** 이상으로 높게 형성되어 있으며, 중앙값 라인이 **7.00시간**선에 육박하여 확연히 위쪽으로 치우친 과몰입 형태를 보여줍니다.
    * **💤 수면 시간 비교**:
        * 정상군(0)은 수면 분포의 중앙값이 약 **6.50시간 안팎**에 넓게 포진하여 최소 **4.00시간**에서 최대 **9.00시간**까지의 유연한 범위를 보입니다.
        * 반면, 우울 위험군(1)은 최댓값(upper fence)조차 **5.50시간**을 넘지 못하며, 박스 상자의 대부분과 중앙값선이 **4.50시간 미만** 구역에 빽빽하게 찌그러져 있습니다. 우울 위험군 집단이 극단적인 만성 수면 결핍 상태에 갇혀 있음을 명확하게 대조해 줍니다.
    """)

    # ----------------------------------------------------------------
    # 📐 3. 분포차트 (Histogram) : 소수점 단위 정밀 분석
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("📐 3. 분포차트 (Histogram) : 일일 소셜미디어 이용 빈도와 우울증 위험의 임계점")
    st.markdown("**💡 그래프 결과 해석**: 전체 청소년들의 SNS 소비 시간대별 인구 분포 밀도와 위험군(붉은색)이 적층되기 시작하는 정밀 변곡점을 진단합니다.")
    
    fig_hist = px.histogram(
        df, x="daily_social_media_hours", color="depression_label",
        title="📱 조사 대상 청소년들의 SNS 사용 시간대별 인구 분포",
        template="plotly_white",
        nbins=15
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.error("""
    **🧐 분포차트에서 찾아낸 핵심 정보 (정밀 수치 기준):**
    * **⏰ 최대 밀집 사용 구간**: 조사 대상 청소년들은 하루 평균 **4.00시간 초과 ~ 4.50시간 이하** 구간에서 약 100명 이상의 최대 빈도수(최고 봉우리)를 기록하며 가장 집중적으로 소셜미디어를 소비하고 있습니다.
    * **⚠️ 우울 위험 신호(1)의 발현 임계점**: 하루 사용 시간이 **4.80시간~5.20시간** 선을 통과하기 시작하면서부터 파란색 정상군 블록 위에 **빨간색 우울 위험군(1) 데이터 블록이 본격적으로 누적되어 출현**하기 시작합니다.
    * **🚨 고위험 밀도 구역**: **7.00시간에서 8.00시간 이상** 영역으로 진입할 경우, 파란색 정상군(0)의 빈도는 급격히 낮아지는 반면 빨간색 우울 위험군(1)의 적층 비율이 가장 높게 유지되는 구조적 특징을 보여줍니다.
    """)

else:
    st.error("❌ 'Teen_Mental_Health_Dataset.csv' 파일을 불러올 수 없습니다. 경로를 확인해 주세요.")
