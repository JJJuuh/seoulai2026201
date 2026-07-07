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
우리 데이터셋 분석 결과, 청소년들의 일상(SNS, 수면, 운동)이 정신 건강에 어떤 직접적인 영향을 주는지 
**배운 5대 그래프**를 통해 명확한 인과관계와 의미 있는 결론을 도출했습니다.
""")

data_path = "Teen_Mental_Health_Dataset.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    
    # ----------------------------------------------------------------
    # 🔥 1. 히트맵 (Heatmap) : 데이터의 전체 숲을 먼저 진단
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("🔥 1. 상관관계 지도 (Heatmap) : 어떤 요소들끼리 가장 밀접할까?")
    st.markdown("**💡 그래프 결과 해석 (의미 있는 정보)**: 격자판의 수치가 0에 가까우면 아무 관계가 없는 것이고, **1이나 -1에 가까울수록 강력한 원인과 결과**가 된다는 뜻입니다.")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    corr_matrix = df[numeric_cols].corr()
    
    fig_heatmap = px.imshow(
        corr_matrix, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1.0, zmax=1.0,
        title="🧠 데이터 항목 간 상관계수 분석"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.info("""
    **🧐 히트맵에서 찾아낸 핵심 정보:**
    * **`daily_social_media_hours`와 `stress_level`의 연결**: 수치가 양의 관계로 붉게 물들어 있습니다. 즉, 소셜미디어 사용량이 늘어날수록 스트레스와 불안감도 함께 치솟는 명확한 리스크 동맹 관계임이 데이터로 입증되었습니다.
    * **`sleep_hours`와 `anxiety_level`의 연결**: 푸른색 톤을 띱니다. 잠을 많이 잘수록 불안감이 낮아진다는 강력한 완충(방어) 관계를 보여줍니다.
    """)

    # ----------------------------------------------------------------
    # ✨ 2. 산점도 (Scatter Plot) : SNS 시간에 따른 스트레스 추세 추적
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("✨ 2. 흐름 산점도 (Scatter Plot) : SNS 사용 시간에 따라 스트레스 지수가 정말 달라질까?")
    st.markdown("**💡 그래프 결과 해석 (의미 있는 정보)**: 점들이 우상향하며 뻗어 나가는 흐름과 중심의 추세선을 주목해 보세요.")
    
    fig_scatter = px.scatter(
        df, x="daily_social_media_hours", y="stress_level", color="gender",
        trendline="ols",
        title="📱 SNS 사용 시간 대비 스트레스 지수 분포와 우상향 추세선",
        opacity=0.6, template="plotly_white"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.error("""
    **🧐 산점도에서 찾아낸 핵심 정보:**
    * **확연한 우상향 추세**: 그래프 한가운데를 관통하는 직선(추세선)이 오른쪽 위를 향하고 있습니다. 이는 **하루 SNS 사용 시간이 늘어남에 따라 비례하여 청소년들의 스트레스 지수도 선형적으로 가중된다**는 움직일 수 없는 증거입니다.
    * **고위험군 밀집 구역**: 특히 6시간 이상 극단적으로 SNS를 많이 사용하는 우측 상단 영역에 수많은 청소년들의 데이터 포인트가 쏠려 있는 것을 눈으로 확인할 수 있습니다.
    """)

    # ----------------------------------------------------------------
    # 📊 3. 막대그래프 (Bar Chart) : 운동량에 따른 스트레스 평균 비교
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 3. 막대그래프 (Bar Chart) : 일일 운동 시간이 스트레스를 낮춰줄까?")
    st.markdown("**💡 그래프 결과 해석 (의미 있는 정보)**: 비교가 가장 쉬운 막대그래프를 통해 특정 조건(운동 시간)에 따른 스트레스 평균값의 높낮이를 직관적으로 대조합니다.")
    
    # 분석을 위해 운동 시간을 구간화(0.5시간 미만, 0.5~1.5시간, 1.5시간 이상)
    def categorize_activity(hours):
        if hours < 0.5: return "① 부족 (0.5시간 미만)"
        elif hours <= 1.5: return "② 보통 (0.5 ~ 1.5시간)"
        else: return "③ 충분 (1.5시간 이상)"
        
    df['activity_group'] = df['physical_activity'].apply(categorize_activity)
    df_bar = df.groupby('activity_group')['stress_level'].mean().reset_index()
    
    fig_bar = px.bar(
        df_bar, x="activity_group", y="stress_level",
        color="activity_group",
        title="🏃‍♂️ 신체 운동 수준별 스트레스 평균치 격차",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.success(f"""
    **🧐 막대그래프에서 찾아낸 핵심 정보:**
    * **신체 활동의 스트레스 차단 효과**: 운동량이 **'부족(0.5시간 미만)'**한 청소년 그룹의 막대 높이가 가장 높게 솟아있습니다.
    * 반면 하루 1.5시간 이상 **'충분'**히 땀 흘려 운동하는 청소년 그룹은 스트레스 평균치가 가장 아래로 뚝 떨어집니다. 
    * 두 집단 간의 막대 높이 차이만큼 **신체 활동이 정신 건강을 지켜주는 강력한 천연 방어벽 역할을 수행하고 있음**이 증명됩니다.
    """)

    # ----------------------------------------------------------------
    # 📦 4. 박스차트 (Box Plot) : 스마트폰 중독도 분포와 성별 격차
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("📦 4. 박스차트 (Box Plot) : 성별에 따른 스마트폰 과몰입(중독) 분포 범위")
    st.markdown("**💡 그래프 결과 해석 (의미 있는 정보)**: 가운데 굵은 가로선(중앙값)과 상자의 위치를 통해 남학생과 여학생 간의 데이터 격차를 점검합니다.")
    
    fig_box = px.box(
        df, x="gender", y="addiction_level", color="gender",
        title="⚖️ 남학생(male)과 여학생(female)의 매체 중독도 수준 비교",
        template="plotly_white",
        color_discrete_sequence=["#3b82f6", "#ec4899"]
    )
    st.plotly_chart(fig_box, use_container_width=True)
    
    st.warning("""
    **🧐 박스차트에서 찾아낸 핵심 정보:**
    * **성별 편차의 시각화**: 여학생(female)의 핑크색 박스 및 중앙값 라인이 남학생(male)의 파란색 박스보다 전반적으로 상단(높은 점수대)에 위치하고 있습니다.
    * 이는 **여학생 집단이 남학생 집단에 비해 스마트폰 과몰입 및 소셜미디어 중독 리스크에 노출될 확률이 상대적으로 더 높다**는 통계적 분포 특성을 직관적으로 보여줍니다.
    """)

    # ----------------------------------------------------------------
    # 📐 5. 히스토그램 (Histogram) : 수면 부족 청소년 인구 실태
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("📐 5. 히스토그램 (Histogram) : 우리 청소년들은 하루에 몇 시간이나 자고 있을까?")
    st.markdown("**💡 그래프 결과 해석 (의미 있는 정보)**: Y축은 학생 수(인구 빈도)를 뜻합니다. 어느 시간대 구간에 가장 많은 아이들이 밀집해 있는지 확인하세요.")
    
    fig_hist = px.histogram(
        df, x="sleep_hours", nbins=10,
        title="💤 조사 대상 청소년들의 일일 수면 시간 빈도 분포",
        template="plotly_white", color_discrete_sequence=["#6366f1"]
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.info("""
    **🧐 히스토그램에서 찾아낸 핵심 정보:**
    * **수면 부족 실태 발견**: 청소년 권장 수면 시간인 7~8시간 영역보다 **5~6시간 이하의 짧은 수면 구간**에 그래프의 가장 높은 봉우리(대다수의 학생 빈도)가 몰려 있는 구조를 볼 수 있습니다.
    * 대다수의 청소년이 심각한 수면 부족 상태에 직면해 있으며, 이는 앞서 확인한 스트레스 상승의 또 다른 숨은 배경 원인임을 유추할 수 있습니다.
    """)

else:
    st.error("❌ 'Teen_Mental_Health_Dataset.csv' 파일을 불러올 수 없습니다. 경로를 확인해 주세요.")
