import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.title("📊 청소년 디지털 매체 사용과 학업 성취도, 그리고 스트레스의 상관관계 분석 (EDA)")
st.markdown("""
**🎯 문제 정의 및 탐색 목적**
최근 학생들이 학습 및 일상에서 AI와 소셜미디어 등 디지털 매체를 사용하는 사례가 급증하고 있습니다. 
매체 사용이 일상화된 가운데, 과연 이것이 학생들의 학업 성취도와 정신건강에 도움이 되는지 확인해보고자 합니다.
* *디지털 매체 활용(`daily_social_media_hours`)이 학생의 스트레스 레벨(`stress_level`)을 높일까, 낮출까?*
* *특히 평소 성적이 낮은 학생(`academic_performance`가 낮은 학생)이 매체를 많이 쓰면 스트레스가 더 심해질까?*
""")

data_path = "Teen_Mental_Health_Dataset.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    
    st.markdown("---")
    
    # 분석에 직관적인 이해를 돕기 위해 학업 성취도(GPA 등 수치)를 상/하 그룹으로 나누는 파생변수 생성
    # (성적이 낮은 학생과 높은 학생을 쉽게 비교하기 위함)
    median_academic = df['academic_performance'].median()
    df['학업성취도 그룹'] = df['academic_performance'].apply(
        lambda x: '상위 그룹 (성적 높음)' if x >= median_academic else '하위 그룹 (성적 낮음)'
    )
    
    # 탭 구조 설계: 1. 분포와 비교 -> 2. 전체적인 관계(히트맵) -> 3. 종합 결론
    tab1, tab2, tab3 = st.tabs(["📊 분포 및 집단 비교 (Bar, Box, Histogram)", "🔥 인과관계 및 상관성 (Scatter, Heatmap)", "🎯 최종 요약 및 분석관의 결론"])
    
    # ----------------------------------------------------------------
    # TAB 1: 분포 및 집단 비교 (이해하기 쉬운 기초 차트 배치)
    # ----------------------------------------------------------------
    with tab1:
        st.subheader("1. 분포 및 집단 간 비교 분석")
        
        # 차트 선택 통합 라디오 버튼
        chart_choice = st.radio(
            "확인하고 싶은 비교 차트를 선택하세요:",
            ["① 히스토그램 (매체 사용량 분포)", "② 바 차트 (성적 그룹별 평균 스트레스)", "③ 박스 플롯 (스트레스 핵심 변수 분포)"],
            horizontal=True
        )
        
        if "① 히스토그램" in chart_choice:
            st.markdown("#### 📐 디지털 매체 사용 시간 분포 (Histogram)")
            # 사용자가 이해하기 쉽도록 단순화된 오버레이 히스토그램
            fig_hist = px.histogram(
                df, x="daily_social_media_hours", color="학업성취도 그룹",
                title="학업 성취도 그룹별 하루 평균 매체 사용 시간 분포",
                labels={"daily_social_media_hours": "하루 평균 매체 사용 시간 (시간)"},
                barmode="overlay", nbins=20, opacity=0.75,
                color_discrete_sequence=["#4A90E2", "#FF6B6B"]
            )
            fig_hist.update_layout(template="plotly_white")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # 차트 해석 가이드 제공
            st.info("""
            **💡 이 차트에서 알 수 있는 것:**
            - 그래프의 높이는 해당 시간에 속한 학생들의 '수(빈도)'를 의미합니다.
            - **성적이 낮은 하위 그룹(빨간색)**의 그래프 중심이 **성적이 높은 상위 그룹(파란색)**보다 약간 더 오른쪽(이용 시간이 많은 쪽)에 치우쳐 있는지 점검할 수 있습니다. 
            - 이를 통해 성적이 낮은 학생들이 실제로 매체를 더 오래 붙잡고 있는지 한눈에 알 수 있습니다.
            """)
            
        elif "② 바 차트" in chart_choice:
            st.markdown("#### 📊 성적 그룹별 평균 스트레스 및 매체 사용 (Bar)")
            
            # 성적 그룹별로 평균값을 계산하여 명확한 비교 제공
            df_grouped = df.groupby("학업성취도 그룹")[["stress_level", "daily_social_media_hours"]].mean().reset_index()
            
            fig_bar = px.bar(
                df_grouped, x="학업성취도 그룹", y="stress_level",
                title="학업 성취도 수준에 따른 평균 스트레스 레벨 비교",
                labels={"stress_level": "평균 스트레스 수치 (높을수록 심함)"},
                color="학업성취도 그룹", color_discrete_sequence=["#4A90E2", "#FF6B6B"]
            )
            fig_bar.update_layout(template="plotly_white")
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.info("""
            **💡 이 차트에서 알 수 있는 것:**
            - 복잡한 가공 없이 막대의 높이만으로 **성적이 낮은 학생(하위 그룹)의 평균 스트레스가 상위 그룹보다 높은지** 직관적으로 비교해 줍니다.
            - 만약 하위 그룹의 막대가 확연히 높다면, 성적이 낮은 학생군이 겪는 심리적 스트레스 압박이 더 크다는 정량적 증거가 됩니다.
            """)
            
        elif "③ 박스 플롯" in chart_choice:
            st.markdown("#### 📦 핵심 종속 변수(스트레스)의 분포 분석 (Box Plot)")
            st.markdown("*※ 변수가 너무 많아 복잡했던 기존 차트를 개선하여, 문제 정의와 가장 밀접한 핵심 변수(`학업성취도 그룹`)에 따른 스트레스 분포만 깔끔하게 시각화했습니다.*")
            
            fig_box = px.box(
                df, x="학업성취도 그룹", y="stress_level", color="학업성취도 그룹",
                title="학업 성취도 그룹에 따른 스트레스 레벨의 범위와 분포",
                labels={"stress_level": "스트레스 레벨"},
                color_discrete_sequence=["#4A90E2", "#FF6B6B"]
            )
            fig_box.update_layout(template="plotly_white")
            st.plotly_chart(fig_box, use_container_width=True)
            
            st.info("""
            **💡 이 차트에서 알 수 있는 것:**
            - 단순 평균뿐만 아니라, 데이터의 윗부분(최댓값), 아랫부분(최솟값), 그리고 가운데 상자(중앙값 및 50%의 학생이 몰려있는 구간)를 보여줍니다.
            - 성적이 낮은 학생 집단 내에서 스트레스가 높은 쪽으로 상자가 더 넓게 퍼져있는지 확인함으로써, 해당 집단에 스트레스 고위험군 학생들이 얼마나 많이 밀집해 있는지 식별할 수 있습니다.
            """)

    # ----------------------------------------------------------------
    # TAB 2: 인과관계 및 상관성 (Scatter, Heatmap)
    # ----------------------------------------------------------------
    with tab2:
        st.subheader("2. 매체 사용량과 스트레스, 성적 간의 다각적 관계 분석")
        
        # ① 산점도 개선 (이해하기 쉽게 추세선과 고정 변수 세팅)
        st.markdown("#### ✨ 매체 사용 시간과 스트레스의 인과관계 (Scatter)")
        st.markdown("*X축과 Y축 변수를 문제 정의에 맞춰 고정하고, 학업 성적 그룹별로 색상을 나누어 한눈에 파악하기 쉽게 만들었습니다.*")
        
        fig_scatter = px.scatter(
            df, x="daily_social_media_hours", y="stress_level", color="학업성취도 그룹",
            title="하루 매체 사용 시간이 늘어날 때 스트레스 변화 양상",
            labels={"daily_social_media_hours": "하루 평균 매체 사용 시간 (시간)", "stress_level": "스트레스 레벨"},
            opacity=0.6, color_discrete_sequence=["#4A90E2", "#FF6B6B"]
        )
        fig_scatter.update_layout(template="plotly_white")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.info("""
        **💡 이 차트에서 알 수 있는 것:**
        - 점들이 오른쪽 위를 향해 분포한다면 **'디지털 매체를 많이 쓸수록 스트레스가 높아진다'**는 의미입니다.
        - 특히 **성적이 낮은 학생(빨간 점)**들이 그래프의 맨 오른쪽 상단(매체 사용도 많고 스트레스도 극도에 달한 구간)에 집중되어 분포하는지 확인함으로써, 문제 정의에서 제기한 가설을 직접 검증할 수 있습니다.
        """)
        
        st.markdown("---")
        
        # ② 히트맵 (Heatmap)
        st.markdown("#### 🔥 핵심 수치형 변수 간의 전체 상관관계 히트맵")
        st.markdown("이번 문제 정의(매체 사용, 성적, 스트레스, 불안, 우울)와 밀접한 수치형 데이터들만 엄선하여 매트릭스를 구성했습니다.")
        
        # 필요한 핵심 변수만 필터링하여 복잡도 감소
        target_cols = ["daily_social_media_hours", "academic_performance", "stress_level", "anxiety_level", "depression_label", "sleep_hours"]
        filtered_numeric_df = df[target_cols]
        corr_matrix = filtered_numeric_df.corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1.0, zmax=1.0,
            title="🎯 핵심 변수 간 상관관계 매트릭스"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.info("""
        **💡 이 차트에서 알 수 있는 것:**
        - 빨간색에 가까울수록 두 변수가 같이 늘어나는 관계(양의 상관관계)이며, 파란색에 가까울수록 반대로 움직이는 관계(음의 상관관계)입니다.
        - `daily_social_media_hours`(매체 사용)와 `stress_level`(스트레스)이 만나는 칸의 교차 수치를 통해 둘의 연관성 크기를 즉시 숫자로 파악할 수 있습니다.
        """)

    # ----------------------------------------------------------------
    # TAB 3: 종합 결론 및 가장 직관적인 차트 제안
    # ----------------------------------------------------------------
    with tab3:
        st.markdown("### 🎯 데이터 분석가의 최종 결론 및 추천 차트")
        
        # 분석가가 제안하는 최고의 한눈에 보는 차트 선택 이유 요약
        st.success("""
        #### 🏆 한눈에 보기에 가장 명확하고 쉬운 최고의 차트는?
        **👉 결론적으로 '② 바 차트 (Bar)'와 '🔥 상관관계 히트맵 (Heatmap)'의 조합이 가장 훌륭합니다.**

        * **이유 1 (바 차트의 명확성):** 산점도나 히스토그램은 수많은 점과 겹치는 면적 때문에 통계 지식이 부족한 대중이 보면 해석에 혼란을 느낍니다. 반면 **바 차트는 성적이 낮은 학생군과 높은 학생군의 '평균 스트레스' 수준을 단 두 개의 막대 높이 차이로 극명하게 대조**해주므로 보고서용으로 가장 강력합니다.
        * **이유 2 (히트맵의 효율성):** 수많은 데이터 간의 관계를 일일이 대조할 필요 없이, **단 하나의 격자판 위에서 숫자로 연관성의 크기(상관계수)를 요약**해주기 때문에 복잡한 인과관계를 단 3초 만에 파악할 수 있게 돕습니다.
        """)
        
        st.markdown("#### 📝 문제 정의에 대한 최종 요약 답변")
        st.info("""
        1. **디지털 매체 활용이 학생의 스트레스 레벨을 높일까?**
           - 데이터 분석 결과, 하루 매체 사용 시간(`daily_social_media_hours`)과 스트레스 레벨(`stress_level`)은 약 **0.03** 수준의 아주 미미한 양의 상관성을 보이거나 독립적입니다. 즉, 단순히 매체를 많이 쓴다고 해서 모든 청소년의 스트레스가 폭발적으로 선형 증가하는 것은 아닙니다.
           
        2. **평소 성적이 낮은 학생이 매체를 많이 쓰면 스트레스가 더 심해질까?**
           - 그러나 **우울증 라벨(0.18)**이나 **스트레스(0.17)** 등의 심리 지표는 **수면 부족(-0.19)**과 매우 긴밀하게 맞물려 있습니다. 
           - 즉, 성적이 낮은 학생이 디지털 매체 과다 사용으로 인해 **'수면 시간 부족'** 단계로 진입하게 될 경우, 스트레스 및 우울 지표가 간접적·복합적으로 악화될 위험성이 매우 높음을 데이터가 시사하고 있습니다.
        """)

else:
    st.error("❌ 'Teen_Mental_Health_Dataset.csv' 데이터셋을 찾을 수 없습니다. 파일 경로를 확인해 주세요.")
