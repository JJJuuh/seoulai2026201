import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.title("📊 청소년 정신 건강 및 소셜미디어 데이터 자유 시각화 (EDA)")
st.markdown("""
이 페이지에서는 데이터셋에 포함된 다양한 요소들을 사용자가 직접 조합하여 자유롭게 분석할 수 있습니다.
왼쪽 선택창에서 알고 싶은 변수들을 골라 조합해 보세요!
""")

data_path = "Teen_Mental_Health_Dataset.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    
    st.markdown("---")
    
    # 분석의 편리함을 위해 '학업 성취도'를 그룹화한 변수 하나를 데이터셋에 기본 포함해 둡니다.
    median_academic = df['academic_performance'].median()
    df['academic_performance_group'] = df['academic_performance'].apply(
        lambda x: '상위권 (성적 높음)' if x >= median_academic else '하위권 (성적 낮음)'
    )
    
    # ----------------------------------------------------------------
    # ⚙️ 사용자가 직접 제어하는 변수 선택 영역
    # ----------------------------------------------------------------
    all_columns = df.columns.tolist()
    
    # 사용자가 보기 편하도록 한글 가이드 제공용 추천 목록 분리
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # X축은 주로 기준이 되는 범주형이나 수치형 변수를 놓습니다.
        default_x = all_columns.index("daily_social_media_hours") if "daily_social_media_hours" in all_columns else 0
        x_axis = st.selectbox("🎯 X축에 놓을 변수 선택", options=all_columns, index=default_x)
        
    with col2:
        # Y축은 변화를 관찰할 수치형 변수를 추천합니다.
        default_y = all_columns.index("stress_level") if "stress_level" in all_columns else min(1, len(all_columns)-1)
        y_axis = st.selectbox("📈 Y축에 놓을 변수 선택 (비교 대상)", options=all_columns, index=default_y)
        
    with col3:
        # 그룹 구분을 위한 옵션 리스트
        color_options = ["None"] + all_columns
        color_target = st.selectbox("🎨 색상 구분 기준 (그룹 쪼개기)", options=color_options, index=0)
        color_var = None if color_target == "None" else color_target

    st.markdown("---")

    # 탭 구조 설계: 1. 자유 변수 시각화 -> 2. 전체 연관성 지도 (히트맵) -> 3. 어떤 차트가 제일 보기 편할까?
    tab1, tab2, tab3 = st.tabs(["📊 내가 고른 변수로 차트 보기", "🔥 변수 간 전체 연관성 지도 (Heatmap)", "🎯 한눈에 보기 쉬운 차트 가이드"])
    
    # ----------------------------------------------------------------
    # TAB 1: 사용자가 고른 변수로 그려지는 자유 차트 영역
    # ----------------------------------------------------------------
    with tab1:
        st.subheader("1. 선택한 변수로 맞춤형 그래프 그리기")
        
        chart_choice = st.radio(
            "그려보고 싶은 차트 종류를 선택하세요:",
            ["① 바 차트 (평균치 비교)", "② 박스 플롯 (분포의 범위 점검)", "③ 산점도 (두 변수의 움직임 흐름)", "④ 히스토그램 (데이터 수 집계)"],
            horizontal=True
        )
        
        # 차트 제목 자동 동적 생성
        chart_title = f"'{x_axis}'와 '{y_axis}'의 관계" + (f" (기준: {color_target})" if color_var else "")
        
        if "① 바 차트" in chart_choice:
            st.markdown(f"#### 📊 {chart_title}")
            # 사용자가 어떤 변수를 고르든 안전하게 평균값으로 집계하여 바 차트 출력
            group_keys = [x_axis]
            if color_var and color_var != x_axis:
                group_keys.append(color_var)
                
            df_bar = df.groupby(group_keys)[y_axis].mean().reset_index()
            
            fig_bar = px.bar(
                df_bar, x=x_axis, y=y_axis, color=color_var,
                title=f"{x_axis}별 평균 {y_axis} 수준 비교",
                barmode="group",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.info(f"""
            **🧐 이 바 차트 팁:**
            - 지금 고르신 **{x_axis}**의 조건에 따라 **{y_axis}**의 평균치가 얼마나 높고 낮은지 막대의 높이로 바로 비교할 수 있습니다. 
            - 집단 간의 '단순 평균 차이'를 누군가에게 설명할 때 가장 직관적이고 강력한 무기가 됩니다.
            """)
            
        elif "② 박스 플롯" in chart_choice:
            st.markdown(f"#### 📦 {chart_title}")
            fig_box = px.box(
                df, x=x_axis, y=y_axis, color=color_var,
                title=f"{x_axis}에 따른 {y_axis}의 세부 분포 범위",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
            st.info(f"""
            **🧐 이 박스 플롯 팁:**
            - 단순 평균값 하나만 보면 집단 내부의 사정을 놓치기 쉽습니다. 
            - 이 상자 그림은 **{x_axis}** 그룹 안에서 **{y_axis}** 점수가 위아래로 얼마나 넓게 퍼져 있는지(최고점, 최저점, 중간에 몰린 범위)를 보여주므로 고위험군이나 특이 집단을 찾아내기 좋습니다.
            """)
            
        elif "③ 산점도" in chart_choice:
            st.markdown(f"#### ✨ {chart_title}")
            fig_scatter = px.scatter(
                df, x=x_axis, y=y_axis, color=color_var,
                title=f"{x_axis} 변화에 따른 {y_axis}의 실제 데이터 분포",
                opacity=0.7,
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.info(f"""
            **🧐 이 산점도 팁:**
            - 흩뿌려진 점들이 오른쪽 위를 향해 올라가는 흐름인지, 반대로 내려가는 흐름인지 보세요.
            - **{x_axis}**가 늘어날 때 **{y_axis}**도 덩달아 증가하는 성향이 있는지, 아니면 아무런 규칙 없이 둥글게 뭉쳐있는지 한눈에 훑어볼 수 있습니다.
            """)
            
        elif "④ 히스토그램" in chart_choice:
            st.markdown(f"#### 📐 {x_axis} 데이터의 빈도 빈도수 분포")
            fig_hist = px.histogram(
                df, x=x_axis, color=color_var,
                title=f"선택한 {x_axis} 데이터의 덩어리 분포 상황",
                barmode="overlay",
                opacity=0.75,
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            st.info(f"""
            **🧐 이 히스토그램 팁:**
            - Y축은 사용자가 따로 고른 변수와 상관없이, 현재 X축에 놓인 **{x_axis}** 데이터에 해당하는 학생이 몇 명이나 존재하느냐(빈도수)를 보여줍니다.
            - 우리 학생들이 주로 어떤 시간대나 점수대에 가장 많이 몰려 분포해 있는지 지형을 파악하기 좋습니다.
            """)

    # ----------------------------------------------------------------
    # TAB 2: 전체 수치형 변수들 간의 연관성 지도 (Heatmap)
    # ----------------------------------------------------------------
    with tab2:
        st.subheader("2. 어떤 변수들끼리 친하고 원수지간일까? (전체 조망)")
        st.markdown("데이터셋에 들어있는 수치 항목들을 한자리에 모아 서로의 밀접도를 격자판 지도로 확인합니다.")
        
        numeric_df = df.select_dtypes(include=['number'])
        corr_matrix = numeric_df.corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1.0, zmax=1.0,
            title="🧠 청소년 마음 및 행동 변수 간 전체 상관관계 격차판"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.info("""
        **🧐 이 히트맵 지도를 읽는 방법:**
        - **+1(진한 빨간색)**에 가까울수록 "A가 커질 때 B도 무조건 커지는 동맹 관계"입니다.
        - **-1(진한 푸른색)**에 가까울수록 "A가 커지면 B는 반대로 뚝 떨어지는 가위바위보 반대 관계"입니다.
        - 수많은 변수를 하나씩 그래프로 그려 대조할 필요 없이, 이 지도 한 장이면 가장 밀접하게 연결된 단짝 변수쌍을 3초 만에 찾아낼 수 있습니다.
        """)

    # ----------------------------------------------------------------
    # TAB 3: 분석가가 추천하는 '가장 직관적인 최고의 차트'
    # ----------------------------------------------------------------
    with tab3:
        st.subheader("🎯 분석가가 전하는 '가장 보기 편한 차트' 선택 가이드")
        
        st.success("""
        #### 🏆 모든 차트 중 직관성 1위는? 👉 단연 '① 바 차트 (Bar Chart)'와 '🔥 히트맵 (Heatmap)'입니다!
        
        **💡 왜 그렇게 생각하나요?**
        1. **바 차트(Bar)가 최고인 이유:** 산점도(Scatter)는 무수한 점들의 잔상이 남고, 박스 플롯(Box)은 통계 기호(사분위수, 수염)의 개념을 모르면 시선이 분산됩니다. 반면 **바 차트는 그냥 '막대의 높이' 그 자체가 데이터의 크기**를 뜻합니다. 초등학생이 보더라도 "이쪽 막대가 더 높으니 여기가 평균 점수가 더 크구나!"라고 즉각 이해할 수 있어서 대중적인 보고용으로 최고의 효율을 자랑합니다.
           
        2. **히트맵(Heatmap)이 최고인 이유:** 여러 변수들의 관계를 파악하기 위해 수십 개의 산점도를 띄워두고 비교하는 번거로움을 **색상 톤(Tone)과 숫자 1장**으로 압축 요약해 버립니다. 전체 데이터의 숲을 먼저 파악하는 데 이보다 효율적인 차트는 없습니다.
           
        **⚙️ 추천하는 EDA 탐색 루틴:**
        - **Step 1:** 우선 `TAB 2`로 가서 히트맵 지도를 보고 숫자가 큰 핵심 변수 커플들을 눈으로 찜합니다.
        - **Step 2:** `TAB 1`로 돌아와 X축과 Y축에 그 변수들을 넣고 `바 차트`나 `박스 플롯`으로 가볍고 명확하게 집단 비교를 완료합니다.
        """)
else:
    st.error("❌ 'Teen_Mental_Health_Dataset.csv' 파일을 불러올 수 없습니다. 경로를 확인해 주세요.")
