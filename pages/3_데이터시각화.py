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
    
    # 변수명 한글 번역 가이드 사전 (설명 및 자동 검증용)
    var_desc = {
        "age": "학생들의 나이(연령대별 차이 확인 가능)",
        "gender": "성별 (남학생과 여학생 간의 심리/행동 격차)",
        "daily_social_media_hours": "하루 평균 소셜미디어(SNS) 사용 시간",
        "platform_usage": "주로 사용하는 SNS 플랫폼 (Instagram, TikTok 등)",
        "sleep_hours": "하루 평균 수면 시간 (기본적인 건강 지표)",
        "screen_time_before_sleep": "자기 전 스마트폰 화면을 보는 시간",
        "academic_performance": "학업 성취도 (학교 성적 및 GPA 수치)",
        "physical_activity": "하루 중 운동 및 신체 활동을 하는 시간",
        "social_interaction_level": "오프라인에서 친구·가족과 소통하는 수준 (상/중/하)",
        "stress_level": "학생이 체감하는 정서적 스트레스 레벨 (높을수록 위험)",
        "anxiety_level": "불안감을 느끼는 정도 (정신 건강 지표)",
        "addiction_level": "스마트폰 및 매체에 과몰입/중독된 수준",
        "depression_label": "우울감 위험 신호 여부 (정서적 고위험군 분류)",
        "academic_performance_group": "성적 중앙값 기준 상위권/하위권 그룹"
    }

    # 범주형(고유값이 적어 쪼개기 좋은 변수)과 수치형을 명확히 구분
    categorical_cols = ["gender", "platform_usage", "social_interaction_level", "depression_label", "academic_performance_group", "age"]
    # 데이터셋에 실제로 존재하는 변수들만 필터링
    categorical_cols = [c for c in categorical_cols if c in all_columns]
    
    # 공통 탭 레이아웃 이전에 차트 종류를 먼저 선택하게 하여 변수 선택창을 유연하게 통제합니다.
    chart_choice = st.radio(
        "📊 먼저 그려보고 싶은 차트 종류를 선택하세요:",
        ["① 바 차트 (평균치 비교)", "② 박스 플롯 (분포의 범위 점검)", "③ 산점도 (두 변수의 움직임 흐름)", "④ 히스토그램 (데이터 수 집계)"],
        horizontal=True
    )
    
    st.markdown("##### ⚙️ 그래프 축 변수 설정")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 🔥 핵심 솔루션: 박스 플롯이나 바 차트처럼 X축이 너무 쪼개지면 안 되는 경우 범주형 변수만 선택하도록 제한
        if "② 박스 플롯" in chart_choice or "① 바 차트" in chart_choice:
            box_x_options = categorical_cols
            default_x = box_x_options.index("academic_performance_group") if "academic_performance_group" in box_x_options else 0
            x_axis = st.selectbox("🎯 X축 변수 선택 (그룹 범주)", options=box_x_options, index=default_x, help="박스 플롯/바 차트가 깨지지 않도록 비교하기 좋은 그룹형 변수만 매칭됩니다.")
        else:
            default_x = all_columns.index("daily_social_media_hours") if "daily_social_media_hours" in all_columns else 0
            x_axis = st.selectbox("🎯 X축 변수 선택 (자유 선택)", options=all_columns, index=default_x)
        
    with col2:
        default_y = all_columns.index("stress_level") if "stress_level" in all_columns else min(1, len(all_columns)-1)
        y_axis = st.selectbox("📈 Y축 변수 선택 (비교 지표)", options=all_columns, index=default_y)
        
    with col3:
        color_options = ["None"] + categorical_cols
        color_target = st.selectbox("🎨 색상 구분 기준 (그룹 쪼개기)", options=color_options, index=0)
        color_var = None if color_target == "None" else color_target

    st.markdown("---")

    # 🔍 선택된 변수 실시간 요약 브리핑
    st.markdown(f"### 🔍 현재 분석 중인 항목 돋보기")
    desc_x = var_desc.get(x_axis, "데이터셋에 포함된 고유 지표")
    desc_y = var_desc.get(y_axis, "데이터셋에 포함된 고유 지표")
    desc_c = var_desc.get(color_target, "그룹 분리 기준 없음")
    
    st.write(f"• **X축 기준 [`{x_axis}`]** : {desc_x}")
    st.write(f"• **Y축 지표 [`{y_axis}`]** : {desc_y}")
    if color_var:
        st.write(f"• **색상 서브 그룹 [`{color_target}`]** : {desc_c}")
    
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 맞춤형 분석 차트 보기", "🔥 변수 간 전체 연관성 지도 (Heatmap)", "🎯 한눈에 보기 쉬운 차트 가이드"])
    
    # ----------------------------------------------------------------
    # TAB 1: 사용자가 고른 변수로 그려지는 자유 차트 영역
    # ----------------------------------------------------------------
    with tab1:
        chart_title = f"'{x_axis}'에 따른 '{y_axis}'의 상세 분석" + (f" (서브 그룹: {color_target})" if color_var else "")
        
        if "① 바 차트" in chart_choice:
            st.markdown(f"#### 📊 {chart_title}")
            group_keys = [x_axis]
            if color_var and color_var != x_axis:
                group_keys.append(color_var)
            
            # 수치형 데이터일 때만 평균 연산이 의미가 있으므로 안전장치 마련
            if pd.api.types.is_numeric_dtype(df[y_axis]):
                df_bar = df.groupby(group_keys)[y_axis].mean().reset_index()
                fig_bar = px.bar(
                    df_bar, x=x_axis, y=y_axis, color=color_var,
                    title=f"{x_axis}별 평균 {y_axis} 수준 비교",
                    barmode="group",
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # 🔍 고도화된 실시간 상세 브리핑
                st.info(f"""
                **👨‍💻 데이터 분석관의 실시간 바 차트 브리핑:**
                * **현상 파악**: 현재 그래프는 **{x_axis}**의 각 집단이 나타내는 **{y_axis}**의 평균값입니다. 가장 높게 솟은 막대를 가진 집단이 현재 청소년 데이터 중 `{y_axis}`의 리스크나 수치가 가장 높은 취약 집단입니다.
                * **수치 해석**: 만약 집단 간 막대 높이 차이가 확연하다면, 청소년들의 `{x_axis}` 특성이 `{y_axis}`를 결정짓거나 예측하는 데 아주 강력한 지표로 작동하고 있음을 의미합니다.
                * **추천 액션**: 이 차트는 복잡한 통계를 모르는 학부모나 대중에게 "어느 그룹이 더 심각한가?"를 단 1초 만에 납득시키기 가장 좋은 리포트용 그래프입니다.
                """)
            else:
                st.warning(f"⚠️ 평균치를 계산하기 위해 Y축(`{y_axis}`)에는 숫자로 된 변수를 선택해 주세요!")
            
        elif "② 박스 플롯" in chart_choice:
            st.markdown(f"#### 📦 {chart_title}")
            fig_box = px.box(
                df, x=x_axis, y=y_axis, color=color_var,
                title=f"{x_axis} 그룹별 {y_axis}의 세부 분포 및 사분위 범위",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
            # 🔍 고도화된 실시간 상세 브리핑
            st.info(f"""
            **👨‍💻 데이터 분석관의 실시간 박스 플롯 브리핑:**
            * **상자의 높낮이와 위치**: 상자의 가운데 선은 해당 그룹의 **'딱 중간 점수(중앙값)'**입니다. 상자 자체가 위쪽에 가 있을수록 정서적 압박이나 행동 수치가 전반적으로 높게 형성된 집단입니다.
            * **상자의 세로 길이 (밀집도)**: 상자가 세로로 길쭉하다면 그 집단 내에서도 아이들의 개인차가 극심하다는 뜻이며, 상자가 납작하고 콤팩트하다면 그 그룹의 아이들은 대부분 엇비슷한 `{y_axis}` 상태를 공유하고 있다는 뜻입니다.
            * **숨겨진 고위험군 (이상치 발견)**: 상자 위아래 수염을 벗어나 유독 홀로 삐져나와 점으로 찍힌 아이들이 있는지 보세요. 이들은 집단의 평균값에 가려져 보이지 않던 **'심각한 불안/우울/스트레스를 겪는 사각지대의 정서적 고위험군 학생'**들입니다. 평균만 보여주는 바 차트에서는 절대 찾아낼 수 없는 박스 플롯만의 강력한 무기입니다.
            """)
            
        elif "③ 산점도" in chart_choice:
            st.markdown(f"#### ✨ {chart_title}")
            fig_scatter = px.scatter(
                df, x=x_axis, y=y_axis, color=color_var,
                title=f"{x_axis}와 {y_axis}의 개별 데이터 매핑 지도",
                opacity=0.7,
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # 🔍 고도화된 실시간 상세 브리핑
            st.info(f"""
            **👨‍💻 데이터 분석관의 실시간 산점도 브리핑:**
            * **동향성 분석 (우상향 vs 우하향)**: 무수히 찍힌 데이터의 점들이 전체적으로 오른쪽 위를 향해 뻗어 나가는지 보세요. 만약 우상향한다면 **{x_axis}**가 증가함에 따라 **{y_axis}**도 강력하게 동반 상승하는 인과/상관 흐름입니다.
            * **독립성 분석 (구름형 패턴)**: 점들이 아무런 규칙 없이 넓고 둥글게 퍼져 있다면, 두 지표는 겉보기와 달리 실제 청소년 생활에서 서로 크게 간섭하지 않고 따로 움직이는 독립적인 요인입니다.
            * **그룹 격차**: 격자망 속에서 색상별로 점들이 완전히 분리되어 노는지, 혹은 빽빽하게 뒤섞여 있는지 확인하여 집단 간 정서 패턴의 동질성을 파악할 수 있습니다.
            """)
            
        elif "④ 히스토그램" in chart_choice:
            st.markdown(f"#### 📐 {x_axis} 데이터의 빈도수 분포 지형도")
            fig_hist = px.histogram(
                df, x=x_axis, color=color_var,
                title=f"선택한 {x_axis} 데이터의 학생 수 분포 상황",
                barmode="overlay",
                opacity=0.75,
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # 🔍 고도화된 실시간 상세 브리핑
            st.info(f"""
            **👨‍💻 데이터 분석관의 실시간 히스토그램 브리핑:**
            * **데이터의 숲 조망**: 이 그래프의 Y축은 오직 **'학생의 수(빈도수)'**만을 나타냅니다. 가장 높게 솟아오른 거대한 봉우리 구간이 우리 청소년 데이터에서 가장 보편적이고 일반적인 다수가 밀집한 생활권입니다.
            * **치우침(Skewness) 점검**: 그래프의 꼬리가 오른쪽으로 길게 늘어져 있다면 대다수 평범한 아이들 사이에 극단적으로 높은 수치를 가진 소수의 아이들이 섞여 있는 지형임을 나타냅니다.
            * **서브 그룹 면적 대조**: 서브 그룹 색상이 지정되었을 때, 두 집단의 지형 덩어리가 완전히 포개져 있는지 한쪽으로 완전히 밀려나 밀도가 다르게 생성되어 있는지 대조해 볼 수 있습니다.
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
        - +1(진한 빨간색)에 가까울수록 "A가 커질 때 B도 무조건 커지는 동맹 관계"입니다.
        - -1(진한 푸른색)에 가까울수록 "A가 커지면 B는 반대로 뚝 떨어지는 가위바위보 반대 관계"입니다.
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
