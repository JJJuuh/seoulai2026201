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
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        default_x = all_columns.index("daily_social_media_hours") if "daily_social_media_hours" in all_columns else 0
        x_axis = st.selectbox("🎯 X축에 놓을 변수 선택 (원인 또는 기준)", options=all_columns, index=default_x)
        
    with col2:
        default_y = all_columns.index("stress_level") if "stress_level" in all_columns else min(1, len(all_columns)-1)
        y_axis = st.selectbox("📈 Y축에 놓을 변수 선택 (결과 또는 비교 대상)", options=all_columns, index=default_y)
        
    with col3:
        color_options = ["None"] + all_columns
        color_target = st.selectbox("🎨 색상 구분 기준 (그룹 쪼개기)", options=color_options, index=0)
        color_var = None if color_target == "None" else color_target

    st.markdown("---")

    # 💡 [AI가 만든 느낌 제거 핵심] 선택된 변수들이 무엇을 뜻하는지 인간적인 언어로 실시간 요약 브리핑 제공
    st.markdown(f"### 🔍 현재 분석 중인 항목 돋보기")
    
    # 변수명 한글 번역 가이드 사전
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
    
    desc_x = var_desc.get(x_axis, "데이터셋에 포함된 고유 지표")
    desc_y = var_desc.get(y_axis, "데이터셋에 포함된 고유 지표")
    desc_c = var_desc.get(color_target, "그룹 분리 기준 없음")
    
    st.write(f"• **X축 [`{x_axis}`]** : {desc_x}")
    st.write(f"• **Y축 [`{{y_axis}}`]** : {desc_y}")
    if color_var:
        st.write(f"• **색상 기준 [`{color_target}`]** : 이 변수를 기준으로 학생들을 쪼개서 비교합니다. ({desc_c})")
    
    st.markdown("---")

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
        
        chart_title = f"'{x_axis}'와 '{y_axis}'의 관계" + (f" (기준: {color_target})" if color_var else "")
        
        if "① 바 차트" in chart_choice:
            st.markdown(f"#### 📊 {chart_title}")
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
            
            # 🔍 세부 설명 강화
            st.info(f"""
            **🧐 이 바 차트를 깊게 읽는 방법 (Deep Insight):**
            1. **막대의 높이가 주는 의미**: 현재 그래프는 **{x_axis}** 조건에 따른 **{y_axis}**의 '평균값'을 나타냅니다. 막대가 유독 높게 솟아오른 구간이 있다면, 그 조건에 처한 청소년들이 해당 문제나 지표에 가장 취약하다는 뜻입니다.
            2. **그룹 간 격차 관전 포인트**: {"오른쪽 선택창에서 고른 `" + color_target + "`에 따라 막대 색상이 나뉩니다. 색상별 높낮이 차이가 뚜렷할수록, 같은 조건 안에서도 그룹에 따라 확연한 차이가 발생함을 의미합니다." if color_var else "현재는 단일 그룹 데이터입니다. 색상 분리 기준을 선택하시면 집단 간 격차를 한눈에 볼 수 있습니다."}
            3. **실전 팁**: 대중적인 보고서나 발표 자료를 만들 때는 복잡한 분포보다 이 '평균 바 차트'가 메시지를 전달하기에 가장 쉽고 명확합니다.
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
            
            # 🔍 세부 설명 강화
            st.info(f"""
            **🧐 이 박스 플롯을 깊게 읽는 방법 (Deep Insight):**
            1. **상자와 수염 모양 해석하기**: 가운데 굵은 가로선은 딱 중간에 위치한 학생의 점수(중앙값)입니다. 상자의 위아래 길이는 학생들의 50%가 밀집해 있는 주된 점수 구간입니다. 위아래로 길게 뻗은 선(수염)은 데이터의 전체적인 범위를 뜻합니다.
            2. **현장 분석관의 눈**: 평균값은 정상이어도, 특정 집단에서 상자가 위쪽으로 길게 늘어지거나 삐져나온 점(이상치)이 많다면 그 집단 내에 정서적·학업적 관리가 시급한 **'고위험군 아이들'이 많이 잠재되어 있다**는 강력한 증거가 됩니다. 
            3. **매칭 분석**: **{x_axis}**가 변화함에 따라 **{y_axis}** 상자의 위치가 위아래로 요동치는지 주목해 보세요.
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
            
            # 🔍 세부 설명 강화
            st.info(f"""
            **🧐 이 산점도를 깊게 읽는 방법 (Deep Insight):**
            1. **점들의 행진 방향 찾기**: 점들이 우상향(오른쪽 위로 대각선 흐름)하면 **{x_axis}**가 많아질수록 **{y_axis}**도 같이 증가하는 강한 동맹 관계입니다. 반대로 우하향하면 하나가 커질 때 다른 하나는 억제되는 반비례 관계입니다.
            2. **구름 형태나 무작위 패턴**: 만약 점들이 사방에 동그란 구름처럼 퍼져 있다면, 두 변수는 서로 아무런 영향을 주지 않고 따로 노는 '독립적인 관계'에 가깝습니다.
            3. **사각지대 탐색**: 그래프의 극단적인 모서리(예: 맨 오른쪽 위, 맨 오른쪽 아래)에 찍힌 아이들이 있는지 보세요. 그 아이들이 우리가 집중 케어해야 하거나, 혹은 아주 모범적인 생활 패턴을 가진 특이 케이스의 학생들입니다.
            """)
            
        elif "④ 히스토그램" in chart_choice:
            st.markdown(f"#### 📐 {x_axis} 데이터의 빈도수 분포")
            fig_hist = px.histogram(
                df, x=x_axis, color=color_var,
                title=f"선택한 {x_axis} 데이터의 덩어리 분포 상황",
                barmode="overlay",
                opacity=0.75,
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # 🔍 세부 설명 강화
            st.info(f"""
            **🧐 이 히스토그램을 깊게 읽는 방법 (Deep Insight):**
            1. **Y축의 비밀**: 히스토그램의 Y축은 사용자가 선택한 Y축 변수와 관계없이 오직 **'학생 수(데이터 개수)'**만 의미합니다. 
            2. **지형지물 분석**: 그래프가 낙타 등처럼 봉우리가 높게 솟은 곳이 우리 청소년들이 대다수 몰려 있는 '평범한 일상 구간'입니다. 반대로 양쪽 끝자락의 낮고 평평한 곳은 아주 드문 케이스입니다.
            3. **집단 간 면적 비교**: {"색상 기준인 `" + color_target + "` 그룹들이 서로 파도처럼 겹쳐 보일 것입니다. 어떤 그룹의 그래프 덩어리가 더 오른쪽(혹은 높은 점수대)에 치우쳐 있는지 비교하면, 두 집단의 전반적인 특성 차이를 직관적으로 파악할 수 있습니다." if color_var else "색상 기준을 선택하시면 집단별 데이터 분포 덩어리를 겹쳐서 비교해 볼 수 있습니다."}
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
