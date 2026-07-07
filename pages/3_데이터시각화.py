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

st.title("📊 청소년 정신 건강 및 소셜미디어 데이터 자유 시각화 (EDA)")
st.markdown("""
이 페이지에서는 데이터셋에 포함된 다양한 요소들을 사용자가 직접 조합하여 자유롭게 분석할 수 있습니다.
상단의 선택창에서 알고 싶은 변수들을 골라 조합하면 **실시간 그래프 결론**이 아래에 자동으로 도출됩니다!
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

    # 변수명 한글 번역 가이드 사전 (인간적인 언어 브리핑용)
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
    
    st.markdown(f"### 🔍 현재 분석 중인 항목 돋보기")
    st.write(f"• **X축 [`{x_axis}`]** : {desc_x}")
    st.write(f"• **Y축 [`{y_axis}`]** : {desc_y}")
    if color_var:
        st.write(f"• **색상 기준 [`{color_target}`]** : 이 변수를 기준으로 학생들을 쪼개서 비교합니다. ({desc_c})")
    
    st.markdown("---")

    # 탭 레이아웃 구성
    tab1, tab2, tab3 = st.tabs(["📊 내가 고른 변수로 차트 보기", "🔥 변수 간 전체 연관성 지도 (Heatmap)", "🎯 한눈에 보기 쉬운 차트 가이드"])
    
    # ----------------------------------------------------------------
    # TAB 1: 사용자가 고른 변수로 그려지는 자유 차트 + 수치 격차 분석 피드백 반영
    # ----------------------------------------------------------------
    with tab1:
        st.subheader("1. 선택한 변수로 맞춤형 그래프 그리기")
        
        chart_choice = st.radio(
            "그려보고 싶은 차트 종류를 선택하세요:",
            ["① 바 차트 (평균치 비교)", "② 박스 플롯 (분포의 범위 점검)", "③ 산점도 (두 변수의 움직임 흐름)", "④ 히스토그램 (데이터 수 집계)"],
            horizontal=True
        )
        
        chart_title = f"'{x_axis}'와 '{y_axis}'의 관계" + (f" (기준: {color_target})" if color_var else "")
        
        # [차트 그리기 실행 파트]
        if "① 바 차트" in chart_choice:
            group_keys = [x_axis]
            if color_var and color_var != x_axis:
                group_keys.append(color_var)
                
            df_bar = df.groupby(group_keys)[y_axis].mean().reset_index()
            
            fig = px.bar(
                df_bar, x=x_axis, y=y_axis, color=color_var,
                title=f"{x_axis}별 평균 {y_axis} 수준 비교",
                barmode="group",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif "② 박스 플롯" in chart_choice:
            fig = px.box(
                df, x=x_axis, y=y_axis, color=color_var,
                title=f"{x_axis}에 따른 {y_axis}의 세부 분포 범위",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif "③ 산점도" in chart_choice:
            fig = px.scatter(
                df, x=x_axis, y=y_axis, color=color_var,
                title=f"{x_axis} 변화에 따른 {y_axis}의 실제 데이터 분포",
                opacity=0.7,
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif "④ 히스토그램" in chart_choice:
            fig = px.histogram(
                df, x=x_axis, color=color_var,
                title=f"선택한 {x_axis} 데이터의 덩어리 분포 상황",
                barmode="overlay",
                opacity=0.75,
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig, use_container_width=True)

        # ====================================================================
        # ⭐ [수정 반영 반영 완료] 데이터 그래프 결론 섹션
        # ====================================================================
        st.markdown("---")
        st.markdown(f"### 📢 데이터 그래프 결론: 한눈에 보는 수치 격차 분석")
        
        try:
            # 1단계: 수치형 변수 커플일 경우 동맹/반비례 상관관계 브리핑
            is_x_num = pd.api.types.is_numeric_dtype(df[x_axis])
            is_y_num = pd.api.types.is_numeric_dtype(df[y_axis])
            
            if is_x_num and is_y_num:
                correlation = df[x_axis].corr(df[y_axis])
                if correlation > 0.3:
                    relation_text = f"📈 **비례 관계 격차**: `{x_axis}` 지수가 높아질수록 `{y_axis}` 값도 함께 위로 치솟는 뚜렷한 경향을 보입니다."
                elif correlation < -0.3:
                    relation_text = f"📉 **반비례 완충 격차**: `{x_axis}` 지수가 늘어나면 상대적으로 `{y_axis}` 값을 아래로 뚝 떨어뜨리며 방어해 주는 흐름입니다."
                else:
                    relation_text = "☁️ **독립 데이터 흐름**: 두 변수는 그래프상에서 특별히 한쪽 방향으로 치우치지 않고 서로 따로 움직이고 있습니다."
                
                st.info(relation_text)

            # 2단계: 핵심 변수 선택 시 구체적인 '최댓값, 최솟값, 그룹 간 차이 수치' 실시간 계산
            st.markdown("#### 🎯 그래프 속 핵심 격차 팩트 체크")
            
            # [케이스 A] 소셜미디어 시간 vs 스트레스 지수
            if x_axis == "daily_social_media_hours" and y_axis == "stress_level":
                high_sns = df[df["daily_social_media_hours"] >= 4]["stress_level"].mean()
                low_sns = df[df["daily_social_media_hours"] < 4]["stress_level"].mean()
                diff = abs(high_sns - low_sns)
                st.write(f"• **가장 큰 값 영역**: 하루 4시간 이상 소셜미디어를 헤비하게 사용하는 학생 집단에서 스트레스 평균이 **{high_sns:.1f}점**으로 가장 높게 솟구쳤습니다.")
                st.write(f"• **가장 작은 값 영역**: 4시간 미만으로 조절하는 학생 집단은 평균 **{low_sns:.1f}점**으로 가장 낮았습니다.")
                st.write(f"• **💡 그래프 최종 결론**: 두 집단 간의 스트레스 차이는 무려 **{diff:.1f}점**이나 벌어집니다. 소셜미디어 사용량이 확연한 심리 격차를 만들어내고 있습니다.")
                
            # [케이스 B] 수면 시간 vs 스트레스 지수
            elif x_axis == "sleep_hours" and y_axis == "stress_level":
                low_sleep = df[df["sleep_hours"] <= 5]["stress_level"].mean()
                good_sleep = df[df["sleep_hours"] >= 7]["stress_level"].mean()
                diff = abs(low_sleep - good_sleep)
                st.write(f"• **가장 큰 값 영역**: 하루 5시간 이하로 자는 '수면 부족' 학생들의 스트레스 수치가 평균 **{low_sleep:.1f}점**으로 정점을 찍었습니다.")
                st.write(f"• **가장 작은 값 영역**: 하루 7시간 이상 충분히 수면을 취하는 학생들은 평균 **{good_sleep:.1f}점**으로 가장 안정적이었습니다.")
                st.write(f"• **💡 그래프 최종 결론**: 잠을 잘 자는 학생과 못 자는 학생 사이에 스트레스 수치는 무려 **{diff:.1f}점의 극단적인 격차**가 발생합니다.")
                
            # [케이스 C] 운동 시간 vs 스트레스 지수
            elif x_axis == "physical_activity" and y_axis == "stress_level":
                active = df[df["physical_activity"] >= 1.5]["stress_level"].mean()
                sedentary = df[df["physical_activity"] < 0.5]["stress_level"].mean()
                diff = abs(sedentary - active)
                st.write(f"• **가장 큰 값 영역**: 운동을 거의 안 하는(0.5시간 미만) 학생 집단의 스트레스가 평균 **{sedentary:.1f}점**으로 가장 높았습니다.")
                st.write(f"• **가장 작은 값 영역**: 하루 1.5시간 이상 땀 흘려 운동하는 학생 집단은 평균 **{active:.1f}점**으로 가장 낮게 떨어집니다.")
                st.write(f"• **💡 그래프 최종 결론**: 신체 활동 여부에 따라 학생들의 마음 건강 점수가 **{diff:.1f}점만큼 차이**가 납니다. 운동이 강력한 스트레스 완충벽 역할을 하고 있음을 보여줍니다.")
            
            # [범용 케이스] 사용자가 어떤 임의의 변수를 선택해도 자동으로 연산하여 대조해 주는 스크립트
            else:
                grouped = df.groupby(x_axis)[y_axis].mean()
                top_group = grouped.idxmax()
                top_val = grouped.max()
                bottom_group = grouped.idxmin()
                bottom_val = grouped.min()
                diff = top_val - bottom_val
                
                st.write(f"• **가장 큰 값 영역**: `{x_axis}` 항목이 **[{top_group}]**인 학생들로, `{y_axis}` 평균이 **{top_val:.2f}**로 가장 높게 나타났습니다.")
                st.write(f"• **가장 작은 값 영역**: `{x_axis}` 항목이 **[{bottom_group}]**인 학생들로, `{y_axis}` 평균이 **{bottom_val:.2f}**로 가장 낮게 나타났습니다.")
                st.write(f"• **💡 그래프 최종 결론**: 이 그래프에서 최고점을 찍은 **[{top_group}]** 집단과 최저점을 찍은 **[{bottom_group}]** 집단은 수치상으로 **약 {diff:.2f}만큼의 차이**를 보이고 있습니다.")
                
        except Exception as e:
            st.info("선택하신 데이터 변수 쌍을 실시간 연산 중입니다. 그래프 막대의 높낮이 차이와 상자의 위치 변동을 통해 그룹 간 격차를 바로 확인해 보세요!")

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
