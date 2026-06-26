import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.title("📊 3. 데이터 시각화 (EDA)")
st.markdown("시각화를 통해 청소년 정신 건강 데이터 속에 숨겨진 패턴을 찾아봅니다.")

data_path = "Teen_Mental_Health_Dataset.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    
    st.markdown("---")
    
    # 1. 데이터 컬럼 분류 (시각화 편의성 제공)
    columns = df.columns.tolist()
    
    # 색상 구분에 사용할 수 있는 범주형/라벨 변수 추천 목록
    color_options = ["None", "depression_label", "gender", "platform_usage", "social_interaction_level"]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # 기본값으로 'daily_social_media_hours' 설정
        default_x = columns.index("daily_social_media_hours") if "daily_social_media_hours" in columns else 0
        x_axis = st.selectbox("X축 변수 선택", options=columns, index=default_x)
    with col2:
        # 기본값으로 'stress_level' 설정
        default_y = columns.index("stress_level") if "stress_level" in columns else min(1, len(columns)-1)
        y_axis = st.selectbox("Y축 변수 선택", options=columns, index=default_y)
    with col3:
        # 데이터의 패턴을 더 쉽게 보기 위한 그룹 색상 축 추가
        color_target = st.selectbox("색상 구분 (Group Color)", options=color_options, index=1)

    # Plotly 전용 옵션: 색상 지정 처리
    color_var = None if color_target == "None" else color_target
    
    # 2. 차트 종류 선택
    chart_type = st.radio(
        "차트 종류 선택", 
        ["산점도 (Scatter)", "바 차트 (Bar)", "박스 플롯 (Box)", "히스토그램 (Histogram)"],
        horizontal=True
    )
    
    st.markdown("#### 📈 생성된 시각화 차트")
    
    # 3. Plotly를 활용한 차트 생성 (Matplotlib 배제)
    if chart_type == "산점도 (Scatter)":
        fig = px.scatter(
            df, x=x_axis, y=y_axis, color=color_var,
            title=f"✨ {x_axis}와 {y_axis}의 상관 관계",
            trendline="ols" if (df[x_axis].dtype != 'object' and df[y_axis].dtype != 'object' and color_var is None) else None
        )
        
    elif chart_type == "바 차트 (Bar)":
        # 범주형 데이터의 평균이나 합계를 보기 좋게 집계 후 출력 기능 지원
        fig = px.bar(
            df, x=x_axis, y=y_axis, color=color_var,
            title=f"📊 {x_axis}별 {y_axis} 데이터 분포",
            barmode="group"
        )
        
    elif chart_type == "박스 플롯 (Box)":
        fig = px.box(
            df, x=x_axis, y=y_axis, color=color_var,
            title=f"📦 {x_axis}에 따른 {y_axis}의 수치 분포 (이상치 확인)"
        )
        
    elif chart_type == "히스토그램 (Histogram)":
        # 단일 변수의 분포 혹은 그룹별 분포를 볼 때 유용
        fig = px.histogram(
            df, x=x_axis, color=color_var,
            title=f"📐 {x_axis}의 데이터 빈도수(Histogram) 분석",
            barmode="overlay"
        )
        
    # Plotly 차트 레이아웃 깔끔하게 조정
    fig.update_layout(
        title_font_size=16,
        hovermode="closest",
        template="plotly_white"  # 깨끗한 화이트 테마 적용
    )
        
    st.plotly_chart(fig, use_container_width=True)
    
    # 4. 간단한 데이터 요약 팁 제공
    with st.expander("💡 데이터 시각화 분석 가이드"):
        st.write(f"- 현재 **X축**: `{x_axis}` / **Y축**: `{y_axis}` 상태입니다.")
        st.write("- Plotly 차트 우측 상단의 툴바를 이용해 **확대(Zoom), 저장(Camera), 초기화** 기능을 사용할 수 있습니다.")
        st.write("- 범주형 데이터(`gender`, `platform_usage` 등)를 X축에 두고, 수치형 데이터(`stress_level` 등)를 Y축에 둔 뒤 **'박스 플롯'**을 그리면 집단 간 차이가 잘 보입니다.")

else:
    st.error("❌ 'Teen_Mental_Health_Dataset.csv' 데이터셋을 찾을 수 없습니다. 파일 경로를 확인해 주세요.")
