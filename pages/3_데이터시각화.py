import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import os

# 페이지 기본 설정
st.set_page_config(
    page_title="데이터 시각화 (EDA)",
    page_icon="📊",
    layout="wide"
)

st.title("📊 청소년 정신 건강 데이터 자유 시각화 존")
st.markdown("데이터셋에 숨겨진 청소년들의 마음 신호를 5가지 핵심 차트로 자유롭게 탐색합니다.")

data_path = "Teen_Mental_Health_Dataset.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    
    st.markdown("---")
    
    # 수치형 컬럼과 범주형 컬럼 분류 (차트 최적화를 위함)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    all_cols = df.columns.tolist()
    
    # ⚙️ 상단 전역 변수 조작 컨트롤러
    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox("🎯 X축 변수 선택 (기준/원인)", options=all_cols, index=all_cols.index("daily_social_media_hours") if "daily_social_media_hours" in all_cols else 0)
    with col2:
        y_axis = st.selectbox("📈 Y축 변수 선택 (비교/결과)", options=numeric_cols, index=numeric_cols.index("stress_level") if "stress_level" in numeric_cols else 0)
    with col3:
        color_options = ["None"] + all_cols
        color_target = st.selectbox("🎨 색상 그룹 기준 (선택사항)", options=color_options, index=all_cols.index("gender") + 1 if "gender" in all_cols else 0)
        color_var = None if color_target == "None" else color_target

    st.markdown("---")

    # 탭 레이아웃으로 5가지 차트 깔끔하게 분할
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📦 1. 박스 차트 (Box Plot)", 
        "🎻 2. 분포 차트 (Violin Plot)", 
        "📐 3. 히스토그램 (Histogram)", 
        "🔥 4. 상관관계 지도 (Heatmap)", 
        "✨ 5. 흐름 산점도 (Scatter Plot)"
    ])

    # ----------------------------------------------------------------
    # 📦 TAB 1: 박스 차트 (이상치 및 중앙값 확인)
    # ----------------------------------------------------------------
    with tab1:
        st.subheader("📦 집단별 데이터 분포 범위 및 이상치 확인")
        fig_box = px.box(
            df, x=x_axis, y=y_axis, color=color_var,
            title=f"[{x_axis}] 조건에 따른 [{y_axis}]의 사분위 분포 범위",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ----------------------------------------------------------------
    # 🎻 TAB 2: 분포 차트 (바이올린 플롯 - 데이터 밀도 확인)
    # ----------------------------------------------------------------
    with tab2:
        st.subheader("🎻 데이터 밀도 지형 구조 분석")
        fig_violin = px.violin(
            df, x=x_axis, y=y_axis, color=color_var,
            box=True, points="all",
            title=f"[{x_axis}] 대비 [{y_axis}] 데이터의 밀집 패턴",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_violin, use_container_width=True)

    # ----------------------------------------------------------------
    # 📐 TAB 3: 히스토그램 (빈도수 분포)
    # ----------------------------------------------------------------
    with tab3:
        st.subheader("📐 선택한 변수의 학생 수 빈도 측정")
        fig_hist = px.histogram(
            df, x=x_axis, color=color_var,
            marginal="rug", # 하단에 조밀한 분포선 추가
            barmode="overlay", opacity=0.75,
            title=f"[{x_axis}] 데이터의 구간별 청소년 인구 분포 상황",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ----------------------------------------------------------------
    # 🔥 TAB 4: 히트맵 (상관관계 판넬)
    # ----------------------------------------------------------------
    with tab4:
        st.subheader("🔥 어떤 변수들끼리 밀접하게 연결되어 있을까?")
        corr_matrix = df[numeric_cols].corr()
        
        fig_heatmap = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1.0, zmax=1.0,
            title="🧠 청소년 마음 및 생활 패턴 변수 간 피어슨 상관관계 지도"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # ----------------------------------------------------------------
    # ✨ TAB 5: 산점도 (두 변수의 움직임 흐름)
    # ----------------------------------------------------------------
    with tab5:
        st.subheader("✨ 실제 개별 데이터 포인트의 흐름 파악")
        fig_scatter = px.scatter(
            df, x=x_axis, y=y_axis, color=color_var,
            trendline="ols", # 데이터 경향을 보여주는 회귀 트렌드선 자동 매칭
            title=f"[{x_axis}] 변화와 [{y_axis}] 추세선의 방향성",
            opacity=0.7, template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

else:
    st.error("❌ 'Teen_Mental_Health_Dataset.csv' 파일을 불러올 수 없습니다. 프로젝트 루트 폴더에 위치시켜 주세요.")
