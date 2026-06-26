import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- 웹 페이지 설정 ---
st.set_page_config(
    page_title="청소년 정신 건강 데이터 EDA",
    page_icon="📊",
    layout="wide"
)

# --- 커스텀 스타일 (CSS 기법 적용) ---
st.markdown("""
    <style>
        .main-title { font-size: 28px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
        .sub-title { font-size: 15px; color: #4B5563; margin-bottom: 25px; }
        .section-box { padding: 18px; background-color: #F3F4F6; border-radius: 8px; margin-bottom: 20px; }
        .success-box { padding: 12px; background-color: #DEF7EC; color: #03543F; border-radius: 6px; font-size: 14px; }
        .info-box { padding: 12px; background-color: #EBF5FF; color: #1E429F; border-radius: 6px; font-size: 14px; }
    </style>
""", unsafe_gradient=True, unsafe_allow_html=True)

# --- 헤더 섹션 ---
st.markdown('<div class="main-title">📊 3. 데이터 시각화 & 전처리 (EDA)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">데이터를 정제하고, 고도화된 Plotly 시각화를 통해 청소년 정신 건강 데이터 속에 숨겨진 핵심 패턴을 탐색합니다.</div>', unsafe_allow_html=True)

data_path = "Teen_Mental_Health_Dataset.csv"

if os.path.exists(data_path):
    # 최초 데이터 로드 (원본 보존)
    df_raw = pd.read_csv(data_path)
    df = df_raw.copy()
    
    # ---------------------------------------------------------
    # 🛠️ [추가] 고도화된 데이터 전처리 섹션
    # ---------------------------------------------------------
    st.markdown("### 🛠️ 1. 데이터 자동 및 수동 전처리")
    
    with st.expander("🔍 데이터 요약 및 정제 옵션 열기", expanded=True):
        col_info1, col_info2 = st.columns([1, 2])
        
        with col_info1:
            st.markdown("##### 📊 데이터 형상 (Shape)")
            st.metric(label="총 샘플(행) 수", value=f"{df.shape[0]}개")
            st.metric(label="총 변수(열) 수", value=f"{df.shape[1]}개")
            
        with col_info2:
            st.markdown("##### 🧼 결측치(NaN) 감지 및 처리")
            null_counts = df.isnull().sum()
            
            if null_counts.sum() > 0:
                st.warning(f"⚠️ 총 {null_counts.sum()}개의 비어있는 데이터(결측치)가 발견되었습니다.")
                st.dataframe(null_counts[null_counts > 0].to_frame(name="결측치 개수").transpose(), use_container_width=True)
                
                # 사용자가 제어할 수 있는 전처리 라디오 버튼
                missing_action = st.radio(
                    "결측치 처리 전략 선택:",
                    ["그대로 유지", "결측치가 포함된 행 삭제", "수치형은 중간값(Median)/범주형은 최빈값(Mode) 채우기"],
                    horizontal=True
                )
                
                if missing_action == "결측치가 포함된 행 삭제":
                    df = df.dropna()
                    st.markdown(f'<div class="success-box">✅ 결측치 데이터가 완벽히 제거되었습니다! (남은 행: {df.shape[0]}개)</div>', unsafe_allow_html=True)
                elif missing_action == "수치형은 중간값(Median)/범주형은 최빈값(Mode) 채우기":
                    for col in df.columns:
                        if df[col].dtype in ['int64', 'float64']:
                            df[col] = df[col].fillna(df[col].median())
                        else:
                            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
                    st.markdown('<div class="success-box">✅ 수치형은 중간값, 범주형은 최빈값으로 자동 결측치 대체 작업이 완료되었습니다.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">✅ 데이터 검사 완료: 데이터셋에 결측치가 존재하지 않는 깨끗한 상태입니다.</div>', unsafe_allow_html=True)
                
        # 데이터 미리보기 추가
        st.markdown("##### 📄 데이터셋 미리보기 (상위 5개 행)")
        st.dataframe(df.head(5), use_container_width=True)

    # ---------------------------------------------------------
    # 📈 데이터 시각화 제어판 설정
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🎛️ 2. 시각화 제어 패널 설정")
    
    columns = df.columns.tolist()
    
    # 가변적 색상 지정을 위해 데이터에 실제 존재하는 컬럼 필터링 처리
    available_colors = ["None"] + [col for col in ["depression_label", "gender", "platform_usage", "social_interaction_level"] if col in columns]
    
    # 카드 레이아웃 형태의 입력창 구현
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        default_x = columns.index("daily_social_media_hours") if "daily_social_media_hours" in columns else 0
        x_axis = st.selectbox("📌 X축 변수 선택 (독립변수)", options=columns, index=default_x)
    with col2:
        default_y = columns.index("stress_level") if "stress_level" in columns else min(1, len(columns)-1)
        y_axis = st.selectbox("📌 Y축 변수 Choice (수치형 권장)", options=columns, index=default_y)
    with col3:
        color_target = st.selectbox("🎨 색상 구분 그룹 (Group Color)", options=available_colors, index=1 if len(available_colors) > 1 else 0)
    st.markdown('</div>', unsafe_allow_html=True)

    color_var = None if color_target == "None" else color_target
    
    chart_type = st.radio(
        "📊 시각화 할 차트 형태 선택", 
        ["산점도 (Scatter)", "바 차트 (Bar)", "박스 플롯 (Box)", "히스토그램 (Histogram)"],
        horizontal=True
    )
    
    # --- 차트 렌더링 영역 ---
    st.markdown("#### 📈 분석 차트 시각화 결과")
    
    if df.empty:
        st.error("❌ 전처리 필터링 결과, 분석할 수 있는 데이터가 존재하지 않습니다.")
    else:
        try:
            # 예외 방어 처리: 문자열 데이터에 OLS 트렌드라인 설정 시 에러 방지
            is_numeric_x = df[x_axis].dtype in ['int64', 'float64']
            is_numeric_y = df[y_axis].dtype in ['int64', 'float64']
            
            if chart_type == "산점도 (Scatter)":
                # 수치형 변수일 때만 트렌드라인 옵션 제공 활성화
                use_trend = "ols" if (is_numeric_x and is_numeric_y and color_var is None) else None
                fig = px.scatter(
                    df, x=x_axis, y=y_axis, color=color_var,
                    title=f"✨ {x_axis}와 {y_axis}의 상호 연관성 분석",
                    trendline=use_trend,
                    opacity=0.7,
                    marginal_x="box" if is_numeric_x else None # 차트 상단 밀도 분포 추가
                )
                
            elif chart_type == "바 차트 (Bar)":
                # 데이터가 너무 많아 렉 걸리는 현상을 방지하기 위해 집계(평균값) 기준으로 시각화 디자인 변경
                df_grouped = df.groupby([x_axis] + ([color_var] if color_var else [])).mean(numeric_only=True).reset_index()
                fig = px.bar(
                    df_grouped, x=x_axis, y=y_axis, color=color_var,
                    title=f"📊 {x_axis} 그룹별 {y_axis}의 평균값 분석",
                    barmode="group"
                )
                
            elif chart_type == "박스 플롯 (Box)":
                fig = px.box(
                    df, x=x_axis, y=y_axis, color=color_var,
                    title=f"📦 {x_axis} 범주에 따른 {y_axis} 수치 분포 및 이상치(Outlier) 확인",
                    points="all" # 전체 데이터 분산 포인트 같이 표현
                )
                
            elif chart_type == "히스토그램 (Histogram)":
                fig = px.histogram(
                    df, x=x_axis, color=color_var,
                    title=f"📐 {x_axis} 변수의 빈도 및 누적 분포 분석",
                    barmode="overlay",
                    marginal="rug" # 하단 데이터 밀도선 표시
                )
                
            # 디자인 테마 고도화 (글꼴 및 모던 룩앤필 설정)
            fig.update_layout(
                title_font=dict(size=18, family="Malgun Gothic, sans-serif", color="#1E3A8A"),
                hovermode="closest",
                template="plotly_white",
                margin=dict(l=40, r=40, t=60, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) # 범례를 차트 상단으로 정렬
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"⚠️ 선택하신 변수 조합({x_axis} - {y_axis})은 해당 차트 유형으로 렌더링할 수 없습니다. 데이터 타입을 확인해주세요.")
            st.info(f"💡 에러 디버깅 정보: {e}")
    
    # --- 정보 가이드 ---
    with st.expander("💡 데이터 시각화 및 인사이트 도출 가이드"):
        st.markdown(f"""
        * 현재 선택된 축 정보 : **X축** `{x_axis}` / **Y축** `{y_axis}` / **그룹화 색상** `{color_target}`
        * **청소년 데이터 분석 매뉴얼**:
          1. 변수의 성격이 **문자형(범주형)**일 때는 **바 차트**나 **박스 플롯**을 사용하세요.
          2. 변수의 성격이 **숫자형(연속형)**일 때는 **산점도**를 선택하면 OLS 선형 회귀 추세선을 통해 양/음의 상관관계를 파악할 수 있습니다.
          3. 단일 변수의 치우침(왜도)이나 몰림 현상을 보고 싶다면 **히스토그램**이 효과적입니다.
        """)

else:
    st.error("❌ 'Teen_Mental_Health_Dataset.csv' 데이터셋을 찾을 수 없습니다. 현재 작업 경로에 파일이 올바르게 존재하는지 확인해 주세요.")
