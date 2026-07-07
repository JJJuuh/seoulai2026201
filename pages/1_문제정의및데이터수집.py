import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

# 페이지 설정
st.set_page_config(
    page_title="청소년 정신건강 데이터 전처리",
    page_icon="📌",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📌 청소년 정신건강 데이터 전처리 및 EDA")

# ===== 탭 구성 =====
tab1, tab2, tab3 = st.tabs(["📊 데이터 로드 및 전처리", "🔍 결측치 분석", "📈 기초 통계"])

# ===== TAB 1: 데이터 로드 및 전처리 =====
with tab1:
    st.header("📊 데이터 로드 및 전처리")
    
    data_path = "Teen_Mental_Health_Dataset.csv"
    
    if os.path.exists(data_path):
        # 원본 데이터 로드
        @st.cache_data
        def load_raw_data():
            return pd.read_csv(data_path)
        
        df_raw = load_raw_data()
        df = df_raw.copy()
        
        # 🔴 원본 데이터 현황
        st.subheader("🔴 원본 데이터 현황")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("표본 수", f"{df.shape[0]:,}명")
        col2.metric("변수 개수", f"{df.shape[1]}개")
        col3.metric("완전성", f"{(100 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.1f}%")
        col4.metric("메모리 사용", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        st.write("**원본 데이터 샘플 (상위 5개 행)**")
        st.dataframe(df.head(), use_container_width=True)
        
        # 🟡 데이터 타입 및 결측치 정보
        st.subheader("🟡 데이터 타입 및 결측치 정보")
        
        data_info = pd.DataFrame({
            "변수명": df.columns,
            "데이터타입": df.dtypes.astype(str),
            "결측치": df.isnull().sum().values,
            "결측률(%)": (df.isnull().sum().values / len(df) * 100).round(2),
            "고유값": [df[col].nunique() for col in df.columns]
        })
        st.dataframe(data_info, use_container_width=True)
        
        # 🟢 데이터 전처리
        st.subheader("🟢 데이터 전처리 실행")
        
        preprocessing_log = []
        
        # 1. 결측치 처리
        st.markdown("#### 1️⃣ 결측치 처리")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**결측치 처리 전략**")
            st.write("""
            - 수치형: 중앙값(median) 대체
            - 범주형: 최빈값(mode) 대체
            - 이상치: IQR 방식으로 처리
            """)
        
        with col2:
            # 결측치 처리
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    preprocessing_log.append(f"✓ {col}: {df[col].isnull().sum()} 값을 중앙값({median_val:.2f})으로 대체")
            
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(mode_val, inplace=True)
                    preprocessing_log.append(f"✓ {col}: {df[col].isnull().sum()} 값을 최빈값({mode_val})으로 대체")
        
        # 전처리 로그 출력
        if preprocessing_log:
            for log in preprocessing_log:
                st.success(log)
        else:
            st.info("결측치가 없습니다.")
        
        # 2. 이상치 처리
        st.markdown("#### 2️⃣ 이상치 처리 (IQR 방식)")
        
        outlier_info = []
        numeric_df = df[numeric_cols].copy()
        
        for col in numeric_cols:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)).sum()
            
            if outliers > 0:
                # 이상치를 경계값으로 클리핑
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                outlier_info.append(f"✓ {col}: {outliers}개 이상치 탐지 및 클리핑 처리")
        
        if outlier_info:
            for info in outlier_info:
                st.warning(info)
        else:
            st.info("주요 이상치가 없습니다.")
        
        # 3. 데이터 타입 최적화
        st.markdown("#### 3️⃣ 데이터 타입 최적화")
        
        # depression_label을 숫자로 변환
        if 'depression_label' in df.columns and df['depression_label'].dtype == 'object':
            df['depression_label'] = (df['depression_label'] == 'Yes').astype(int)
            st.success("✓ depression_label: object → int (Yes=1, No=0)")
        
        # gender를 카테고리로 변환
        if 'gender' in df.columns:
            df['gender'] = df['gender'].astype('category')
            st.success("✓ gender: object → category")
        
        # 🟢 전처리 완료 데이터 현황
        st.subheader("🟢 전처리 완료 데이터 현황")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("표본 수", f"{df.shape[0]:,}명")
        col2.metric("결측치", f"{df.isnull().sum().sum()}개")
        col3.metric("완전성", f"{100 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.1f}%")
        col4.metric("메모리 절감", f"{((df_raw.memory_usage(deep=True).sum() - df.memory_usage(deep=True).sum()) / 1024**2):.2f} MB")
        
        st.success("✅ 데이터 전처리 완료!")
        
        # 전처리 완료 데이터 미리보기
        st.write("**전처리 완료 데이터 샘플**")
        st.dataframe(df.head(), use_container_width=True)
        
        # 전처리 완료 데이터 저장
        if st.button("💾 전처리 완료 데이터 저장", use_container_width=True):
            output_path = "Teen_Mental_Health_Cleaned.csv"
            df.to_csv(output_path, index=False)
            st.success(f"✅ 데이터가 저장되었습니다: {output_path}")
    
    else:
        st.error(f"❌ `{data_path}` 파일이 없습니다.")
        st.info("💡 CSV 파일을 업로드해주세요.")

# ===== TAB 2: 결측치 분석 =====
with tab2:
    st.header("🔍 결측치 분석")
    
    if os.path.exists(data_path):
        df = load_raw_data()
        
        # 결측치 요약
        st.subheader("결측치 요약")
        
        missing_df = pd.DataFrame({
            "변수명": df.columns,
            "결측치 수": df.isnull().sum().values,
            "결측률(%)": (df.isnull().sum().values / len(df) * 100).round(2)
        }).sort_values("결측치 수", ascending=False)
        
        missing_df = missing_df[missing_df["결측치 수"] > 0]
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
            
            # 결측치 시각화
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[
                go.Bar(x=missing_df["변수명"], y=missing_df["결측률(%)"], 
                       marker_color='#FF6B6B', text=missing_df["결측률(%)"],
                       textposition='auto')
            ])
            fig.update_layout(
                title="변수별 결측률",
                xaxis_title="변수",
                yaxis_title="결측률 (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ 결측치가 없습니다!")
        
        # 결측 패턴 분석
        st.subheader("결측 패턴 분석")
        
        st.info(f"""
        **전체 셀 개수**: {df.shape[0] * df.shape[1]:,}
        
        **결측 셀 개수**: {df.isnull().sum().sum()}
        
        **결측 비율**: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%
        """)

# ===== TAB 3: 기초 통계 =====
with tab3:
    st.header("📈 기초 통계 분석")
    
    if os.path.exists(data_path):
        df = load_raw_data()
        
        # 수치형 변수 기초 통계
        st.subheader("수치형 변수 기초 통계")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)
        
        # 범주형 변수 분포
        st.subheader("범주형 변수 분포")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            st.write(f"**{col}**")
            value_counts = df[col].value_counts()
            st.bar_chart(value_counts)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
📌 청소년 정신건강 데이터 전처리 시스템 | 2024
</div>
""", unsafe_allow_html=True)
