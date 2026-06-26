import streamlit as st
import pandas as pd
import os

st.title("📌 1. 문제 정의 및 데이터 수집")

st.markdown("""
### 🔍 문제 정의 (Problem Definition)
- **배경**: 학업 스트레스, 대인관계, 가정환경 등 다양한 요인이 청소년 정신 건강에 미치는 영향 분석 필요
- **목적**: 정신 건강에 가장 큰 영향을 주는 요인을 파악하고, 위험군 청소년을 조기에 예측하여 지원 체계 마련
""")

st.markdown("---")
st.subheader("💾 수집된 데이터셋 확인")

data_path = "Teen_Mental_Health_Dataset.csv"

if os.path.exists(data_path):
    # 데이터 로드
    @st.cache_data
    def load_raw_data():
        return pd.read_csv(data_path)
    
    df = load_raw_data()
    
    st.success(f"✅ 데이터셋을 성공적으로 불러왔습니다! (총 {df.shape[0]}행, {df.shape[1]}열)")
    
    # 상위 데이터 미리보기
    st.markdown("#### 📄 원본 데이터 미리보기 (상위 5개 행)")
    st.dataframe(df.head(), use_container_width=True)
    
    # 데이터 구조 정보
    st.markdown("#### 📊 데이터 기본 정보")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**수치형 변수 요약통계량**")
        st.dataframe(df.describe(), use_container_width=True)
    with col2:
        st.markdown("**데이터 컬럼 및 타입 정보**")
        buffer = pd.DataFrame({
            "컬럼명": df.columns,
            "데이터 타입": df.dtypes.astype(str),
            "결측치 개수": df.isnull().sum().values
        })
        st.dataframe(buffer, use_container_width=True)
else:
    st.error(f"❌ `{data_path}` 파일이 루트 폴더에 존재하지 않습니다. 파일을 업로드해 주세요.")
