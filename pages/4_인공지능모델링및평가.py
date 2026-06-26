import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.title("🤖 4. 인공지능 모델링 및 평가")
st.markdown("업로드된 청소년 정신 건강 데이터를 활용해 모델을 학습시키고, 실시간으로 **스트레스 지수(1~10)**를 예측해 봅니다.")

# 1. 데이터 로드 및 모델 학습 (캐싱을 통해 속도 최적화)
@st.cache_data
def load_and_train_model():
    # 데이터 불러오기
    df = pd.read_csv("Teen_Mental_Health_Dataset.csv")
    
    # 사용할 독립변수(Features)와 종속변수(Target) 설정
    # 주 성격이 비슷한 수치형 변수들을 Feature로 활용합니다.
    features = ['daily_social_media_hours', 'sleep_hours', 'screen_time_before_sleep', 'physical_activity']
    target = 'stress_level'  # 종속변수 변경
    
    X = df[features]
    y = df[target]
    
    # 데이터 분할 (학습용 80%, 검증용 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 모델 생성 및 학습 (다중 분류를 위한 RandomForestClassifier)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 예측 및 평가 지표 계산 (다중 분류이므로 average='macro' 적용)
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
        "f1": f1_score(y_test, y_pred, average='macro', zero_division=0)
    }
    
    return model, metrics

# 모델 학습 실행
try:
    model, metrics = load_and_train_model()
    data_loaded = True
except FileNotFoundError:
    st.error("파일을 찾을 수 없습니다. 'Teen_Mental_Health_Dataset.csv' 파일이 올바른 경로에 있는지 확인해주세요.")
    data_loaded = False

if data_loaded:
    st.markdown("---")
    st.subheader("📊 실제 모델 평가 지표 (Random Forest 다중 분류 모델)")
    st.markdown("데이터셋을 학습시켜 청소년의 스트레스 지수(1~10)를 맞추는 분류 모델의 테스트 결과입니다.")

    # 평가 지표 시각화 (실제 데이터 학습 결과 반영)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="정확도 (Accuracy)", value=f"{metrics['accuracy']*100:.1f}%")
    col2.metric(label="정밀도 (Precision)", value=f"{metrics['precision']:.2f}")
    col3.metric(label="재현율 (Recall)", value=f"{metrics['recall']:.2f}")
    col4.metric(label="F1-Score", value=f"{metrics['f1']:.2f}")

    st.markdown("---")
    st.subheader("🔮 실시간 스트레스 지수 예측기")
    st.markdown("아래 청소년의 생활 패턴 정보를 입력하면, 학습된 AI 모델이 **스트레스 레벨(1~10)**을 예측합니다.")

    # 입력 폼 구성 (스트레스를 예측하기 위한 독립변수 입력)
    predict_col1, predict_col2 = st.columns(2)

    with predict_col1:
        sns_hours = st.slider("하루 평균 SNS 사용 시간 (시간)", 0.0, 10.0, 3.0, step=0.5)
        sleep_hours = st.slider("하루 평균 수면 시간 (시간)", 3.0, 12.0, 7.0, step=0.5)

    with predict_col2:
        screen_time_before_sleep = st.slider("취침 전 전자기기 화면 사용 시간 (시간)", 0.0, 4.0, 1.5, step=0.1)
        exercise_hours = st.slider("하루 평균 운동 시간 (시간)", 0.0, 3.0, 1.0, step=0.1)

    if st.button("🧠 AI 스트레스 예측 결과 보기"):
        # 입력 데이터를 모델 피처 형태로 변환 (순서 일치 필수)
        # ['daily_social_media_hours', 'sleep_hours', 'screen_time_before_sleep', 'physical_activity']
        input_data = np.array([[sns_hours, sleep_hours, screen_time_before_sleep, exercise_hours]])
        
        # 모델 예측 (1~10 사이의 스트레스 레벨 반환)
        predicted_stress = model.predict(input_data)[0]
        
        st.markdown("### 🎯 AI 예측 결과")
        
        # 스트레스 단계별 시각적 분기 처리 (1~3: 낮음, 4~7: 보통, 8~10: 높음)
        if predicted_stress >= 8:
            st.error(f"🚨 **위험 수준: 높음 (예측 스트레스 지수: {predicted_stress} / 10)**")
            st.markdown("AI 분석 결과 스트레스 수치가 매우 높게 예측되었습니다. 과도한 스마트폰 및 전자기기 사용을 줄이고 일찍 숙면을 취하거나 위클래스 상담을 받아보는 것을 추천합니다.")
        elif predicted_stress >= 4:
            st.warning(f"⚠️ **위험 수준: 보통 (예측 스트레스 지수: {predicted_stress} / 10)**")
            st.markdown("적당한 수준의 스트레스를 겪고 있는 상태입니다. 가벼운 운동이나 취미 활동을 통해 스트레스가 누적되지 않도록 관리해 주세요.")
        else:
            st.success(f"✅ **위험 수준: 낮음 (예측 스트레스 지수: {predicted_stress} / 10)**")
            st.markdown("스트레스 지수가 매우 안정적입니다! 현재의 건강한 생활 패턴과 수면 습관을 꾸준히 유지해 주세요.")
