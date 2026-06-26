import streamlit as st
import pandas as pd
import numpy as np
import os

st.title("🤖 4. 인공지능 모델링 및 평가")
st.markdown("학습된 인공지능 모델의 성능을 확인하고, 청소년의 정신 건강 상태를 예측해 봅니다.")

st.markdown("---")
st.subheader("📊 모델 성능 지표 (가상 결과)")

# 가상의 가이드 성능 지표
col1, col2, col3 = st.columns(3)
col1.metric(label="모델 정확도 (Accuracy)", value="87.5%", delta="Random Forest")
col2.metric(label="F1-Score", value="0.86", delta="우수")
col3.metric(label="정밀도 (Precision)", value="88.1%")

st.markdown("---")
st.subheader("🔮 실시간 정신 건강 위험도 예측기")

st.markdown("아래 정보를 입력하면 AI 모델이 위험도를 예측합니다.")

# 입력 폼 구성 (예시 변수들)
predict_col1, predict_col2 = st.columns(2)

with predict_col1:
    stress_level = st.slider("평소 스트레스 지수 (1~10)", 1, 10, 5)
    sleep_hours = st.slider("하루 평균 수면 시간", 3, 12, 7)

with predict_col2:
    sns_hours = st.slider("하루 평균 SNS 사용 시간 (시간)", 0, 8, 2)
    exercise = st.selectbox("주간 운동 횟수", ["0회", "1~2회", "3회 이상"])

if st.button("🧠 위험도 예측 결과 보기"):
    # 간단한 가상 예측 알고리즘 (실제 모델 학습 후엔 model.predict()로 대체)
    score = (stress_level * 0.5) + (8 - sleep_hours) * 0.3 + (sns_hours * 0.2)
    
    st.markdown("### 🎯 AI 예측 결과")
    if score > 5.5:
        st.error(f"🚨 **위험군 (예측 스코어: {score:.2f})**: 전문가 상담이나 휴식이 필요한 상태로 예측됩니다.")
    elif score > 3.5:
        st.warning(f"⚠️ **주의군 (예측 스코어: {score:.2f})**: 약간의 스트레스 관리가 필요합니다.")
    else:
        st.success(f"✅ **정상 (예측 스코어: {score:.2f})**: 비교적 건강한 정신 상태를 유지하고 있습니다.")
