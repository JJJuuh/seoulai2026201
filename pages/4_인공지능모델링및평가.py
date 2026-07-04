import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

st.title("🤖 4. 인공지능 모델링 및 평가")
st.markdown("다양한 AI 모델을 학습시켜 성능을 비교하고, 가장 우수한 모델을 선택해 청소년의 **스트레스 지수(1~10)**를 예측합니다.")

# 1. 사이드바 조절 기능 (실험 환경 구축)
st.sidebar.header("🛠️ AI 모델 하이퍼파라미터 설정")
rf_trees = st.sidebar.slider("랜덤 포레스트 나무 개수 (n_estimators)", 10, 200, 100, step=10)
dt_depth = st.sidebar.slider("의사결정나무 최대 깊이 (max_depth)", 3, 15, 7)

# 2. 데이터 로드 및 여러 모델 학습/평가 (캐싱 적용)
@st.cache_data
def train_and_compare_models(trees, depth):
    # 데이터 불러오기
    df = pd.read_csv("Teen_Mental_Health_Dataset.csv")
    
    # 독립변수(Features)와 종속변수(Target) 설정
    features = ['daily_social_media_hours', 'sleep_hours', 'screen_time_before_sleep', 'physical_activity']
    target = 'stress_level'
    
    X = df[features]
    y = df[target]
    
    # 데이터 분할 (80% 학습, 20% 검증)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 비교할 3가지 모델 정의
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=depth, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=trees, random_state=42)
    }
    
    model_results = {}
    best_model_name = None
    best_f1 = -1
    
    # 각 모델별 학습 및 평가
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        model_results[name] = {
            "model": model,
            "accuracy": acc,
            "f1_score": f1
        }
        
        # F1-Score 기준으로 가장 우수한 모델 선정
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    return model_results, best_model_name, features

# 모델 학습 실행
try:
    results, best_name, feature_names = train_and_compare_models(rf_trees, dt_depth)
    data_loaded = True
except FileNotFoundError:
    st.error("파일을 찾을 수 없습니다. 'Teen_Mental_Health_Dataset.csv' 파일이 올바른 경로에 있는지 확인해주세요.")
    data_loaded = False

if data_loaded:
    st.markdown("---")
    st.subheader("📊 AI 모델별 성능 비교")
    st.markdown("스트레스 지수를 예측하기 위해 3가지 알고리즘을 학습시킨 결과입니다.")

    # 표(DataFrame) 형태로 출력
    summary_data = []
    for name, info in results.items():
        summary_data.append({
            "AI 모델 이름": name,
            "정확도 (Accuracy)": f"{info['accuracy']*100:.1f}%",
            "F1-Score (Macro)": f"{info['f1_score']:.2f}"
        })
    st.table(pd.DataFrame(summary_data))

    # 최적 모델 결과 강조
    st.success(f"🏆 **선택된 최적의 모델:** `{best_name}` (F1-Score가 가장 높아 최종 예측기로 채택되었습니다.)")

    # 모델 상세 설명 (접어두기)
    with st.expander("ℹ️ 학습된 AI 모델 알고리즘 설명 보기"):
        st.markdown("""
        * **로지스틱 회귀 (Logistic Regression)**
            * 데이터 간의 선형적인 관계를 확률로 계산하여 분류하는 알고리즘입니다. 빠른 연산이 장점입니다.
        * **의사결정나무 (Decision Tree)**
            * 스무고개처럼 조건문 분기를 생성하여 데이터를 쪼개나가는 알고리즘입니다. 직관적이지만 깊어질수록 과적합의 위험이 있습니다.
        * **랜덤 포레스트 (Random Forest)**
            * 수많은 의사결정나무를 만들고 투표(다수결)를 통해 최종 예측을 수행하는 강력한 앙상블 모델입니다.
        """)

    # 3. 피처 중요도(Feature Importance) 시각화 섹션
    st.markdown("---")
    st.subheader("🔑 무엇이 청소년 스트레스에 가장 큰 영향을 줄까요?")
    st.markdown("최적 모델이 청소년의 스트레스를 분석할 때 가장 주목한 생활 패턴 요소 순위입니다.")
    
    best_model = results[best_name]["model"]
    
    # 트리 기반 모델인 경우 피처 중요도 그리기
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # 그래프 한글 깨짐을 방지하기 위해 영문 표기 사용
        feature_labels = {
            'daily_social_media_hours': 'SNS Usage (Hours)',
            'sleep_hours': 'Sleep (Hours)',
            'screen_time_before_sleep': 'Screen Time Before Sleep',
            'physical_activity': 'Physical Activity'
        }
        plot_labels = [feature_labels[feature_names[i]] for i in indices]
        
        # Matplotlib & Seaborn 그래프 그리기
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.barplot(x=importances[indices], y=plot_labels, ax=ax, palette="viridis")
        ax.set_xlabel("Importance Score")
        st.pyplot(fig)
    else:
        st.info("선택된 모델(Logistic Regression)은 선형 계수 기반 모델이므로 트리 방식의 중요도 그래프 대신 수식 가중치로 분석을 진행합니다.")

    # 4. 실시간 스트레스 지수 예측기
    st.markdown("---")
    st.subheader(f"🔮 실시간 스트레스 지수 예측기 (최적 모델: {best_name})")
    st.markdown("아래 정보를 입력하면 선정된 최적의 AI 모델이 스트레스 레벨(1~10)을 예측합니다.")

    predict_col1, predict_col2 = st.columns(2)

    with predict_col1:
        sns_hours = st.slider("하루 평균 SNS 사용 시간 (시간)", 0.0, 10.0, 3.0, step=0.5)
        sleep_hours = st.slider("하루 평균 수면 시간 (시간)", 3.0, 12.0, 7.0, step=0.5)

    with predict_col2:
        screen_time_before_sleep = st.slider("취침 전 전자기기 화면 사용 시간 (시간)", 0.0, 4.0, 1.5, step=0.1)
        exercise_hours = st.slider("하루 평균 운동 시간 (시간)", 0.0, 3.0, 1.0, step=0.1)

    if st.button("🧠 AI 스트레스 예측 결과 보기"):
        input_data = np.array([[sns_hours, sleep_hours, screen_time_before_sleep, exercise_hours]])
        predicted_stress = best_model.predict(input_data)[0]
        
        st.markdown("### 🎯 AI 예측 결과")
        
        if predicted_stress >= 8:
            st.error(f"🚨 **위험 수준: 높음 (예측 스트레스 지수: {predicted_stress} / 10)**")
            st.markdown("AI 분석 결과 스트레스 수치가 매우 높게 예측되었습니다. 충분한 수면을 취하고 전자기기 사용을 줄이는 등의 관리가 필요합니다.")
        elif predicted_stress >= 4:
            st.warning(f"⚠️ **위험 수준: 보통 (예측 스트레스 지수: {predicted_stress} / 10)**")
            st.markdown("적당한 수준의 스트레스를 겪고 있습니다. 누적되지 않도록 취미나 가벼운 운동으로 환기해 주세요.")
        else:
            st.success(f"✅ **위험 수준: 낮음 (예측 스트레스 지수: {predicted_stress} / 10)**")
            st.markdown("스트레스 지수가 아주 안정적입니다! 현재의 건강한 생활 패턴을 잘 유지해 주세요.")
