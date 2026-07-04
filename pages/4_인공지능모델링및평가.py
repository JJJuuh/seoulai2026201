import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

st.set_page_config(layout="wide") # 다양한 그래프를 넓게 보기 위해 와이드 레이아웃 적용

st.title("🤖 4. 인공지능 모델링 및 시각화 평가")
st.markdown("다양한 AI 모델과 수치를 직접 조절하며, 모델의 내부 동작과 평가 결과를 다양한 시각적 그래프로 확인해 보세요.")

# -----------------------------------------------------
# 1. 사이드바 제어판 (모델 선택 및 하이퍼파라미터 조절)
# -----------------------------------------------------
st.sidebar.header("🛠️ AI 실험실 설정")

# 사용자가 직접 예측에 사용할 메인 모델 선택
selected_model_name = st.sidebar.selectbox(
    "실시간 예측에 사용할 AI 모델 선택", 
    ["Random Forest", "Decision Tree", "Logistic Regression"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 모델별 세부 수치(하이퍼파라미터) 조절")
rf_trees = st.sidebar.slider("랜덤 포레스트 나무 개수 (n_estimators)", 10, 200, 100, step=10)
dt_depth = st.sidebar.slider("의사결정나무 최대 깊이 (max_depth)", 3, 15, 7)
lr_c = st.sidebar.slider("로지스틱 규제 강도 (C값 - 클수록 규제 약함)", 0.01, 10.0, 1.0, step=0.1)

# -----------------------------------------------------
# 2. 데이터 학습 및 전체 모델 평가 데이터 생성 (캐싱)
# -----------------------------------------------------
@st.cache_data
def train_and_evaluate(trees, depth, c_val):
    df = pd.read_csv("Teen_Mental_Health_Dataset.csv")
    
    features = ['daily_social_media_hours', 'sleep_hours', 'screen_time_before_sleep', 'physical_activity']
    target = 'stress_level'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 사용자가 조절한 하이퍼파라미터 주입
    models = {
        "Logistic Regression": LogisticRegression(C=c_val, max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=depth, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=trees, random_state=42)
    }
    
    model_results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        model_results[name] = {
            "model": model,
            "accuracy": acc,
            "f1_score": f1,
            "confusion_matrix": cm,
            "y_test": y_test,
            "y_pred": y_pred
        }
        
    return model_results, features, X_test

try:
    results, feature_names, X_test_df = train_and_evaluate(rf_trees, dt_depth, lr_c)
    data_loaded = True
except FileNotFoundError:
    st.error("데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    data_loaded = False

if data_loaded:
    # -----------------------------------------------------
    # 3. 다양한 시각화 대시보드 배치 (2x2 구조)
    # -----------------------------------------------------
    st.subheader("📊 실시간 변동 데이터 시각화 리포트")
    st.markdown("왼쪽 제어판에서 수치를 바꾸면 아래 그래프들이 실시간으로 다시 그려집니다.")
    
    # 레이아웃을 위한 2개의 행(Row)과 각 2개의 열(Column) 구성
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    # --- [그래프 1] 세 모델의 성능 비교 바 차트 ---
    with row1_col1:
        st.markdown("##### ① AI 모델별 성능 지표 비교")
        names = list(results.keys())
        accs = [results[n]["accuracy"] for n in names]
        f1s = [results[n]["f1_score"] for n in names]
        
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        x = np.arange(len(names))
        width = 0.35
        
        ax1.bar(x - width/2, accs, width, label='Accuracy', color='#4e79a7')
        ax1.bar(x + width/2, f1s, width, label='F1-Score', color='#f28e2b')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names)
        ax1.set_ylim(0, 1.0)
        ax1.legend()
        st.pyplot(fig1)
        st.caption("세 알고리즘의 정확도와 전체 균형 점수(F1)를 직접 비교합니다.")

    # --- [그래프 2] 선택된 모델의 피처 중요도 차트 ---
    with row1_col2:
        st.markdown(f"##### ② {selected_model_name}의 핵심 영향 요인")
        current_model = results[selected_model_name]["model"]
        
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        feature_labels_eng = ['SNS Hours', 'Sleep Hours', 'Screen Before Sleep', 'Physical Activity']
        
        if hasattr(current_model, 'feature_importances_'):
            importances = current_model.feature_importances_
            sns.barplot(x=importances, y=feature_labels_eng, ax=ax2, palette="crest")
            ax2.set_xlabel("Importance")
        elif hasattr(current_model, 'coef_'):
            # 로지스틱 회귀의 경우 계수(coefficients)의 절대값을 중요도로 시각화
            importances = np.abs(current_model.coef_[0])[:4] 
            sns.barplot(x=importances, y=feature_labels_eng, ax=ax2, palette="flare")
            ax2.set_xlabel("Absolute Coefficient")
        
        st.pyplot(fig2)
        st.caption("선택한 모델이 스트레스를 판단할 때 가장 중요하게 체크한 요소입니다.")

    # --- [그래프 3] 실제값 vs AI 예측값 분포 차트 ---
    with row2_col1:
        st.markdown(f"##### ③ {selected_model_name}의 예측 트렌드 분석")
        y_true = results[selected_model_name]["y_test"]
        y_pred = results[selected_model_name]["y_pred"]
        
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        # 수면시간 대비 스트레스 지수 분포 위에 예측 경향을 산점도로 시각화
        sns.scatterplot(x=X_test_df['sleep_hours'], y=y_true, alpha=0.5, label='Actual Data', color='gray', ax=ax3)
        sns.lineplot(x=X_test_df['sleep_hours'], y=y_pred, color='red', marker='o', label='AI Predict Trend', ax=ax3, errorbar=None)
        ax3.set_xlabel("Sleep Hours")
        ax3.set_ylabel("Stress Level (1~10)")
        ax3.legend()
        st.pyplot(fig3)
        st.caption("실제 데이터의 분포(회색) 대비 AI가 수면시간별로 스트레스를 어떻게 예측(빨간선)하는지 트렌드를 보여줍니다.")

    # --- [그래프 4] 혼동 행렬(Confusion Matrix) 히트맵 ---
    with row2_col2:
        st.markdown(f"##### ④ {selected_model_name} 혼동 행렬 (오답 격자판)")
        cm = results[selected_model_name]["confusion_matrix"]
        
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        # 데이터가 깔끔하게 보이도록 상위 몇 개 클래스 위주 혹은 전체 격자 히트맵 시각화
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax4)
        ax4.set_xlabel("Predicted Label")
        ax4.set_ylabel("True Label")
        st.pyplot(fig4)
        st.caption("대각선 방향에 숫자가 몰려있을수록 AI가 오답 없이 정확하게 분류해냈음을 의미합니다.")

    # -----------------------------------------------------
    # 4. 실시간 스트레스 지수 예측기 (선택된 모델 활용)
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader(f"🔮 실시간 스트레스 지수 예측기 (선택된 모델: {selected_model_name})")
    st.markdown("선택 및 수치 튜닝이 완료된 최적의 모델을 가지고 실제 청소년의 생활 속 스트레스를 시뮬레이션합니다.")

    predict_col1, predict_col2 = st.columns(2)

    with predict_col1:
        sns_hours = st.slider("하루 평균 SNS 사용 시간 (시간)", 0.0, 10.0, 3.0, step=0.5)
        sleep_hours = st.slider("하루 평균 수면 시간 (시간)", 3.0, 12.0, 7.0, step=0.5)

    with predict_col2:
        screen_time_before_sleep = st.slider("취침 전 전자기기 화면 사용 시간 (시간)", 0.0, 4.0, 1.5, step=0.1)
        exercise_hours = st.slider("하루 평균 운동 시간 (시간)", 0.0, 3.0, 1.0, step=0.1)

    if st.button("🧠 AI 스트레스 예측 결과 보기"):
        input_data = np.array([[sns_hours, sleep_hours, screen_time_before_sleep, exercise_hours]])
        active_model = results[selected_model_name]["model"]
        predicted_stress = active_model.predict(input_data)[0]
        
        st.markdown("### 🎯 AI 예측 결과")
        if predicted_stress >= 8:
            st.error(f"🚨 **위험 수준: 높음 (예측 스트레스 지수: {predicted_stress} / 10)**")
        elif predicted_stress >= 4:
            st.warning(f"⚠️ **위험 수준: 보통 (예측 스트레스 지수: {predicted_stress} / 10)**")
        else:
            st.success(f"✅ **위험 수준: 낮음 (예측 스트레스 지수: {predicted_stress} / 10)**")
