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

st.set_page_config(layout="wide") # 그래프들을 와이드 화면에 넓게 배치

st.title("🤖 4. 인공지능 모델링 및 평가 대시보드")
st.markdown("다양한 AI 모델과 수치를 직접 조절하며, 모델의 내부 분류 작동 방식과 평가 결과를 여러 종류의 그래프로 확인해 보세요.")

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
    # 3. 그래프 대시보드 배치 (2x2 구조로 화면에 직접 랜더링)
    # -----------------------------------------------------
    st.subheader("📊 실시간 분석 그래프 리포트")
    st.markdown("왼쪽 제어판에서 수치를 바꾸면 데이터셋을 즉시 재학습하여 아래 그래프들이 실시간으로 갱신됩니다.")
    
    # 레이아웃 분할
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    # --- [그래프 1] 세 모델의 성능 비교 막대 그래프 ---
    with row1_col1:
        st.markdown("##### ① AI 모델별 성능 지표 비교 그래프")
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
        st.caption("세 알고리즘의 정확도와 균형 점수(F1)를 직접 비교하는 막대 그래프입니다.")

    # --- [그래프 2] 선택된 모델의 피처 중요도 가로 막대 그래프 ---
    with row1_col2:
        st.markdown(f"##### ② {selected_model_name}의 핵심 영향 요인 분석 그래프")
        current_model = results[selected_model_name]["model"]
        
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        feature_labels_eng = ['SNS Hours', 'Sleep Hours', 'Screen Before Sleep', 'Physical Activity']
        
        if hasattr(current_model, 'feature_importances_'):
            importances = current_model.feature_importances_
            sns.barplot(x=importances, y=feature_labels_eng, ax=ax2, palette="crest")
            ax2.set_xlabel("Importance")
        elif hasattr(current_model, 'coef_'):
            importances = np.abs(current_model.coef_[0])[:4] 
            sns.barplot(x=importances, y=feature_labels_eng, ax=ax2, palette="flare")
            ax2.set_xlabel("Absolute Coefficient")
        
        st.pyplot(fig2)
        st.caption("선택한 알고리즘이 스트레스를 판단할 때 가장 중요하게 반영한 생활 패턴 순위 그래프입니다.")

    # --- [그래프 3] 수면시간 대비 예측값 경향성 그래프 ---
    with row2_col1:
        st.markdown(f"##### ③ {selected_model_name}의 수면시간별 예측 경향 그래프")
        y_true = results[selected_model_name]["y_test"]
        y_pred = results[selected_model_name]["y_pred"]
        
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=X_test_df['sleep_hours'], y=y_true, alpha=0.5, label='Actual Data', color='gray', ax=ax3)
        sns.lineplot(x=X_test_df['sleep_hours'], y=y_pred, color='red', marker='o', label='AI Predict Trend', ax=ax3, errorbar=None)
        ax3.set_xlabel("Sleep Hours")
        ax3.set_ylabel("Stress Level (1~10)")
        ax3.legend()
        st.pyplot(fig3)
        st.caption("실제 데이터 분포(회색 점) 위에 AI의 수면시간별 스트레스 예측 경향(빨간 선)을 나타낸 그래프입니다.")

    # --- [그래프 4] 혼동 행렬(Confusion Matrix) 히트맵 매트릭스 그래프 ---
    with row2_col2:
        st.markdown(f"##### ④ {selected_model_name} 혼동 행렬 격자 그래프")
        cm = results[selected_model_name]["confusion_matrix"]
        
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax4)
        ax4.set_xlabel("Predicted Label")
        ax4.set_ylabel("True Label")
        st.pyplot(fig4)
        st.caption("실제 스트레스 점수(True)와 모델이 예측한 점수(Predicted)가 일치하는 칸을 밝게 표시하는 매트릭스 그래프입니다.")

    # 최종 선정 안내
    st.table(pd.DataFrame([
        {"AI 모델 이름": name, "정확도 (Accuracy)": f"{info['accuracy']*100:.1f}%", "F1-Score (Macro)": f"{info['f1_score']:.2f}"}
        for name, info in results.items()
    ]))
    st.success(f"🏆 **예측에 적용 중인 알고리즘:** `{selected_model_name}`")

    # 모델 상세 설명 (접어두기)
    with st.expander("ℹ️ 학습된 AI 모델 알고리즘 설명 보기"):
        st.markdown("""
        * **로지스틱 회귀 (Logistic Regression)**
            * 데이터 간의 선형적인 관계를 확률로 계산하여 분류하는 알고리즘입니다. 빠른 연산이 장점입니다.
        * **의사결정나무 (Decision Tree)**
            * 스무고개처럼 조건문 분기를 생성하여 데이터를 쪼개나가는 알고리즘입니다. 깊어질수록 과적합의 위험이 있습니다.
        * **랜덤 포레스트 (Random Forest)**
            * 수많은 의사결정나무를 만들고 투표(다수결)를 통해 최종 예측을 수행하는 강력한 앙상블 모델입니다.
        """)

    # 4. 실시간 스트레스 지수 예측기
    st.markdown("---")
    st.subheader(f"🔮 실시간 스트레스 지수 예측기 (선택된 모델: {selected_model_name})")
    st.markdown("수치 튜닝이 완료된 모델을 가지고 실제 청소년의 생활 속 스트레스를 시뮬레이션합니다.")

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
            st.markdown("AI 분석 결과 스트레스 수치가 매우 높게 예측되었습니다. 충분한 수면을 취하고 전자기기 사용을 줄이는 등의 관리가 필요합니다.")
        elif predicted_stress >= 4:
            st.warning(f"⚠️ **위험 수준: 보통 (예측 스트레스 지수: {predicted_stress} / 10)**")
            st.markdown("적당한 수준의 스트레스를 겪고 있습니다. 누적되지 않도록 취미나 가벼운 운동으로 환기해 주세요.")
        else:
            st.success(f"✅ **위험 수준: 낮음 (예측 스트레스 지수: {predicted_stress} / 10)**")
            st.markdown("스트레스 지수가 아주 안정적입니다! 현재의 건강한 생활 패턴을 잘 유지해 주세요.")
