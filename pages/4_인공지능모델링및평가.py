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

st.set_page_config(layout="wide") # 화면을 넓게 쓰기 위한 설정

st.title("🤖 4. 인공지능 모델링 및 평가 대시보드")
st.markdown("다양한 AI 모델과 수치를 조절하며 모델의 성능과 생활 패턴 분석 결과를 확인해 보세요.")

# -----------------------------------------------------
# 1. 화면 중간/상단 가로 제어판 (사이드바 제거)
# -----------------------------------------------------
st.markdown("### 🛠️ AI 실험실 설정 제어판")
control_col1, control_col2, control_col3, control_col4 = st.columns(4)

with control_col1:
    selected_model_name = st.selectbox(
        "사용할 AI 모델 선택", 
        ["Random Forest", "Decision Tree", "Logistic Regression"]
    )

with control_col2:
    rf_trees = st.slider("랜덤 포레스트 나무 개수", 10, 200, 100, step=10)

with control_col3:
    dt_depth = st.slider("의사결정나무 최대 깊이", 3, 15, 7)

with control_col4:
    lr_c = st.slider("로지스틱 규제 강도(C값)", 0.01, 10.0, 1.0, step=0.1)

# -----------------------------------------------------
# 2. 데이터 학습 및 평가 연산 (캐싱 적용)
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
    st.markdown("---")
    st.subheader("📊 실시간 분석 결과 리포트")
    
    # -----------------------------------------------------
    # 3. 탭 기능을 활용한 그래프 종류별 창 구분 및 배경 제거
    # -----------------------------------------------------
    # 그래프 스타일을 격자 없는 깔끔한 흰색 배경으로 통일
    sns.set_style("white")
    plt.rcParams['axes.facecolor'] = 'none'
    plt.rcParams['figure.facecolor'] = 'none'

    # 4개의 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs([
        "🥇 모델별 성능 비교", 
        "🔑 핵심 영향 요인 분석", 
        "📈 수면시간별 예측 경향", 
        "🔲 혼동 행렬 격자 점검"
    ])
    
    # --- [탭 1] 세 모델의 성능 비교 그래프 ---
    with tab1:
        st.markdown("##### AI 모델별 성능 지표 비교 그래프")
        names = list(results.keys())
        accs = [results[n]["accuracy"] for n in names]
        f1s = [results[n]["f1_score"] for n in names]
        
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        x = np.arange(len(names))
        width = 0.35
        
        ax1.bar(x - width/2, accs, width, label='Accuracy', color='#4e79a7')
        ax1.bar(x + width/2, f1s, width, label='F1-Score', color='#f28e2b')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names)
        ax1.set_ylim(0, 1.0)
        ax1.legend()
        ax1.grid(False) # 배경 격자 제거
        sns.despine()   # 그래프 테두리 깔끔하게 정리
        st.pyplot(fig1)

    # --- [탭 2] 핵심 영향 요인 그래프 ---
    with tab2:
        st.markdown(f"##### {selected_model_name}의 핵심 영향 요인 분석 그래프")
        current_model = results[selected_model_name]["model"]
        
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        feature_labels_eng = ['SNS Hours', 'Sleep Hours', 'Screen Before Sleep', 'Physical Activity']
        
        if hasattr(current_model, 'feature_importances_'):
            importances = current_model.feature_importances_
            sns.barplot(x=importances, y=feature_labels_eng, ax=ax2, palette="crest")
        elif hasattr(current_model, 'coef_'):
            importances = np.abs(current_model.coef_[0])[:4] 
            sns.barplot(x=importances, y=feature_labels_eng, ax=ax2, palette="flare")
            
        ax2.set_xlabel("Importance")
        ax2.grid(False)
        sns.despine()
        st.pyplot(fig2)

    # --- [탭 3] 예측 경향성 그래프 ---
    with tab3:
        st.markdown(f"##### {selected_model_name}의 수면시간별 예측 경향 그래프")
        y_true = results[selected_model_name]["y_test"]
        y_pred = results[selected_model_name]["y_pred"]
        
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.scatterplot(x=X_test_df['sleep_hours'], y=y_true, alpha=0.4, label='Actual Data', color='gray', ax=ax3)
        sns.lineplot(x=X_test_df['sleep_hours'], y=y_pred, color='red', marker='o', label='AI Predict Trend', ax=ax3, errorbar=None)
        ax3.set_xlabel("Sleep Hours")
        ax3.set_ylabel("Stress Level (1~10)")
        ax3.legend()
        ax3.grid(False)
        sns.despine()
        st.pyplot(fig3)

    # --- [탭 4] 혼동 행렬 히트맵 그래프 ---
    with tab4:
        st.markdown(f"##### {selected_model_name} 혼동 행렬 격자 그래프")
        cm = results[selected_model_name]["confusion_matrix"]
        
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax4)
        ax4.set_xlabel("Predicted Label")
        ax4.set_ylabel("True Label")
        ax4.grid(False)
        st.pyplot(fig4)

    # 하단 전체 스코어 보드 요약
    st.markdown("---")
    st.table(pd.DataFrame([
        {"AI 모델 이름": name, "정확도 (Accuracy)": f"{info['accuracy']*100:.1f}%", "F1-Score (Macro)": f"{info['f1_score']:.2f}"}
        for name, info in results.items()
    ]))
    st.success(f"🏆 **예측에 시뮬레이션 중인 알고리즘:** `{selected_model_name}`")

    # 4. 실시간 스트레스 지수 예측기
    st.markdown("---")
    st.subheader(f"🔮 실시간 스트레스 지수 예측기 (선택된 모델: {selected_model_name})")
    st.markdown("수치 설정이 완료된 모델을 가지고 실제 청소년의 생활 속 스트레스를 예측합니다.")

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
