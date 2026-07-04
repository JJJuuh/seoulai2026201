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

st.set_page_config(layout="wide") 

st.title("🤖 4. 인공지능 모델링 및 평가 대시보드")
st.markdown("다양한 AI 모델과 수치를 조절하며 모델의 성능과 생활 패턴 분석 결과를 확인해 보세요.")

# -----------------------------------------------------
# 1. 화면 중간/상단 가로 제어판
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
    # 3. 탭 구성 및 배경 제거 그래프
    # -----------------------------------------------------
    sns.set_style("white")
    plt.rcParams['axes.facecolor'] = 'none'
    plt.rcParams['figure.facecolor'] = 'none'

    tab1, tab2, tab3, tab4 = st.tabs([
        "🥇 모델별 성능 비교", 
        "🔑 핵심 영향 요인 분석", 
        "📈 수면시간별 예측 경향", 
        "🔲 혼동 행렬 격자 점검"
    ])
    
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
        ax1.grid(False)
        sns.despine()
        st.pyplot(fig1)

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

    with tab4:
        st.markdown(f"##### {selected_model_name} 혼동 행렬 격자 그래프")
        cm = results[selected_model_name]["confusion_matrix"]
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax4)
        ax4.set_xlabel("Predicted Label")
        ax4.set_ylabel("True Label")
        ax4.grid(False)
        st.pyplot(fig4)

    st.markdown("---")
    st.table(pd.DataFrame([
        {"AI 모델 이름": name, "정확도 (Accuracy)": f"{info['accuracy']*100:.1f}%", "F1-Score (Macro)": f"{info['f1_score']:.2f}"}
        for name, info in results.items()
    ]))
    st.success(f"🏆 **예측에 시뮬레이션 중인 알고리즘:** `{selected_model_name}`")

    # -----------------------------------------------------
    # 4. 실시간 스트레스 지수 예측기 및 종합 평가 리포트
    # -----------------------------------------------------
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
        
        st.markdown("---")
        st.markdown("### 🎯 AI 종합 분석 결과 보고서")
        
        # 1단계: 스트레스 등급 판정
        if predicted_stress >= 8:
            st.error(f"🚨 **위험 등급: 고위험군 (예측 스트레스 지수: {predicted_stress} / 10)**")
        elif predicted_stress >= 4:
            st.warning(f"⚠️ **위험 등급: 주의군 (예측 스트레스 지수: {predicted_stress} / 10)**")
        else:
            st.success(f"✅ **위험 등급: 안정군 (예측 스트레스 지수: {predicted_stress} / 10)**")
            
        # 2단계: 라이프스타일 지표 종합 피드백 생성 (룰 베이스 결합 요약)
        st.markdown("#### 📝 입력 데이터 기반 맞춤형 생활 패턴 피드백")
        
        feedback_list = []
        
        if sleep_hours < 6.0:
            feedback_list.append("❌ **심각한 수면 부족:** 하루 수면 시간이 6시간 미만으로, 신체 및 대뇌 회복이 정상적으로 이루어지지 않아 스트레스에 매우 취약한 상태를 유발하고 있습니다.")
        elif sleep_hours >= 8.0:
            feedback_list.append("👍 **적절한 수면:** 권장 수면 시간을 충족하고 있어 스트레스를 방어하는 좋은 기반이 됩니다.")
            
        if sns_hours >= 5.0:
            feedback_list.append("❌ **과도한 SNS 이용:** 하루 5시간 이상의 SNS 사용은 타인과의 비교 심리를 자극하고 주의력을 분산시켜 정신적 피로도를 급격히 높일 수 있습니다.")
            
        if screen_time_before_sleep >= 2.0:
            feedback_list.append("❌ **취침 전 스마트폰 중독 위험:** 잠들기 전 2시간 이상의 전자기기 노출은 멜라토닌 분비를 억제해 얕은 수면을 유발하고 스트레스 저항력을 떨어뜨립니다.")
            
        if exercise_hours < 0.5:
            feedback_list.append("❌ **신체 활동 부족:** 하루 운동량이 30분 미만입니다. 가벼운 유산소 운동은 스트레스 호르몬인 코르티솔을 분해하므로 일상적 신체 활동을 늘려야 합니다.")
        elif exercise_hours >= 1.0:
            feedback_list.append("👍 **활발한 신체 활동:** 하루 1시간 이상의 규칙적인 운동이 스트레스 해소에 든든한 버팀목 역할을 해주고 있습니다.")
            
        # 피드백 리스트 출력
        if feedback_list:
            for item in feedback_list:
                st.markdown(item)
        else:
            st.markdown("정상적이고 균형 잡힌 생활 패턴을 보이고 있습니다.")
            
        # 3단계: 종합 정리 및 조언
        st.markdown("#### 💡 AI 마인드 케어 처방전")
        if predicted_stress >= 8:
            st.markdown(f"현재 AI 모델 분석 결과, 입력하신 생활 패턴은 청소년 고위험 스트레스 군의 전형적인 수치와 일치합니다. 특히 **수면 개선과 취침 전 전자기기 차단**이 가장 시급합니다. 혼자 고민하기보다는 학교 내 위클래스(Wee 클래스)나 청소년 상담 전화(1388)를 통해 대화를 나누어 보는 것을 적극 권장합니다.")
        elif predicted_stress >= 4:
            st.markdown(f"현재는 일상적인 스트레스를 겪고 있는 단계이지만, 불규칙한 생활 습관이 지속된다면 만성 피로나 무기력증으로 이어질 수 있습니다. 주말을 이용해 모바일 디톡스(SNS 끊기)를 실천하거나 운동 시간을 조금 더 확보하여 내면의 에너지를 충전해 주세요.")
        else:
            st.markdown(f"매우 모범적이고 건강한 라이프 사이클을 유지하고 있습니다! AI가 예측한 낮은 스트레스 지수는 우수한 수면 습관과 철저한 전자기기 절제력이 만들어낸 결과입니다. 지금처럼 자신만의 밸런스를 계속 유지해 나가시길 바랍니다.")
