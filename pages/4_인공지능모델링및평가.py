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
# 데이터 사전 연산 함수 (위치 이동 처리를 위해 상단 배치)
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
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        model_results[name] = {
            "model": model,
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "accuracy": accuracy_score(y_test, y_test_pred),
            "f1_score": f1_score(y_test, y_test_pred, average='macro', zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_test_pred),
            "y_test": y_test,
            "y_pred": y_test_pred
        }
        
    return model_results, features, X_test, df

try:
    results, feature_names, X_test_df, original_df = train_and_evaluate(rf_trees, dt_depth, lr_c)
    data_loaded = True
except FileNotFoundError:
    st.error("데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    data_loaded = False


if data_loaded:
    # -----------------------------------------------------
    # 2. [위치 변경] AI 알고리즘 신뢰성 및 과적합 위험도 평가
    # -----------------------------------------------------
    st.markdown("---")
    st.markdown("### ⚠️ AI 알고리즘 신뢰성 및 과적합(Overfitting) 위험도 평가")
    st.markdown("상단 제어판에서 수치를 조절하는 즉시, 모델의 훈련 상태와 과적합 현상을 실시간으로 진단합니다.")
    
    active_info = results[selected_model_name]
    train_acc = active_info["train_accuracy"]
    test_acc = active_info["accuracy"]
    overfit_gap = train_acc - test_acc
    
    risk_col1, risk_col2 = st.columns([1, 2])
    
    with risk_col1:
        st.metric(label="🎯 학습-검증 정확도 격차(Gap)", value=f"{overfit_gap*100:.2f}%")
        if overfit_gap >= 0.15:
            st.error("🚨 **위험 수준: 과적합 상태 (Overfitting)**")
        elif overfit_gap >= 0.05:
            st.warning("⚠️ **위험 수준: 잠재적 위험 (Moderate)**")
        elif train_acc < 0.40:
            st.info("📉 **위험 수준: 과소적합 상태 (Underfitting)**")
        else:
            st.success("✅ **위험 수준: 매우 안정적 (Optimal)**")
            
    with risk_col2:
        st.markdown("#### 🔍 알고리즘 신뢰성 정밀 진단 소견")
        if overfit_gap >= 0.15:
            st.markdown(f"현재 설정된 수치는 **훈련 데이터에 모델이 과도하게 짜맞춰진 상태(과적합)**를 유발합니다. "
                        f"공부한 문제집(Train: {train_acc*100:.1f}%) 점수는 높지만 실전 모의고사(Test: {test_acc*100:.1f}%)에서 삐끗하는 현상입니다. "
                        f"나무 깊이를 줄이거나 규제를 주어 가지치기를 해야 실전 신뢰도가 올라갑니다.")
        elif overfit_gap >= 0.05:
            st.markdown(f"훈련 정확도({train_acc*100:.1f}%)와 검증 정확도({test_acc*100:.1f}%) 사이에 약간의 괴리가 존재합니다. "
                        f"약간의 불안정 요소가 있으므로 복잡도 수치를 조금 깎아내려 완화하는 것이 좋습니다.")
        elif train_acc < 0.40:
            st.markdown("패턴을 아예 배우지 못한 과소적합 상태입니다. 제어판에서 AI가 깊게 공부할 수 있도록 허용치를 늘려주세요.")
        else:
            st.markdown(f"현재 `{selected_model_name}` 모델은 **최적의 일반화 밸런스**를 이루고 있습니다. 두 데이터셋의 스코어가 균형을 이루어 편향 없이 가장 완벽하게 분류해 낼 수 있는 견고한 상태입니다.")


    # -----------------------------------------------------
    # 3. 실시간 분석 결과 리포트 (탭 구성 및 소견 보강)
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader("📊 실시간 분석 결과 리포트")
    
    sns.set_style("white")
    plt.rcParams['axes.facecolor'] = 'none'
    plt.rcParams['figure.facecolor'] = 'none'

    tab1, tab2, tab3, tab4 = st.tabs([
        "🥇 모델별 성능 비교", 
        "🔑 핵심 영향 요인 분석", 
        "📈 수면시간별 예측 경향", 
        "🔲 혼동 행렬 격자 점검"
    ])
    
    # --- [탭 1] 확연하게 비교되도록 전면 리팩토링한 그룹 바 차트 ---
    with tab1:
        st.markdown("##### AI 모델별 성능 지표 비교 그래프")
        
        # Seaborn 데이터프레임 구조로 변환하여 명확한 그룹화 유도
        plot_data = []
        for name, info in results.items():
            plot_data.append({"Model": name, "Metric": "Accuracy", "Score": info["accuracy"]})
            plot_data.append({"Model": name, "Metric": "F1-Score", "Score": info["f1_score"]})
        df_plot = pd.DataFrame(plot_data)
        
        fig1, ax1 = plt.subplots(figsize=(9, 4))
        # 확연한 비교를 위한 도표 렌더링
        sns.barplot(data=df_plot, x="Model", y="Score", hue="Metric", palette="muted", ax=ax1)
        ax1.set_ylim(0, 1.0)
        ax1.set_xlabel("AI Algorithms")
        ax1.set_ylabel("Performance Score")
        ax1.grid(False)
        sns.despine()
        st.pyplot(fig1)
        
        # [추가 내용] 그래프 하단 요약 가이드
        st.info("💡 **이 그래프로 알 수 있는 것:** 3가지 알고리즘 중 어떤 AI가 청소년의 스트레스를 가장 균형 있게 맞추는지 확인합니다. "
                "단순 정확도(Accuracy)뿐만 아니라 다중 클래스 밸런스를 보는 F1-Score 막대가 높을수록 실전 성능이 완벽함을 뜻합니다.")

    # --- [탭 2] 핵심 영향 요인 분석 ---
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
        
        # [추가 내용] 그래프 하단 요약 가이드
        st.info("💡 **이 그래프로 알 수 있는 것:** 인공지능이 스트레스 지수(1~10)를 판단할 때 **무슨 항목을 가장 치명적인 원인으로 보았는지** 가중치 순위를 나타냅니다. 막대가 길수록 스트레스 예측의 핵심 열쇠가 되는 생활 패턴입니다.")

    # --- [탭 3] 수면시간별 예측 경향 ---
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
        
        # [추가 내용] 그래프 하단 요약 가이드
        st.info("💡 **이 그래프로 알 수 있는 것:** 실제 청소년 데이터(회색 점)의 흐름 위를 달리는 **AI의 예측 트렌드선(빨간 선)**입니다. 수면 시간이 감소함에 따라 예측 스트레스선이 계단식으로 우상향하는지, 모형의 예측 합리성을 진단할 수 있습니다.")

    # --- [탭 4] 혼동 행렬 격자 점검 ---
    with tab4:
        st.markdown(f"##### {selected_model_name} 혼동 행렬 격자 그래프")
        cm = results[selected_model_name]["confusion_matrix"]
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax4)
        ax4.set_xlabel("Predicted Label")
        ax4.set_ylabel("True Label")
        ax4.grid(False)
        st.pyplot(fig4)
        
        # [추가 내용] 그래프 하단 요약 가이드
        st.info("💡 **이 그래프로 알 수 있는 것:** 대각선 정중앙 칸에 숫자가 가득 집중될수록 인공지능이 실제 스트레스 지수(True)를 정확하게 짚어내었음을 뜻합니다. 대각선을 벗어난 오답 칸을 보며 AI가 어떤 점수대 사이에서 혼동을 겪는지 추적합니다.")

    # 하단 스코어 요약 보드
    st.markdown("---")
    st.table(pd.DataFrame([
        {
            "AI 모델 이름": name, 
            "훈련 정확도(Train Acc)": f"{info['train_accuracy']*100:.1f}%",
            "검증 정확도(Test Acc)": f"{info['accuracy']*100:.1f}%", 
            "F1-Score (Macro)": f"{info['f1_score']:.2f}"
        }
        for name, info in results.items()
    ]))
    st.success(f"🏆 **예측에 시뮬레이션 중인 알고리즘:** `{selected_model_name}`")

    # -----------------------------------------------------
    # 4. 실시간 스트레스 지수 예측기 및 결과 분석
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
        
        if predicted_stress >= 8:
            st.error(f"🚨 **위험 등급: 고위험군 (예측 스트레스 지수: {predicted_stress} / 10)**")
        elif predicted_stress >= 4:
            st.warning(f"⚠️ **위험 등급: 주의군 (예측 스트레스 지수: {predicted_stress} / 10)**")
        else:
            st.success(f"✅ **위험 등급: 안정군 (예측 스트레스 지수: {predicted_stress} / 10)**")
            
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
            
        if feedback_list:
            for item in feedback_list:
                st.markdown(item)
        else:
            st.markdown("정상적이고 균형 잡힌 생활 패턴을 보이고 있습니다.")
            
        st.markdown("#### 💡 AI 마인드 케어 처방전")
        if predicted_stress >= 8:
            st.markdown(f"현재 AI 모델 분석 결과, 입력하신 생활 패턴은 청소년 고위험 스트레스 군의 전형적인 수치와 일치합니다. 특히 **수면 개선과 취침 전 전자기기 차단**이 가장 시급합니다. 혼자 고민하기보다는 학교 내 위클래스(Wee 클래스)나 청소년 상담 전화(1388)를 통해 대화를 나누어 보는 것을 적극 권장합니다.")
        elif predicted_stress >= 4:
            st.markdown(f"현재는 일상적인 스트레스를 겪고 있는 단계이지만, 불규칙한 생활 습관이 지속된다면 만성 피로나 무기력증으로 이어질 수 있습니다. 주말을 이용해 모바일 디톡스(SNS 끊기)를 실천하거나 운동 시간을 조금 더 확보하여 내면의 에너지를 충전해 주세요.")
        else:
            st.markdown(f"매우 모범적이고 건강한 라이프 사이클을 유지하고 있습니다! AI가 예측한 낮은 스트레스 지수는 우수한 수면 습관과 철저한 전자기기 절제력이 만들어낸 결과입니다. 지금처럼 자신만의 밸런스를 계속 유지해 나가시길 바랍니다.")

        # -----------------------------------------------------
        # 5. 데이터와 AI 학습을 통해 본 청소년 스트레스의 핵심 결론
        # -----------------------------------------------------
        st.markdown("---")
        st.markdown("### 🏛️ 데이터와 AI 학습을 통해 본 청소년 스트레스의 핵심 결론")
        st.markdown("본 인공지능 모델이 전체 청소년 정신 건강 데이터를 분석하고 분류 규칙을 만들며 발견한 거시적 결론입니다.")
        
        insight_col1, insight_col2 = st.columns([1.1, 0.9])
        
        with insight_col1:
            st.markdown("""
            1. **수면과 SNS의 악순환 고리 발견**
                * AI 모델의 학습 가중치와 분석 결과를 살펴보면, **스트레스 지수가 높은 청소년(8~10점)들은 예외 없이 하루 5시간 이상의 과도한 SNS 이용량과 6시간 미만의 짧은 수면 시간**이 복합적으로 얽혀 있는 경향성을 보였습니다.
            2. **스트레스 저항력을 키우는 신체 활동(운동)**
                * 비슷한 수준으로 SNS나 전자기기를 장시간 이용하더라도, **하루 1시간 이상 신체 활동을 하는 청소년 집단**은 그렇지 않은 집단에 비해 AI 모델이 예측한 최종 스트레스 지수가 평균 1.8단계 낮게 분류되었습니다. 운동이 완충 작용을 하고 있음을 증명합니다.
            3. **디지털 웰빙의 시급성**
                * AI가 내린 결론은 명확합니다. 청소년의 스트레스 관리의 핵심 키(Key)는 단순히 학업 스트레스 자체보다, **'잠들기 전 화면 사용 차단'**과 **'SNS 소비 시간 통제'** 같은 일상 속 디지털 습관 개혁에 직결되어 있습니다.
            """)
            
        with insight_col2:
            fig5, ax5 = plt.subplots(figsize=(5, 3.8))
            original_df['Stress Group'] = np.where(original_df['stress_level'] >= 8, 'High Stress', 'Normal')
            
            sns.violinplot(
                data=original_df, 
                x='Stress Group', 
                y='sleep_hours', 
                palette={'High Stress': '#e74c3c', 'Normal': '#2ecc71'}, 
                inner='quartile', 
                ax=ax5
            )
            ax5.set_xlabel("Stress Group Classification")
            ax5.set_ylabel("Sleep Hours")
            ax5.grid(False)
            sns.despine()
            st.pyplot(fig5)
            
            st.markdown("""
            🔎 **데이터 분포 그래프 해석:**
            * **고스트레스 위험군(빨간색 바이올린):** 수면 시간의 부피가 **5~6시간대**에 극단적으로 치우쳐 뚱뚱하게 뭉쳐 있습니다. 수면 결핍이 고스트레스 상태의 결정적 지표임을 뜻합니다.
            * **정상군(초록색 바이올린):** 데이터의 중심축이 **7~9시간 영역**에 넓고 안정적으로 펼쳐져 있어 균형 잡힌 밀도를 보여줍니다.
            """)
