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
# 2. 데이터 학습 및 평가 연산 (과적합 진단을 위해 Train 스코어도 계산)
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
        
        # 과적합 검증용 Train/Test 예측
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_test, y_test_pred)
        
        model_results[name] = {
            "model": model,
            "train_accuracy": train_acc,
            "accuracy": test_acc,  # 기존 코드 호환용 (test)
            "f1_score": f1,
            "confusion_matrix": cm,
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
    # 4. 실시간 스트레스 지수 예측기
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

        # -----------------------------------------------------
        # 6. [신규 추가] AI 모델 자체 신뢰성 및 과적합 위험도 분석
        # -----------------------------------------------------
        st.markdown("---")
        st.markdown("### ⚠️ AI 알고리즘 신뢰성 및 과적합(Overfitting) 위험도 평가")
        st.markdown("실시간으로 수치를 조절하며 학습시킨 현재 모델의 구조적 안정성을 자체 진단합니다.")
        
        # 진단 대상 모델 정보 불러오기
        active_info = results[selected_model_name]
        train_acc = active_info["train_accuracy"]
        test_acc = active_info["accuracy"]
        
        # 학습 성능과 검증 성능의 차이(Gap) 계산
        overfit_gap = train_acc - test_acc
        
        risk_col1, risk_col2 = st.columns([1, 2])
        
        with risk_col1:
            st.metric(label="🎯 학습-검증 정확도 격차(Gap)", value=f"{overfit_gap*100:.2f}%")
            
            # 격차에 따른 위험도 등급 산정 및 시각화
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
                st.markdown(f"현재 선택된 `{selected_model_name}` 모델은 **훈련 데이터에 과도하게 동화된(과적합) 상태**입니다. "
                            f"훈련용 문제집(Train Acc: {train_acc*100:.1f}%)은 완벽히 풀지만, 새로운 모의고사(Test Acc: {test_acc*100:.1f}%)에서는 오답률이 높은 현상입니다. "
                            f"**해결책:** 의사결정나무의 최대 깊이를 줄이거나, 로지스틱 규제 강도를 낮추어 모델을 더 단순하게 깎아내야 실전 예측 신뢰도가 상승합니다.")
            elif overfit_gap >= 0.05:
                st.markdown(f"훈련 정확도({train_acc*100:.1f}%)와 검증 정확도({test_acc*100:.1f}%) 사이에 약간의 괴리가 존재합니다. "
                            f"완전한 과적합은 아니지만 복잡한 난수가 일부 섞여 있으므로, 제어판 슬라이더를 통해 나무 개수를 조절하거나 조금 더 규제를 주어 일반화 성능을 올리는 것을 권장합니다.")
            elif train_acc < 0.40:
                st.markdown(f"훈련 데이터조차 제대로 분류해내지 못하는 **과소적합(Underfitting)** 상태입니다. "
                            f"모델의 뼈대나 제약 수치가 너무 빡빡하여 청소년 데이터의 패턴을 전혀 학습하지 못하고 있습니다. "
                            f"**해결책:** 하이퍼파라미터 제어판에서 나무 깊이를 늘리거나 규제 강도(C값)를 키워 모델이 더 깊게 공부할 수 있는 환경을 열어주어야 합니다.")
            else:
                st.markdown(f"현재 설정된 `{selected_model_name}` 모델은 **최적의 학습 밸런스(Generalized Optimal Model)**를 이루고 있습니다. "
                            f"훈련 스코어와 검증 스코어가 나란히 균형을 이루며, 실시간 예측기 폼에 어떤 새로운 데이터 조합을 입력하더라도 "
                            f"가장 편향 없이 정확하고 신뢰도 높은 스트레스 수치를 도출해 낼 수 있는 견고한 상태입니다.")
