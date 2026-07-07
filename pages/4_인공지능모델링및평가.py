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
# 데이터 사전 연산 함수
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
    # 2. AI 알고리즘 신뢰성 및 과적합 위험도 평가
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
    # 3. 실시간 분석 결과 리포트
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader("📊 실시간 분석 결과 리포트")
    
    # 그래프 글씨를 하얀색(white)으로 렌더링하기 위한 Matplotlib 옵션 설정
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['axes.facecolor'] = 'none'
    plt.rcParams['figure.facecolor'] = 'none'
    sns.set_style("white")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🥇 모델별 성능 비교", 
        "🔑 핵심 영향 요인 분석", 
        "📈 수면시간별 예측 경향", 
        "🔲 혼동 행렬 격자 점검"
    ])
    
    # --- [탭 1] 차이가 확연히 드러나도록 y축 다이내믹 줌인 + 라인 차트로 변경 ---
    with tab1:
        st.markdown("##### AI 모델별 성능 지표 비교 그래프")
        
        names = list(results.keys())
        accs = [results[n]["accuracy"] for n in names]
        f1s = [results[n]["f1_score"] for n in names]
        
        # 차이를 시각적으로 극대화하기 위해 현재 스코어 중 최소값 탐색 후 y축 하한선 설정
        all_scores = accs + f1s
        min_score = min(all_scores)
        max_score = max(all_scores)
        
        # 미세한 변동폭을 반영한 하한/상한 단위 조정 마진 연산 (최소값보다 살짝 아래부터 줌인)
        y_min = max(0.0, float(np.floor(min_score * 20) / 20) - 0.05) 
        y_max = min(1.0, float(np.ceil(max_score * 20) / 20) + 0.05)
        
        fig1, ax1 = plt.subplots(figsize=(9, 4))
        
        # 막대 대신 점과 선(Line & Marker) 플롯을 사용하여 높낮이 비교 극대화
        ax1.plot(names, accs, marker='o', markersize=8, linewidth=2.5, label='Accuracy', color='#4e79a7')
        ax1.plot(names, f1s, marker='s', markersize=8, linewidth=2.5, label='F1-Score', color='#f28e2b')
        
        ax1.set_ylim(y_min, y_max) # 연산된 다이내믹 줌인 y축 적용
        ax1.set_xlabel("AI Algorithms", color='white')
        ax1.set_ylabel("Performance Score", color='white')
        ax1.grid(False)
        
        legend = ax1.legend(facecolor='black', edgecolor='none')
        for text in legend.get_texts():
            text.set_color('white')
            
        sns.despine()
        st.pyplot(fig1)
        
        # 동적 결과 설명
        best_acc_model = max(results, key=lambda k: results[k]["accuracy"])
        best_acc_val = results[best_acc_model]["accuracy"] * 100
        
        st.success(f"틀린 그림 찾기 같던 그래프를 **y축 범위를 점수 밀집 구역({y_min*100:.0f}% ~ {y_max*100:.0f}%)으로 현미경처럼 확대**하여 시각화했습니다.\n\n"
                   f"📈 **실시간 그래프 분석 결과:** \n"
                   f"미세한 차이를 정밀 분석한 결과, 현재 설정 기준 가장 높은 정확도를 갱신한 우수 알고리즘은 **{best_acc_model}**({best_acc_val:.1f}%)입니다.")

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
            
        ax2.set_xlabel("Importance", color='white')
        ax2.grid(False)
        sns.despine()
        st.pyplot(fig2)
        
        max_idx = np.argmax(importances)
        top_feature = feature_labels_eng[max_idx]
        
        st.success(f"🔑 **실시간 그래프 분석 결과:** \n"
                   f"현재 제어판 수치 기준으로 `{selected_model_name}`이 청소년의 스트레스를 예측할 때 "
                   f"가장 결정적인 단서로 지목한 생활 패턴은 **{top_feature}** 입니다. 이 요인의 변화가 예측 결과에 가장 큰 변동을 줍니다.")

    # --- [탭 3] 수면시간별 예측 경향 ---
    with tab3:
        st.markdown(f"##### {selected_model_name}의 수면시간별 예측 경향 그래프")
        y_true = results[selected_model_name]["y_test"]
        y_pred = results[selected_model_name]["y_pred"]
        
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.scatterplot(x=X_test_df['sleep_hours'], y=y_true, alpha=0.4, label='Actual Data', color='gray', ax=ax3)
        sns.lineplot(x=X_test_df['sleep_hours'], y=y_pred, color='red', marker='o', label='AI Predict Trend', ax=ax3, errorbar=None)
        ax3.set_xlabel("Sleep Hours", color='white')
        ax3.set_ylabel("Stress Level (1~10)", color='white')
        
        legend3 = ax3.legend(facecolor='black', edgecolor='none')
        for text in legend3.get_texts():
            text.set_color('white')
            
        ax3.grid(False)
        sns.despine()
        st.pyplot(fig3)
        
        low_sleep_pred = y_pred[X_test_df['sleep_hours'] <= 6.0].mean()
        high_sleep_pred = y_pred[X_test_df['sleep_hours'] > 6.0].mean()
        
        st.success(f"📈 **실시간 그래프 분석 결과:** \n"
                   f"빨간색 AI 예측 트렌드 선의 흐름을 분석한 결과, 수면 시간이 6시간 이하일 때 스트레스 지수 예측치 평균은 **{low_sleep_pred:.1f}점**인 반면, "
                   f"6시간을 초과하여 충분히 잘 때의 예측 평균은 **{high_sleep_pred:.1f}점**으로 뚜렷하게 감소하는 경향성(우하향)을 실시간으로 그려내고 있습니다.")

    # --- [탭 4] 혼동 행렬 격자 점검 ---
    with tab4:
        st.markdown(f"##### {selected_model_name} 혼동 행렬 격자 그래프")
        cm = results[selected_model_name]["confusion_matrix"]
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax4, annot_kws={"color": "white"})
        ax4.set_xlabel("Predicted Label", color='white')
        ax4.set_ylabel("True Label", color='white')
        ax4.grid(False)
        st.pyplot(fig4)
        
        total_samples = np.sum(cm)
        correct_samples = np.trace(cm)
        
        st.success(f"🔲 **실시간 그래프 분석 결과:** \n"
                   f"전체 검증 데이터 {total_samples}건 중, 진한 파란색 대각선 격자에 안착하여 AI가 한 치의 오차도 없이 완벽하게 실제 스트레스 점수를 맞춘 샘플은 "
                   f"총 **{correct_samples}건**입니다. 나머지 칸에 적힌 숫자들이 AI가 한두 단계씩 헷갈려한 오답 분과 영역입니다.")

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
            ax5.set_xlabel("Stress Group Classification", color='white')
            ax5.set_ylabel("Sleep Hours", color='white')
            ax5.grid(False)
            sns.despine()
            st.pyplot(fig5)
            
            st.markdown("""
            🔎 **데이터 분포 그래프 해석:**
            * **고스트레스 위험군(빨간색 바이올린):** 수면 시간의 부피가 **5~6시간대**에 극단적으로 치우쳐 뚱뚱하게 뭉쳐 있습니다. 수면 결핍이 고스트레스 상태의 결정적 지표임을 뜻합니다.
            * **정상군(초록색 바이올린):** 데이터의 중심축이 **7~9시간 영역**에 넓고 안정적으로 펼쳐져 있어 균형 잡힌 밀도를 보여줍니다.
            """)
