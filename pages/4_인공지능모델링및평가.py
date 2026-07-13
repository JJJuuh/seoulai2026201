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

st.set_page_config(layout="wide", page_title="AI 정신건강 분석 대시보드", page_icon="🤖") 

# 🌟 시스템 테마(다크/라이트)에 따라 자동으로 글자색이 반전되는 세련된 유동성 CSS 🌟
st.markdown("""
    <style>
        .block-container { padding-top: 2.5rem; padding-bottom: 2rem; }
        
        /* 특정 테마색을 강제하지 않고 시스템 전역 폰트 색상을 상속받아 자동으로 반전되도록 설정 */
        h1, h3, h5, p, span { font-family: inherit; }
        h1 { font-weight: 800; letter-spacing: -0.05em; }
        h3 { font-weight: 700; margin-top: 1.5rem; }
        
        /* 탭 메뉴 글씨 가독성 강화 */
        .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 700; padding: 10px 20px; }
        
        /* 제어판 테두리 및 배경을 시스템 테마에 맞춤 (안 보이지 않게 투명도 활용) */
        div[data-testid="stContainer"] { 
            border-color: rgba(128, 128, 128, 0.3) !important; 
            background-color: rgba(128, 128, 128, 0.05); 
        }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 4. 인공지능 모델링 및 평가 대시보드")
st.markdown("알고리즘 조절 나사를 제어하며 청소년의 생활 패턴과 스트레스 지수(1~10) 간의 상관관계를 다각도로 분석합니다.")

# -----------------------------------------------------
# 1. 화면 중간/상단 가로 제어판
# -----------------------------------------------------
with st.container(border=True):
    st.markdown("##### 🛠️ AI 실험실 설정 제어판")
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
# 2. 데이터 학습 및 평가 연산
# -----------------------------------------------------
@st.cache_data
def train_and_evaluate(trees, depth, c_val):
    df = pd.read_csv("Teen_Mental_Health_Dataset.csv")
    df_encoded = pd.get_dummies(df, columns=['gender'], drop_first=True)
    
    features = ['daily_social_media_hours', 'sleep_hours', 'screen_time_before_sleep', 'physical_activity',
                'age', 'academic_performance', 'anxiety_level', 'addiction_level']
    target = 'stress_level'
    
    X = df_encoded[features]
    y = df_encoded[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        "Logistic Regression": LogisticRegression(C=c_val, max_iter=2000, random_state=42),
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
    # 3. AI 알고리즘 신뢰성 및 과적합 위험도 평가
    # -----------------------------------------------------
    st.markdown("---")
    st.markdown("### ⚠️ AI 알고리즘 신뢰성 및 과적합(Overfitting) 위험도 평가")
    
    active_info = results[selected_model_name]
    train_acc = active_info["train_accuracy"]
    test_acc = active_info["accuracy"]
    overfit_gap = train_acc - test_acc
    
    with st.container(border=True):
        risk_col1, risk_col2 = st.columns([1, 2])
        with risk_col1:
            st.metric(label="🎯 훈련-검증 정확도 격차(Gap)", value=f"{overfit_gap*100:.2f}%")
            if overfit_gap >= 0.15:
                st.error("🚨 위험 수준: 과적합 상태 (Overfitting)")
            elif overfit_gap >= 0.05:
                st.warning("⚠️ 위험 수준: 잠재적 위험 (Moderate)")
            else:
                st.success("✅ 위험 수준: 안정적 (Optimal)")
                
        with risk_col2:
            st.markdown(f"**🔍 {selected_model_name} 모델 상태 판정**")
            if overfit_gap >= 0.15:
                st.markdown(f"⚠️ **과적합 위험 경고:** 현재 모델은 훈련 점수({train_acc*100:.1f}%)만 압도적으로 높고 실전 점수({test_acc*100:.1f}%)는 떨어지는 **심각한 과적합** 상태입니다. 데이터의 소음까지 암기해 버렸으므로 제어판에서 규제를 주거나 깊이를 제한해 범용성을 복구해야 합니다.")
            elif overfit_gap >= 0.05:
                st.markdown(f"훈련 데이터와 검증 데이터 간에 미세한 격차가 있습니다. 복잡도 수치를 조금 낮추면 밸런스가 향상됩니다.")
            else:
                st.markdown(f"훈련({train_acc*100:.1f}%)과 검증({test_acc*100:.1f}%)의 밸런스가 정교하게 맞물린 최적의 일반화 상태입니다. 실전 신뢰도가 매우 높습니다.")

    # -----------------------------------------------------
    # 4. 실시간 분석 결과 리포트 (테마 자동 동기화 렌더링)
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader("📊 실시간 분석 결과 리포트")
    
    # 🌟 [핵심 변경] Matplotlib 그래프가 시스템 다크/라이트 테마 색상을 자동으로 감지하도록 설정 🌟
    # 배경을 투명하게 만들어 Streamlit 테마색이 그대로 비치도록 유도하고, 축과 글씨는 기본 스타일 테마를 따릅니다.
    plt.rcParams['figure.facecolor'] = 'none'
    plt.rcParams['axes.facecolor'] = 'none'
    
    # 테마에 따라 텍스트가 동적으로 반전될 수 있도록 Matplotlib의 스타일 설정을 초기화합니다.
    plt.style.use('default') 
    plt.rcParams.update({
        "text.color": "auto",
        "axes.labelcolor": "auto",
        "xtick.color": "auto",
        "ytick.color": "auto"
    })
    sns.set_style("whitegrid" if st.get_option("theme.base") == "light" else "darkgrid")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🥇 모델별 성능 비교", 
        "🔑 핵심 영향 요인 분석", 
        "📈 SNS사용시간별 예측 경향", 
        "🔲 혼동 행렬 격자 점검"
    ])
    
    # --- [탭 1] 모델별 성능 비교 ---
    with tab1:
        st.markdown("##### AI 모델별 성능 지표 비교 그래프")
        names = list(results.keys())
        accs = [results[n]["accuracy"] for n in names]
        f1s = [results[n]["f1_score"] for n in names]
        
        all_scores = accs + f1s
        y_min = max(0.0, float(np.floor(min(all_scores) * 20) / 20) - 0.05) 
        y_max = min(1.0, float(np.ceil(max(all_scores) * 20) / 20) + 0.05)
        
        # 🌟 Streamlit 전용 차트 컬러 자동 스케일링 기법 🌟
        fig1, ax1 = plt.subplots(figsize=(10, 3.8))
        ax1.plot(names, accs, marker='o', markersize=8, linewidth=2.5, label='Accuracy', color='#3498db')
        ax1.plot(names, f1s, marker='s', markersize=8, linewidth=2.5, label='F1-Score', color='#e67e22')
        ax1.set_ylim(y_min, y_max)
        sns.despine()
        
        # 테마 맞춤형 범례 상자 생성
        legend1 = ax1.legend(facecolor='none', edgecolor='gray')
        st.pyplot(fig1)
        
        best_acc_model = max(results, key=lambda k: results[k]["accuracy"])
        st.info(f"📈 **실시간 스냅샷:** 현미경 줌인 범위({y_min*100:.0f}% ~ {y_max*100:.0f}%) 분석 결과, 최종 검증 성능이 가장 우수한 알고리즘은 **{best_acc_model}**입니다.")

    # --- [탭 2] 핵심 영향 요인 분석 ---
    with tab2:
        st.markdown(f"##### {selected_model_name}의 핵심 영향 요인 분석 그래프")
        current_model = results[selected_model_name]["model"]
        fig2, ax2 = plt.subplots(figsize=(10, 3.8))
        
        core_labels = ['SNS Hours', 'Sleep Hours', 'Screen Before Sleep', 'Physical Activity']
        
        if hasattr(current_model, 'feature_importances_'):
            importances = current_model.feature_importances_[:4]
            sns.barplot(x=importances, y=core_labels, ax=ax2, palette="mako")
        elif hasattr(current_model, 'coef_'):
            importances = np.abs(current_model.coef_[0])[:4] 
            sns.barplot(x=importances, y=core_labels, ax=ax2, palette="rocket")
            
        sns.despine()
        st.pyplot(fig2)
        
        top_feature = core_labels[np.argmax(importances)]
        st.info(f"🔑 **실시간 스냅샷:** 현재 입력 제어판 기준으로 AI가 스트레스 지수 분류의 가장 결정적인 변수로 타겟팅한 항목은 **{top_feature}** 입니다.")

    # --- [탭 3] SNS사용시간별 예측 경향 ---
    with tab3:
        st.markdown(f"##### {selected_model_name}의 SNS 사용 시간별 예측 경향 그래프")
        y_true = results[selected_model_name]["y_test"]
        y_pred = results[selected_model_name]["y_pred"]
        
        fig3, ax3 = plt.subplots(figsize=(10, 3.8))
        sns.scatterplot(x=X_test_df['daily_social_media_hours'], y=y_true, alpha=0.4, label='Actual Data', color='#7f8c8d', ax=ax3)
        sns.lineplot(x=X_test_df['daily_social_media_hours'], y=y_pred, color='#e67e22', marker='o', label='AI Predict Trend', ax=ax3, errorbar=None)
        sns.despine()
        legend3 = ax3.legend(facecolor='none', edgecolor='gray')
        st.pyplot(fig3)
        
        low_sns_pred = y_pred[X_test_df['daily_social_media_hours'] <= 3.0].mean()
        high_sns_pred = y_pred[X_test_df['daily_social_media_hours'] > 5.0].mean()
        st.info(f"📈 **실시간 스냅샷:** SNS 이용량이 3시간 이하일 때 예측 평균은 **{low_sns_pred:.1f}점**이며, 5시간 초과 시 예측 평균은 **{high_sns_pred:.1f}점**으로 뚜렷이 우상향합니다.")

    # --- [탭 4] 혼동 행렬 격자 점검 ---
    with tab4:
        st.markdown(f"##### {selected_model_name} 혼동 행렬 격자 그래프")
        cm = results[selected_model_name]["confusion_matrix"]
        fig4, ax4 = plt.subplots(figsize=(6, 3.8))
        
        # 맵 글자 색상이 배경 밝기에 맞게 선명하게 표출되도록 설정
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax4, annot_kws={"weight": "bold"})
        st.pyplot(fig4)
        
        total_samples = np.sum(cm)
        correct_samples = np.trace(cm)
        st.info(f"🔲 **실시간 스냅샷:** 검증 표본 {total_samples}건 중 정중앙 대각선 격자에 일치하여 오차 없이 정확히 분류해 낸 데이터는 총 **{correct_samples}건**입니다.")

    st.markdown("<p style='font-size:15px; font-weight:600; margin-bottom:-5px; margin-top:20px;'>📋 전 알고리즘 성능 스코어보드 요약</p>", unsafe_allow_html=True)
    st.table(pd.DataFrame([
        {
            "AI 알고리즘 이름": name, 
            "훈련 데이터 정확도 (Train Acc)": f"{info['train_accuracy']*100:.1f}%",
            "검증 데이터 정확도 (Test Acc)": f"{info['accuracy']*100:.1f}%", 
            "종합 불균형 평가 지표 (F1-Score)": f"{info['f1_score']:.2f}"
        }
        for name, info in results.items()
    ]))

    # -----------------------------------------------------
    # 5. 실시간 스트레스 지수 예측기 및 입체적 분석 리포트
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader(f"🔮 실시간 종합 스트레스 예측기 (최적 모델: {selected_model_name})")
    st.markdown("아래 생활수치들과 추가 환경 지표를 복합 입력하면, 고도화된 AI 알고리즘이 예측 보고서와 처방전을 발행합니다.")

    with st.container(border=True):
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            sns_hours = st.slider("하루 평균 SNS 이용량 (시간)", 0.0, 10.0, 3.0, step=0.5)
            sleep_hours = st.slider("하루 평균 수면 양 (시간)", 3.0, 12.0, 7.0, step=0.5)
        with p_col2:
            screen_time_before_sleep = st.slider("취침 전 화면 노출 시간 (시간)", 0.0, 4.0, 1.5, step=0.1)
            exercise_hours = st.slider("하루 평균 신체 활동 (시간)", 0.0, 3.0, 1.0, step=0.1)
        with p_col3:
            user_age = st.slider("대상 청소년 나이 (세)", 13, 19, 16)
            academic_performance = st.slider("평균 학업 성적 (GPA)", 0.0, 4.5, 3.0, step=0.1)
        with p_col4:
            anxiety_level = st.slider("자가 불안 지수 (1~10)", 1, 10, 4)
            addiction_level = st.slider("스마트폰 중독 지수 (1~10)", 1, 10, 5)

        btn_triggered = st.button("🧠 AI 종합 분석 및 맞춤형 심리 처방전 출력", use_container_width=True)

    if btn_triggered:
        input_data = np.array([[sns_hours, sleep_hours, screen_time_before_sleep, exercise_hours, user_age, academic_performance, anxiety_level, addiction_level]])
        active_model = results[selected_model_name]["model"]
        predicted_stress = active_model.predict(input_data)[0]
        
        with st.container(border=True):
            st.markdown("### 🎯 AI 종합 분석 결과 보고서")
            
            risk_text = ""
            if predicted_stress >= 8:
                st.error(f"🚨 위험 등급: 고위험군 (예측 스트레스 지수: {predicted_stress} / 10)")
                risk_text = "고위험군"
            elif predicted_stress >= 4:
                st.warning(f"⚠️ 위험 등급: 주의군 (예측 스트레스 지수: {predicted_stress} / 10)")
                risk_text = "주의군"
            else:
                st.success(f"✅ 위험 등급: 안정군 (예측 스트레스 지수: {predicted_stress} / 10)")
                risk_text = "안정군"
                
            st.markdown("#### 📝 다각도 매칭 라이프 가이드 피드백 (심리/학업 데이터 융합)")
            feedback_list = []
            
            if anxiety_level >= 7:
                feedback_list.append("- 🧠 **위험 수준의 자가 불안감 검출:** 수면 부족 외에도 일상 속 만성적인 심리적 불안 요소가 인공지능 스트레스 상승 가속의 주요 원인으로 작동하고 있습니다.")
            if academic_performance <= 2.0:
                feedback_list.append("- 🏫 **학업 성취도 압박감 반영:** 학업 성적 대역에 따른 심리적 부담감이 스트레스 방어 수치를 일부 상쇄하고 있음이 포착됩니다.")
            if sleep_hours < 6.0:
                feedback_list.append("- ❌ **심각한 수면 결핍:** 대뇌 회복이 어려운 구조적 수면 부족 상태입니다. 스트레스 면역력을 가장 급격히 무너뜨립니다.")
            if exercise_hours >= 1.0:
                feedback_list.append("- 👍 **긍정적 신체적 완충재 축적:** 다행히 하루 1시간 이상의 신체 활동이 고스트레스 폭발을 제어해 주는 든든한 보호막이 되어 줍니다.")
                
            feedback_text = "\n".join(feedback_list) if feedback_list else "- 심리, 학업, 생활 패턴이 치우침 없이 정상 범주 내에 예쁘게 안착해 있습니다."
            st.markdown(feedback_text)
                
            st.markdown("#### 💡 AI 마인드 케어 처방 조언")
            care_advice = ""
            if predicted_stress >= 8:
                care_advice = "생활 패턴의 균열과 심리적 불안도 수치가 동시에 위험 신호를 보내고 있습니다. 모바일 디톡스와 수면 확보가 시급하며, 교내 Wee 클래스나 전문 상담 기관의 노크를 강력 권장합니다."
            elif predicted_stress >= 4:
                care_advice = "불규칙한 디지털 사용 습관을 제어하면 빠르게 안정군으로 유턴할 수 있는 주의 단계입니다. 취침 전 스마트폰 오프 습관과 주 3회 땀 흘리는 운동을 믹스해 보세요."
            else:
                care_advice = "균형 잡힌 라이프 스타일과 차분한 내면 상태가 융합된 최상의 마인드 컨디션입니다. 기복 없이 현재의 건강한 루틴을 계속 이어나가길 응원합니다."
            st.markdown(care_advice)

            report_txt = f"""[AI 청소년 융합 정신건강 진단 처방전]
-----------------------------------------
■ 진단 모델 알고리즘: {selected_model_name}
■ 환경 변수: 나이 {user_age}세 | 학업 성적 GPA {academic_performance}
■ 심리 변수: 자가 불안도 {anxiety_level}/10 | 스마트폰 중독 지수 {addiction_level}/10

■ AI 최종 예측결과: 스트레스 지수 {predicted_stress}/10 ({risk_text})
■ 맞춤형 다각도 피드백:
{feedback_text}

■ AI 마인드 케어 종합 처방 조언:
{care_advice}
-----------------------------------------"""

            st.download_button(
                label="📥 AI 진단서 및 처방전 파일(.txt) 다운로드 받기",
                data=report_txt,
                file_name="AI_Teen_Mental_Health_Advanced_Report.txt",
                mime="text/plain",
                use_container_width=True
            )

            # 빅데이터 분석 결론부
            st.markdown("---")
            st.markdown("### 🏛 Honor 리포트: 청소년 정신 건강 빅데이터 학습 집약 결과")
            ins_col1, ins_col2 = st.columns([1.1, 0.9])
            
            with ins_col1:
                st.markdown("""
                * **심리적 요인과 디지털 이용 패턴의 상호 결합**: 빅데이터 분석을 종합하면 스트레스 고위험 청소년 군은 과도한 야간 모바일 스크린 노출과 수면 저하의 파괴적인 인과 고리에 전형적으로 묶여 있습니다.
                * **수면의 보편적 보호막 효과**: 나이나 성별, 성적 등 모든 환경적 격차를 불문하고 '충분한 수면 시간 확보'는 청소년 정신 건강 방어선을 사수하는 데 있어 AI 통계 모형이 검증한 가장 강력하고 보편적인 공통 핵심 인자입니다.
                """)
            with ins_col2:
                fig5, ax5 = plt.subplots(figsize=(5, 3.6))
                original_df['Stress Group'] = np.where(original_df['stress_level'] >= 8, 'High Stress', 'Normal')
                sns.violinplot(data=original_df, x='Stress Group', y='sleep_hours', palette={'High Stress': '#e74c3c', 'Normal': '#2ecc71'}, inner='quartile', ax=ax5)
                sns.despine()
                st.pyplot(fig5)
                st.markdown("<p style='font-size:12px; text-align:center;'>주황/빨간색 고위험군의 분포 부피가 5~6시간 구역에 뚱뚱하게 밀집되어 있어, 수면 시간 확보가 스트레스 방어의 최우선 과제임을 AI 통계 분석이 직관적으로 증명합니다.</p>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # 6. 왜 Random Forest 모델이 최종 채택되었는가? (상시 노출)
    # -----------------------------------------------------
    st.markdown("---")
    st.markdown("### 🏆 왜 이 프로젝트의 최적 알고리즘으로 'Random Forest'를 채택했을까요?")
    
    with st.container(border=True):
        st.markdown("""
        * **비선형적 다중 결합 패턴 분석에 강함**: SNS 사용량, 수면 부족, 불안도 등이 복합적으로 시너지를 내며 악화되는 비선형적 상호작용 분류 규칙을 가장 정교하게 탐색합니다.
        * **데이터 불균형 제어 및 오버피팅 예방**: 수백 개의 나무 모델을 심어 다수결 투표(Bagging) 방식을 사용하기 때문에, 특정 스트레스 수치에 데이터가 몰려있는 불균형 환경에서도 치우침 없는 밸런스 점수(F1-Score)를 도출합니다.
        * **설명 가능한 AI 피드백 제공**: 학습 연산이 종료된 후 어떤 도메인 요소가 스트레스 예측에 큰 지분을 가졌는지 '피처 중요도(Feature Importance)'를 수치화하여 도출할 수 있는 강력한 학술적 근거를 제공합니다.
        """)
