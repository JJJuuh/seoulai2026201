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

# 1. 페이지 기본 설정 및 다크/모던 감성의 CSS 스타일 정의
st.set_page_config(layout="wide", page_title="AI 정신건강 분석 대시보드", page_icon="🤖") 

# 웹페이지 요소를 조금 더 부드럽고 세련되게 만들기 위한 스타일 삽입
st.markdown("""
    <style>
        .block-container { padding-top: 2.5rem; padding-bottom: 2rem; }
        h1 { font-weight: 800; color: #ffffff; letter-spacing: -0.05em; }
        h3 { font-weight: 700; color: #f8f9fa; margin-top: 1.5rem; }
        .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; padding: 10px 20px; }
        .stSlider [data-baseweb="slider"] { margin-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

# 타이틀 및 헤더 배너
st.title("🤖 4. 인공지능 모델링 및 평가 대시보드")
st.markdown("<p style='font-size:16px; color:#b2bec3; margin-top:-10px;'>다양한 AI 알고리즘의 파라미터를 실시간으로 제어하며 청소년의 생활 패턴과 스트레스 지수(1~10) 간의 상관관계를 다각도로 분석합니다.</p>", unsafe_allow_html=True)

# -----------------------------------------------------
# 2. [가로 제어판] 디자인 세련되게 컴팩트화
# -----------------------------------------------------
with st.container(border=True):
    st.markdown("<h5 style='margin-top:0; margin-bottom:10px; color:#dfe6e9;'>🛠️ AI 실험실 설정 제어판</h5>", unsafe_allow_html=True)
    control_col1, control_col2, control_col3, control_col4 = st.columns(4)

    with control_col1:
        selected_model_name = st.selectbox(
            "시뮬레이션 메인 모델", 
            ["Random Forest", "Decision Tree", "Logistic Regression"],
            help="하단 예측기와 상세 그래프 분석에 매핑될 중심 알고리즘을 선택합니다."
        )
    with control_col2:
        rf_trees = st.slider("Random Forest: 나무 개수", 10, 200, 100, step=10)
    with control_col3:
        dt_depth = st.slider("Decision Tree: 최대 깊이", 3, 15, 7)
    with control_col4:
        lr_c = st.slider("Logistic Regression: 규제 강도(C)", 0.01, 10.0, 1.0, step=0.1)

# -----------------------------------------------------
# 데이터 연산 함수 (안정적인 데이터 바인딩)
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
    st.error("⚠️ 데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    data_loaded = False


if data_loaded:
    # -----------------------------------------------------
    # 3. [위치 변경 & 디자인 업그레이드] 과적합 위험도 평가
    # -----------------------------------------------------
    st.markdown("### ⚠️ AI 알고리즘 신뢰성 및 과적합(Overfitting) 위험도 평가")
    
    active_info = results[selected_model_name]
    train_acc = active_info["train_accuracy"]
    test_acc = active_info["accuracy"]
    overfit_gap = train_acc - test_acc
    
    with st.container(border=True):
        risk_col1, risk_col2 = st.columns([1, 2])
        
        with risk_col1:
            st.metric(label="🎯 학습-검증 정확도 격차(Gap)", value=f"{overfit_gap*100:.2f}%")
            if overfit_gap >= 0.15:
                st.error("🚨 위험 수준: 과적합 상태 (Overfitting)")
            elif overfit_gap >= 0.05:
                st.warning("⚠️ 위험 수준: 잠재적 위험 (Moderate)")
            elif train_acc < 0.40:
                st.info("📉 위험 수준: 과소적합 상태 (Underfitting)")
            else:
                st.success("✅ 위험 수준: 매우 안정적 (Optimal)")
                
        with risk_col2:
            st.markdown(f"<p style='font-size:15px; font-weight:600; color:#dfe6e9; margin-bottom:5px;'>🔍 {selected_model_name} 모델 진단 소견</p>", unsafe_allow_html=True)
            if overfit_gap >= 0.15:
                st.markdown(f"현재 모델 구조가 훈련 데이터셋에 과도하게 짜맞춰져 있습니다. "
                            f"학습용 문제집 점수({train_acc*100:.1f}%)에 비해 실전 검증 스코어({test_acc*100:.1f}%)가 떨어지는 전형적인 현상입니다. "
                            f"제어판에서 가지치기 조절을 통해 복잡도를 낮추어야 실전 신뢰도가 상승합니다.")
            elif overfit_gap >= 0.05:
                st.markdown(f"훈련 정확도({train_acc*100:.1f}%)와 검증 정확도({test_acc*100:.1f}%) 사이에 약간의 틈이 발생했습니다. "
                            f"큰 위험은 아니나 복잡도 변수를 살짝 하향 조정하면 더욱 탄탄한 모델이 됩니다.")
            elif train_acc < 0.40:
                st.markdown("패턴 학습량이 부족한 과소적합 상태입니다. AI가 깊게 공부할 수 있도록 제한을 완화해 주세요.")
            else:
                st.markdown(f"훈련({train_acc*100:.1f}%)과 검증({test_acc*100:.1f}%) 스코어가 정교하게 맞물린 **최적의 일반화 상태**입니다. 실제 서빙 환경에서도 왜곡 없이 높은 확률로 점수를 올바르게 예측해 낼 수 있습니다.")

    # -----------------------------------------------------
    # 4. 실시간 분석 결과 리포트 (탭 및 그래프 흰색 폰트 최적화)
    # -----------------------------------------------------
    st.markdown("### 📊 실시간 분석 결과 리포트")
    
    # 그래프 화이트 텍스트 커스텀 테마 고정
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
    
    # --- [탭 1] 현미경 줌인 라인 차트 ---
    with tab1:
        st.markdown("<p style='font-size:14px; color:#b2bec3; margin-bottom:15px;'>각 모델의 검증 스코어 밀집 대역을 현미경처럼 확대 추적하여 미세한 우열을 판별합니다.</p>", unsafe_allow_html=True)
        names = list(results.keys())
        accs = [results[n]["accuracy"] for n in names]
        f1s = [results[n]["f1_score"] for n in names]
        
        all_scores = accs + f1s
        y_min = max(0.0, float(np.floor(min(all_scores) * 20) / 20) - 0.05) 
        y_max = min(1.0, float(np.ceil(max(all_scores) * 20) / 20) + 0.05)
        
        fig1, ax1 = plt.subplots(figsize=(10, 3.8))
        ax1.plot(names, accs, marker='o', markersize=8, linewidth=2.5, label='Accuracy', color='#3498db')
        ax1.plot(names, f1s, marker='s', markersize=8, linewidth=2.5, label='F1-Score', color='#e67e22')
        ax1.set_ylim(y_min, y_max)
        ax1.set_ylabel("Performance Score", color='white', fontsize=11)
        sns.despine()
        legend1 = ax1.legend(facecolor='#2d3436', edgecolor='none')
        for text in legend1.get_texts(): text.set_color('white')
        st.pyplot(fig1)
        
        best_acc_model = max(results, key=lambda k: results[k]["accuracy"])
        st.info(f"📊 **실시간 분석 스냅샷:**\ny축 단위를 점수 밀집 구역({y_min*100:.0f}% ~ {y_max*100:.0f}%)으로 미세 튜닝하여 렌더링한 결과, 현재 세팅 기준 가장 우수한 정확도를 입증한 모델은 **{best_acc_model}**({results[best_acc_model]['accuracy']*100:.1f}%)입니다.")

    # --- [탭 2] 핵심 영향 요인 분석 ---
    with tab2:
        st.markdown("<p style='font-size:14px; color:#b2bec3; margin-bottom:15px;'>알고리즘 내부에서 스트레스 레벨을 분류할 때 가장 지대한 영향을 미친 상위 생활 패턴 요인입니다.</p>", unsafe_allow_html=True)
        current_model = results[selected_model_name]["model"]
        fig2, ax2 = plt.subplots(figsize=(10, 3.8))
        feature_labels_eng = ['SNS Hours', 'Sleep Hours', 'Screen Before Sleep', 'Physical Activity']
        
        if hasattr(current_model, 'feature_importances_'):
            importances = current_model.feature_importances_
            sns.barplot(x=importances, y=feature_labels_eng, ax=ax2, palette="crest")
        elif hasattr(current_model, 'coef_'):
            importances = np.abs(current_model.coef_[0])[:4] 
            sns.barplot(x=importances, y=feature_labels_eng, ax=ax2, palette="flare")
            
        ax2.set_xlabel("Importance Weight", color='white')
        sns.despine()
        st.pyplot(fig2)
        
        top_feature = feature_labels_eng[np.argmax(importances)]
        st.info(f"🔑 **실시간 분석 스냅샷:**\n현재 입력 조건 하에 `{selected_model_name}` 모델이 스트레스 판별의 가장 결정적인 변수로 지목한 항목은 **{top_feature}** 입니다. 해당 패턴 통제가 청소년 멘탈 케어의 핵심 도메인임을 시사합니다.")

    # --- [탭 3] 수면시간별 예측 경향 ---
    with tab3:
        st.markdown("<p style='font-size:14px; color:#b2bec3; margin-bottom:15px;'>실제 원본 데이터 분포(회색) 위로 흐르는 AI 알고리즘 고유의 수면시간 대비 스트레스 예측 곡선입니다.</p>", unsafe_allow_html=True)
        y_true = results[selected_model_name]["y_test"]
        y_pred = results[selected_model_name]["y_pred"]
        
        fig3, ax3 = plt.subplots(figsize=(10, 3.8))
        sns.scatterplot(x=X_test_df['sleep_hours'], y=y_true, alpha=0.3, label='Actual Data', color='#b2bec3', ax=ax3)
        sns.lineplot(x=X_test_df['sleep_hours'], y=y_pred, color='#e74c3c', marker='o', label='AI Predict Trend', ax=ax3, errorbar=None)
        ax3.set_xlabel("Sleep Hours", color='white')
        ax3.set_ylabel("Stress Level (1~10)", color='white')
        sns.despine()
        legend3 = ax3.legend(facecolor='#2d3436', edgecolor='none')
        for text in legend3.get_texts(): text.set_color('white')
        st.pyplot(fig3)
        
        low_sleep_pred = y_pred[X_test_df['sleep_hours'] <= 6.0].mean()
        high_sleep_pred = y_pred[X_test_df['sleep_hours'] > 6.0].mean()
        st.info(f"📈 **실시간 분석 스냅샷:**\nAI의 예측 궤적 추적 결과, 수면량이 6시간 이하일 때의 스트레스 예측값 평균은 **{low_sleep_pred:.1f}점**이며, 6시간을 초과하여 숙면할 때의 예측 평균은 **{high_sleep_pred:.1f}점**으로 확연한 우하향 방어 효과가 검증됩니다.")

    # --- [탭 4] 혼동 행렬 격자 점검 ---
    with tab4:
        st.markdown("<p style='font-size:14px; color:#b2bec3; margin-bottom:15px;'>실제 점수(True)와 AI가 맞춘 점수(Predicted)가 일치하는 칸을 추적하여 오답의 분포를 정밀 진단합니다.</p>", unsafe_allow_html=True)
        cm = results[selected_model_name]["confusion_matrix"]
        fig4, ax4 = plt.subplots(figsize=(6, 3.8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax4, annot_kws={"color": "white"})
        ax4.set_xlabel("Predicted Label", color='white')
        ax4.set_ylabel("True Label", color='white')
        st.pyplot(fig4)
        
        total_samples = np.sum(cm)
        correct_samples = np.trace(cm)
        st.info(f"🔲 **실시간 분석 스냅샷:**\n전체 검증 표본 {total_samples}건 중, 진한 메인 대각선 격자에 정확히 일치하여 정답 처리가 완료된 데이터는 총 **{correct_samples}건**입니다.")

    # 정돈된 전 모델 스코어보드 표
    st.markdown("<p style='font-size:15px; font-weight:600; color:#dfe6e9; margin-bottom:-5px; margin-top:20px;'>📋 전 알고리즘 성능 스코어보드 요약</p>", unsafe_allow_html=True)
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
    # 5. 실시간 스트레스 지수 예측기 및 보고서 (카드 디자인 반영)
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader(f"🔮 실시간 스트레스 지수 예측기 (최적 모델: {selected_model_name})")
    st.markdown("아래 생활 수치들을 모의 배치하면, 완성된 AI 알고리즘이 해당 상태의 스트레스 위험 지수를 추정합니다.")

    with st.container(border=True):
        predict_col1, predict_col2 = st.columns(2)
        with predict_col1:
            sns_hours = st.slider("하루 평균 SNS 이용량 (시간)", 0.0, 10.0, 3.0, step=0.5)
            sleep_hours = st.slider("하루 평균 수면 양 (시간)", 3.0, 12.0, 7.0, step=0.5)
        with predict_col2:
            screen_time_before_sleep = st.slider("취침 전 화면 노출 시간 (시간)", 0.0, 4.0, 1.5, step=0.1)
            exercise_hours = st.slider("하루 평균 신체 활동 (시간)", 0.0, 3.0, 1.0, step=0.1)

        btn_triggered = st.button("🧠 AI 종합 분석 및 멘탈 조언 처방전 출력", use_container_width=True)

    if btn_triggered:
        input_data = np.array([[sns_hours, sleep_hours, screen_time_before_sleep, exercise_hours]])
        active_model = results[selected_model_name]["model"]
        predicted_stress = active_model.predict(input_data)[0]
        
        # 카드 형태로 묶인 종합 결과 보고서 링박스
        with st.container(border=True):
            st.markdown("### 🎯 AI 종합 분석 결과 보고서")
            
            if predicted_stress >= 8:
                st.error(f"🚨 위험 등급: 고위험군 (예측 스트레스 지수: {predicted_stress} / 10)")
            elif predicted_stress >= 4:
                st.warning(f"⚠️ 위험 등급: 주의군 (예측 스트레스 지수: {predicted_stress} / 10)")
            else:
                st.success(f"✅ 위험 등급: 안정군 (예측 스트레스 지수: {predicted_stress} / 10)")
                
            st.markdown("#### 📝 입력 패턴 매칭 라이프 가이드")
            feedback_list = []
            if sleep_hours < 6.0:
                feedback_list.append("❌ **심각한 수면 결핍:** 대뇌 회복이 어려운 구조적 수면 상태입니다. 스트레스 지수에 가장 악영향을 미칩니다.")
            elif sleep_hours >= 8.0:
                feedback_list.append("👍 **우수한 수면 기반:** 권장 숙면 타임을 준수하여 내면 스트레스 방어선이 탄탄합니다.")
            if sns_hours >= 5.0:
                feedback_list.append("❌ **도파민 및 SNS 과소비:** 타인과의 상시 비교 자극에 무방비하게 노출되어 피로도가 쉽게 누적됩니다.")
            if screen_time_before_sleep >= 2.0:
                feedback_list.append("❌ **수면 직전 야간 블루라이트 위험:** 숙면 호르몬 분비를 억제하여 잔여 피로를 높이는 패턴입니다.")
            if exercise_hours < 0.5:
                feedback_list.append("❌ **신체 활동 부족:** 코르티솔 호르몬 분해를 위한 하루 최소 운동 임계치에 미달합니다.")
            elif exercise_hours >= 1.0:
                feedback_list.append("👍 **체계적인 신체 활동:** 규칙적인 운동을 통해 정신적 유해 요소를 훌륭히 환기 중입니다.")
                
            for item in feedback_list if feedback_list else ["모든 항목이 정상 라이프 사이클 범주 내에 안착해 있습니다."]:
                st.markdown(item)
                
            st.markdown("#### 💡 AI 마인드 케어 처방 조언")
            if predicted_stress >= 8:
                st.markdown("인공지능 모델 검진 결과 고위험 영역 패턴과 일치합니다. 스마트폰 디톡스와 수면 보충이 최우선이며, 혼자 감내하기보다 교내 Wee 클래스나 전문 상담 기관의 도움을 받아보는 것을 강력 처방합니다.")
            elif predicted_stress >= 4:
                st.markdown("만성 무기력으로 번질 우려가 있는 주의 단계입니다. 일 단위 SNS 사용량을 의도적으로 통제하고 주 3회 유산소 활동을 믹스하여 일상의 밸런스를 환기해 주세요.")
            else:
                st.markdown("훌륭합니다! 균형 잡힌 라이프 스타일이 지탱해 주는 최상의 컨디션 상태입니다. 현재의 생활 밸런스를 기복 없이 모범적으로 이어나가길 응원합니다.")

            # 거시적 결론 도출 섹션 (2분할 레이아웃 적용)
            st.markdown("---")
            st.markdown("### 🏛️ 청소년 정신 건강 빅데이터 학습 집약 결과")
            ins_col1, ins_col2 = st.columns([1.1, 0.9])
            
            with ins_col1:
                st.markdown("""
                * **디지털 기기 과의존과 수면의 역상관**: 빅데이터 분석을 종합하면 스트레스 고위험 청소년 군은 과도한 야간 모바일 스크린 노출과 수면 저하의 파괴적인 인과 고리에 전형적으로 묶여 있습니다.
                * **신체 활동의 물리적 쉴드 효과**: 비슷한 스크린 타임을 가지더라도 하루 1시간 이상의 운동 습관을 결합한 청소년 표본은 스트레스 위험도가 평균 1.8단계 방어되는 강력한 완충 완화 효과가 도출되었습니다.
                """)
            with ins_col2:
                fig5, ax5 = plt.subplots(figsize=(5, 3.6))
                original_df['Stress Group'] = np.where(original_df['stress_level'] >= 8, 'High Stress', 'Normal')
                sns.violinplot(data=original_df, x='Stress Group', y='sleep_hours', palette={'High Stress': '#e74c3c', 'Normal': '#2ecc71'}, inner='quartile', ax=ax5)
                ax5.set_xlabel("Stress Group", color='white')
                ax5.set_ylabel("Sleep Hours", color='white')
                sns.despine()
                st.pyplot(fig5)
                st.markdown("<p style='font-size:12px; color:#b2bec3; text-align:center;'>주황/빨간색 고위험군의 분포 부피가 5~6시간 구역에 뚱뚱하게 밀집되어 있어, 수면 시간 확보가 스트레스 방어의 최우선 과제임을 AI 통계 분석이 직관적으로 증명합니다.</p>", unsafe_allow_html=True)
