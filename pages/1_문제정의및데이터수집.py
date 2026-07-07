import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="청소년 정신건강 AI 모델 프로젝트",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 60px 40px;
        border-radius: 0;
        color: white;
        text-align: center;
        margin-bottom: 40px;
    }
    
    .main-header h1 {
        font-size: 48px;
        margin-bottom: 15px;
        font-weight: bold;
    }
    
    .main-header p {
        font-size: 20px;
        opacity: 0.95;
    }
    
    .grid-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin-bottom: 50px;
    }
    
    .grid-item {
        background: white;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-top: 4px solid;
        text-align: center;
        transition: transform 0.3s;
    }
    
    .grid-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .grid-item h3 {
        font-size: 18px;
        margin-bottom: 15px;
        color: #333;
    }
    
    .grid-item-number {
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .grid-item-unit {
        font-size: 14px;
        color: #666;
    }
    
    .item1 { border-top-color: #667eea; }
    .item1 .grid-item-number { color: #667eea; }
    
    .item2 { border-top-color: #f093fb; }
    .item2 .grid-item-number { color: #f093fb; }
    
    .item3 { border-top-color: #4facfe; }
    .item3 .grid-item-number { color: #4facfe; }
    
    .item4 { border-top-color: #43e97b; }
    .item4 .grid-item-number { color: #43e97b; }
    
    .section {
        margin-bottom: 50px;
    }
    
    .section-title {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 30px;
        color: #333;
        padding-bottom: 15px;
        border-bottom: 3px solid #667eea;
    }
    
    .two-column {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 30px;
        margin-bottom: 30px;
    }
    
    .info-card {
        background: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .info-card h4 {
        font-size: 18px;
        margin-bottom: 15px;
        color: #333;
    }
    
    .info-card ul {
        list-style: none;
        padding-left: 0;
    }
    
    .info-card li {
        padding: 8px 0;
        color: #555;
        line-height: 1.6;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .info-card li:before {
        content: "▸ ";
        color: #667eea;
        font-weight: bold;
        margin-right: 8px;
    }
    
    .info-card li:last-child {
        border-bottom: none;
    }
    
    .highlight-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 20px;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        margin: 20px 0;
    }
    
    .highlight-box strong {
        color: #667eea;
    }
    
    .flow-step {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .flow-number {
        width: 50px;
        height: 50px;
        background: #667eea;
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 20px;
        margin-right: 20px;
        flex-shrink: 0;
    }
    
    .flow-content {
        background: white;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        flex-grow: 1;
    }
    
    .flow-content strong {
        color: #333;
        display: block;
        margin-bottom: 5px;
    }
    
    .flow-content p {
        color: #666;
        font-size: 14px;
        margin: 0;
    }
    
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
    }
    
    .comparison-table th {
        background: #667eea;
        color: white;
        padding: 15px;
        text-align: left;
        font-weight: bold;
    }
    
    .comparison-table td {
        padding: 15px;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .comparison-table tr:hover {
        background: #f9f9f9;
    }
    
    .metric-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin-bottom: 30px;
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        color: #667eea;
        margin-bottom: 8px;
    }
    
    .metric-label {
        font-size: 14px;
        color: #666;
    }
    
    .footer {
        text-align: center;
        padding: 30px;
        color: #999;
        border-top: 1px solid #e0e0e0;
        margin-top: 60px;
    }
    </style>
""", unsafe_allow_html=True)

# ===== 메인 헤더 =====
st.markdown("""
<div class="main-header">
    <h1>🧠 청소년 정신건강 AI 예측 모델</h1>
    <p>데이터 기반 우울증 조기 발견 시스템</p>
</div>
""", unsafe_allow_html=True)

# ===== 핵심 수치 =====
st.markdown("""
<div class="grid-container">
    <div class="grid-item item1">
        <h3>분석 대상</h3>
        <div class="grid-item-number">1,000</div>
        <div class="grid-item-unit">명의 청소년</div>
    </div>
    <div class="grid-item item2">
        <h3>입력 변수</h3>
        <div class="grid-item-number">12</div>
        <div class="grid-item-unit">개의 생활 지표</div>
    </div>
    <div class="grid-item item3">
        <h3>목표 정확도</h3>
        <div class="grid-item-number">80%</div>
        <div class="grid-item-unit">이상 달성</div>
    </div>
    <div class="grid-item item4">
        <h3>예상 효과</h3>
        <div class="grid-item-number">30%</div>
        <div class="grid-item-unit">자살 예방율</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===== 1. 문제 정의 =====
st.markdown("""
<div class="section">
    <div class="section-title">1. 연구 배경 및 문제 정의</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-card">
        <h4>현황</h4>
        <ul>
            <li>전 세계 청소년 10-20% 정신질환 유병</li>
            <li>한국 청소년 우울증 유병률 15-20% (상승 추세)</li>
            <li>청소년 자살은 사망 원인 2위</li>
            <li>코로나 이후 정신건강 문제 심화</li>
            <li>조기 발견 및 개입 시스템 부족</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>연구 목표</h4>
        <ul>
            <li>생활 습관 데이터로 우울증 조기 예측</li>
            <li>주요 위험 요인 규명</li>
            <li>위험군 청소년 자동 식별 시스템 구축</li>
            <li>맞춤형 개입 전략 수립 근거 제공</li>
            <li>자살 예방을 위한 사전 대응 체계 마련</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ===== 2. 데이터 =====
st.markdown("""
<div class="section">
    <div class="section-title">2. 데이터 정의 및 수집</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 0.8])

with col1:
    st.markdown("""
    <div class="info-card">
        <h4>입력 변수 (Features)</h4>
        <ul>
            <li><strong>행동 요인:</strong> SNS 사용 시간, 수면 시간, 신체활동, 학업 성과</li>
            <li><strong>환경 요인:</strong> 화면 노출 시간 (취침 전), 플랫폼 유형</li>
            <li><strong>심리 요인:</strong> 스트레스 수준, 불안 수준, 중독 수준</li>
            <li><strong>사회 요인:</strong> 사회적 상호작용 정도</li>
            <li><strong>인구통계:</strong> 나이, 성별</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>목표 변수 (Target)</h4>
        <ul>
            <li><strong>우울증 진단:</strong> Yes/No</li>
            <li><strong>예측 형태:</strong> 이진 분류</li>
            <li><strong>결과:</strong> 확률값 (0-1)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="highlight-box">
    <strong>데이터 특성:</strong> 1,000명 청소년 대상 자기보고식 설문조사 (Cross-sectional study). 연령 13-18세, 결측률 5% 이하, 온라인 수집 방식
</div>
""", unsafe_allow_html=True)

# ===== 데이터 전처리 =====
st.subheader("데이터 전처리 프로세스")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card" style="background: linear-gradient(135deg, #f093fb15 0%, #f5576c15 100%);">
        <h4>1단계: 결측치 처리</h4>
        <ul>
            <li>수치형: 중앙값 대체</li>
            <li>범주형: 최빈값 대체</li>
            <li>기준: < 5% 결측률</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card" style="background: linear-gradient(135deg, #4facfe15 0%, #00f2fe15 100%);">
        <h4>2단계: 이상치 제거</h4>
        <ul>
            <li>IQR 방식 적용</li>
            <li>범위 외 값 클리핑</li>
            <li>논리 검증</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card" style="background: linear-gradient(135deg, #43e97b15 0%, #38f9d715 100%);">
        <h4>3단계: 정규화 & 인코딩</h4>
        <ul>
            <li>수치형 스케일링</li>
            <li>범주형 인코딩</li>
            <li>특성 표준화</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ===== 3. 모델 선택 =====
st.markdown("""
<div class="section">
    <div class="section-title">3. 모델 선정 논리</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="highlight-box">
    <strong>문제 유형:</strong> 이진 분류 (Binary Classification)
    <br>목표 변수가 범주형(우울증 있음/없음)이므로 회귀가 아닌 분류 모델 적용
</div>
""", unsafe_allow_html=True)

# 모델 비교
st.subheader("후보 모델 비교")

model_data = {
    "모델": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "SVM"],
    "정확도": ["78%", "75%", "83%", "86%", "80%"],
    "해석성": ["높음", "높음", "중간", "중간", "낮음"],
    "훈련속도": ["빠름", "빠름", "보통", "보통", "느림"],
    "과적합 위험": ["낮음", "높음", "낮음", "낮음", "중간"],
    "선택": ["", "", "✓", "✓", ""]
}

model_df = pd.DataFrame(model_data)
st.dataframe(model_df, use_container_width=True, hide_index=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-card">
        <h4>Random Forest (1순위)</h4>
        <ul>
            <li>다양한 트리로 과적합 방지</li>
            <li>특성 중요도 제공 → 위험요인 파악</li>
            <li>비선형 관계 포착 가능</li>
            <li>불균형 클래스 처리 우수</li>
            <li>튜닝이 상대적으로 쉬움</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>XGBoost (검증 모델)</h4>
        <ul>
            <li>최고 성능 달성 (86%)</li>
            <li>강력한 정규화로 과적합 방지</li>
            <li>순차적 개선을 통한 성능 최적화</li>
            <li>클래스 불균형 처리 능력</li>
            <li>Random Forest 성능 검증용</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ===== 4. 평가 지표 =====
st.markdown("""
<div class="section">
    <div class="section-title">4. 모델 평가 지표</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("정확도 (Accuracy)", "80%+", "전체 예측 중 정확한 비율")

with col2:
    st.metric("재현율 (Recall)", "85%+", "실제 위험군 발견 비율")

with col3:
    st.metric("ROC-AUC", "0.85+", "모델 판별 능력")

evaluation_details = """
| 평가 지표 | 의미 | 목표 |
|----------|------|------|
| **정확도** | 전체 예측 중 맞은 비율 | 80% 이상 |
| **정밀도** | "양성"이라 한 것 중 실제 양성 비율 | 75% 이상 |
| **재현율** | 실제 양성 중 발견한 비율 | 85% 이상 (중요) |
| **F1-Score** | 정밀도와 재현율의 조화평균 | 80% 이상 |
| **ROC-AUC** | 모델의 전체 판별 능력 | 0.85 이상 |
| **특이도** | 실제 음성 중 올바르게 판정한 비율 | 75% 이상 |
"""

st.markdown(evaluation_details)

st.markdown("""
<div class="highlight-box">
    <strong>중요 포인트:</strong> 재현율(Recall)을 가장 중시하는 이유는 우울증 위험군을 놓칠 경우 매우 심각한 결과가 초래될 수 있기 때문. 거짓 음성(위험군을 정상으로 판정)의 비용이 거짓 양성(정상을 위험군으로 판정)보다 훨씬 크다.
</div>
""", unsafe_allow_html=True)

# ===== 5. 개발 프로세스 =====
st.markdown("""
<div class="section">
    <div class="section-title">5. 모델 개발 프로세스</div>
</div>
""", unsafe_allow_html=True)

steps = [
    {
        "title": "데이터 분할",
        "desc": "Train (70%) / Validation (10%) / Test (20%) - Stratified K-Fold (5-fold)"
    },
    {
        "title": "특성 공학",
        "desc": "다항 특성 생성, 상호작용 항 추가, 특성 선택 (SelectKBest)"
    },
    {
        "title": "모델 학습",
        "desc": "Random Forest, XGBoost 학습 - GridSearchCV로 하이퍼파라미터 튜닝"
    },
    {
        "title": "교차 검증",
        "desc": "5-Fold Cross-Validation으로 일반화 성능 평가"
    },
    {
        "title": "성능 평가",
        "desc": "Confusion Matrix, 정밀도-재현율 곡선, ROC 곡선 분석"
    },
    {
        "title": "모델 해석",
        "desc": "특성 중요도, SHAP 분석으로 위험 요인 규명"
    }
]

for i, step in enumerate(steps, 1):
    st.markdown(f"""
    <div class="flow-step">
        <div class="flow-number">{i}</div>
        <div class="flow-content">
            <strong>{step['title']}</strong>
            <p>{step['desc']}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ===== 6. 기대효과 =====
st.markdown("""
<div class="section">
    <div class="section-title">6. 기대효과 및 활용 방안</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value" style="color: #f093fb;">35%</div>
        <div class="metric-label">자살 예방율</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value" style="color: #4facfe;">45%</div>
        <div class="metric-label">조기 개입율</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value" style="color: #43e97b;">40%</div>
        <div class="metric-label">치료 성공율 개선</div>
    </div>
    """, unsafe_allow_html=True)

st.subheader("활용 시나리오")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h4>학교 현장</h4>
        <ul>
            <li>학생 정신건강 선별검사</li>
            <li>위험군 자동 식별</li>
            <li>상담사 알림</li>
            <li>조기 개입</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>의료 기관</h4>
        <ul>
            <li>임상의 진단 보조</li>
            <li>위험도 평가 지원</li>
            <li>치료 계획 수립</li>
            <li>진료 효율성 증대</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h4>모바일 앱</h4>
        <ul>
            <li>자가진단 기능</li>
            <li>실시간 피드백</li>
            <li>개선 권고안 제시</li>
            <li>전문가 연결</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ===== 7. 한계 및 개선안 =====
st.markdown("""
<div class="section">
    <div class="section-title">7. 연구 한계 및 개선 방안</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-card" style="border-left-color: #f5576c; background: linear-gradient(135deg, #f5576c15 0%, #f093fb15 100%);">
        <h4>현재 한계</h4>
        <ul>
            <li>Cross-sectional 설계 (특정 시점 조사)</li>
            <li>자기보고식 데이터 (주관성)</li>
            <li>인과관계 도출 불가</li>
            <li>특정 지역 표본 편향</li>
            <li>임상 진단 미포함</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card" style="border-left-color: #43e97b; background: linear-gradient(135deg, #43e97b15 0%, #38f9d715 100%);">
        <h4>개선 계획</h4>
        <ul>
            <li>Longitudinal Study로 업그레이드 (추적조사)</li>
            <li>임상 진단 데이터 통합</li>
            <li>다국가 표본 확대</li>
            <li>딥러닝 모델 도입 검토</li>
            <li>실시간 모니터링 시스템</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ===== 로드맵 =====
st.subheader("개발 로드맵")

roadmap_col1, roadmap_col2, roadmap_col3, roadmap_col4, roadmap_col5 = st.columns(5)

phases = [
    {"phase": "Phase 1", "duration": "1-2개월", "activity": "문제 정의 & 데이터 수집", "color": "#667eea"},
    {"phase": "Phase 2", "duration": "2-3개월", "activity": "모델 개발 & 학습", "color": "#f093fb"},
    {"phase": "Phase 3", "duration": "2-3개월", "activity": "검증 & 평가", "color": "#4facfe"},
    {"phase": "Phase 4", "duration": "2-3개월", "activity": "서비스 개발", "color": "#43e97b"},
    {"phase": "Phase 5", "duration": "6개월+", "activity": "임상 검증 & 모니터링", "color": "#FFA500"}
]

for i, phase_info in enumerate(phases):
    with [roadmap_col1, roadmap_col2, roadmap_col3, roadmap_col4, roadmap_col5][i]:
        st.markdown(f"""
        <div class="info-card" style="background: linear-gradient(135deg, {phase_info['color']}20 0%, {phase_info['color']}10 100%); border-left-color: {phase_info['color']};">
            <div style="font-weight: bold; color: {phase_info['color']}; margin-bottom: 8px;">{phase_info['phase']}</div>
            <div style="font-size: 12px; color: #666; margin-bottom: 10px;">{phase_info['duration']}</div>
            <div style="font-size: 13px; color: #333;">{phase_info['activity']}</div>
        </div>
        """, unsafe_allow_html=True)

# ===== 윤리 및 주의사항 =====
st.markdown("""
<div class="section">
    <div class="section-title">8. 윤리 고려사항</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-card">
        <h4>프라이버시 보호</h4>
        <ul>
            <li>개인정보 암호화</li>
            <li>익명처리 (De-identification)</li>
            <li>접근 제한 (Role-based Access Control)</li>
            <li>감사 기록 (Audit Log)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>공정성 및 책임성</h4>
        <ul>
            <li>의료 전문가의 최종 판단 필수</li>
            <li>치료 대체 불가 (보조 도구)</li>
            <li>편향 모니터링</li>
            <li>설명 가능성 (Explainability)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    청소년 정신건강 AI 예측 모델 | 2024
</div>
""", unsafe_allow_html=True)
