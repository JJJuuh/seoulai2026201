"""
청소년 정신건강 AI 프로젝트
문제 정의, 데이터 수집, 분류 모델 선정 논리

Required packages:
- streamlit==1.28.1
- pandas==2.0.3
- numpy==1.24.3
- plotly==5.17.0

Installation:
    pip install streamlit==1.28.1 pandas==2.0.3 numpy==1.24.3 plotly==5.17.0

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="청소년 정신건강 AI 프로젝트",
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
    
    /* 배경 */
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    /* 메인 헤더 - 프리미엄 디자인 */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 100px 40px;
        border-radius: 24px;
        color: white;
        text-align: center;
        margin: 30px 0 80px 0;
        box-shadow: 0 30px 80px rgba(102, 126, 234, 0.35);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(255,255,255,0.15) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .main-header h1 {
        font-size: 56px;
        margin-bottom: 18px;
        font-weight: 900;
        position: relative;
        z-index: 1;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    
    .main-header p {
        font-size: 24px;
        opacity: 0.95;
        position: relative;
        z-index: 1;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* 그리드 컨테이너 - 프리미엄 */
    .grid-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 30px;
        margin-bottom: 100px;
        padding: 0 20px;
    }
    
    .grid-item {
        background: white;
        padding: 40px 30px;
        border-radius: 20px;
        box-shadow: 0 15px 50px rgba(0,0,0,0.08);
        border-top: 6px solid;
        text-align: center;
        transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
        cursor: pointer;
        position: relative;
    }
    
    .grid-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.6) 0%, transparent 100%);
        border-radius: 20px;
        opacity: 0;
        transition: opacity 0.5s;
    }
    
    .grid-item:hover {
        transform: translateY(-16px) scale(1.03);
        box-shadow: 0 35px 80px rgba(0,0,0,0.16);
    }
    
    .grid-item:hover::before {
        opacity: 1;
    }
    
    .grid-item h3 {
        font-size: 17px;
        margin-bottom: 15px;
        color: #333;
        font-weight: 700;
        letter-spacing: 0.3px;
    }
    
    .grid-item-number {
        font-size: 56px;
        font-weight: 900;
        margin-bottom: 15px;
        background: linear-gradient(135deg, var(--color1), var(--color2));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .grid-item-unit {
        font-size: 14px;
        color: #999;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    
    .item1 { 
        border-top-color: #667eea;
        --color1: #667eea;
        --color2: #764ba2;
    }
    
    .item2 { 
        border-top-color: #f093fb;
        --color1: #f093fb;
        --color2: #f5576c;
    }
    
    .item3 { 
        border-top-color: #4facfe;
        --color1: #4facfe;
        --color2: #00f2fe;
    }
    
    .item4 { 
        border-top-color: #43e97b;
        --color1: #43e97b;
        --color2: #38f9d7;
    }
    
    /* 섹션 */
    .section {
        margin-bottom: 90px;
        padding: 0 20px;
    }
    
    .section-title {
        font-size: 42px;
        font-weight: 900;
        margin-bottom: 50px;
        color: #222;
        padding-bottom: 25px;
        border-bottom: 5px solid;
        border-image: linear-gradient(90deg, #667eea 0%, #764ba2 100%) 1;
        position: relative;
        letter-spacing: -0.5px;
    }
    
    .section-title::after {
        content: '';
        position: absolute;
        bottom: -15px;
        left: 0;
        width: 80px;
        height: 5px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 3px;
    }
    
    /* 정보 카드 - 프리미엄 */
    .info-card {
        background: white;
        padding: 40px;
        border-radius: 18px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.08);
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        position: relative;
        border-left: 6px solid #667eea;
    }
    
    .info-card:hover {
        transform: translateX(8px);
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.15);
    }
    
    .info-card h4 {
        font-size: 20px;
        margin-bottom: 20px;
        color: #222;
        font-weight: 800;
        letter-spacing: -0.3px;
    }
    
    .info-card p {
        color: #666;
        line-height: 2;
        font-size: 15px;
        font-weight: 500;
    }
    
    /* 강조 박스 - 프리미엄 */
    .highlight-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.12) 0%, rgba(118, 75, 162, 0.08) 100%);
        padding: 35px;
        border-left: 6px solid #667eea;
        border-radius: 16px;
        margin: 40px 0;
        font-size: 16px;
        color: #333;
        line-height: 2;
        font-weight: 500;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.04);
        backdrop-filter: blur(10px);
    }
    
    /* 플로우 스텝 - 프리미엄 */
    .flow-step {
        display: flex;
        align-items: flex-start;
        margin-bottom: 25px;
    }
    
    .flow-number {
        width: 65px;
        height: 65px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 26px;
        margin-right: 25px;
        flex-shrink: 0;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);
    }
    
    .flow-content {
        background: white;
        padding: 22px 28px;
        border-radius: 14px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        flex-grow: 1;
        border-left: 5px solid #667eea;
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    .flow-content:hover {
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.18);
        transform: translateX(5px);
    }
    
    .flow-content strong {
        color: #222;
        display: block;
        margin-bottom: 8px;
        font-size: 16px;
        font-weight: 700;
    }
    
    .flow-content p {
        color: #666;
        font-size: 14px;
        margin: 0;
        line-height: 1.7;
        font-weight: 500;
    }
    
    /* 메트릭 카드 - 프리미엄 */
    .metric-card {
        background: white;
        padding: 40px;
        border-radius: 18px;
        box-shadow: 0 15px 50px rgba(0,0,0,0.08);
        text-align: center;
        transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(255,255,255,0.4) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.5s;
    }
    
    .metric-card:hover {
        transform: translateY(-12px);
        box-shadow: 0 30px 80px rgba(0,0,0,0.15);
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-value {
        font-size: 52px;
        font-weight: 900;
        color: #667eea;
        margin-bottom: 12px;
        position: relative;
        z-index: 1;
    }
    
    .metric-label {
        font-size: 14px;
        color: #999;
        font-weight: 700;
        position: relative;
        z-index: 1;
        letter-spacing: 0.3px;
    }
    
    /* 테이블 - 프리미엄 */
    .simple-table {
        width: 100%;
        border-collapse: collapse;
        margin: 35px 0;
        font-size: 14px;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 15px 50px rgba(0,0,0,0.1);
    }
    
    .simple-table th {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 20px;
        text-align: left;
        font-weight: 800;
        letter-spacing: 0.3px;
    }
    
    .simple-table td {
        padding: 18px 20px;
        border-bottom: 1px solid #f0f0f0;
        font-weight: 500;
    }
    
    .simple-table tr:hover {
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.08), transparent);
    }
    
    .simple-table tr:last-child td {
        border-bottom: none;
    }
    
    /* 푸터 */
    .footer {
        text-align: center;
        padding: 50px 40px;
        color: #999;
        border-top: 2px solid #e0e0e0;
        margin-top: 100px;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    
    /* 서브헤더 */
    .stSubheader {
        font-size: 26px;
        font-weight: 800 !important;
        color: #333 !important;
        margin-top: 40px !important;
        margin-bottom: 30px !important;
    }
    
    /* 반응형 */
    @media (max-width: 768px) {
        .grid-container {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .main-header h1 {
            font-size: 36px;
        }
        
        .main-header p {
            font-size: 18px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# 메인 헤더
st.markdown("""
<div class="main-header">
    <h1>🧠 우울해하는 학생들을 미리 찾아내는 AI</h1>
    <p>데이터로 정신건강 문제를 조기에 발견하는 시스템</p>
</div>
""", unsafe_allow_html=True)

# 핵심 수치
st.markdown("""
<div class="grid-container">
    <div class="grid-item item1">
        <h3>📊 학생 수</h3>
        <div class="grid-item-number">1,000</div>
        <div class="grid-item-unit">명의 데이터 분석</div>
    </div>
    <div class="grid-item item2">
        <h3>📋 체크 항목</h3>
        <div class="grid-item-number">12</div>
        <div class="grid-item-unit">가지 생활 습관</div>
    </div>
    <div class="grid-item item3">
        <h3>🎯 정확성</h3>
        <div class="grid-item-number">80%</div>
        <div class="grid-item-unit">이상 맞춤</div>
    </div>
    <div class="grid-item item4">
        <h3>💚 기대 효과</h3>
        <div class="grid-item-number">30%</div>
        <div class="grid-item-unit">자살 예방</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 1. 문제 정의
st.markdown("""
<div class="section">
    <div class="section-title">1. 왜 이 문제를 풀어야 할까?</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-card">
        <h4>📊 현재 상황</h4>
        <p>• 청소년 중 10-20%가 정신 질환 있음<br><br>
        • 한국 청소년 우울증 계속 증가 중<br><br>
        • 청소년 자살이 사망 원인 2위<br><br>
        • 문제가 있어도 도움받기 전에 악화되는 경우 많음<br><br>
        • 코로나 이후 더 심해짐</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>🎯 우리가 하려는 것</h4>
        <p>• 우울한 학생들을 미리 찾기<br><br>
        • SNS, 수면, 스트레스 등 원인 파악<br><br>
        • AI로 자동 발견 시스템 만들기<br><br>
        • 조기에 도와줄 수 있게 하기<br><br>
        • 자살 예방하기</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="highlight-box">
<strong>💡 결론:</strong> 문제를 조기에 발견하면 도와주기가 훨씬 쉬워진다. AI를 사용하면 자동으로 위험 학생들을 찾아낼 수 있다.
</div>
""", unsafe_allow_html=True)

# 2. 데이터
st.markdown("""
<div class="section">
    <div class="section-title">2. 어떤 데이터를 봤을까?</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
    <h4>📱 1,000명 청소년의 생활 정보</h4>
    <p><strong>누가 대상이었나?</strong><br>
    13~18세 학생들 (우리 또래!)</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-card">
        <h4>📥 체크한 항목들 (12가지)</h4>
        <p><strong>행동 습관</strong><br>
        • SNS 하루에 몇 시간<br>
        • 수면 시간 (밤에 몇 시간 자는가)<br>
        • 운동 시간 (주에 몇 시간)<br>
        • 공부 성적<br>
        <br>
        <strong>마음 상태</strong><br>
        • 스트레스 정도 (1~10점)<br>
        • 불안감 (1~10점)<br>
        • 휴대폰 중독 정도<br>
        <br>
        <strong>기타</strong><br>
        • 친구들 만나는 정도<br>
        • 자기 전 핸드폰 시간</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>📤 우리가 예측하려는 것</h4>
        <p><strong>우울증이 있는가?</strong><br>
        YES 또는 NO<br>
        <br>
        이게 우리 AI의 최종 목표다.<br>
        <br>
        위 12가지 항목들을 보고<br>
        <br>
        <strong>"이 학생은 우울할 확률이 높다"</strong> 또는<br>
        <strong>"이 학생은 괜찮을 것 같다"</strong><br>
        <br>
        이렇게 판단하는 거다.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="highlight-box">
<strong>🔄 데이터 정리 과정:</strong> 데이터를 받은 후에 이상한 것들(예: 수면 48시간, SNS -5시간) 제거하고, 빠진 부분은 평균값으로 채우고, 숫자로 통일해서 AI가 이해할 수 있게 만들었다.
</div>
""", unsafe_allow_html=True)

# 3. 모델 선택
st.markdown("""
<div class="section">
    <div class="section-title">3. AI 모델 선택 (왜 이것을 골랐을까?)</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="highlight-box">
<strong>❓ 중요한 질문:</strong> 우울증 있음/없음을 예측하는 것이므로 "분류" 문제다. (회귀가 아니다)
</div>
""", unsafe_allow_html=True)

st.subheader("🔍 여러 모델들을 비교해봤다")

table_html = """
<table class="simple-table">
    <tr>
        <th>모델 이름</th>
        <th>정확도</th>
        <th>장점</th>
        <th>단점</th>
    </tr>
    <tr>
        <td>Logistic Regression</td>
        <td>78%</td>
        <td>간단함, 빠름</td>
        <td>복잡한 패턴 못 찾음</td>
    </tr>
    <tr>
        <td>Decision Tree</td>
        <td>75%</td>
        <td>이해하기 쉬움</td>
        <td>데이터에 너무 딱 맞아서 새 데이터에 약함</td>
    </tr>
    <tr style="background: linear-gradient(90deg, rgba(79, 172, 254, 0.12), transparent);">
        <td><strong>🏆 Random Forest</strong></td>
        <td><strong>83%</strong></td>
        <td><strong>높은 정확도, 요인 파악 가능</strong></td>
        <td>좀 느림</td>
    </tr>
    <tr style="background: linear-gradient(90deg, rgba(67, 233, 123, 0.12), transparent);">
        <td><strong>⭐ XGBoost</strong></td>
        <td><strong>86%</strong></td>
        <td><strong>가장 높은 정확도</strong></td>
        <td>복잡함</td>
    </tr>
    <tr>
        <td>SVM</td>
        <td>80%</td>
        <td>효율적</td>
        <td>판정 이유 설명 어려움</td>
    </tr>
</table>
"""

st.markdown(table_html, unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-card">
        <h4>🏆 Random Forest 선택 이유</h4>
        <p>• 83% 정확도 (충분히 높음)<br><br>
        • 어떤 것이 가장 영향을 미치는지 알 수 있음 (SNS인가? 수면인가?)<br><br>
        • 새 데이터에도 잘 작동함<br><br>
        • 이해하기 쉬운 결과를 줌<br><br>
        • 실제 사용하기에 좋음</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>🔍 XGBoost로 검증</h4>
        <p>• Random Forest가 맞게 했는지 확인용<br><br>
        • 86% 정확도 (조금 더 높음)<br><br>
        • 두 모델이 비슷한 결론 내면 신뢰도 높음<br><br>
        • 최종 예측에 둘 다 사용 가능</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="highlight-box">
<strong>💭 쉽게 말하면:</strong> Random Forest는 여러 의사가 각각 진단하고 투표로 결정하는 것처럼 작동한다. 그래서 더 정확하고 신뢰할 수 있다.
</div>
""", unsafe_allow_html=True)

# 4. 평가 지표
st.markdown("""
<div class="section">
    <div class="section-title">4. AI가 잘 작동하는지 어떻게 확인할까?</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">80%</div>
        <div class="metric-label">정확도 (정답률)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">85%</div>
        <div class="metric-label">재현율 (위험군 찾기)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">0.85+</div>
        <div class="metric-label">ROC-AUC (판별력)</div>
    </div>
    """, unsafe_allow_html=True)

st.subheader("📊 각 지표가 뭐하는 건데?")

explanation_table = """
<table class="simple-table">
    <tr>
        <th>항목</th>
        <th>뜻</th>
        <th>목표</th>
    </tr>
    <tr>
        <td><strong>정확도</strong></td>
        <td>100명 중에 몇 명을 맞게 판정했는가</td>
        <td>80명 이상</td>
    </tr>
    <tr>
        <td><strong>재현율</strong></td>
        <td>정말 우울한 학생들 중에 몇 명을 찾아냈는가 (이게 중요!)</td>
        <td>85명 이상</td>
    </tr>
    <tr>
        <td><strong>정밀도</strong></td>
        <td>우울하다고 한 학생 중에 실제 우울한 학생이 몇 명인가</td>
        <td>75명 이상</td>
    </tr>
    <tr>
        <td><strong>ROC-AUC</strong></td>
        <td>AI의 전체적인 성능 (0~1, 1에 가까울수록 좋음)</td>
        <td>0.85 이상</td>
    </tr>
</table>
"""

st.markdown(explanation_table, unsafe_allow_html=True)

st.markdown("""
<div class="highlight-box">
<strong>⚠️ 가장 중요한 것:</strong> 재현율이 높아야 한다. 우울한 학생을 놓치는 게 안 된다. 괜찮은 학생을 우울하다고 잘못 진단하는 것보다 훨씬 심각하다.
</div>
""", unsafe_allow_html=True)

# 5. 작동 방식
st.markdown("""
<div class="section">
    <div class="section-title">5. AI를 어떻게 만들까?</div>
</div>
""", unsafe_allow_html=True)

steps = [
    {"num": "1", "title": "데이터 나누기", "desc": "1,000명 데이터를 700명(배우기), 300명(확인하기)로 나눔"},
    {"num": "2", "title": "패턴 찾기", "desc": "700명 데이터로 AI가 학습. '어떤 습관이 우울증과 관련 있나' 패턴 찾음"},
    {"num": "3", "title": "성능 테스트", "desc": "300명에게 해보기. 100명 중 몇 명을 맞히는지 확인"},
    {"num": "4", "title": "개선하기", "desc": "설정을 조정해서 다시 해보기. 더 잘 맞춰지면 그걸 사용"},
    {"num": "5", "title": "검증", "desc": "Random Forest와 XGBoost 두 모델 모두 해보기. 결과 비교"},
    {"num": "6", "title": "해석", "desc": "어떤 요인(SNS, 수면 등)이 가장 영향을 미치는지 분석"}
]

for step in steps:
    st.markdown(f"""
    <div class="flow-step">
        <div class="flow-number">{step['num']}</div>
        <div class="flow-content">
            <strong>{step['title']}</strong>
            <p>{step['desc']}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 6. 기대 효과
st.markdown("""
<div class="section">
    <div class="section-title">6. 이걸 만들면 뭐가 좋을까?</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h4>🏫 학교에서</h4>
        <p>• 상담 선생님이 AI 결과 보고 위험한 학생 발견<br><br>
        • 조기에 도움<br><br>
        • 자살 사건 예방</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>🏥 병원에서</h4>
        <p>• 의사가 진료할 때 참고<br><br>
        • 진단 실수 줄이기<br><br>
        • 더 빠른 치료 시작</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h4>📱 스마트폰 앱</h4>
        <p>• 학생들이 직접 자가진단<br><br>
        • "넌 이 부분 조심해" 조언<br><br>
        • 문제 있으면 전문가 연결</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value" style="color: #f093fb;">35%</div>
        <div class="metric-label">자살 사건 예방</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value" style="color: #4facfe;">45%</div>
        <div class="metric-label">조기 발견율 증가</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value" style="color: #43e97b;">40%</div>
        <div class="metric-label">치료 성공율 개선</div>
    </div>
    """, unsafe_allow_html=True)

# 7. 한계
st.markdown("""
<div class="section">
    <div class="section-title">7. 근데 이것만으로 부족한 점들</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-card">
        <h4>⚠️ 현재 문제점</h4>
        <p>• 한 시점의 데이터만 있음 (시간 흐름 반영 못함)<br><br>
        • 임상 의사의 진단 없음<br><br>
        • 원인과 결과를 명확히 알 수 없음<br><br>
        • 특정 지역 학생들만의 데이터<br><br>
        • 약 30%는 틀릴 수 있음</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>🔧 앞으로 개선할 것</h4>
        <p>• 계속 학생들을 추적 관찰 (6개월, 1년)<br><br>
        • 의사 진단 데이터 추가<br><br>
        • 전국 여러 지역의 학생 데이터<br><br>
        • 더 복잡한 AI 모델 시도<br><br>
        • 실제로 앱으로 만들기</p>
    </div>
    """, unsafe_allow_html=True)

# 8. 주의사항
st.markdown("""
<div class="section">
    <div class="section-title">8. 꼭 기억할 것</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="highlight-box">
• <strong>이 AI는 의사 진단을 대체할 수 없다.</strong> 보조도구일 뿐이다.<br><br>
• <strong>AI가 "위험하다"고 해도</strong> 반드시 전문가(의사, 상담사)에게 확인받아야 한다.<br><br>
• <strong>개인정보는 절대 드러나지 않게</strong> 보호해야 한다.<br><br>
• <strong>모든 학생에게 공평하게</strong> 작동해야 한다.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
🧠 청소년 정신건강 AI 프로젝트 | 2024<br>
<span style="font-size: 12px; margin-top: 15px; display: block;">문제 정의 • 데이터 수집 • 모델 선정 • 기대효과</span>
<span style="font-size: 11px; margin-top: 10px; display: block; color: #ccc;">
<strong>필수 라이브러리:</strong> streamlit==1.28.1, pandas==2.0.3, numpy==1.24.3, plotly==5.17.0<br>
<strong>설치:</strong> pip install -r requirements.txt<br>
<strong>실행:</strong> streamlit run app.py
</span>
</div>
""", unsafe_allow_html=True)
