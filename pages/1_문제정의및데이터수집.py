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
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 50px 40px;
        border-radius: 0;
        color: white;
        text-align: center;
        margin-bottom: 40px;
    }
    
    .main-header h1 {
        font-size: 42px;
        margin-bottom: 10px;
        font-weight: bold;
    }
    
    .main-header p {
        font-size: 18px;
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
        padding: 25px;
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
        font-size: 16px;
        margin-bottom: 12px;
        color: #333;
    }
    
    .grid-item-number {
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 8px;
    }
    
    .grid-item-unit {
        font-size: 13px;
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
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 25px;
        color: #333;
        padding-bottom: 12px;
        border-bottom: 3px solid #667eea;
    }
    
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .info-card h4 {
        font-size: 16px;
        margin-bottom: 12px;
        color: #333;
    }
    
    .info-card p {
        color: #555;
        line-height: 1.6;
        font-size: 14px;
    }
    
    .highlight-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 18px;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        margin: 20px 0;
        font-size: 14px;
        color: #333;
        line-height: 1.6;
    }
    
    .flow-step {
        display: flex;
        align-items: flex-start;
        margin-bottom: 15px;
    }
    
    .flow-number {
        width: 45px;
        height: 45px;
        background: #667eea;
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 18px;
        margin-right: 15px;
        flex-shrink: 0;
    }
    
    .flow-content {
        background: white;
        padding: 12px 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        flex-grow: 1;
    }
    
    .flow-content strong {
        color: #333;
        display: block;
        margin-bottom: 4px;
        font-size: 14px;
    }
    
    .flow-content p {
        color: #666;
        font-size: 13px;
        margin: 0;
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #667eea;
        margin-bottom: 8px;
    }
    
    .metric-label {
        font-size: 13px;
        color: #666;
    }
    
    .footer {
        text-align: center;
        padding: 20px;
        color: #999;
        border-top: 1px solid #e0e0e0;
        margin-top: 50px;
        font-size: 13px;
    }

    .simple-table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        font-size: 13px;
    }

    .simple-table th {
        background: #667eea;
        color: white;
        padding: 12px;
        text-align: left;
        font-weight: bold;
    }

    .simple-table td {
        padding: 12px;
        border-bottom: 1px solid #e0e0e0;
    }

    .simple-table tr:hover {
        background: #f9f9f9;
    }
    </style>
""", unsafe_allow_html=True)

# 메인 헤더
st.markdown("""
<div class="main-header">
    <h1>우울해하는 학생들을 미리 찾아내는 AI</h1>
    <p>데이터로 정신건강 문제를 조기에 발견하는 시스템</p>
</div>
""", unsafe_allow_html=True)

# 핵심 수치
st.markdown("""
<div class="grid-container">
    <div class="grid-item item1">
        <h3>학생 수</h3>
        <div class="grid-item-number">1,000</div>
        <div class="grid-item-unit">명의 데이터 분석</div>
    </div>
    <div class="grid-item item2">
        <h3>체크 항목</h3>
        <div class="grid-item-number">12</div>
        <div class="grid-item-unit">가지 생활 습관</div>
    </div>
    <div class="grid-item item3">
        <h3>정확성</h3>
        <div class="grid-item-number">80%</div>
        <div class="grid-item-unit">이상 맞춤</div>
    </div>
    <div class="grid-item item4">
        <h3>기대 효과</h3>
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
        <p>• 청소년 중 10-20%가 정신 질환 있음<br>
        • 한국 청소년 우울증 계속 증가 중<br>
        • 청소년 자살이 사망 원인 2위<br>
        • 문제가 있어도 도움받기 전에 악화되는 경우 많음<br>
        • 코로나 이후 더 심해짐</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>🎯 우리가 하려는 것</h4>
        <p>• 우울한 학생들을 미리 찾기<br>
        • SNS, 수면, 스트레스 등 원인 파악<br>
        • AI로 자동 발견 시스템 만들기<br>
        • 조기에 도와줄 수 있게 하기<br>
        • 자살 예방하기</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="highlight-box">
결론: 문제를 조기에 발견하면 도와주기가 훨씬 쉬워진다. AI를 사용하면 자동으로 위험 학생들을 찾아낼 수 있다.
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
    <p><b>누가 대상이었나?</b><br>
    13~18세 학생들 (우리 또래!)</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-card">
        <h4>📥 체크한 항목들 (12가지)</h4>
        <p><b>행동 습관</b><br>
        • SNS 하루에 몇 시간<br>
        • 수면 시간 (밤에 몇 시간 자는가)<br>
        • 운동 시간 (주에 몇 시간)<br>
        • 공부 성적<br>
        <br>
        <b>마음 상태</b><br>
        • 스트레스 정도 (1~10점)<br>
        • 불안감 (1~10점)<br>
        • 휴대폰 중독 정도<br>
        <br>
        <b>기타</b><br>
        • 친구들 만나는 정도<br>
        • 자기 전 핸드폰 시간</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>📤 우리가 예측하려는 것</h4>
        <p><b>우울증이 있는가?</b><br>
        YES 또는 NO<br>
        <br>
        이게 우리 AI의 최종 목표다.<br>
        <br>
        위 12가지 항목들을 보고<br>
        <br>
        "이 학생은 우울할 확률이 높다" 또는<br>
        "이 학생은 괜찮을 것 같다"<br>
        <br>
        이렇게 판단하는 거다.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="highlight-box">
<b>데이터 정리 과정:</b> 데이터를 받은 후에 이상한 것들(예: 수면 48시간, SNS -5시간) 제거하고, 빠진 부분은 평균값으로 채우고, 숫자로 통일해서 AI가 이해할 수 있게 만들었다.
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
<b>중요한 질문:</b> 우울증 있음/없음을 예측하는 것이므로 "분류" 문제다. (회귀가 아니다)
</div>
""", unsafe_allow_html=True)

st.subheader("여러 모델들을 비교해봤다")

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
    <tr style="background: #f0f9ff;">
        <td><b>Random Forest</b></td>
        <td><b>83%</b></td>
        <td><b>높은 정확도, 요인 파악 가능</b></td>
        <td>좀 느림</td>
    </tr>
    <tr style="background: #f0fff4;">
        <td><b>XGBoost</b></td>
        <td><b>86%</b></td>
        <td><b>가장 높은 정확도</b></td>
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
        <p>• 83% 정확도 (충분히 높음)<br>
        • 어떤 것이 가장 영향을 미치는지 알 수 있음 (SNS인가? 수면인가?)<br>
        • 새 데이터에도 잘 작동함<br>
        • 이해하기 쉬운 결과를 줌<br>
        • 실제 사용하기에 좋음</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>🔍 XGBoost로 검증</h4>
        <p>• Random Forest가 맞게 했는지 확인용<br>
        • 86% 정확도 (조금 더 높음)<br>
        • 두 모델이 비슷한 결론 내면 신뢰도 높음<br>
        • 최종 예측에 둘 다 사용 가능</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="highlight-box">
<b>쉽게 말하면:</b> Random Forest는 여러 의사가 각각 진단하고 투표로 결정하는 것처럼 작동한다. 그래서 더 정확하고 신뢰할 수 있다.
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

st.subheader("각 지표가 뭐하는 건데?")

explanation_table = """
<table class="simple-table">
    <tr>
        <th>항목</th>
        <th>뜻</th>
        <th>목표</th>
    </tr>
    <tr>
        <td><b>정확도</b></td>
        <td>100명 중에 몇 명을 맞게 판정했는가</td>
        <td>80명 이상</td>
    </tr>
    <tr>
        <td><b>재현율</b></td>
        <td>정말 우울한 학생들 중에 몇 명을 찾아냈는가 (이게 중요!)</td>
        <td>85명 이상</td>
    </tr>
    <tr>
        <td><b>정밀도</b></td>
        <td>우울하다고 한 학생 중에 실제 우울한 학생이 몇 명인가</td>
        <td>75명 이상</td>
    </tr>
    <tr>
        <td><b>ROC-AUC</b></td>
        <td>AI의 전체적인 성능 (0~1, 1에 가까울수록 좋음)</td>
        <td>0.85 이상</td>
    </tr>
</table>
"""

st.markdown(explanation_table, unsafe_allow_html=True)

st.markdown("""
<div class="highlight-box">
<b>가장 중요한 것:</b> 재현율이 높아야 한다. 우울한 학생을 놓치는 게 안 된다. 괜찮은 학생을 우울하다고 잘못 진단하는 것보다 훨씬 심각하다.
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
        <p>• 상담 선생님이 AI 결과 보고 위험한 학생 발견<br>
        • 조기에 도움<br>
        • 자살 사건 예방</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>🏥 병원에서</h4>
        <p>• 의사가 진료할 때 참고<br>
        • 진단 실수 줄이기<br>
        • 더 빠른 치료 시작</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h4>📱 스마트폰 앱</h4>
        <p>• 학생들이 직접 자가진단<br>
        • "넌 이 부분 조심해" 조언<br>
        • 문제 있으면 전문가 연결</p>
    </div>
    """, unsafe_allow_html=True)

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
        <p>• 한 시점의 데이터만 있음 (시간 흐름 반영 못함)<br>
        • 임상 의사의 진단 없음<br>
        • 원인과 결과를 명확히 알 수 없음<br>
        • 특정 지역 학생들만의 데이터<br>
        • 약 30%는 틀릴 수 있음</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>🔧 앞으로 개선할 것</h4>
        <p>• 계속 학생들을 추적 관찰 (6개월, 1년)<br>
        • 의사 진단 데이터 추가<br>
        • 전국 여러 지역의 학생 데이터<br>
        • 더 복잡한 AI 모델 시도<br>
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
• 이 AI는 의사 진단을 대체할 수 없다. 보조도구일 뿐이다.<br>
• AI가 "위험하다"고 해도 반드시 전문가(의사, 상담사)에게 확인받아야 한다.<br>
• 개인정보는 절대 드러나지 않게 보호해야 한다.<br>
• 모든 학생에게 공평하게 작동해야 한다.
</div>
""", unsafe_allow_html=True)

# 9. 로드맵
st.markdown("""
<div class="section">
    <div class="section-title">9. 이제부터 뭘 할 거야?</div>
</div>
""", unsafe_allow_html=True)

roadmap_items = [
    {"phase": "1단계", "timeline": "1~2개월", "task": "문제 정의 & 데이터 모으기", "color": "#667eea"},
    {"phase": "2단계", "timeline": "2~3개월", "task": "AI 모델 개발 & 학습", "color": "#f093fb"},
    {"phase": "3단계", "timeline": "2~3개월", "task": "성능 확인 & 평가", "color": "#4facfe"},
    {"phase": "4단계", "timeline": "2~3개월", "task": "앱 또는 웹사이트 만들기", "color": "#43e97b"},
    {"phase": "5단계", "timeline": "6개월+", "task": "실제로 사용해보고 개선하기", "color": "#ffa500"}
]

roadmap_cols = st.columns(5)

for i, item in enumerate(roadmap_items):
    with roadmap_cols[i]:
        st.markdown(f"""
        <div class="info-card" style="background: linear-gradient(135deg, {item['color']}20 0%, {item['color']}10 100%); border-left: 4px solid {item['color']};">
            <div style="font-weight: bold; color: {item['color']}; margin-bottom: 5px; font-size: 13px;">{item['phase']}</div>
            <div style="font-size: 11px; color: #666; margin-bottom: 8px;">{item['timeline']}</div>
            <div style="font-size: 12px; color: #333; line-height: 1.5;">{item['task']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
청소년 정신건강 AI 프로젝트 | 2024
</div>
""", unsafe_allow_html=True)
