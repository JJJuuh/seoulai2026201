"""
🧠 청소년 정신건강 AI 프로젝트
문제 정의, 데이터 수집, 분류 모델 선정 논리

Required packages:
- streamlit==1.28.1
- pandas==2.0.3
- numpy==1.24.3

Installation:
    pip install streamlit==1.28.1 pandas==2.0.3 numpy==1.24.3

Run:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="청소년 정신건강 AI 프로젝트",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GOAT 디자인 스타일
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800;900&family=Playfair+Display:wght@700;900&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        font-family: 'Poppins', sans-serif;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        min-height: 100vh;
    }
    
    /* ━━━━━━━━━━━━━━ 메인 헤더 ━━━━━━━━━━━━━━ */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 80px 50px;
        border-radius: 25px;
        color: white;
        text-align: center;
        margin: 30px 20px 60px 20px;
        box-shadow: 0 50px 120px rgba(102, 126, 234, 0.45), 0 0 80px rgba(240, 147, 251, 0.3);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(255,255,255,0.2) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .main-header h1 {
        font-size: 48px;
        margin-bottom: 15px;
        font-weight: 900;
        position: relative;
        z-index: 1;
        letter-spacing: -1px;
        font-family: 'Playfair Display', serif;
    }
    
    .main-header p {
        font-size: 20px;
        opacity: 0.98;
        position: relative;
        z-index: 1;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    /* ━━━━━━━━━━━━━━ 핵심 수치 ━━━━━━━━━━━━━━ */
    .grid-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 25px;
        margin-bottom: 60px;
        padding: 0 20px;
    }
    
    .grid-item {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.85) 100%);
        padding: 35px 25px;
        border-radius: 20px;
        box-shadow: 0 25px 60px rgba(0,0,0,0.15), 0 0 40px rgba(102, 126, 234, 0.1);
        border-top: 6px solid;
        text-align: center;
        transition: all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
        cursor: pointer;
        position: relative;
    }
    
    .grid-item:hover {
        transform: translateY(-15px) scale(1.05);
        box-shadow: 0 40px 100px rgba(0,0,0,0.2), 0 0 60px rgba(102, 126, 234, 0.25);
    }
    
    .grid-item h3 {
        font-size: 16px;
        margin-bottom: 12px;
        color: #333;
        font-weight: 800;
    }
    
    .grid-item-number {
        font-size: 48px;
        font-weight: 900;
        margin-bottom: 10px;
        background: linear-gradient(135deg, var(--color1), var(--color2));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Playfair Display', serif;
    }
    
    .grid-item-unit {
        font-size: 13px;
        color: #999;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .item1 { border-top-color: #667eea; --color1: #667eea; --color2: #764ba2; }
    .item2 { border-top-color: #f093fb; --color1: #f093fb; --color2: #f5576c; }
    .item3 { border-top-color: #4facfe; --color1: #4facfe; --color2: #00f2fe; }
    .item4 { border-top-color: #43e97b; --color1: #43e97b; --color2: #38f9d7; }
    
    /* ━━━━━━━━━━━━━━ 섹션 ━━━━━━━━━━━━━━ */
    .section {
        margin-bottom: 70px;
        padding: 0 20px;
    }
    
    .section-title {
        font-size: 40px;
        font-weight: 900;
        margin-bottom: 40px;
        color: #ffffff;
        padding-bottom: 20px;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%) 1;
        letter-spacing: -1px;
        font-family: 'Playfair Display', serif;
    }
    
    /* ━━━━━━━━━━━━━━ 카드 ━━━━━━━━━━━━━━ */
    .info-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.88) 100%);
        padding: 35px;
        border-radius: 18px;
        box-shadow: 0 15px 50px rgba(0,0,0,0.12), 0 0 30px rgba(102, 126, 234, 0.08);
        transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
        position: relative;
        border-left: 6px solid #667eea;
        border: 1px solid rgba(255,255,255,0.5);
    }
    
    .info-card:hover {
        transform: translateX(8px);
        box-shadow: 0 25px 70px rgba(0,0,0,0.15), 0 0 40px rgba(102, 126, 234, 0.2);
    }
    
    .info-card h4 {
        font-size: 20px;
        margin-bottom: 18px;
        color: #222;
        font-weight: 800;
        font-family: 'Playfair Display', serif;
    }
    
    .info-card p {
        color: #666;
        line-height: 1.8;
        font-size: 14px;
        font-weight: 500;
    }
    
    /* ━━━━━━━━━━━━━━ 하이라이트 ━━━━━━━━━━━━━━ */
    .highlight-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 30px;
        border-left: 6px solid #667eea;
        border-radius: 16px;
        margin: 35px 0;
        font-size: 15px;
        color: #f0f0f0;
        line-height: 1.8;
        font-weight: 500;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.05), 0 10px 30px rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102,126,234,0.3);
    }
    
    /* ━━━━━━━━━━━━━━ 테이블 ━━━━━━━━━━━━━━ */
    .simple-table {
        width: 100%;
        border-collapse: collapse;
        margin: 30px 0;
        font-size: 13px;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 15px 50px rgba(0,0,0,0.15);
    }
    
    .simple-table th {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 18px;
        text-align: left;
        font-weight: 800;
        font-size: 14px;
    }
    
    .simple-table td {
        padding: 16px 18px;
        border-bottom: 1px solid rgba(102,126,234,0.05);
        font-weight: 500;
        color: #666;
    }
    
    .simple-table tr:hover {
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.08), transparent);
    }
    
    /* ━━━━━━━━━━━━━━ 메트릭 ━━━━━━━━━━━━━━ */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.88) 100%);
        padding: 35px;
        border-radius: 16px;
        box-shadow: 0 15px 50px rgba(0,0,0,0.12), 0 0 40px rgba(102, 126, 234, 0.12);
        text-align: center;
        transition: all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
        cursor: pointer;
        border: 1px solid rgba(255,255,255,0.5);
    }
    
    .metric-card:hover {
        transform: translateY(-12px) scale(1.05);
        box-shadow: 0 35px 90px rgba(0,0,0,0.18), 0 0 60px rgba(102, 126, 234, 0.3);
    }
    
    .metric-value {
        font-size: 48px;
        font-weight: 900;
        color: #667eea;
        margin-bottom: 10px;
        font-family: 'Playfair Display', serif;
    }
    
    .metric-label {
        font-size: 13px;
        color: #999;
        font-weight: 800;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    /* ━━━━━━━━━━━━━━ 반응형 ━━━━━━━━━━━━━━ */
    @media (max-width: 768px) {
        .grid-container {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    </style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 사이드바 메뉴
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:
    st.markdown("### 📚 목차")
    page = st.radio("", [
        "🎯 핵심 요약",
        "1️⃣ 문제 정의",
        "2️⃣ 데이터 수집",
        "3️⃣ 모델 선택",
    ], label_visibility="collapsed")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 페이지 1: 핵심 요약
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if page == "🎯 핵심 요약":
    st.markdown("""
    <div class="main-header">
        <h1>🧠 우울해하는 학생들을 미리 찾아내는 AI</h1>
        <p>데이터로 정신건강 문제를 조기에 발견하는 시스템</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="grid-container">
        <div class="grid-item item1">
            <h3>📊 학생 수</h3>
            <div class="grid-item-number">1,000</div>
            <div class="grid-item-unit">명 데이터</div>
        </div>
        <div class="grid-item item2">
            <h3>📋 체크 항목</h3>
            <div class="grid-item-number">12</div>
            <div class="grid-item-unit">가지 항목</div>
        </div>
        <div class="grid-item item3">
            <h3>🎯 정확도</h3>
            <div class="grid-item-number">83%</div>
            <div class="grid-item-unit">Random Forest</div>
        </div>
        <div class="grid-item item4">
            <h3>💚 목표</h3>
            <div class="grid-item-number">30%</div>
            <div class="grid-item-unit">자살 예방</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>🔴 문제</h4>
            <p>• 청소년 우울증 증가 중<br>
            • 자살이 사망 원인 2위<br>
            • 조기 발견 어려움</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>🟢 해결책</h4>
            <p>• AI로 자동 발견<br>
            • 조기 개입 가능<br>
            • 생명 구하기</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight-box">
    <strong>💡 핵심:</strong> AI가 SNS, 수면, 스트레스 등 12가지 생활 습관 데이터를 분석해서 우울증 위험이 높은 학생을 미리 찾아낸다.
    </div>
    """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 페이지 2: 문제 정의
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "1️⃣ 문제 정의":
    st.markdown("""
    <div class="section">
        <div class="section-title">왜 이 문제를 풀어야 할까?</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>📊 현재 상황</h4>
            <p><strong>청소년 정신건강 위기</strong><br>
            • 10-20%가 정신질환 있음<br>
            • 우울증 계속 증가 중<br>
            • 자살이 사망 원인 2위<br>
            • 코로나 이후 악화됨<br>
            • 조기 발견 어려움</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>🎯 우리의 목표</h4>
            <p><strong>AI로 조기 발견</strong><br>
            • 위험 학생 자동 찾기<br>
            • 원인 요인 파악<br>
            • 빠른 개입 가능<br>
            • 치료 성공율 향상<br>
            • 자살 예방</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight-box">
    <strong>⚡ 핵심 인사이트:</strong> 정신건강 문제는 발견이 빠를수록 치료 성공률이 높다. AI는 패턴을 찾아 인간이 놓치는 신호를 감지할 수 있다.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="section">
        <div class="section-title">💰 기대 효과</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">35%</div>
            <div class="metric-label">자살 예방</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">45%</div>
            <div class="metric-label">조기 발견율 증가</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">40%</div>
            <div class="metric-label">치료 성공율</div>
        </div>
        """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 페이지 3: 데이터 수집
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "2️⃣ 데이터 수집":
    st.markdown("""
    <div class="section">
        <div class="section-title">어떤 데이터를 수집했나?</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4>📱 대상: 1,000명 청소년 (13~18세)</h4>
        <p>우리 또래 학생들의 생활 패턴 데이터</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>📥 수집한 12가지 항목</h4>
            <p><strong>행동 습관</strong><br>
            ✓ SNS 시간<br>
            ✓ 수면 시간<br>
            ✓ 운동 시간<br>
            ✓ 공부 성적<br>
            <br>
            <strong>마음 상태</strong><br>
            ✓ 스트레스 (1~10)<br>
            ✓ 불안감 (1~10)<br>
            ✓ 휴대폰 중독<br>
            <br>
            <strong>기타</strong><br>
            ✓ 친구 만남 정도<br>
            ✓ 자기 전 핸드폰</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>📤 예측 목표</h4>
            <p><strong>우울증 있음/없음?</strong><br>
            YES 또는 NO<br>
            <br>
            위 12가지 데이터를 보고<br>
            <br>
            ✓ "이 학생은 위험하다"<br>
            또는<br>
            ✓ "이 학생은 괜찮다"<br>
            <br>
            이렇게 판단하는 것!</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight-box">
    <strong>🔄 데이터 정리 과정:</strong> 이상치 제거 → 결측치 채우기 → 정규화 → AI 학습 가능한 형태로 변환
    </div>
    """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 페이지 4: 모델 선택
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "3️⃣ 모델 선택":
    st.markdown("""
    <div class="section">
        <div class="section-title">AI 모델 선택 (왜 이것?)</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight-box">
    <strong>❓ 중요한 질문:</strong> 이 문제는 "분류" 문제다!<br>
    우울증이 있다/없다 → 두 개 범주 중 하나를 고르는 것 (회귀가 아님)
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section">
        <div class="section-title">모델 비교</div>
    </div>
    """, unsafe_allow_html=True)
    
    table_html = """
    <table class="simple-table">
        <tr>
            <th>모델</th>
            <th>정확도</th>
            <th>장점</th>
            <th>단점</th>
        </tr>
        <tr>
            <td>Logistic Regression</td>
            <td>78%</td>
            <td>간단함</td>
            <td>패턴 못 찾음</td>
        </tr>
        <tr>
            <td>Decision Tree</td>
            <td>75%</td>
            <td>이해 쉬움</td>
            <td>과적합</td>
        </tr>
        <tr style="background: linear-gradient(90deg, rgba(79, 172, 254, 0.12), transparent);">
            <td><strong>🏆 Random Forest</strong></td>
            <td><strong>83%</strong></td>
            <td><strong>높은 정확도</strong></td>
            <td>느림</td>
        </tr>
        <tr style="background: linear-gradient(90deg, rgba(67, 233, 123, 0.12), transparent);">
            <td><strong>⭐ XGBoost</strong></td>
            <td><strong>86%</strong></td>
            <td><strong>최고 정확도</strong></td>
            <td>복잡함</td>
        </tr>
        <tr>
            <td>SVM</td>
            <td>80%</td>
            <td>효율적</td>
            <td>설명 어려움</td>
        </tr>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section">
        <div class="section-title">✅ 최종 선택: Random Forest</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>왜 Random Forest?</h4>
            <p>• 83% 정확도 (충분히 높음)<br>
            • 어떤 요인이 중요한지 알 수 있음<br>
            • 새 데이터에도 잘 작동<br>
            • 결과 해석이 쉬움<br>
            • 실제 사용 최적</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>XGBoost는 검증용</h4>
            <p>• 86% 정확도 (더 높음)<br>
            • Random Forest 맞는지 확인<br>
            • 두 모델이 일치하면 신뢰도 ↑<br>
            • 최종 예측에 함께 사용</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight-box">
    <strong>💭 쉽게 말하면:</strong> Random Forest는 여러 의사가 각각 진단하고 투표로 결정하는 것처럼 작동한다. 그래서 더 정확하고 신뢰할 수 있다!
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="section">
        <div class="section-title">📊 평가 지표</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">80%</div>
            <div class="metric-label">정확도</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">85%</div>
            <div class="metric-label">재현율 (중요!)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">0.85+</div>
            <div class="metric-label">ROC-AUC</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight-box">
    <strong>⚠️ 가장 중요한 것:</strong> 재현율이 높아야 한다!<br>
    우울한 학생을 놓치는 것이 괜찮은 학생을 우울하다고 진단하는 것보다 훨씬 심각하다.
    </div>
    """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 푸터
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(240,147,251,0.05)); border-radius: 12px; border-left: 4px solid #667eea;">
    <strong>🔗 필수 라이브러리</strong><br>
    streamlit • pandas • numpy
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(240,147,251,0.05)); border-radius: 12px; border-left: 4px solid #667eea;">
    <strong>📦 설치</strong><br>
    pip install -r requirements.txt
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(240,147,251,0.05)); border-radius: 12px; border-left: 4px solid #667eea;">
    <strong>▶️ 실행</strong><br>
    streamlit run app.py
    </div>
    """, unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: #999; margin-top: 30px;'>🧠 청소년 정신건강 AI 프로젝트 | 2024</p>", unsafe_allow_html=True)
