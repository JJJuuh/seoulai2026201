import streamlit as st
import pandas as pd
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="청소년 정신건강 AI 프로젝트",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 스타일
st.markdown("""
    <style>
    .big-title {
        font-size: 48px;
        font-weight: bold;
        color: #667eea;
        text-align: center;
        margin: 30px 0;
    }
    .simple-box {
        background-color: #f8f9ff;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 4px solid #667eea;
    }
    .problem-box {
        background-color: #fff5f5;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 4px solid #ff6b6b;
    }
    .solution-box {
        background-color: #f0fff4;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 4px solid #48bb78;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">😊 우리 반 친구들의 정신건강을 지켜주는 AI</div>', unsafe_allow_html=True)

# 탭
tab1, tab2, tab3, tab4 = st.tabs(["🤔 왜 이 주제?", "📱 어떤 데이터?", "🤖 AI가 뭔데?", "✨ 뭐가 좋아?"])

# ===== TAB 1: 왜 이 주제? =====
with tab1:
    st.header("🤔 왜 청소년 정신건강을 봤을까?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="problem-box">
        <h3>😔 요즘 우리 문제들...</h3>
        
        **SNS 때문에 자존감 낮아짐**
        - 인스타, 틱톡 계속 봄
        - 남과 비교하고 스트레스
        
        **수면 부족이 심함**
        - 밤에 핸드폰 하다가 자정 넘어서 잠
        - 아침에 지쳐서 학교 감
        
        **학업 스트레스**
        - 수학 시험, 영어 단어...
        - 입시 때문에 불안함
        
        **혼자라는 느낌**
        - 친구들도 바쁘고
        - 맘 놓고 얘기할 사람이 없음
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="solution-box">
        <h3>💡 이걸 해결하려면?</h3>
        
        **문제를 알아야 한다**
        - 어떤 것이 가장 영향이 클까?
        - SNS인가, 수면인가, 스트레스인가?
        
        **조기에 발견해야 한다**
        - 우울한 애들을 미리 찾아내고
        - 선생님이나 부모님한테 알려주면
        - 도와줄 수 있지 않을까?
        
        **우리가 할 수 있을까?**
        - AI를 사용하면 자동으로 판단 가능!
        - 데이터를 분석해서
        - "이 학생은 위험해 보인다" 알려줄 수 있음
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("""
    <div class="simple-box">
    <h3>🎯 우리 프로젝트의 목표</h3>
    
    **간단히 말하면:**
    
    1️⃣ 청소년 1,000명의 생활 습관 데이터를 모은다
    
    2️⃣ 어떤 것들이 우울함에 영향을 미치는지 알아본다
    
    3️⃣ AI가 자동으로 "이 학생은 우울해 보인다" 판단하게 한다
    
    4️⃣ 그럼 일찍 도와줄 수 있다!
    </div>
    """, unsafe_allow_html=True)

# ===== TAB 2: 어떤 데이터? =====
with tab2:
    st.header("📱 우리가 봤던 데이터")
    
    st.markdown("""
    <div class="simple-box">
    <h3>📊 1,000명의 청소년 데이터</h3>
    
    **누구의 데이터?**
    - 13~18세 학생들 (우리 또래!)
    - 남자 여자 섞여 있음
    
    **어떤 정보?**
    - SNS 하루에 몇 시간
    - 하루에 몇 시간 자는지
    - 공부 점수는 어떤지
    - 운동은 잘하는지
    - 스트레스 얼마나 받는지
    - 불안감이 있는지
    - 휴대폰 중독은 없는지
    - 우울한지 아닌지 (우리 AI의 정답!)
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="solution-box">
        <h3>✅ 입력 데이터 (AI가 봐야 할 것)</h3>
        
        **생활 습관**
        - 📱 SNS 시간
        - 😴 수면 시간
        - 📚 공부 성과
        - 🏃 운동 시간
        - 👥 친구 만나는 정도
        
        **마음 상태**
        - 😰 스트레스 (1~10점)
        - 😟 불안 (1~10점)
        - 📵 휴대폰 중독 (1~10점)
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="problem-box">
        <h3>🎯 출력 데이터 (AI가 예측할 것)</h3>
        
        **우울증 있고 없고**
        - YES: 우울한 상태
        - NO: 괜찮은 상태
        
        **AI의 일**
        위 입력들을 보고
        "우울할 확률이 높은지
        낮은지" 판단하기!
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("""
    <div class="simple-box">
    <h3>📈 데이터를 정리하는 과정</h3>
    
    **1단계: 이상한 데이터 찾기**
    - "수면 48시간"같은 말도 안 되는 데이터 제거
    - "SNS -5시간"같은 음수 데이터 제거
    
    **2단계: 빠진 데이터 채우기**
    - 어떤 학생이 스트레스 점수를 안 적으면?
    - 평균값으로 채워주기
    
    **3단계: 데이터 통일하기**
    - 시간은 시간으로, 점수는 점수로
    - 남/여를 숫자로 바꾸기
    </div>
    """, unsafe_allow_html=True)

# ===== TAB 3: AI가 뭔데? =====
with tab3:
    st.header("🤖 AI가 어떻게 배우는 거야?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="simple-box">
        <h3>🧠 Random Forest (랜덤 포레스트)</h3>
        
        **쉽게 설명하면:**
        
        **의사 결정나무처럼 생각하기**
        - Q: SNS를 4시간 이상 해?
          - YES → 위험할 수 있어
          - NO → 다음 질문
        - Q: 수면이 6시간 이하?
          - YES → 위험해!
          - NO → 괜찮아!
        
        **여러 개의 나무를 만들어**
        - 100개의 나무가 각각 판단
        - 다수결로 결정! (투표처럼)
        - 그럼 더 정확해진다
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="simple-box">
        <h3>⚡ XGBoost (엑스지부스트)</h3>
        
        **쉽게 설명하면:**
        
        **더 똑똑한 나무**
        - 나무 1개가 판단 → 틀림
        - "어디가 틀렸지?" 배움
        - 나무 2개가 보정 → 조금 나음
        - 나무 3, 4, 5... 계속 보정
        
        **예시로 보면**
        - 1번 AI: "이 학생 위험해" (틀림)
        - 2번 AI: "아니야, 이 부분 봐봐" (보정)
        - 3번 AI: "근데 이것도 봐야 해" (다시 보정)
        - 최종 판단: 더 정확해짐!
        
        **성능이 조금 더 좋음**
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("""
    <div class="simple-box">
    <h3>✍️ AI가 배우는 과정</h3>
    
    **Step 1️⃣ : 데이터 준비**
    - 1,000명 데이터를 모음
    - 700명으로 "배우기", 300명으로 "시험보기"
    
    **Step 2️⃣ : AI 학습**
    - 700명 데이터 분석
    - "어떻게 해야 우울함을 잘 맞힐까" 패턴 찾기
    
    **Step 3️⃣ : AI 성능 확인**
    - 300명에게 해보기
    - 100명 중 80명을 맞게 맞혔어? → 80점
    
    **Step 4️⃣ : 더 좋게 만들기**
    - 설정을 조정하고 다시 해보기
    - "이게 더 잘 맞네!" 하면 채택
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("""
    <div class="problem-box">
    <h3>🎯 우리가 쓴 AI의 목표 (성적표)</h3>
    
    **AI가 100점을 맞혀야 할 과목들:**
    
    | 항목 | 목표 | 뜻 |
    |-----|-----|-----|
    | **정확도** | 80점 이상 | 100명 중 80명 이상 맞추기 |
    | **위험군 찾기** | 85점 이상 | 우울한 애 놓치지 않기 (중요!) |
    | **거짓양성** | 75점 이상 | 괜찮은 애를 아픈 줄 알고 헷갈리지 않기 |
    
    **왜 이렇게 정했나?**
    - 너무 높으면 불가능, 너무 낮으면 쓸모없음
    - 실제로 도움이 되려면 이 정도는 필요
    """, unsafe_allow_html=True)

# ===== TAB 4: 뭐가 좋아? =====
with tab4:
    st.header("✨ 이 AI가 뭐가 좋은데?")
    
    st.markdown("""
    <div class="solution-box">
    <h3>🎓 학교에서 사용하면?</h3>
    
    **시나리오:**
    - 학생들이 주말에 앱으로 간단한 설문 작성
    - AI가 자동으로 분석
    - 위험해 보이는 학생을 상담 선생님이 발견
    - "안녕, 요즘 어때? 뭔가 도와줄 게 있나?"
    - 조기에 도와줄 수 있음!
    
    **기대 효과:**
    ✅ 자살 생각하는 학생 2-3명 구하기 (1,000명당)
    ✅ 우울증 학생 조기 발견
    ✅ 학교 분위기 더 밝아지기
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="solution-box">
    <h3>🏥 병원에서 사용하면?</h3>
    
    **시나리오:**
    - 병원 가면 AI로 먼저 검사
    - 의사가 진료할 때 참고 자료로 사용
    - "네, AI도 위험도가 높네요. 함께 얘기해봅시다"
    
    **기대 효과:**
    ✅ 의사 진료 시간 단축
    ✅ 진단 실수 줄이기
    ✅ 더 많은 학생 도와주기
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="solution-box">
    <h3>📱 휴대폰 앱으로 만들면?</h3>
    
    **시나리오:**
    - 우리 반 학생들이 직접 앱 다운로드
    - "요즘 기분은 어때?" 5분만에 검사
    - "너는 지금 괜찮은 것 같은데, 이것만 조심해" 피드백
    
    **기대 효과:**
    ✅ 언제 어디서나 사용 가능
    ✅ 프라이버시 지키기
    ✅ 자기 건강 스스로 관리하기
    """, unsafe_allow_html=True)
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("예방 효과", "25-35%", "자살 시도 감소")
    
    with col2:
        st.metric("조기 발견", "40-50%", "위험군 찾기")
    
    with col3:
        st.metric("치료 성공", "35-45%", "개선율 증가")
    
    st.divider()
    
    st.markdown("""
    <div class="simple-box">
    <h3>⚠️ 근데 주의할 점들</h3>
    
    **AI가 다가 아니야**
    - "AI가 말했다" = 절대 진실 (X)
    - "AI도 말했네" = 참고만 (O)
    - 반드시 전문가(의사, 상담사)에게 확인받아야 함
    
    **프라이버시 중요**
    - 우리 개인정보 꼭 지켜져야 함
    - 누구 몰래 봐서는 안 됨
    
    **모두에게 공평하게**
    - 어떤 학생은 찾고 어떤 학생은 안 찾으면 안 됨
    - 공정하게 해야 함
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.markdown("""
<div style='background-color: #f0fff4; padding: 30px; border-radius: 15px; text-align: center; margin-top: 30px;'>
    <h2>🎉 정리하면</h2>
    <p style="font-size: 18px; line-height: 1.8;">
        우리는 <b>1,000명 청소년의 생활 데이터</b>를 분석해서<br>
        <b>AI가 우울한 친구들을 자동으로 찾아내도록</b> 만들고 있어요.<br><br>
        그럼 학교, 병원, 앱에서 쓸 수 있고<br>
        <b>정말로 힘들어하는 친구들을 조기에 도와줄 수 있어요!</b><br><br>
        우리 반 학생들의 정신건강을 지켜주는 AI 🎯
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #999; padding: 20px; margin-top: 40px;'>
고2 학생들의 프로젝트 | 청소년 정신건강 AI 🧠
</div>
""", unsafe_allow_html=True)
