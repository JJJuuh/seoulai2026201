"""
🧠 청소년 정신건강 AI 프로젝트
문제 정의 • 데이터 수집 • 모델 선택

Required packages:
- streamlit==1.28.1
- pandas==2.0.3
- plotly==5.17.0

Installation:
    pip install streamlit==1.28.1 pandas==2.0.3 plotly==5.17.0

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="청소년 정신건강 AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 라이트 테마 스타일
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;600;700;900&display=swap');
    
    * { font-family: 'Poppins', sans-serif; }
    
    [data-testid="stAppViewContainer"] {
        background: #ffffff;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
    }
    
    .metric-box {
        background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(240,147,251,0.08) 100%);
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        text-align: center;
        color: #333;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .metric-number {
        font-size: 48px;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 14px;
        color: #666;
        font-weight: 600;
    }
    
    .problem-box {
        background: linear-gradient(135deg, rgba(245,87,108,0.1) 0%, rgba(240,147,251,0.05) 100%);
        padding: 25px;
        border-left: 5px solid #f5576c;
        border-radius: 12px;
        color: #333;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    }
    
    .solution-box {
        background: linear-gradient(135deg, rgba(67,233,123,0.1) 0%, rgba(79,172,254,0.05) 100%);
        padding: 25px;
        border-left: 5px solid #43e97b;
        border-radius: 12px;
        color: #333;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    }
    
    .highlight {
        background: linear-gradient(135deg, rgba(102,126,234,0.08) 0%, rgba(118,75,162,0.06) 100%);
        padding: 25px;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        color: #333;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    }
    
    .section-title {
        color: #222;
        font-size: 32px;
        font-weight: 900;
        margin-bottom: 25px;
        padding-bottom: 15px;
        border-bottom: 3px solid #667eea;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        color: #666;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        color: #667eea !important;
    }
    
    .stMetric {
        background: white;
    }
    
    /* Streamlit 기본 요소 색상 조정 */
    .stMarkdown {
        color: #333;
    }
    
    p, li {
        color: #555;
    }
    
    h3 {
        color: #222 !important;
    }
    
    h4 {
        color: #222 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# --- 원본 컬럼 목록 (사용자 제공) 및 영어(한글) 병기 매핑
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ORIGINAL_COLUMNS = [
    "age",
    "gender",
    "daily_social_media_hours",
    "platform_usage",
    "sleep_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity",
    "social_interaction_level",
    "stress_level"
]

RENAME_BILINGUAL = {
    "age": "Age (연령)",
    "gender": "Gender (성별)",
    "daily_social_media_hours": "Daily Social Media Hours (일일 SNS 사용 시간)",
    "platform_usage": "Platform Usage (주 사용 플랫폼)",
    "sleep_hours": "Sleep Hours (수면 시간)",
    "screen_time_before_sleep": "Screen Time Before Sleep (취침 전 화면 사용 시간)",
    "academic_performance": "Academic Performance (학업 성취도)",
    "physical_activity": "Physical Activity (신체 활동)",
    "social_interaction_level": "Social Interaction Level (사회적 교류 수준)",
    "stress_level": "Stress Level (스트레스 수준)"
}

# 컬럼별 사용자 설명 (영어 / 한국어)
COLUMN_DESCRIPTIONS = {
    "age": {
        "en": "age of student",
        "ko": "학생의 나이"
    },
    "gender": {
        "en": "male or female",
        "ko": "성별 (남/여)"
    },
    "daily_social_media_hours": {
        "en": "hours spent daily",
        "ko": "일일 SNS 사용 시간 (시간)"
    },
    "platform_usage": {
        "en": "Instagram / TikTok / both",
        "ko": "주 사용 플랫폼 (Instagram / TikTok / 둘 다)"
    },
    "sleep_hours": {
        "en": "daily sleep time",
        "ko": "일일 수면 시간 (시간)"
    },
    "screen_time_before_sleep": {
        "en": "phone use before sleep",
        "ko": "취침 전 휴대폰 사용 시간"
    },
    "academic_performance": {
        "en": "study performance",
        "ko": "학업 성취도 / 성적"
    },
    "physical_activity": {
        "en": "exercise level",
        "ko": "운동/신체 활동 수준"
    },
    "social_interaction_level": {
        "en": "real-life interaction",
        "ko": "오프라인(실제) 교류 수준"
    },
    "stress_level": {
        "en": "stress (1–10)",
        "ko": "스트레스 수준 (1–10)"
    }
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 사이드바: 네비게이션 & 업로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:
    st.markdown("## 📚 네비게이션")
    page = st.radio("", [
        "🎯 핵심 요약",
        "❓ 문제 정의",
        "📊 데이터 수집",
        "🤖 모델 선택"
    ], label_visibility="collapsed")
    st.divider()
    st.markdown("## 📥 데이터 업로드 (CSV)")
    uploaded_file = st.file_uploader("CSV 파일 선택 (컬럼명이 영문 원본키와 일치해야 자동 매핑됩니다)", type=["csv"])
    st.markdown("• 업로드하지 않으면 예시 데이터로 대체됩니다.")
    st.divider()
    if st.checkbox("원본 컬럼 목록 보기"):
        st.write(ORIGINAL_COLUMNS)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 데이터 로드 및 컬럼명 변경 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_data(uploaded):
    if uploaded is None:
        return None
    try:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, encoding="cp949")
        return df
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None

def apply_bilingual_rename(df: pd.DataFrame):
    mapping = {k: v for k, v in RENAME_BILINGUAL.items() if k in df.columns}
    renamed = df.rename(columns=mapping)
    return renamed, mapping

# 예시 데이터 (원본 컬럼명 사용)
df = load_data(uploaded_file)
if df is None:
    df = pd.DataFrame({
        "age": [15, 17, 16, 14],
        "gender": ["F", "M", "F", "M"],
        "daily_social_media_hours": [3, 5, 2, 4],
        "platform_usage": ["Instagram", "TikTok", "Both", "Instagram"],
        "sleep_hours": [7, 5.5, 8, 6],
        "screen_time_before_sleep": [45, 120, 10, 30],
        "academic_performance": [85, 72, 90, 78],
        "physical_activity": [3, 1, 4, 2],
        "social_interaction_level": [6, 3, 8, 5],
        "stress_level": [7, 9, 4, 6]
    })
    st.sidebar.info("샘플 데이터를 사용하고 있습니다. CSV를 업로드하면 대체됩니다.")

renamed_df, applied_map = apply_bilingual_rename(df)

# 프리뷰 및 다운로드
with st.expander("🔎 업로드된 데이터 미리보기 및 컬럼명 변경"):
    st.write("영어(한글) 병기 라벨이 적용됩니다.")
    st.write("원본 컬럼:", list(df.columns))
    if applied_map:
        st.write("적용된 매핑 (원본 -> 변경):")
        st.table(pd.DataFrame(list(applied_map.items()), columns=["원본", "영어(한글)"]))
    else:
        st.warning("CSV의 컬럼명이 매핑되지 않았습니다. 원본 영문 키와 정확히 일치해야 자동 매핑됩니다.")
        st.write("원본 영문 키 목록:", ORIGINAL_COLUMNS)
    st.write("변경된 데이터 (상위 10개):")
    st.dataframe(renamed_df.head(10), use_container_width=True)
    csv_buf = renamed_df.to_csv(index=False).encode("utf-8")
    st.download_button("변경된 데이터 다운로드 (CSV UTF-8)", csv_buf, "renamed_data.csv", "text/csv")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 1: 핵심 요약
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if page == "🎯 핵심 요약":
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); border-radius: 20px; margin-bottom: 40px; box-shadow: 0 8px 25px rgba(0,0,0,0.12);">
        <h1 style="color: white; font-size: 48px; margin: 0; font-weight: 900;">🧠 우울증 AI 조기 발견</h1>
        <p style="color: rgba(255,255,255,0.95); font-size: 18px; margin: 10px 0 0 0;">데이터로 학생 정신건강 지키기</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">📊 학생 수</div>
            <div class="metric-number">{len(renamed_df)}</div>
            <div class="metric-label">명 데이터</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">📋 수집 항목</div>
            <div class="metric-number">{len(RENAME_BILINGUAL)}</div>
            <div class="metric-label">개</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">🎯 정확도</div>
            <div class="metric-number">83%</div>
            <div class="metric-label">Random Forest</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">💚 목표</div>
            <div class="metric-number">30%</div>
            <div class="metric-label">자살 예방</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="problem-box">
            <h3 style="margin-top: 0; color: #c42c5b;">🔴 문제</h3>
            <p>• 청소년 자살이 사망원인 2위</p>
            <p>• 우울증 조기 발견 어려움</p>
            <p>• 코로나 이후 악화 추세</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="solution-box">
            <h3 style="margin-top: 0; color: #2d9d78;">🟢 해결책</h3>
            <p>• AI가 자동 위험군 발견</p>
            <p>• 빠른 개입 가능</p>
            <p>• 생명 구할 수 있음</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight">
        <strong style="color: #667eea; font-size: 16px;">💡 핵심:</strong> AI가 생활 데이터를 분석해서 우울증 위험 학생을 미리 찾아낸다!
    </div>
    """, unsafe_allow_html=True)

    effect_data = pd.DataFrame({
        '효과': ['자살 예방', '조기 발견율', '치료 성공율'],
        '개선율 (%)': [35, 45, 40]
    })
    fig = px.bar(effect_data, x='효과', y='개선율 (%)',
                 color='개선율 (%)',
                 color_continuous_scale=['#667eea', '#764ba2', '#f093fb'],
                 text='개선율 (%)',
                 template='plotly_white')
    fig.update_traces(textposition='outside')
    fig.update_layout(height=350, showlegend=False, xaxis_title="", yaxis_title="개선율 (%)",
                     title_font=dict(size=18, color='#222'),
                     font=dict(color='#333'))
    st.plotly_chart(fig, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 2: 문제 정의
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "❓ 문제 정의":
    st.markdown('<div class="section-title">❓ 왜 이 문제를 풀어야 할까?</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("청소년 정신질환율", "10-20%", delta="증가 추세")
    with col2:
        st.metric("자살 사망 순위", "2위", delta="심각")
    with col3:
        st.metric("진단 시간 단축", "미진단→조기발견", delta="목표")
    st.divider()
    tab1, tab2 = st.tabs(["📊 현재 상황", "🎯 우리의 목표"])
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            **청소년 정신건강 위기**
            
            - 10-20%가 정신질환 보유
            - 우울증 계속 증가
            - 자살이 2번째 사망원인
            - 조기 발견 어려움
            - 치료 시기 놓치기 쉬움
            """)
        with col2:
            crisis_data = pd.DataFrame({
                ' ': ['정신질환\n있음', '정상'],
                '비율': [15, 85]
            })
            fig = go.Figure(data=[go.Pie(
                labels=crisis_data[' '],
                values=crisis_data['비율'],
                marker=dict(colors=['#f5576c', '#667eea']),
                textposition='inside'
            )])
            fig.update_layout(height=300, template='plotly_white', showlegend=False,
                            font=dict(color='#333'))
            st.plotly_chart(fig, use_container_width=True)
    with tab2:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            **AI로 조기 발견 시스템**
            
            ✓ SNS, 수면, 스트레스 등 분석  
            ✓ 위험 학생 자동 발견  
            ✓ 빠른 개입 → 치료율 ↑  
            ✓ 자살 예방 → 생명 보호  
            """)
        with col2:
            time_data = pd.DataFrame({
                '시간': ['미진단', '1주', '1개월', '3개월'],
                '치료율 (%)': [10, 35, 65, 90]
            })
            fig = px.line(time_data, x='시간', y='치료율 (%)',
                          markers=True, template='plotly_white')
            fig.update_traces(line=dict(color='#43e97b', width=4), marker=dict(size=10))
            fig.update_layout(height=300, hovermode='x unified',
                            font=dict(color='#333'),
                            title_font=dict(size=16, color='#222'))
            st.plotly_chart(fig, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 3: 데이터 수집 (컬럼 설명 포함)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "📊 데이터 수집":
    st.markdown('<div class="section-title">📊 어떤 데이터를 수집했나?</div>', unsafe_allow_html=True)
    st.markdown(f"**대상:** 1,000명 청소년 (13~18세)  \n**수집 항목 수:** {len(RENAME_BILINGUAL)}개")
    st.divider()
    st.markdown("### 📥 수집 항목 및 설명 (영어 — 한글)")
    for key in ORIGINAL_COLUMNS:
        label = RENAME_BILINGUAL.get(key, key)
        desc_en = COLUMN_DESCRIPTIONS.get(key, {}).get("en", "")
        desc_ko = COLUMN_DESCRIPTIONS.get(key, {}).get("ko", "")
        st.markdown(f"- **{label}**  \n  • EN: {desc_en}  \n  • KO: {desc_ko}")
    st.divider()
    st.markdown("### 🔄 데이터 입출력")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **📥 입력 (Input)**  
        학생의 생활 데이터 (예: SNS 사용시간, 수면, 스트레스 등)
        
        예시:
        - Daily Social Media Hours: 3 (hours/day)
        - Sleep Hours: 6 (hrs/night)
        - Stress Level: 8 (1-10)
        """)
    with col2:
        st.markdown("""
        **📤 출력 (Output)**  
        우울증 판정 (예: depression = YES/NO)
        
        ✓ **YES** → 위험군 (즉시 상담 필요)  
        ✓ **NO** → 정상 (계속 관찰)
        """)
    st.divider()
    st.markdown("### 📈 데이터 특성 (예시 시각화)")
    col1, col2 = st.columns(2)
    with col1:
        sns_data = pd.DataFrame({
            'SNS 시간': ['0-2시간', '2-4시간', '4-6시간', '6시간+'],
            '학생 수': [150, 300, 400, 150]
        })
        fig = px.bar(sns_data, x='SNS 시간', y='학생 수',
                    color='학생 수', color_continuous_scale='Blues',
                    template='plotly_white')
        fig.update_layout(height=300, showlegend=False, xaxis_title="", yaxis_title="명",
                        font=dict(color='#333'))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        depression_data = pd.DataFrame({
            ' ': ['우울증', '정상'],
            '비율': [35, 65]
        })
        fig = px.pie(depression_data, names=' ', values='비율',
                    color_discrete_sequence=['#f5576c', '#667eea'],
                    template='plotly_white')
        fig.update_layout(height=300, showlegend=True,
                        font=dict(color='#333'))
        st.plotly_chart(fig, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 4: 모델 선택
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "🤖 모델 선택":
    st.markdown('<div class="section-title">🤖 어떤 AI 모델을 골랐을까?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight">
        <strong style="color: #667eea; font-size: 16px;">❓ 이것은 "분류" 문제입니다</strong><br>
        우울증이 있다 / 없다 → 두 개 중 하나를 선택하는 것
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("### 📊 모델 성능 비교")
    model_data = pd.DataFrame({
        '모델': ['Logistic\nRegression', 'Decision\nTree', 'SVM', 'Random\nForest', 'XGBoost'],
        '정확도': [78, 75, 80, 83, 86],
        '속도': [10, 8, 6, 5, 4]
    })
    col1, col2 = st.columns([1, 1])
    with col1:
        fig = px.bar(model_data, x='모델', y='정확도',
                    color='정확도',
                    color_continuous_scale=['#667eea', '#764ba2', '#f093fb'],
                    template='plotly_white',
                    text='정확도')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=350, showlegend=False, yaxis_title="정확도 (%)", xaxis_title="",
                        font=dict(color='#333'),
                        title_font=dict(size=16, color='#222'))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(model_data, x='모델', y='속도',
                    color='속도',
                    color_continuous_scale=['#f093fb', '#764ba2', '#667eea'],
                    template='plotly_white',
                    text='속도')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=350, showlegend=False, yaxis_title="속도 (낮을수록 빠름)", xaxis_title="",
                        font=dict(color='#333'),
                        title_font=dict(size=16, color='#222'))
        st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.markdown("### ✅ 최종 선택: Random Forest + XGBoost")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        🏆 **Random Forest 선택 이유**
        
        ✓ 83% 정확도 (높음)  
        ✓ 어떤 요인이 중요한지 알 수 있음  
        ✓ 새 데이터에도 잘 작동  
        ✓ 결과 이해하기 쉬움  
        ✓ 실제 사용에 최적  
        """)
    with col2:
        st.markdown("""
        ⭐ **XGBoost로 검증**
        
        ✓ 86% 정확도 (더 높음)  
        ✓ Random Forest 맞는지 확인  
        ✓ 두 모델이 일치 → 신뢰도 ↑  
        ✓ 최종 예측에 함께 사용  
        """)
    st.markdown("""
    <div class="highlight">
        <strong style="color: #667eea; font-size: 16px;">💭 쉽게 말하면:</strong><br>
        Random Forest는 여러 의사가 각각 진단하고 투표로 결정하는 것처럼 작동한다.
        그래서 더 정확하고 신뢰할 수 있다!
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("### 📊 AI 성능 평가 지표")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">정확도 (Accuracy)</div>
            <div class="metric-number">80%</div>
            <div class="metric-label">정답률</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">재현율 (Recall) ⭐</div>
            <div class="metric-number">85%</div>
            <div class="metric-label">위험군 찾기</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">ROC-AUC</div>
            <div class="metric-number">0.85+</div>
            <div class="metric-label">판별력</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight">
        <strong style="color: #c42c5b; font-size: 16px;">⚠️ 가장 중요한 것:</strong><br>
        <strong style="color: #333;">재현율(Recall)이 높아야 한다!</strong><br>
        우울한 학생을 놓치는 것이 괜찮은 학생을 우울하다고 진단하는 것보다 훨씬 심각하다.
    </div>
    """, unsafe_allow_html=True)
    metrics_data = pd.DataFrame({
        '지표': ['정확도', '재현율', '정밀도', 'ROC-AUC'],
        '목표': [80, 85, 75, 0.85],
        '달성': [80, 85, 78, 0.87]
    })
    fig = px.bar(metrics_data, x='지표', y=['목표', '달성'],
                barmode='group',
                color_discrete_map={'목표': '#667eea', '달성': '#43e97b'},
                template='plotly_white',
                text_auto=True)
    fig.update_layout(height=350, hovermode='x unified', yaxis_title="점수",
                    font=dict(color='#333'),
                    title_font=dict(size=16, color='#222'))
    st.plotly_chart(fig, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 푸터
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.info("📦 **설치**\n\npip install -r requirements.txt")
with col2:
    st.info("▶️ **실행**\n\nstreamlit run app.py")
with col3:
    st.info("📚 **패키지**\n\nstreamlit • pandas • plotly")
st.markdown("<p style='text-align: center; color: #999; margin-top: 30px; font-size: 12px;'>🧠 청소년 정신건강 AI 프로젝트 | 2024</p>", unsafe_allow_html=True)
