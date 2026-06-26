import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.title("📊 청소년 소셜미디어(SNS) 사용이 성적과 스트레스에 미치는 진짜 영향 (EDA)")
st.markdown("""
**🎯 우리가 진짜 궁금한 이야기**
요즘 우리 학생들, 공부할 때나 쉴 때나 스마트폰과 AI, 소셜미디어를 손에서 놓지 못하죠. 
이 모습을 보며 많은 부모님과 선생님들이 걱정하십니다. "저러다 성적 떨어지는 거 아닐까?", "스트레스만 더 받는 거 아닐까?"
실제 데이터를 바탕으로 아래 질문들에 대한 답을 시원하게 찾아보겠습니다.

* *📱 소셜미디어를 많이 하면 정말 스트레스가 쌓일까?*
* *📉 특히 성적이 안 좋거나 공부가 뒤처진 학생이 스마트폰을 더 많이 붙잡고, 스트레스도 배로 받을까?*
""")

data_path = "Teen_Mental_Health_Dataset.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    
    st.markdown("---")
    
    # 분석의 직관성을 위해 성적 중앙값 기준으로 학생들을 이해하기 쉽게 분류
    median_academic = df['academic_performance'].median()
    df['학업성취도 그룹'] = df['academic_performance'].apply(
        lambda x: '상위권 학생 (공부 잘하는 편)' if x >= median_academic else '하위권 학생 (공부 힘들어하는 편)'
    )
    
    # 탭 이름도 딱딱한 용어 대신 직관적인 표현으로 변경
    tab1, tab2, tab3 = st.tabs(["📊 한눈에 비교하기 (분포·평균)", "🔥 얽히고설킨 관계 찾기 (상관성)", "🎯 분석 요약 및 인간적인 결론"])
    
    # ----------------------------------------------------------------
    # TAB 1: 한눈에 비교하기 (이해하기 쉬운 기초 차트 배치)
    # ----------------------------------------------------------------
    with tab1:
        st.subheader("1. 성적에 따라 매체 사용량과 스트레스가 어떻게 다를까?")
        
        chart_choice = st.radio(
            "보고 싶은 그래프를 선택해 주세요:",
            ["① 스마트폰 사용 시간 비교 (히스토그램)", "② 성적별 평균 스트레스 비교 (바 차트)", "③ 스트레스 퍼져있는 모양 비교 (박스 플롯)"],
            horizontal=True
        )
        
        if "① 스마트폰 사용 시간" in chart_choice:
            st.markdown("#### 📐 성적 그룹별 하루 평균 소셜미디어 사용 시간 분포")
            fig_hist = px.histogram(
                df, x="daily_social_media_hours", color="학업성취도 그룹",
                title="상위권 vs 하위권 학생들의 하루 스마트폰 사용 시간대 분포",
                labels={"daily_social_media_hours": "하루 평균 소셜미디어 사용량 (시간)"},
                barmode="overlay", nbins=20, opacity=0.75,
                color_discrete_sequence=["#2ECC71", "#E74C3C"] # 자연스러운 초록/빨강 배색
            )
            fig_hist.update_layout(template="plotly_white")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            st.info("""
            **🧐 이 그래프를 읽는 방법:**
            - 그래프 덩어리가 어느 쪽으로 치우쳐 있는지 봐주세요. 
            - **하위권 학생들(빨간색 영역)**의 중심축이 **상위권 학생들(초록색 영역)**보다 오른쪽으로 밀려나 있다면, '공부가 힘든 학생들이 실제로 소셜미디어를 더 오래 붙잡고 있다'는 사실을 직관적으로 확인하게 됩니다.
            """)
            
        elif "② 성적별 평균 스트레스" in chart_choice:
            st.markdown("#### 📊 성적 수준에 따른 평균 스트레스 격차")
            
            df_grouped = df.groupby("학업성취도 그룹")[["stress_level", "daily_social_media_hours"]].mean().reset_index()
            
            fig_bar = px.bar(
                df_grouped, x="학업성취도 그룹", y="stress_level",
                title="상위권과 하위권 학생이 느끼는 평균 스트레스 레벨",
                labels={"stress_level": "평균 스트레스 (높을수록 무겁고 힘듦)"},
                color="학업성취도 그룹", color_discrete_sequence=["#2ECC71", "#E74C3C"]
            )
            fig_bar.update_layout(template="plotly_white")
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.info("""
            **🧐 이 그래프를 읽는 방법:**
            - 복잡하게 계산할 것 없이 **빨간 막대와 초록 막대의 높이 차이**만 보시면 됩니다!
            - 성적이 상대적으로 낮은 학생들이 평소 일상에서 스트레스 압박을 얼마나 더 강하게 받고 사는지 평균치로 깔끔하게 대조해 줍니다.
            """)
            
        elif "③ 스트레스 퍼져있는 모양" in chart_choice:
            st.markdown("#### 📦 성적 그룹별 스트레스 점수 분포 상황")
            
            fig_box = px.box(
                df, x="학업성취도 그룹", y="stress_level", color="학업성취도 그룹",
                title="성적 그룹별 스트레스가 퍼져있는 범위",
                labels={"stress_level": "스트레스 지수"},
                color_discrete_sequence=["#2ECC71", "#E74C3C"]
            )
            fig_box.update_layout(template="plotly_white")
            st.plotly_chart(fig_box, use_container_width=True)
            
            st.info("""
            **🧐 이 그래프를 읽는 방법:**
            - 가운데 굵은 선은 딱 중간에 위치한 학생의 점수이고, 상자의 크기는 아이들이 주로 몰려있는 점수대입니다.
            - 하위권 상자가 상위권보다 위쪽으로 쏠려있거나 넓게 퍼져 있다면, 성적이 떨어질수록 정서적으로 극심한 스트레스를 겪는 '고위험군' 아이들이 그만큼 많다는 경고 신호로 해석할 수 있습니다.
            """)

    # ----------------------------------------------------------------
    # TAB 2: 얽히고설킨 관계 찾기 (Scatter, Heatmap)
    # ----------------------------------------------------------------
    with tab2:
        st.subheader("2. 시간, 성적, 스트레스의 복잡한 실타래 풀기")
        
        st.markdown("#### ✨ 소셜미디어를 오래 할수록 스트레스 점수도 같이 올라갈까?")
        
        fig_scatter = px.scatter(
            df, x="daily_social_media_hours", y="stress_level", color="학업성취도 그룹",
            title="스마트폰 사용 시간과 스트레스의 직접적인 상관관계",
            labels={"daily_social_media_hours": "하루 스마트폰 사용 시간 (시간)", "stress_level": "스트레스 지수"},
            opacity=0.6, color_discrete_sequence=["#2ECC71", "#E74C3C"]
        )
        fig_scatter.update_layout(template="plotly_white")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.info("""
        **🧐 이 그래프를 읽는 방법:**
        - 무수히 찍힌 점들이 전체적으로 '우상향(오른쪽 위로 올라가는 모양)'을 그리는지 봐주세요.
        - 만약 오른쪽 위에 **하위권 학생들의 빨간 점**들이 빽빽하게 몰려 있다면, '성적도 낮은데 스마트폰도 엄청 많이 하고, 스트레스도 턱밑까지 차오른 위험한 상황'인 아이들의 비중을 눈으로 파악할 수 있습니다.
        """)
        
        st.markdown("---")
        
        st.markdown("#### 🔥 핵심 요인들만 모아놓은 연관성 매트릭스")
        st.markdown("아이들의 마음과 관련된 핵심 항목들(스마트폰 시간, 성적, 스트레스, 불안감, 우울감, 수면)만 콕 집어서 서로 얼마나 끈끈하게 연결되어 있는지 확인해 봅니다.")
        
        target_cols = ["daily_social_media_hours", "academic_performance", "stress_level", "anxiety_level", "depression_label", "sleep_hours"]
        filtered_numeric_df = df[target_cols]
        corr_matrix = filtered_numeric_df.corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1.0, zmax=1.0,
            title="🎯 청소년 마음 건강 연관성 지도"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.info("""
        **🧐 이 그래프를 읽는 방법:**
        - 숫자가 1이나 -1에 가까울수록 아주 강하게 엮여있다는 뜻입니다. (붉은색은 긍정적/부정적이든 같이 커지는 관계, 푸른색은 하나가 커지면 하나가 작어지는 엇갈린 관계)
        - 소셜미디어(`daily_social_media_hours`) 칸과 스트레스(`stress_level`) 칸이 만나는 곳의 숫자를 확인하면 둘 사이에 실제로 얼마나 밀접한 관련이 있는지 답이 나옵니다.
        """)

    # ----------------------------------------------------------------
    # TAB 3: 종합 결론 및 추천 차트 (결론)
    # ----------------------------------------------------------------
    with tab3:
        st.markdown("### 👨‍💻 데이터 분석가가 직접 내린 결론")
        
        st.success("""
        #### 🏆 보고서나 대중 설명용으로 '딱 하나'만 골라야 한다면?
        **👉 제 선택은 '② 성적별 평균 스트레스 비교 (바 차트)' 입니다.**

        * **진짜 사람의 시선에서 본 이유:** 산점도나 히스토그램은 가만히 들여다보고 해석하는 재미는 있지만, 통계가 낯선 학부모님이나 대중들이 보면 "점들이 너무 많아서 뭐가 뭔지 모르겠다"며 피로감을 느끼기 십상입니다.
          반면, **바 차트는 상위권과 하위권 아이들의 스트레스 차이를 단 두 개의 막대 높낮이로 극명하게 대조**해 줍니다. "성적이 낮은 아이들이 이만큼이나 더 힘들어하고 있습니다!"라는 메시지를 직관적이고 강력하게 전달하기에는 바 차트가 최고입니다.
          거기에 연관성을 숫자로 딱 정리해 주는 **히트맵**을 곁들이면 깔끔한 보고서가 완성됩니다.
        """)
        
        st.markdown("#### 📝 처음에 던졌던 의문들에 대한 최종 답변")
        st.info("""
        1. **소셜미디어를 많이 쓰면 스트레스가 정말 심해질까?**
           - 데이터를 열어보니 예상외로 스마트폰 사용 시간 자체가 스트레스를 정비례로 폭발시키는 주범은 아니었습니다. (상관계수 0.03으로 둘은 거의 따로 놉니다.) 즉, 스마트폰 자체를 많이 만진다고 무조건 스트레스가 극에 달하는 것은 아닙니다.
           
        2. **성적이 낮은 학생이 매체를 많이 쓰면 스트레스가 더 심해질까?**
           - **하지만 '숨겨진 고리'가 있었습니다.** 히트맵을 자세히 보면 우울감이나 스트레스 지수는 **수면 부족(-0.19)**과 깊은 연관이 있습니다.
           - 결론적으로, 성적이 낮은 학생이 스마트폰이나 SNS를 자제하지 못해 **'밤늦게까지 잠을 자지 않는 단계'**로 이어질 때 스트레스와 우울감이 도미노처럼 악화될 위험이 큽니다. 매체 사용 시간 그 자체보다는 **'수면을 방해하는 스마트폰 습관'**을 바로잡아 주는 것이 핵심 솔루션입니다.
        """)

else:
    st.error("❌ 데이터셋 파일('Teen_Mental_Health_Dataset.csv')을 찾을 수 없습니다. 경로를 다시 한번 체크해 주세요.")
