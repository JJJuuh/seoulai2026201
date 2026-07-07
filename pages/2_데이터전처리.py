import streamlit as st
import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 윈도우 커스텀 스타일 가속 파트 (SaaS App Theme)
# ==========================================
st.set_page_config(
    page_title="DataClean Studio Ultra",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 고급 콤포넌트 섀도우 및 프레임워크 럭셔리 스킨 적용
st.markdown("""
    <style>
        /* 메인 프레임 워크 디자인 */
        .stApp { background-color: #0F172A; color: #F8FAFC; }
        
        /* 신급 앱 탑바 */
        .premium-topbar {
            background: linear-gradient(90deg, #1E1B4B 0%, #311042 100%);
            padding: 30px;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 30px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
        }
        .topbar-title { font-size: 32px; font-weight: 900; letter-spacing: -1px; background: linear-gradient(to right, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .topbar-subtitle { font-size: 14px; color: #94A3B8; margin-top: 8px; font-weight: 400; }
        
        /* 하이엔드 글래스모피즘 카드 컴포넌트 */
        .premium-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            margin-bottom: 25px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .card-badge { display: inline-block; background: rgba(56, 189, 248, 0.15); color: #38BDF8; font-size: 11px; font-weight: 700; padding: 4px 12px; border-radius: 9999px; margin-bottom: 12px; letter-spacing: 0.5px; }
        .card-title { font-size: 20px; font-weight: 700; color: #F8FAFC; margin-bottom: 15px;
