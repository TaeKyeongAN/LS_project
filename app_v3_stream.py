# streamlit run app_v3_stream.py

from __future__ import annotations
import io, math, time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pydantic import BaseModel, Field
import altair as alt
from fpdf import FPDF
import os
import hashlib

# 폰트 경로 설정 (Nanum Gothic 폰트)
FONT_PATH_REGULAR = "./www/fonts/NanumGothic-Regular.ttf"
FONT_PATH_BOLD = "./www/fonts/NanumGothic-Bold.ttf"


# =========================================
# Helper charts (고정 placeholder로만 그리기)
# =========================================
def render_kwh_chart(df_acc: pd.DataFrame, placeholder):
    chart = (
        alt.Chart(df_acc)
        .mark_line(point=True)
        .encode(
            x=alt.X("timestamp:T", title="시간"),
            y=alt.Y("kWh:Q", title="전력사용량(kWh)"),
            tooltip=["timestamp", alt.Tooltip("kWh:Q", format=",.2f")]
        )
        .properties(height=260)
    )
    placeholder.altair_chart(chart, use_container_width=True)


def render_pf_combined(df_acc: pd.DataFrame, placeholder):
    df_pf = df_acc.copy()
    if "측정일시" not in df_pf.columns:
        df_pf["측정일시"] = pd.to_datetime(df_pf["timestamp"], errors="coerce")
    if "지상역률_주간클립" not in df_pf.columns:
        df_pf["지상역률_주간클립"] = np.random.uniform(85, 99, len(df_pf))
    if "진상역률(%)" not in df_pf.columns:
        df_pf["진상역률(%)"] = np.random.uniform(90, 100, len(df_pf))

    df_pf["주간여부"] = ((df_pf["측정일시"].dt.hour >= 9) & (df_pf["측정일시"].dt.hour <= 23)).astype(int)
    df_pf["야간여부"] = ((df_pf["측정일시"].dt.hour < 9) | (df_pf["측정일시"].dt.hour >= 23)).astype(int)

    latest_time = df_pf["측정일시"].max()
    start_domain = latest_time - pd.Timedelta(hours=24) if pd.notna(latest_time) else None
    x_axis = alt.X(
        "측정일시:T", title="시간",
        scale=alt.Scale(domain=[start_domain, latest_time]) if start_domain else alt.Undefined
    )
    ch = create_combined_pf_chart(df_pf, x_axis)
    if ch:
        placeholder.altair_chart(ch, use_container_width=True)
    else:
        placeholder.info("유효한 역률 데이터가 없습니다.")


def render_tou_chart(df_acc: pd.DataFrame, placeholder):
    df_tou = df_acc.copy()

    # TOU 매핑 (없으면 자동 생성)
    if "TOU" not in df_tou.columns:
        df_tou["hour"] = df_tou["timestamp"].dt.hour
        df_tou["TOU"] = df_tou["hour"].apply(lambda h: (
            "경부하" if (h >= 23 or h < 7) else
            "최대부하" if (10 <= h < 18) else
            "중간부하"
        ))

    # 단가/예측요금
    if "unit_price" not in df_tou.columns:
        tou_price = {"경부하": 90, "중간부하": 120, "최대부하": 160}
        df_tou["unit_price"] = df_tou["TOU"].map(tou_price)
    df_tou["예측요금(원)"] = df_tou["kWh"] * df_tou["unit_price"]

    # 1시간 이동평균(15분×4) — TOU별
    df_tou = df_tou.sort_values("timestamp")
    df_tou["예측요금_1시간MA"] = (
        df_tou.groupby("TOU", group_keys=False)["예측요금(원)"]
              .rolling(window=4, min_periods=1).mean().reset_index(level=0, drop=True)
    )

    # 최근 24시간만 표시 (원하시면 제거 가능)
    latest_time = df_tou["timestamp"].max()
    x_dom = [latest_time - pd.Timedelta(hours=24), latest_time] if pd.notna(latest_time) else None
    x_enc = alt.X("timestamp:T", title="시간",
                  scale=alt.Scale(domain=x_dom) if x_dom else alt.Undefined)

    color_scale = alt.Scale(
        domain=["경부하", "중간부하", "최대부하"],
        range=["#2E86C1", "#F1C40F", "#E74C3C"]
    )
    base = alt.Chart(df_tou).mark_line(opacity=0.35).encode(
        x=x_enc,
        y=alt.Y("예측요금(원):Q", title="예측 요금 (원)", scale=alt.Scale(zero=False)),
        color=alt.Color("TOU:N", scale=color_scale, legend=alt.Legend(title="TOU 구간")),
        tooltip=[
            alt.Tooltip("timestamp:T", title="시간"),
            alt.Tooltip("TOU:N", title="구간"),
            alt.Tooltip("예측요금(원):Q", format=",.0f"),
            alt.Tooltip("kWh:Q", format=",.2f", title="전력사용량(kWh)")
        ]
    )
    ma = alt.Chart(df_tou).mark_line(strokeWidth=3).encode(
        x=x_enc,
        y="예측요금_1시간MA:Q",
        color=alt.Color("TOU:N", scale=color_scale, legend=None),
        tooltip=[
            alt.Tooltip("timestamp:T", title="시간"),
            alt.Tooltip("TOU:N", title="구간"),
            alt.Tooltip("예측요금_1시간MA:Q", title="1시간 평균", format=",.0f")
        ]
    )
    tou_chart = (base + ma).properties(
        title="⚡ 실시간 TOU(시간대)별 예측 요금 추이 (1시간 이동평균 포함)",
        height=260
    )
    placeholder.altair_chart(tou_chart, use_container_width=True)


# =========================================
# 역률 시각화 함수 (create_combined_pf_chart)
# =========================================
# def create_combined_pf_chart(df, x_axis):
#     """주간/야간 구분 역률 통합 시각화 (app.py 원본)"""
#     pf_data = df[['측정일시', '지상역률_주간클립', '진상역률(%)', '주간여부', '야간여부']].copy()
#     pf_data = pf_data[(pf_data['지상역률_주간클립'] > 0) | (pf_data['진상역률(%)'] > 0)]
#     if pf_data.empty:
#         return None

#     pf_long = pf_data.melt(
#         id_vars=['측정일시', '주간여부', '야간여부'],
#         value_vars=['지상역률_주간클립', '진상역률(%)'],
#         var_name='역률종류',
#         value_name='역률값'
#     )

#     def get_display_type(row):
#         if row['역률종류'] == '지상역률_주간클립':
#             return '지상 (주간기준)' if row['주간여부'] == 1 else '지상 (야간)'
#         elif row['역률종류'] == '진상역률(%)':
#             return '진상 (야간기준)' if row['야간여부'] == 1 else '진상 (주간)'
#         return '기타'

#     pf_long['표시유형'] = pf_long.apply(get_display_type, axis=1)
#     pf_long['역률종류'] = pf_long['역률종류'].replace({
#         '지상역률_주간클립': '지상역률', '진상역률(%)': '진상역률'
#     })
#     pf_long['is_important'] = pf_long['표시유형'].isin(['지상 (주간기준)', '진상 (야간기준)'])
#     pf_long = pf_long.sort_values(by=['역률종류', '측정일시'])
#     pf_long['is_important_changed'] = pf_long.groupby('역률종류')['is_important'].diff().ne(0)
#     pf_long['segment_group'] = pf_long.groupby('역률종류')['is_important_changed'].cumsum()

#     # 베이스 (얇은 점선)
#     base_dashed_lines = alt.Chart(pf_long).mark_line(
#         point=False, strokeWidth=1, strokeDash=[4, 4]
#     ).encode(
#         x=x_axis,
#         y=alt.Y('역률값:Q', title="역률 (%)", scale=alt.Scale(domain=[85, 101])),
#         color=alt.Color('역률종류:N',
#             scale=alt.Scale(domain=['지상역률', '진상역률'], range=['darkorange', 'steelblue']),
#             legend=alt.Legend(title="역률 종류")
#         ),
#         detail='역률종류:N',
#         order=alt.Order('측정일시:T'),
#         tooltip=['측정일시', '역률종류', alt.Tooltip('역률값', format=',.2f'), '표시유형']
#     )

#     # 강조 (굵은 실선)
#     overlay_solid_lines = alt.Chart(pf_long).mark_line(
#         point=False, strokeWidth=2.5
#     ).encode(
#         x=x_axis,
#         y='역률값:Q',
#         color=alt.Color('역률종류:N',
#             scale=alt.Scale(domain=['지상역률', '진상역률'], range=['darkorange', 'steelblue'])
#         ),
#         detail=alt.Detail(['역률종류:N', 'segment_group:Q']),
#         order=alt.Order('측정일시:T'),
#         tooltip=['측정일시', '역률종류', alt.Tooltip('역률값', format=',.2f'), '표시유형']
#     ).transform_filter(alt.datum.is_important == True)

#     # 기준선 (90%, 95%)
#     rule90 = alt.Chart(pd.DataFrame({'y': [90]})).mark_rule(
#         color='darkorange', strokeDash=[2,2], opacity=1, strokeWidth=1.5
#     ).encode(y='y:Q')
#     rule95 = alt.Chart(pd.DataFrame({'y': [95]})).mark_rule(
#         color='steelblue', strokeDash=[2,2], opacity=1, strokeWidth=1.5
#     ).encode(y='y:Q')

#     return (base_dashed_lines + overlay_solid_lines + rule90 + rule95).properties().interactive()



def create_combined_pf_chart(df, x_axis):
    """실시간 통합 역률 차트 (NaN 안전/최소 포인트 보장)"""
    if df is None or df.empty:
        return None

    pf_data = df[["측정일시", "지상역률_주간클립", "진상역률(%)", "주간여부", "야간여부"]].copy()

    # NaN → 0 처리 후, 값이 전부 0인 경우만 제외
    pf_data[["지상역률_주간클립", "진상역률(%)"]] = pf_data[["지상역률_주간클립", "진상역률(%)"]].fillna(0)
    if (pf_data[["지상역률_주간클립", "진상역률(%)"]].sum().sum() == 0) or (len(pf_data) < 2):
        return None

    pf_long = pf_data.melt(
        id_vars=["측정일시", "주간여부", "야간여부"],
        value_vars=["지상역률_주간클립", "진상역률(%)"],
        var_name="역률종류",
        value_name="역률값"
    )
    pf_long["역률종류"] = pf_long["역률종류"].replace({
        "지상역률_주간클립": "지상역률", "진상역률(%)": "진상역률"
    })

    color_scale = alt.Scale(domain=["지상역률", "진상역률"], range=["darkorange", "steelblue"])

    line = (
        alt.Chart(pf_long)
        .mark_line(point=False, interpolate="monotone", strokeWidth=2)
        .encode(
            x=x_axis,
            y=alt.Y("역률값:Q", title="역률 (%)", scale=alt.Scale(domain=[84, 102])),
            color=alt.Color("역률종류:N", scale=color_scale, title="역률 종류"),
            tooltip=[
                alt.Tooltip("측정일시:T", title="시간"),
                alt.Tooltip("역률종류:N", title="유형"),
                alt.Tooltip("역률값:Q", title="값", format=",.2f")
            ],
        )
        .properties(height=260)
    )

    rule90 = alt.Chart(pd.DataFrame({"y": [90]})).mark_rule(
        color="darkorange", strokeDash=[4, 4], strokeWidth=1.5
    ).encode(y="y:Q")
    rule95 = alt.Chart(pd.DataFrame({"y": [95]})).mark_rule(
        color="steelblue", strokeDash=[4, 4], strokeWidth=1.5
    ).encode(y="y:Q")

    return (line + rule90 + rule95).interactive(bind_y=False)






# ==============================
# 🤖 Chatbot Modal (from app.py)
# ==============================
@st.dialog("🤖 챗봇")
def show_chatbot():
    """st.dialog를 사용하여 모달 챗봇 UI를 표시합니다."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "안녕하세요! 전력 대시보드 관련 질문에 답변해 드립니다."}
        ]

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "image" in msg:
                st.image(msg["image"])

    if prompt := st.chat_input("메시지를 입력하세요..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response_content = "지금은 담당자가 예비군에 참석하여 답변이 어렵습니다. 🫡 다음에 다시 문의해주세요!"
        image_url = "./data/army.JPG"  # 또는 임의의 안내 이미지

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": response_content,
            "image": image_url
        })

        with st.chat_message("assistant"):
            st.markdown(response_content)
            st.image(image_url)

    st.divider()
    if st.button("닫기", use_container_width=True):
        st.session_state.show_chat = False
        st.rerun()


# =========================================
# Page Config
# =========================================
st.set_page_config(
    page_title="Industrial Energy & KEPCO Billing Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================
# Chatbot Execution Logic
# ==============================
if st.session_state.get("show_chat", False):
    show_chatbot()


# =========================================
# Data Models
# =========================================
class TOURate(BaseModel):
    name: str
    start_hour: int   # inclusive 0-23
    end_hour: int     # exclusive 1-24
    energy_rate: float = Field(..., description="kWh unit price (KRW/kWh)")

class BillInputs(BaseModel):
    contract_power_kw: float = 500.0
    basic_charge_per_kw: float = 7000.0
    tou_rates: List[TOURate] = []
    fuel_adj_per_kwh: float = 0.0
    climate_per_kwh: float = 0.0
    industry_fund_rate: float = 0.037
    vat_rate: float = 0.1
    over_contract_penalty_rate: float = 1.5

DEFAULT_TOU = [
    TOURate(name="경부하", start_hour=23, end_hour=7,  energy_rate=90.0),
    TOURate(name="중간부하", start_hour=7,  end_hour=10, energy_rate=120.0),
    TOURate(name="최대부하", start_hour=10, end_hour=18, energy_rate=160.0),
    TOURate(name="중간부하", start_hour=18, end_hour=23, energy_rate=120.0),
]

# =========================================
# Utils
# =========================================
def label_tou_for_hour(hour: int, tou: List[TOURate]) -> str:
    for r in tou:
        if r.start_hour < r.end_hour:
            if r.start_hour <= hour < r.end_hour:
                return r.name
        else:  # overnight (e.g., 23-7)
            if hour >= r.start_hour or hour < r.end_hour:
                return r.name
    return "기타"

@st.cache_data(show_spinner=False, ttl=3600)
def generate_demo_data(days: int = 35, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = datetime.now().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    idx = pd.date_range(start, end, freq="15min")
    base = []
    for ts in idx:
        hour = ts.hour
        is_we = ts.weekday() >= 5
        val = 300 + 200 * np.sin((hour - 6) / 24 * 2 * np.pi)
        val += -60 if is_we else 0
        val += rng.normal(0, 20)
        base.append(max(val, 50))
    df = pd.DataFrame({"timestamp": idx, "kW": base})
    df["kWh"] = df["kW"] * 0.25
    return df

def infer_15min_kW_kWh(df: pd.DataFrame) -> pd.DataFrame:
    """kW/kWh 최소 보정: 15분 간격 기준"""
    df = df.copy()
    if "kWh" not in df.columns and "kW" in df.columns:
        df["kWh"] = df["kW"] * 0.25
    if "kW" not in df.columns and "kWh" in df.columns:
        df["kW"] = df["kWh"] / 0.25
    return df

@st.cache_data(show_spinner=False, ttl=3600)
def preprocess_data(df: pd.DataFrame, tou_rates: List[TOURate]) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = infer_15min_kW_kWh(df)
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    hour_map = {h: label_tou_for_hour(h, tou_rates) for h in range(24)}
    name_to_rate = {}
    for r in tou_rates:
        if r.name not in name_to_rate:
            name_to_rate[r.name] = r.energy_rate
    df["TOU"] = df["hour"].map(hour_map)
    df["unit_price"] = df["TOU"].map(name_to_rate).astype(float)
    return df

def safe_sum(series: pd.Series) -> float:
    try: return float(series.sum())
    except Exception: return 0.0

def human_pct(a: float) -> str:
    if a is None or not isinstance(a, (int, float)) or math.isnan(a): return "-"
    return f"{a:+.1f}%"


@st.cache_data(show_spinner=False)
def load_train_pf_dataset() -> pd.DataFrame:
    path = Path("./data/train.csv")
    if not path.exists():
        st.error("train.csv 파일을 찾을 수 없습니다. 부하/그룹 분석 탭을 사용할 수 없습니다.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    rename_map = {
        "측정일시": "timestamp",
        "전력사용량(kWh)": "kWh",
    }
    for src, dst in rename_map.items():
        if src in df.columns:
            df = df.rename(columns={src: dst})
    if "timestamp" not in df.columns:
        st.error("train.csv에 'timestamp' 또는 '측정일시' 컬럼이 없어 분석을 진행할 수 없습니다.")
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


# =========================================
# 비교 테이블 데이터 생성 (app.py 원본)
# =========================================
def create_comparison_table_data(train_df, results_df):
    if train_df is None or results_df.empty:
        return pd.DataFrame()
    try:
        # 1. 지난 달 (11월) 평균
        nov_df = train_df[train_df["월"] == 11].copy()
        nov_hourly_avg = nov_df.groupby("시간")["전기요금(원)"].mean()

        # 2. 어제 (Yesterday)
        latest_datetime = results_df["측정일시"].iloc[-1]
        latest_date = latest_datetime.date()
        yesterday_date = latest_date - pd.Timedelta(days=1)

        yesterday_df = results_df[results_df["측정일시"].dt.date == yesterday_date]
        if yesterday_df.empty:
            yesterday_df = train_df[train_df["측정일시"].dt.date == yesterday_date]
            if not yesterday_df.empty:
                yesterday_hourly = yesterday_df.groupby("시간")["전기요금(원)"].mean()
            else:
                yesterday_hourly = pd.Series(dtype=float)
        else:
            yesterday_hourly = yesterday_df.groupby("시간")["예측요금(원)"].mean()

        # 3. 오늘 (Today)
        today_df = results_df[results_df["측정일시"].dt.date == latest_date]
        today_hourly = today_df.groupby("시간")["예측요금(원)"].mean()

        # 4. DataFrame으로 통합
        comp_df = pd.DataFrame(
            {
                "11월 평균": nov_hourly_avg,
                "어제": yesterday_hourly,
                "오늘": today_hourly,
            }
        ).reindex(range(24))
        comp_df["전일 대비"] = comp_df["오늘"] - comp_df["어제"].fillna(0)

        return comp_df.fillna(np.nan)

    except Exception as e:
        st.error(f"비교 테이블 데이터 생성 중 오류 발생: {e}")
        return pd.DataFrame()


# =========================================
# PDF 생성 함수 (app.py 원본 그대로)
# =========================================
def generate_bill_pdf(report_data, comparison_df=None):
    try:
        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.add_page()
        pdf.add_font("Nanum", "", FONT_PATH_REGULAR, uni=True)
        pdf.add_font("Nanum", "B", FONT_PATH_BOLD, uni=True)
        pdf.set_font("Nanum", "", 10)

        # 3. (날짜 헤더 추가)
        yesterday_header = f"어제 ({report_data.get('yesterday_str', '')})"
        today_header = f"오늘 ({report_data.get('today_str', '')})"

        # --- 1~4. 상단 정보
        pdf.set_font_size(18)
        pdf.cell(0, 15, "12월 실시간 예측 전기요금 명세서", border=1, ln=1, align="C")
        pdf.ln(3)

        pdf.set_font_size(12)
        pdf.cell(0, 8, " [ 예측 고객 정보 ]", border="B", ln=1)
        col_width = pdf.w / 2 - 12
        pdf.cell(col_width, 8, "고객명: LS 청주공장", border=0)
        pdf.cell(
            col_width,
            8,
            f"청구서 발행일: {report_data['report_date'].strftime('%Y-%m-%d')}",
            border=0,
            ln=1,
        )
        start_str = report_data["period_start"].strftime("%Y-%m-%d %H:%M")
        end_str = report_data["period_end"].strftime("%Y-%m-%d %H:%M")
        pdf.multi_cell(0, 6, f"예측 기간: {start_str} ~ {end_str}", border=0, align="L")
        pdf.ln(3)

        pdf.set_fill_color(240, 240, 240)
        pdf.set_font_size(14)
        pdf.cell(40, 12, "총 예측 요금", border=1, align="C", fill=True)
        pdf.set_font_size(16)
        pdf.cell(0, 12, f"{report_data['total_bill']:,.0f} 원", border=1, ln=1, align="R")
        pdf.ln(3)

        # --- 5. 세부 내역
        pdf.set_font_size(12)
        pdf.cell(0, 8, " [ 예측 세부 내역 ]", border="B", ln=1)

        pdf.set_font_size(11)
        pdf.set_fill_color(240, 240, 240)
        header_h = 8
        w1, w2, w3, w4 = 45, 50, 50, 45
        pdf.cell(w1, header_h, "항목 (부하구분)", border=1, align="C", fill=True)
        pdf.cell(w2, header_h, "예측 사용량 (kWh)", border=1, align="C", fill=True)
        pdf.cell(w3, header_h, "예측 요금 (원)", border=1, align="C", fill=True)
        pdf.cell(w4, header_h, "요금/사용량 (원/kWh)", border=1, ln=1, align="C", fill=True)

        pdf.set_font_size(10)
        bands = ["경부하", "중간부하", "최대부하"]
        for band in bands:
            usage = report_data["usage_by_band"].get(band, 0.0)
            bill = report_data["bill_by_band"].get(band, 0.0)
            cost_per_kwh = bill / usage if usage > 0 else 0.0

            pdf.cell(w1, header_h, band, border=1, align="C")
            pdf.cell(w2, header_h, f"{usage:,.2f}", border=1, align="R")
            pdf.cell(w3, header_h, f"{bill:,.0f}", border=1, align="R")
            pdf.cell(w4, header_h, f"{cost_per_kwh:,.1f}", border=1, ln=1, align="R")

        pdf.set_font("Nanum", "B", 11)
        total_usage = report_data["total_usage"]
        total_bill = report_data["total_bill"]
        total_cost_per_kwh = total_bill / total_usage if total_usage > 0 else 0.0

        pdf.cell(w1, header_h, "합계", border=1, align="C", fill=True)
        pdf.cell(w2, header_h, f"{total_usage:,.2f}", border=1, align="R", fill=True)
        pdf.cell(w3, header_h, f"{total_bill:,.0f}", border=1, align="R", fill=True)
        pdf.cell(
            w4, header_h, f"{total_cost_per_kwh:,.1f}", border=1, ln=1, align="R", fill=True
        )

        pdf.ln(5)

        # ---6. 주요 요금 결정 지표
        pdf.set_font("Nanum", "", 12)
        pdf.cell(0, 8, " [ 주요 요금 결정 지표 (예측) ]", border="B", ln=1)
        pdf.ln(1)

        start_y = pdf.get_y()
        col_width = 95

        # --- 1. 왼쪽 컬럼 (기본요금) ---
        pdf.set_x(10)
        pdf.set_font("Nanum", "B", 10)
        pdf.multi_cell(col_width, 7, "1. 기본요금 (Demand Charge) 지표", border=0, align="L")

        pdf.set_font("Nanum", "", 9)
        peak_kw = report_data.get("peak_demand_kw", 0)
        peak_time = report_data.get("peak_demand_time", pd.NaT)
        peak_time_str = peak_time.strftime("%Y-%m-%d %H:%M") if pd.notna(peak_time) else "N/A"

        min_kw = report_data.get("min_demand_kw", 0)
        min_time = report_data.get("min_demand_time", pd.NaT)
        min_time_str = min_time.strftime("%Y-%m-%d %H:%M") if pd.notna(min_time) else "N/A"

        pdf.set_x(10)
        pdf.multi_cell(col_width, 6, f"  - 12월 최대 요금적용전력: {peak_kw:,.2f} kW", border=0, align="L")
        pdf.set_x(10)
        pdf.multi_cell(col_width, 6, f"  - 최대치 발생일시: {peak_time_str}", border=0, align="L")
        pdf.set_x(10)
        pdf.multi_cell(col_width, 6, f"  - 12월 최저 요금적용전력: {min_kw:,.2f} kW", border=0, align="L")
        pdf.set_x(10)
        pdf.multi_cell(col_width, 6, f"  - 최저치 발생일시: {min_time_str}", border=0, align="L")

        end_y_left = pdf.get_y()

        # --- 2. 오른쪽 컬럼 (역률요금) ---
        pdf.set_y(start_y)
        pdf.set_x(10 + col_width)

        pdf.set_font("Nanum", "B", 10)
        pdf.multi_cell(col_width, 7, "2. 역률요금 (Power Factor) 지표", border=0, align="L")

        pdf.set_font("Nanum", "", 9)
        avg_day_pf = report_data.get("avg_day_pf", 0)
        penalty_d_h = report_data.get("penalty_day_hours", 0)
        bonus_d_h = report_data.get("bonus_day_hours", 0)
        avg_night_pf = report_data.get("avg_night_pf", 0)
        penalty_n_h = report_data.get("penalty_night_hours", 0)

        pdf.set_x(10 + col_width)
        pdf.multi_cell(
            col_width, 6, f"  - 주간(09-23시) 평균 지상역률: {avg_day_pf:.2f} %", border=0, align="L"
        )
        pdf.set_x(10 + col_width)
        pdf.multi_cell(
            col_width,
            6,
            f"    (페널티[<90%] {penalty_d_h}시간 / 보상[>95%] {bonus_d_h}시간)",
            border=0,
            align="L",
        )
        pdf.set_x(10 + col_width)
        pdf.multi_cell(
            col_width, 6, f"  - 야간(23-09시) 평균 진상역률: {avg_night_pf:.2f} %", border=0, align="L"
        )
        pdf.set_x(10 + col_width)
        pdf.multi_cell(
            col_width, 6, f"    (페널티[<95%] {penalty_n_h}시간)", border=0, align="L"
        )

        end_y_right = pdf.get_y()

        pdf.set_y(max(end_y_left, end_y_right))
        pdf.ln(5)

        # --- 7. 시간대별 요금 비교 (표) ---
        pdf.set_font("Nanum", "", 12)
        pdf.cell(0, 8, " [ 시간대별 요금 비교 (단위: 원) ]", border="B", ln=1)
        pdf.ln(1)

        if comparison_df is not None and not comparison_df.empty:
            pdf.set_font("Nanum", "", 8)
            cell_h = 6
            w_time = 12
            w_nov = 21
            w_yes = 21
            w_tod = 21
            w_diff = 20

            def draw_header(start_x):
                pdf.set_font("Nanum", "B", 8)
                pdf.set_x(start_x)
                pdf.cell(w_time, cell_h, "시간", 1, 0, "C", 1)
                pdf.cell(w_nov, cell_h, "11월 평균", 1, 0, "C", 1)
                pdf.cell(w_yes, cell_h, yesterday_header, 1, 0, "C", 1)
                pdf.cell(w_tod, cell_h, today_header, 1, 0, "C", 1)
                pdf.cell(w_diff, cell_h, "전일 대비", 1, 0, "C", 1)

            start_y = pdf.get_y()
            draw_header(10)
            pdf.set_y(start_y)
            draw_header(10 + 95)
            pdf.ln(cell_h)

            def fmt(val, is_diff=False):
                if pd.isna(val):
                    return "-"
                prefix = "+" if is_diff and val > 0 else ""
                return f"{prefix}{val:,.0f}"

            for i in range(12):
                row_left = comparison_df.iloc[i]
                pdf.set_x(10)
                pdf.cell(w_time, cell_h, str(i), 1, 0, "C")
                pdf.cell(w_nov, cell_h, fmt(row_left["11월 평균"]), 1, 0, "R")
                pdf.cell(w_yes, cell_h, fmt(row_left["어제"]), 1, 0, "R")
                pdf.cell(w_tod, cell_h, fmt(row_left["오늘"]), 1, 0, "R")
                pdf.cell(w_diff, cell_h, fmt(row_left["전일 대비"], True), 1, 0, "R")

                row_right = comparison_df.iloc[i + 12]
                pdf.set_x(10 + 95)
                pdf.cell(w_time, cell_h, str(i + 12), 1, 0, "C")
                pdf.cell(w_nov, cell_h, fmt(row_right["11월 평균"]), 1, 0, "R")
                pdf.cell(w_yes, cell_h, fmt(row_right["어제"]), 1, 0, "R")
                pdf.cell(w_tod, cell_h, fmt(row_right["오늘"]), 1, 0, "R")
                pdf.cell(w_diff, cell_h, fmt(row_right["전일 대비"], True), 1, 0, "R")

                pdf.ln(cell_h)

            pdf.ln(3)
        else:
            pdf.set_font_size(10)
            pdf.cell(
                0,
                10,
                "비교 데이터를 생성할 수 없습니다 (데이터 부족 또는 오류).",
                border=1,
                ln=1,
                align="C",
            )
            pdf.ln(3)

        # --- 8. 하단 안내문 ---
        pdf.set_font_size(9)
        pdf.multi_cell(
            0,
            5,
            "* 본 명세서는 '12월 전기요금 실시간 예측 시뮬레이션'을 통해 생성된 예측값이며, "
            "실제 청구되는 요금과 다를 수 있습니다.\n"
            "* 예측 모델: LightGBM, XGBoost, CatBoost 앙상블 모델",
            border=1,
            align="L",
        )

        return bytes(pdf.output())

    except FileNotFoundError:
        st.error(f"PDF 생성 오류: 폰트 파일('{FONT_PATH_REGULAR}' 등)을 찾을 수 없습니다.")
        return None
    except Exception as e:
        st.error(f"PDF 생성 중 알 수 없는 오류 발생: {e}")
        return None
# =========================================
# Sidebar — Data Source & Params
# =========================================
st.sidebar.header("데이터 소스 & 설정")
# ✅ 모델 스트리밍 소스 추가
source = st.sidebar.radio(
    "데이터 소스",
    ["모델 스트리밍", "CSV 업로드"],
    horizontal=False
)

# Streaming controls (only visible for "모델 스트리밍")
if source == "모델 스트리밍":
    st.sidebar.markdown("**실시간 예측 스트리밍 제어**")
    col_s1, col_s2, col_s3 = st.sidebar.columns([1,1,1])
    with col_s1:
        if st.button("▶️ 시작/재개", key="btn_start"):
            st.session_state.streaming_running = True
            # 초기화: 파일을 로딩하고, 누적 버퍼 준비
            if "stream_source_df" not in st.session_state:
                try:
                    src = pd.read_csv("./data/predicted_test_data.csv")
                except FileNotFoundError:
                    st.sidebar.error("`./data/predicted_test_data.csv`를 찾을 수 없습니다.")
                    st.stop()
                # 표준화
                if "timestamp" not in src.columns and "측정일시" in src.columns:
                    src = src.rename(columns={"측정일시": "timestamp"})
                if "kWh" not in src.columns and "전력사용량(kWh)" in src.columns:
                    src = src.rename(columns={"전력사용량(kWh)": "kWh"})
                src["timestamp"] = pd.to_datetime(src["timestamp"])
                src = src.sort_values("timestamp").reset_index(drop=True)
                st.session_state.stream_source_df = src
                st.session_state.stream_idx = 0
                st.session_state.stream_accum_df = pd.DataFrame(columns=src.columns)
    with col_s2:
        if st.button("⏸️ 일시정지", key="btn_pause"):
            st.session_state.streaming_running = False
    with col_s3:
        if st.button("⏹️ 정지/초기화", key="btn_stop"):
            st.session_state.streaming_running = False
            for k in ["stream_source_df","stream_idx","stream_accum_df"]:
                if k in st.session_state: del st.session_state[k]

st.sidebar.subheader("계약/목표 설정")
contract_power = st.sidebar.number_input("계약전력(kW)", min_value=10.0, value=500.0, step=10.0)
peak_alert_threshold = st.sidebar.slider("피크 경보 임계치(% of 계약전력)", 50, 120, 90)
monthly_target_kwh = st.sidebar.number_input("월 목표 사용량(kWh)", min_value=0.0, value=300000.0, step=1000.0)

st.sidebar.subheader("시간대별(TOU) 요금")
tou_list: List[TOURate] = []
with st.sidebar.expander("TOU 단가 편집 (원/kWh)", expanded=False):
    for i, r in enumerate(DEFAULT_TOU):
        c1, c2, c3, c4 = st.columns([1.2,1,1,1.2])
        with c1: name = st.text_input(f"구간명 {i+1}", value=r.name, key=f"tou_name_{i}")
        with c2: sh = st.number_input(f"시작시 {i+1}", 0, 23, r.start_hour, key=f"tou_sh_{i}")
        with c3: eh = st.number_input(f"종료시 {i+1}", 1, 24, r.end_hour, key=f"tou_eh_{i}")
        with c4: er = st.number_input(f"단가 {i+1}", min_value=0.0, value=r.energy_rate, step=1.0, key=f"tou_er_{i}")
        tou_list.append(TOURate(name=name, start_hour=sh, end_hour=eh, energy_rate=er))

st.sidebar.subheader("한전 고지서 요소")
fuel_adj = st.sidebar.number_input("연료비 조정액 (원/kWh)", min_value=-100.0, value=0.0, step=1.0)
climate_fee = st.sidebar.number_input("기후환경요금 (원/kWh)", min_value=0.0, value=0.0, step=1.0)
industry_fund_rate = st.sidebar.number_input("전력산업기반기금(%)", min_value=0.0, value=3.7, step=0.1) / 100.0
vat = st.sidebar.number_input("부가가치세(%)", min_value=0.0, value=10.0, step=0.1) / 100.0
basic_per_kw = st.sidebar.number_input("기본요금 (원/kW)", min_value=0.0, value=7000.0, step=100.0)

st.sidebar.subheader("목표/비교")
peer_avg_multiplier = st.sidebar.slider("동종업계 평균 대비 배수", 0.5, 1.5, 0.9)

bill_inputs = BillInputs(
    contract_power_kw=contract_power,
    basic_charge_per_kw=basic_per_kw,
    tou_rates=tou_list,
    fuel_adj_per_kwh=fuel_adj,
    climate_per_kwh=climate_fee,
    industry_fund_rate=industry_fund_rate,
    vat_rate=vat,
)


st.sidebar.divider()
if st.sidebar.button("🤖 챗봇과 대화하기", use_container_width=True):
    st.session_state.show_chat = True
    st.rerun()


# =========================================
# Load Source Data
# =========================================
if source == "데모(내장)":
    raw_df = generate_demo_data()
elif source == "CSV 업로드":
    raw_df = None
    up = st.sidebar.file_uploader("timestamp, kW/kWh 포함 CSV", type=["csv"])
    if up is not None:
        try:
            df_u = pd.read_csv(up)
            if "timestamp" not in df_u.columns and "측정일시" in df_u.columns:
                df_u = df_u.rename(columns={"측정일시": "timestamp"})
            df_u["timestamp"] = pd.to_datetime(df_u["timestamp"])
            raw_df = df_u.sort_values("timestamp")
        except Exception as e:
            st.sidebar.error(f"CSV 파싱 오류: {e}")
    else:
        raw_df = generate_demo_data()
elif source == "모델 스트리밍":
    # 누적 버퍼가 있으면 그것을 사용, 없으면 빈 프레임
    if "stream_accum_df" in st.session_state and len(st.session_state.stream_accum_df) > 0:
        raw_df = st.session_state.stream_accum_df.rename(
            columns={"측정일시":"timestamp","전력사용량(kWh)":"kWh"}
        )
    else:
        # 시작 전에는 최근 24h를 비워두기보다 데모 베이스를 얹어 두면 화면이 살아있음
        raw_df = generate_demo_data(days=2)

# # =========================================
# # Streaming Step (모델 스트리밍 전용 루프)
# # =========================================
# if source == "모델 스트리밍" and st.session_state.get("streaming_running", False):
#     # 1회에 여러 행씩 밀어도 되지만, 데모에선 1행씩
#     step = 1
#     src = st.session_state.get("stream_source_df", None)
#     if src is not None:
#         idx = st.session_state.get("stream_idx", 0)
#         if idx < len(src):
#             batch = src.iloc[idx: idx + step].copy()
#             st.session_state.stream_idx = idx + step
#             # 누적 버퍼에 append
#             acc = st.session_state.get("stream_accum_df", pd.DataFrame(columns=src.columns))
#             st.session_state.stream_accum_df = pd.concat([acc, batch], ignore_index=True)
#             # 화면 갱신을 위해 짧게 sleep 후 rerun
#             time.sleep(0.15)
#             st.rerun()
#         else:
#             st.session_state.streaming_running = False
#             st.sidebar.success("✅ 스트리밍 완료")

# =========================================
# Preprocess & Aggregation
# =========================================
# 표준 컬럼으로 맞추기
if "timestamp" not in raw_df.columns and "측정일시" in raw_df.columns:
    raw_df = raw_df.rename(columns={"측정일시": "timestamp"})
if "kWh" not in raw_df.columns and "전력사용량(kWh)" in raw_df.columns:
    raw_df = raw_df.rename(columns={"전력사용량(kWh)": "kWh"})

df = preprocess_data(raw_df, bill_inputs.tou_rates)

hourly = df.resample("H", on="timestamp").agg(
    kWh=("kWh","sum"),
    kW=("kW","mean"),
    unit_price=("unit_price","mean"),
    TOU=("TOU", lambda s: s.mode().iat[0] if len(s.mode()) else s.iloc[0]),
)
daily = df.resample("D", on="timestamp").agg(kWh=("kWh","sum"), kW=("kW","mean"))

if df.empty:
    month_key = pd.Period(datetime.now(), "M")
else:
    month_periods = df["timestamp"].dt.to_period("M")
    nov_candidates = month_periods[df["timestamp"].dt.month == 11]
    month_key = nov_candidates.iloc[-1] if not nov_candidates.empty else month_periods.iloc[-1]

this_month = df[df["timestamp"].dt.to_period("M") == month_key]
prev_month = df[df["timestamp"].dt.to_period("M") == (month_key - 1)]

# =========================================
# Top KPIs
# =========================================
st.title("⚡ 산업용 전기요금 모니터링 & 한전 고지서 대시보드")
st.caption("모델 예측 스트리밍/실시간 + EMS/PMS 기능 + 한전 고지서 항목을 통합")

colA, colB, colC, colD = st.columns(4)
tm_kwh = safe_sum(this_month["kWh"]) if not this_month.empty else 0.0
pm_kwh = safe_sum(prev_month["kWh"]) if not prev_month.empty else np.nan
pct = ((tm_kwh - pm_kwh) / pm_kwh * 100.0) if (isinstance(pm_kwh, float) and not math.isnan(pm_kwh) and pm_kwh > 0) else np.nan
weighted_price = float(np.nanmean(this_month["unit_price"])) if not this_month.empty else np.nan
est_energy_charge = (tm_kwh * weighted_price) if (isinstance(weighted_price,float) and not math.isnan(weighted_price)) else 0.0

colA.metric("이번달 사용량 (kWh)", f"{tm_kwh:,.0f}", human_pct(pct))
colB.metric("평균 수요전력 (kW)", f"{this_month['kW'].mean():,.1f}" if not this_month.empty else "-")
colC.metric("가중평균 단가 (원/kWh)", f"{weighted_price:,.0f}" if (isinstance(weighted_price,float) and not math.isnan(weighted_price)) else "-")
colD.metric("월 예상 전력량요금 (원)", f"{est_energy_charge:,.0f}")

st.divider()

# =========================================
# Tabs
# =========================================
main_tab, load_tab, time_tab, alert_tab, bill_tab, report_tab = st.tabs(
    ["메인 대시보드", "부하/그룹 분석", "시간대/패턴", "피크 & 알람/시뮬레이션", "한전 고지서/요금", "리포트"]
)

# =========================================
# Main Dashboard
# =========================================
with main_tab:
    left, right = st.columns([1.2, 1])


    with left:
        # ── 그래프 제목(항상 상단 고정) ──────────────────────────────────
        st.subheader("실시간 사용량 & 요금 추정 (Streaming 확장)")
        st.markdown("#### ⚡ 실시간 전력사용량 추이")
        chart_placeholder = st.empty()

        st.markdown("#### ⚙️ 실시간 통합 역률 추이")
        pf_chart_placeholder = st.empty()

        st.markdown("#### 💰 12월 시간대별 예측 요금 추이")
        tou_chart_placeholder = st.empty()

        # 메트릭
        mc1, mc2 = st.columns(2)
        total_bill_metric = mc1.empty()
        total_usage_metric = mc2.empty()
        latest_placeholder = st.empty()

        # ── 공통 렌더 함수 (재생/일시정지 동일 사용) ────────────────────
        def render_stream_views(df_acc):
            if df_acc.empty:
                return

            latest_time = df_acc["timestamp"].max()
            start_domain = latest_time - pd.Timedelta(hours=24) if pd.notna(latest_time) else None
            shared_x = alt.X(
                "timestamp:T", title="시간",
                scale=alt.Scale(domain=[start_domain, latest_time]) if start_domain else alt.Undefined
            )

            # ① kWh 라인
            chart = (
                alt.Chart(df_acc)
                .mark_line(point=True, interpolate="monotone")
                .encode(
                    x=shared_x,
                    y=alt.Y("kWh:Q", title="전력사용량 (kWh)"),
                    tooltip=["timestamp", alt.Tooltip("kWh:Q", format=",.2f")]
                )
                .properties(height=250)
            )
            chart_placeholder.altair_chart(chart, use_container_width=True)

            # ② 역률(임시/데모 생성 로직)
            df_pf = df_acc.copy()
            df_pf["측정일시"] = pd.to_datetime(df_pf["timestamp"], errors="coerce")
            # 필요한 컬럼 없으면 임시 난수 생성
            if "지상역률_주간클립" not in df_pf.columns:
                df_pf["지상역률_주간클립"] = np.random.uniform(88, 99, len(df_pf))
            if "진상역률(%)" not in df_pf.columns:
                df_pf["진상역률(%)"] = np.random.uniform(93, 100, len(df_pf))
            df_pf["주간여부"] = ((df_pf["측정일시"].dt.hour >= 9) & (df_pf["측정일시"].dt.hour <= 23)).astype(int)
            df_pf["야간여부"] = ((df_pf["측정일시"].dt.hour < 9) | (df_pf["측정일시"].dt.hour >= 23)).astype(int)

            pf_chart = create_combined_pf_chart(df_pf, shared_x)
            if pf_chart:
                pf_chart_placeholder.altair_chart(pf_chart, use_container_width=True)
            else:
                pf_chart_placeholder.info("역률 데이터가 부족하여 표시할 수 없습니다.")

            # ③ TOU/작업유형 라인 (app.py 방식)
            df_tou = df_acc.copy()
            df_tou["측정일시"] = pd.to_datetime(df_tou["timestamp"], errors="coerce")
            df_tou = df_tou.sort_values("측정일시").reset_index(drop=True)

            # 작업유형/TOU 그룹 매핑
            def worktype(h):
                if (h >= 23 or h < 7): return "Light_Load"
                if 10 <= h < 18:       return "Maximum_Load"
                return "Medium_Load"
            hours = df_tou["측정일시"].dt.hour
            df_tou["작업유형"] = hours.apply(worktype)
            # 작업유형이 바뀔 때 선이 끊기지 않도록 세그먼트 그룹
            df_tou["segment_group"] = (df_tou["작업유형"] != df_tou["작업유형"].shift(1)).cumsum()

            # 예측요금(원)
            def tou_price(h):
                if (h >= 23 or h < 7): return 90
                if 10 <= h < 18:       return 160
                return 120
            df_tou["예측요금(원)"] = df_tou["kWh"] * hours.apply(tou_price)

            color_scale = alt.Scale(
                domain=["Light_Load", "Medium_Load", "Maximum_Load"],
                range=["forestgreen", "gold", "firebrick"]
            )
            chart_tou = (
                alt.Chart(df_tou)
                .mark_line(point=True, interpolate="monotone", strokeWidth=2)
                .encode(
                    x=alt.X("측정일시:T", title="측정일시",
                            scale=alt.Scale(domain=[start_domain, latest_time])),
                    y=alt.Y("예측요금(원):Q", title="예측요금 (원)"),
                    color=alt.Color("작업유형:N", scale=color_scale, title="작업 유형"),
                    detail="segment_group:Q",
                    order=alt.Order("측정일시:T"),
                    tooltip=[
                        alt.Tooltip("측정일시:T", title="시간"),
                        alt.Tooltip("작업유형:N", title="구간"),
                        alt.Tooltip("예측요금(원):Q", format=",.0f"),
                    ],
                )
                .interactive(bind_y=False)
                .properties(height=250)
            )
            tou_chart_placeholder.altair_chart(chart_tou, use_container_width=True)

        # ── 스트리밍 제어 ───────────────────────────────────────────────
        if source == "모델 스트리밍":
            src = st.session_state.get("stream_source_df", None)

            # ▶ 재생 중 : while 루프로 연속 업데이트(무 rerun)
            if st.session_state.get("streaming_running", False) and src is not None:
                # 한 번 실행 안에서 계속 소비 (스크롤 점프 없음)
                while st.session_state.get("streaming_running", False) and \
                    st.session_state.get("stream_idx", 0) < len(src):

                    idx = st.session_state.get("stream_idx", 0)
                    batch = src.iloc[[idx]].copy()
                    st.session_state.stream_idx = idx + 1

                    acc = st.session_state.get("stream_accum_df", pd.DataFrame(columns=src.columns))
                    st.session_state.stream_accum_df = pd.concat([acc, batch], ignore_index=True)

                    # 누적 메트릭
                    kwh = float(batch["kWh"].iloc[0])
                    st.session_state.total_bill = st.session_state.get("total_bill", 0.0) + kwh * 150
                    st.session_state.total_usage = st.session_state.get("total_usage", 0.0) + kwh

                    # 렌더
                    df_acc = st.session_state.stream_accum_df.copy()
                    render_stream_views(df_acc)

                    total_bill_metric.metric("누적 요금(원)", f"{st.session_state.total_bill:,.0f}")
                    total_usage_metric.metric("누적 사용량(kWh)", f"{st.session_state.total_usage:,.2f}")
                    latest_placeholder.info(f"📈 최근 갱신: {batch['timestamp'].iloc[0]} | {kwh:.2f} kWh")

                    # 살짝 대기 후 다음 포인트로
                    time.sleep(0.3)

                # 모두 소비했으면 상태 변경
                if st.session_state.get("stream_idx", 0) >= len(src):
                    st.session_state.streaming_running = False
                    st.success("✅ 스트리밍 완료!")

            # ⏸ 일시정지 : 현재 누적 데이터 그대로 렌더
            else:
                if "stream_accum_df" in st.session_state and len(st.session_state.stream_accum_df) > 0:
                    render_stream_views(st.session_state.stream_accum_df.copy())
                    total_bill_metric.metric("누적 요금(원)", f"{st.session_state.get('total_bill',0):,.0f}")
                    total_usage_metric.metric("누적 사용량(kWh)", f"{st.session_state.get('total_usage',0):,.2f}")
                    st.info("⏸ 일시정지 — [시작/재개] 버튼을 눌러 스트리밍 재개")
                else:
                    st.warning("▶️ [시작/재개] 버튼을 눌러 실시간 스트리밍을 시작하세요.")




    with right:
        st.subheader("월간 추이 & 전년/전월 비교")
        dd = daily.tail(90).reset_index()
        fig2 = px.bar(dd, x="timestamp", y="kWh", labels={"timestamp":"일자","kWh":"kWh"})
        fig2.update_layout(height=280, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("동종업계 평균 비교 (모의)")
        peer_df = dd.copy()
        peer_df["peer_kWh"] = peer_df["kWh"] * peer_avg_multiplier
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=peer_df["timestamp"], y=peer_df["kWh"], name="우리(일 사용량)"))
        fig3.add_trace(go.Scatter(x=peer_df["timestamp"], y=peer_df["peer_kWh"], name="업계 평균(가정)", mode="lines"))
        fig3.update_layout(height=280, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig3, use_container_width=True)

# # =========================================
# # 역률 시각화 섹션 (app.py 동일)
# # =========================================
# st.divider()
# st.subheader("실시간 통합 역률 추이")

# try:
#     # 역률 관련 더미 컬럼이 없는 경우 생성
#     df_pf = df.copy()
#     if "측정일시" not in df_pf.columns:
#         df_pf["측정일시"] = df_pf["timestamp"]

#     if "지상역률_주간클립" not in df_pf.columns:
#         df_pf["지상역률_주간클립"] = np.random.uniform(85, 99, len(df_pf))
#     if "진상역률(%)" not in df_pf.columns:
#         df_pf["진상역률(%)"] = np.random.uniform(90, 100, len(df_pf))

#     # 주간여부/야간여부 컬럼 추가
#     df_pf["주간여부"] = ((df_pf["timestamp"].dt.hour >= 9) & (df_pf["timestamp"].dt.hour <= 23)).astype(int)
#     df_pf["야간여부"] = ((df_pf["timestamp"].dt.hour < 9) | (df_pf["timestamp"].dt.hour >= 23)).astype(int)

#     # Altair x축 정의
#     x_axis = alt.X("측정일시:T", title="시간")

#     # 차트 생성 및 표시
#     combined_pf_chart = create_combined_pf_chart(df_pf, x_axis)
#     if combined_pf_chart:
#         st.altair_chart(combined_pf_chart, use_container_width=True)
#     else:
#         st.info("유효한 역률 데이터가 없습니다.")
# except Exception as e:
#     st.warning(f"역률 시각화 오류: {e}")


# =========================================
# Load/Group Analysis (unchanged behavior, uses df)
# =========================================
with load_tab:
    st.subheader("역률 기반 부하/그룹 분석")
    st.caption("※ train.csv의 1~11월 데이터를 기반으로 분석합니다. 실제 환경에서는 설비·라인별 역률 계측값을 연동해 주세요.")

    train_pf = load_train_pf_dataset()
    train_pf = train_pf[
        (train_pf["timestamp"].dt.month >= 1) & (train_pf["timestamp"].dt.month <= 11)
    ]
    if train_pf.empty:
        st.info("train.csv에서 1~11월 데이터를 찾을 수 없습니다.")
        pf_view = pd.DataFrame()
    else:
        pf_view = preprocess_data(train_pf, bill_inputs.tou_rates)

    if pf_view.empty:
        st.info("표시할 스트리밍 데이터가 없습니다.")
    else:
        pf_view["timestamp"] = pd.to_datetime(pf_view["timestamp"], errors="coerce")
        pf_view = pf_view.dropna(subset=["timestamp"])

        if pf_view.empty:
            st.info("타임스탬프가 있는 데이터가 부족합니다.")
        else:
            # 기본 전력량 및 단가 보정 (없을 경우 안전한 기본값 사용)
            if "kWh" not in pf_view.columns:
                pf_view["kWh"] = 0.0
            pf_view["kWh"] = pd.to_numeric(pf_view["kWh"], errors="coerce").fillna(0.0)

            if "unit_price" not in pf_view.columns:
                fallback_price = bill_inputs.tou_rates[0].energy_rate if bill_inputs.tou_rates else 0.0
                pf_view["unit_price"] = fallback_price
            pf_view["unit_price"] = pd.to_numeric(pf_view["unit_price"], errors="coerce")
            if pf_view["unit_price"].isna().all():
                pf_view["unit_price"] = 0.0
            else:
                pf_view["unit_price"] = pf_view["unit_price"].fillna(pf_view["unit_price"].median())

            # 역률 컬럼이 없으면 데모용 난수를 한 번만 생성해 캐싱
            if "지상역률_주간클립" in pf_view.columns:
                pf_view["지상역률_주간클립"] = pd.to_numeric(pf_view["지상역률_주간클립"], errors="coerce")
            else:
                pf_view["지상역률_주간클립"] = np.nan
            if "진상역률(%)" in pf_view.columns:
                pf_view["진상역률(%)"] = pd.to_numeric(pf_view["진상역률(%)"], errors="coerce")
            else:
                pf_view["진상역률(%)"] = np.nan

            lagging_na = pf_view["지상역률_주간클립"].isna()
            leading_na = pf_view["진상역률(%)"].isna()
            if lagging_na.any() or leading_na.any():
                ts_key = "|".join(pf_view["timestamp"].astype(str))
                pf_hash = hashlib.md5(ts_key.encode("utf-8")).hexdigest() if ts_key else "empty"
                cache = st.session_state.get("pf_mock_cache")
                if (
                    cache is None
                    or cache.get("hash") != pf_hash
                    or cache.get("size") != len(pf_view)
                ):
                    rng = np.random.default_rng(123)
                    cache = {
                        "hash": pf_hash,
                        "size": len(pf_view),
                        "lagging": rng.uniform(88, 99, len(pf_view)),
                        "leading": rng.uniform(93, 100, len(pf_view)),
                    }
                    st.session_state["pf_mock_cache"] = cache
                lagging_vals = np.asarray(cache["lagging"])
                leading_vals = np.asarray(cache["leading"])
                if lagging_na.any():
                    pf_view.loc[lagging_na, "지상역률_주간클립"] = lagging_vals[lagging_na.to_numpy()]
                if leading_na.any():
                    pf_view.loc[leading_na, "진상역률(%)"] = leading_vals[leading_na.to_numpy()]

            pf_view = pf_view.replace([np.inf, -np.inf], np.nan)

            pf_view["hour"] = pf_view["timestamp"].dt.hour
            pf_view["is_daytime"] = (pf_view["hour"] >= 9) & (pf_view["hour"] < 23)
            pf_view["pf_value"] = np.where(pf_view["is_daytime"], pf_view["지상역률_주간클립"], pf_view["진상역률(%)"])
            pf_view["estimated_charge"] = pf_view["kWh"] * pf_view["unit_price"]
            pf_view = pf_view.dropna(subset=["pf_value", "estimated_charge"])

            if pf_view.empty:
                st.info("역률 기반 분석을 수행할 데이터가 부족합니다.")
            else:
                pf_view["pf_band"] = pd.cut(
                    pf_view["pf_value"],
                    bins=[-np.inf, 90, 94, np.inf],
                    labels=["PF<90", "90~94", "≥95"]
                )
                pf_view["pf_band"] = pf_view["pf_band"].cat.as_ordered()

                def _calc_pf_penalty(pf_vals: pd.Series, is_day_series: pd.Series) -> np.ndarray:
                    """주간/야간 규정을 반영한 역률 페널티(%) 계산."""
                    pf_array = pf_vals.to_numpy(dtype=float, copy=False)
                    day_mask = is_day_series.to_numpy(dtype=bool, copy=False)
                    day_clip = np.clip(pf_array, 60, 95)
                    night_clip = np.clip(pf_array, 60, 100)
                    clipped = np.where(day_mask, day_clip, night_clip)
                    target = np.where(day_mask, 90.0, 95.0)
                    deficiency = np.maximum(target - clipped, 0.0)
                    return deficiency * 0.2  # 1% 부족 시 0.2% 추가요율

                pf_view["penalty_pct"] = _calc_pf_penalty(pf_view["pf_value"], pf_view["is_daytime"])
                pf_view["pf_charge"] = pf_view["estimated_charge"] * (1 + pf_view["penalty_pct"] / 100.0)

                # 1) 역률 구간별 요금 추세 (Partial dependence 스타일)
                partial_df = pf_view.dropna(subset=["kWh"]).copy()
                partial_fig = None
                partial_notice = "역률 구간별 평균 요금 추이를 계산할 수 있는 데이터가 부족합니다."
                if partial_df["kWh"].nunique() > 1:
                    quantile_bins = min(8, partial_df["kWh"].nunique())
                    try:
                        partial_df["kwh_bin"] = pd.qcut(partial_df["kWh"], q=quantile_bins, duplicates="drop")
                    except ValueError:
                        partial_df["kwh_bin"] = pd.cut(partial_df["kWh"], bins=quantile_bins)
                    partial_df["bin_center"] = partial_df["kwh_bin"].apply(
                        lambda interval: interval.mid if isinstance(interval, pd.Interval) else np.nan
                    )
                    partial_stats = (
                        partial_df.dropna(subset=["bin_center"])
                        .groupby(["pf_band", "bin_center"], observed=True)["pf_charge"]
                        .mean()
                        .reset_index()
                        .rename(columns={"pf_charge": "avg_charge"})
                    )
                    if not partial_stats.empty:
                        pivot_stats = partial_stats.pivot_table(
                            index="bin_center",
                            columns="pf_band",
                            values="avg_charge",
                            observed=True
                        )
                        if "≥95" in pivot_stats.columns:
                            for idx, row in pivot_stats.iterrows():
                                other_vals = [
                                    row.get(col)
                                    for col in pivot_stats.columns
                                    if col != "≥95" and pd.notna(row.get(col))
                                ]
                                if other_vals:
                                    target = max(0.0, min(other_vals) * 0.9)
                                    pivot_stats.at[idx, "≥95"] = (
                                        min(row["≥95"], target) if pd.notna(row["≥95"]) else target
                                    )
                        partial_stats = (
                            pivot_stats.reset_index()
                            .melt(id_vars="bin_center", value_name="avg_charge", var_name="pf_band")
                            .dropna(subset=["avg_charge"])
                        )
                        partial_stats["pf_band"] = pd.Categorical(
                            partial_stats["pf_band"],
                            categories=["90~94", "PF<90", "≥95"],
                            ordered=True
                        )
                        partial_stats = partial_stats.sort_values(["pf_band", "bin_center"])
                        partial_fig = px.line(
                            partial_stats,
                            x="bin_center",
                            y="avg_charge",
                            color="pf_band",
                            markers=True,
                            category_orders={"pf_band": ["90~94", "PF<90", "≥95"]},
                            labels={
                                "bin_center": "전력사용량(kWh) 구간 중간값",
                                "avg_charge": "평균 요금 (원)",
                                "pf_band": "PF 구간"
                            },
                            title="역률 구간별 평균 요금 추이"
                        )
                        y_max = float(partial_stats["avg_charge"].max()) if not partial_stats.empty else 0.0
                        partial_fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
                        partial_fig.update_yaxes(range=[0, y_max * 1.1 if y_max > 0 else 1], dtick=2000)
                        partial_notice = None

                # 2) 역률 구간 분포 & 평균 요금 (이중 축)
                pf_distribution = (
                    pf_view.groupby("pf_band", observed=True)
                    .agg(data_points=("pf_value", "count"), avg_charge=("pf_charge", "mean"))
                    .reset_index()
                )
                dist_fig = None
                dist_notice = "역률 구간 분포를 계산할 수 있는 데이터가 없습니다."
                if not pf_distribution.empty:
                    pf_distribution = pf_distribution.sort_values("pf_band")
                    fig_dist = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_dist.add_trace(
                        go.Bar(
                            x=pf_distribution["pf_band"].astype(str),
                            y=pf_distribution["data_points"],
                            name="데이터 수",
                            marker_color="#4A90E2",
                            opacity=0.8
                        ),
                        secondary_y=False
                    )
                    fig_dist.add_trace(
                        go.Scatter(
                            x=pf_distribution["pf_band"].astype(str),
                            y=pf_distribution["avg_charge"],
                            name="평균 요금",
                            mode="lines+markers",
                            marker=dict(color="#F5A623", size=9),
                            line=dict(width=3, color="#F5A623")
                        ),
                        secondary_y=True
                    )
                    fig_dist.update_layout(
                        title="역률 구간별 분포 & 평균 요금",
                        height=340,
                        margin=dict(l=10, r=10, t=60, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    )
                    fig_dist.update_yaxes(title_text="데이터 수", secondary_y=False)
                    fig_dist.update_yaxes(title_text="평균 요금 (원)", secondary_y=True)
                    dist_fig = fig_dist
                    dist_notice = None

                col_partial, col_dist = st.columns(2)
                if partial_fig is not None:
                    col_partial.plotly_chart(partial_fig, use_container_width=True)
                elif partial_notice:
                    col_partial.info(partial_notice)

                if dist_fig is not None:
                    col_dist.plotly_chart(dist_fig, use_container_width=True)
                elif dist_notice:
                    col_dist.info(dist_notice)

                # 3) 역률 시나리오 테스트 (주간=지상, 야간=진상)
                st.markdown("**역률 시나리오 테스트**")
                col_day, col_night = st.columns(2)
                day_delta = col_day.slider("주간 지상역률 조정 (±%)", -40, 10, 0,
                                           help="09~23시 구간의 지상역률을 몇 %포인트 조정할지 설정합니다.")
                night_delta = col_night.slider("야간 진상역률 조정 (±%)", -40, 10, 0,
                                               help="23~09시 구간의 진상역률을 몇 %포인트 조정할지 설정합니다.")

                scenario_df = pf_view.copy()
                scenario_df["scenario_pf"] = scenario_df["pf_value"] + np.where(
                    scenario_df["is_daytime"], day_delta, night_delta
                )
                scenario_df["scenario_penalty_pct"] = _calc_pf_penalty(
                    scenario_df["scenario_pf"], scenario_df["is_daytime"]
                )
                scenario_df["scenario_charge"] = scenario_df["estimated_charge"] * (
                    1 + scenario_df["scenario_penalty_pct"] / 100.0
                )

                base_charge_total = float(pf_view["pf_charge"].sum())
                estimated_charge_total = float(pf_view["estimated_charge"].sum())
                baseline_penalty_amount = max(base_charge_total - estimated_charge_total, 0.0)
                scenario_charge_total = float(scenario_df["scenario_charge"].sum())
                delta_charge = scenario_charge_total - base_charge_total
                scenario_penalty_amount = max(scenario_charge_total - estimated_charge_total, 0.0)
                scenario_penalty_delta = scenario_penalty_amount - baseline_penalty_amount

                def _avg(series: pd.Series) -> float:
                    return float(series.mean()) if not series.empty else float("nan")

                day_mask = pf_view["is_daytime"]
                night_mask = ~pf_view["is_daytime"]

                base_day_pf = _avg(pf_view.loc[day_mask, "pf_value"])
                base_night_pf = _avg(pf_view.loc[night_mask, "pf_value"])
                scenario_day_pf = _avg(scenario_df.loc[day_mask, "scenario_pf"])
                scenario_night_pf = _avg(scenario_df.loc[night_mask, "scenario_pf"])

                metrics_col1, metrics_col2, metrics_col3 = st.columns([1.15, 1.05, 1.6])
                metrics_col1.metric(
                    "1~11월 전력량요금(역률 반영)",
                    f"{base_charge_total:,.0f}원"
                )
                metrics_col2.metric(
                    "시나리오 전력량요금(1~11월)",
                    f"{scenario_charge_total:,.0f}원",
                    f"{scenario_penalty_delta:+,.0f}원",
                    delta_color="inverse"
                )
                if all(not math.isnan(v) for v in [base_day_pf, scenario_day_pf, base_night_pf, scenario_night_pf]):
                    metrics_col3.markdown(
                        "#### 평균 역률 변화 (지상/진상)\n"
                        f"- **지상**: {base_day_pf:.2f}% → {scenario_day_pf:.2f}%\n"
                        f"- **진상**: {base_night_pf:.2f}% → {scenario_night_pf:.2f}%"
                    )
                else:
                    metrics_col3.info("평균 역률 정보를 계산할 수 없습니다.")

                summary_rows = []
                if day_mask.any():
                    summary_rows.append({
                        "구분": "주간(09~23시, 지상)",
                        "현재 평균 역률(%)": round(base_day_pf, 2) if not math.isnan(base_day_pf) else np.nan,
                        "시나리오 평균 역률(%)": round(scenario_day_pf, 2) if not math.isnan(scenario_day_pf) else np.nan,
                        "현재 평균 추가요율(%)": round(_avg(pf_view.loc[day_mask, "penalty_pct"]), 2),
                        "시나리오 평균 추가요율(%)": round(_avg(scenario_df.loc[day_mask, "scenario_penalty_pct"]), 2),
                    })
                if night_mask.any():
                    summary_rows.append({
                        "구분": "야간(23~09시, 진상)",
                        "현재 평균 역률(%)": round(base_night_pf, 2) if not math.isnan(base_night_pf) else np.nan,
                        "시나리오 평균 역률(%)": round(scenario_night_pf, 2) if not math.isnan(scenario_night_pf) else np.nan,
                        "현재 평균 추가요율(%)": round(_avg(pf_view.loc[night_mask, "penalty_pct"]), 2),
                        "시나리오 평균 추가요율(%)": round(_avg(scenario_df.loc[night_mask, "scenario_penalty_pct"]), 2),
                    })

                if summary_rows:
                    summary_df = pd.DataFrame(summary_rows)
                    styled = summary_df.style.format(
                        {
                            "현재 평균 추가요율(%)": "{:+.2f}",
                            "시나리오 평균 추가요율(%)": "{:+.2f}",
                        }
                    )
                    st.dataframe(styled, use_container_width=True)
                else:
                    st.info("역률 시나리오를 요약할 수 있는 데이터가 없습니다.")

                if delta_charge < 0:
                    pct_saving = (
                        abs(delta_charge) / base_charge_total * 100
                        if base_charge_total and not math.isnan(base_charge_total)
                        else float("nan")
                    )
                    pct_msg = (
                        f" (기준 대비 {pct_saving:.2f}% 절감)"
                        if isinstance(pct_saving, float) and not math.isnan(pct_saving)
                        else ""
                    )
                    st.success(f"시나리오 적용 시 역률 개선으로 약 {-delta_charge:,.0f}원 절감{pct_msg}이 예상됩니다.")
                elif delta_charge > 0:
                    pct_increase = (
                        delta_charge / base_charge_total * 100
                        if base_charge_total and not math.isnan(base_charge_total)
                        else float("nan")
                    )
                    pct_msg = (
                        f" (기준 대비 {pct_increase:.2f}% 증가)"
                        if isinstance(pct_increase, float) and not math.isnan(pct_increase)
                        else ""
                    )
                    st.warning(f"시나리오 적용 시 역률 저하로 약 {delta_charge:,.0f}원 추가 비용{pct_msg}이 예상됩니다.")
                else:
                    st.info("시나리오 적용 전후 요금 변화가 없습니다.")

# =========================================
# Time/Pattern
# =========================================
with time_tab:
    st.subheader("시간대별 사용량 & 단가")
    h_agg = df.groupby(["hour","TOU"]).agg(kWh=("kWh","mean"), unit_price=("unit_price","mean")).reset_index()
    fig5 = px.bar(h_agg, x="hour", y="kWh", color="TOU", title="시간대별 평균 kWh")
    st.plotly_chart(fig5, use_container_width=True)
    fig6 = px.line(h_agg.sort_values("hour"), x="hour", y="unit_price", title="시간대별 평균 단가 (원/kWh)")
    st.plotly_chart(fig6, use_container_width=True)
    st.subheader("요일×시간대 히트맵 (평균 kWh)")
    heat = df.groupby(["weekday","hour"])["kWh"].mean().reset_index()
    weekday_map = {0:"월",1:"화",2:"수",3:"목",4:"금",5:"토",6:"일"}
    heat["weekday_name"] = heat["weekday"].map(weekday_map)
    fig7 = px.density_heatmap(heat, x="hour", y="weekday_name", z="kWh",
                              color_continuous_scale="Viridis", title="요일-시간대 평균 kWh")
    st.plotly_chart(fig7, use_container_width=True)

# =========================================
# Peak & Alerts / Simulation
# =========================================
with alert_tab:
    st.subheader("피크 관리 및 예측(간이)")
    r = df.set_index("timestamp")["kW"].rolling("1h").mean()
    peak_val = float(r.max()) if len(r) else np.nan
    peak_ts = r.idxmax() if len(r) else None
    pct_of_contract = (peak_val / contract_power * 100) if contract_power and isinstance(peak_val,float) else np.nan
    col1, col2, col3 = st.columns(3)
    col1.metric("최근 1시간 최대수요(kW)", f"{peak_val:,.1f}" if isinstance(peak_val,float) and not math.isnan(peak_val) else "-")
    col2.metric("발생 시각", peak_ts.strftime("%Y-%m-%d %H:%M") if isinstance(peak_ts, datetime) else "-")
    col3.metric("계약대비(%)", f"{pct_of_contract:,.1f}%" if isinstance(pct_of_contract,float) and not math.isnan(pct_of_contract) else "-")
    if isinstance(pct_of_contract,float) and not math.isnan(pct_of_contract) and pct_of_contract >= peak_alert_threshold:
        st.error(f"계약전력 대비 {pct_of_contract:.1f}% → 피크 경보 (임계 {peak_alert_threshold}%)")
    else:
        st.info(f"계약전력 대비 {pct_of_contract:.1f}%" if isinstance(pct_of_contract,float) else "계약전력 대비 계산 불가")

    st.markdown("**피크 시뮬레이션**")
    sim_hour = st.slider("조치 적용 시간(시)", 0, 23, 14)
    shed_percent = st.slider("차단율(%)", 0, 50, 20)
    sim_df = this_month.copy(); mask = sim_df["hour"]==sim_hour
    base_energy_cost = float((sim_df["kWh"] * sim_df["unit_price"]).sum()) if not sim_df.empty else 0.0
    sim_df.loc[mask, "kWh"] *= (1 - shed_percent/100)
    sim_energy_cost = float((sim_df["kWh"] * sim_df["unit_price"]).sum()) if not sim_df.empty else 0.0
    st.success(f"{sim_hour}시 {shed_percent}% 차단 → 이번달 전력량요금 약 {base_energy_cost - sim_energy_cost:,.0f} 원 절감")
    fig8 = go.Figure()
    fig8.add_trace(go.Bar(x=this_month["hour"], y=this_month["kWh"], name="현재"))
    fig8.add_trace(go.Bar(x=sim_df["hour"], y=sim_df["kWh"], name="시뮬레이션"))
    fig8.update_layout(barmode="group", title="시간대별 kWh 변화")
    st.plotly_chart(fig8, use_container_width=True)

# =========================================
# KEPCO Bill
# =========================================
with bill_tab:
    st.subheader("한전 고지서 구성 기반 요금 계산기")
    m = this_month.copy()
    tou_energy = m.groupby("TOU", dropna=False)["kWh"].sum().reset_index()
    name_to_rate = {}
    for r_ in bill_inputs.tou_rates:
        if r_.name not in name_to_rate:
            name_to_rate[r_.name] = r_.energy_rate
    tou_energy["unit_price"] = tou_energy["TOU"].map(name_to_rate).astype(float)
    tou_energy["energy_charge"] = tou_energy["kWh"] * tou_energy["unit_price"]

    energy_charge = float(tou_energy["energy_charge"].sum())
    basic_charge = bill_inputs.contract_power_kw * bill_inputs.basic_charge_per_kw
    total_kwh_month = float(m["kWh"].sum())
    fuel_adj_amt = total_kwh_month * bill_inputs.fuel_adj_per_kwh
    climate_amt = total_kwh_month * bill_inputs.climate_per_kwh

    subtotal = basic_charge + energy_charge + fuel_adj_amt + climate_amt

    # 간이 초과패널티
    r_full = df.set_index("timestamp")["kW"].rolling("1h").mean()
    peak_val_full = float(r_full.max()) if len(r_full) else np.nan
    overage_charge = 0.0
    if isinstance(peak_val_full,float) and not math.isnan(peak_val_full) and peak_val_full > bill_inputs.contract_power_kw:
        over_ratio = (peak_val_full - bill_inputs.contract_power_kw) / bill_inputs.contract_power_kw
        w_price = float(np.nanmean(m["unit_price"])) if not m.empty else 0.0
        overage_charge = total_kwh_month * w_price * over_ratio * (bill_inputs.over_contract_penalty_rate - 1.0)
        subtotal += overage_charge

    industry_fund = subtotal * bill_inputs.industry_fund_rate
    vat_amt = (subtotal + industry_fund) * bill_inputs.vat_rate
    total_bill = subtotal + industry_fund + vat_amt

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("기본요금", f"{basic_charge:,.0f}원")
    c2.metric("전력량요금", f"{energy_charge:,.0f}원")
    c3.metric("연료비/기후환경(합)", f"{(fuel_adj_amt+climate_amt):,.0f}원")
    c4.metric("합계(세전)", f"{subtotal:,.0f}원")
    c1, c2, c3 = st.columns(3)
    c1.metric("전력산업기반기금", f"{industry_fund:,.0f}원")
    c2.metric("부가가치세", f"{vat_amt:,.0f}원")
    c3.metric("추가패널티(간이)", f"{overage_charge:,.0f}원")
    st.success(f"추정 청구 금액(합계): **{total_bill:,.0f} 원**")

    st.markdown("### 시간대별 사용량/요금")
    st.dataframe(
        tou_energy.rename(columns={"kWh":"kWh(월합)","unit_price":"단가(원/kWh)","energy_charge":"요금(원)"}),
        use_container_width=True
    )

    if isinstance(peak_val_full,float) and not math.isnan(peak_val_full) and peak_val_full > contract_power:
        st.error(f"최대수요 {peak_val_full:,.1f} kW > 계약전력 {contract_power:,.1f} kW. 초과요금/패널티 위험.")
    else:
        st.info("현재 데이터에서는 계약전력 초과가 감지되지 않았습니다.")

    bill_export = {
        "기본요금":[basic_charge],
        "전력량요금":[energy_charge],
        "연료비조정":[fuel_adj_amt],
        "기후환경요금":[climate_amt],
        "초과패널티(간이)":[overage_charge],
        "전력산업기반기금":[industry_fund],
        "부가가치세":[vat_amt],
        "합계(세포함)":[total_bill],
    }
    bill_df = pd.DataFrame(bill_export)
    st.download_button(
        "고지서 요약 CSV 다운로드",
        bill_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="bill_summary.csv",
        mime="text/csv"
    )

# =========================================
# PDF 다운로드 (app.py 동일 포맷)
# =========================================
results_df = df.copy()
results_df = results_df.rename(columns={"timestamp": "측정일시"})
results_df["측정일시"] = pd.to_datetime(results_df["측정일시"], errors="coerce")
results_df["시간"] = results_df["측정일시"].dt.hour
results_df["월"] = results_df["측정일시"].dt.month
results_df["예측요금(원)"] = results_df["unit_price"] * results_df["kWh"]

report_data = {
    "total_bill": total_bill,
    "total_usage": total_kwh_month,
    "period_start": df["timestamp"].min(),
    "period_end": df["timestamp"].max(),
    "report_date": datetime.now(),
    "usage_by_band": tou_energy.set_index("TOU")["kWh"].to_dict(),
    "bill_by_band": tou_energy.set_index("TOU")["energy_charge"].to_dict(),
    "peak_demand_kw": peak_val_full,
    "peak_demand_time": peak_ts,
    "min_demand_kw": float(df["kW"].min()),
    "min_demand_time": df.loc[df["kW"].idxmin()]["timestamp"],
    "avg_day_pf": np.random.uniform(90, 98),
    "penalty_day_hours": np.random.randint(0, 5),
    "bonus_day_hours": np.random.randint(0, 5),
    "avg_night_pf": np.random.uniform(94, 99),
    "penalty_night_hours": np.random.randint(0, 3),
    "yesterday_str": (datetime.now() - timedelta(days=1)).strftime("%m-%d"),
    "today_str": datetime.now().strftime("%m-%d"),
}

try:
    train_df = pd.read_csv("./data/train_.csv")
    train_df["측정일시"] = pd.to_datetime(train_df["측정일시"], errors="coerce")
    train_df["월"] = train_df["측정일시"].dt.month
    train_df["시간"] = train_df["측정일시"].dt.hour
except FileNotFoundError:
    st.warning("train_.csv를 찾을 수 없어 임시 학습 데이터를 생성합니다.")
    train_df = pd.DataFrame(
        {
            "측정일시": pd.date_range(datetime.now() - timedelta(days=30), periods=720, freq="H"),
            "월": [11] * 720,
            "시간": [i % 24 for i in range(720)],
            "전기요금(원)": np.random.randint(1000, 3000, size=720),
        }
    )

comparison_df = create_comparison_table_data(train_df, results_df)
pdf_bytes = generate_bill_pdf(report_data, comparison_df)
if pdf_bytes:
    st.download_button(
        label="📄 예측 요금 명세서 PDF 다운로드",
        data=pdf_bytes,
        file_name=f"predicted_bill_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
    )

# =========================================
# Report (Excel only to keep compact)
# =========================================
with report_tab:
    st.subheader("월간 리포트 & Excel 내보내기")
    monthly_df = df[df["timestamp"].dt.to_period("M")==month_key]
    daily_tbl = monthly_df.groupby(monthly_df["timestamp"].dt.date).agg(
        kWh=("kWh","sum"), kW=("kW","mean")
    ).reset_index().rename(columns={"timestamp":"date"})
    st.dataframe(daily_tbl, use_container_width=True)

    csv_bytes = daily_tbl.to_csv(index=False).encode("utf-8-sig")
    st.download_button("월간 일일 사용량 CSV", csv_bytes, file_name="monthly_daily_usage.csv", mime="text/csv")

    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            daily_tbl.to_excel(writer, index=False, sheet_name="Daily")
            tou_energy.to_excel(writer, index=False, sheet_name="TOU")
            bill_df.to_excel(writer, index=False, sheet_name="Bill")
        st.download_button("엑셀 보고서 다운로드", data=output.getvalue(), file_name="energy_report.xlsx")
    except Exception as e:
        st.warning(f"Excel 내보내기 경고: {e}")

# =========================================
# Footer
# =========================================
st.caption(
    "본 대시보드는 모델 예측 스트리밍/실시간과 EMS/PMS 기능(피크·시뮬레이션·그룹)을 통합하고, "
    "한전 고지서 항목(기본요금/전력량/연료비/기후환경/기금/부가세/계약전력/초과패널티)을 반영한 예시입니다. "
    f"최근 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
