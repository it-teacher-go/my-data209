
import streamlit as st
import pandas as pd

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="서울 100년 기온 변화",
    page_icon="🌡️",
    layout="wide"
)

# -----------------------------
# 제목
# -----------------------------
st.title("🌡️ 서울의 100년 연평균 기온 변화")
st.write("서울의 일별 기온 데이터를 연도별 평균으로 계산하여 기온 변화를 살펴봅니다.")

# -----------------------------
# 데이터 불러오기
# -----------------------------
url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(url)

    # 열 이름의 불필요한 공백 제거
    df.columns = df.columns.str.strip()

    # 날짜 형식 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 연도 열 만들기
    df["연도"] = df["날짜"].dt.year

    return df

df = load_data()

# -----------------------------
# 데이터 미리보기
# -----------------------------
with st.expander("📋 원본 데이터 확인"):
    st.dataframe(df.head(20), use_container_width=True)

# -----------------------------
# 연평균 기온 계산
# -----------------------------
annual_temp = (
    df.groupby("연도")["평균기온"]
    .mean()
    .reset_index()
)

# 최근 100년만 선택
latest_year = int(annual_temp["연도"].max())
start_year = latest_year - 99

annual_100 = annual_temp[
    annual_temp["연도"] >= start_year
].copy()

# -----------------------------
# 기본 정보
# -----------------------------
st.subheader("📊 데이터 분석 결과")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "분석 시작 연도",
        f"{int(annual_100['연도'].min())}년"
    )

with col2:
    st.metric(
        "최근 연도",
        f"{int(annual_100['연도'].max())}년"
    )

with col3:
    st.metric(
        "최근 연평균 기온",
        f"{annual_100.iloc[-1]['평균기온']:.1f} ℃"
    )

# -----------------------------
# 그래프
# -----------------------------
st.subheader("📈 서울의 연평균 기온 변화")

chart_data = annual_100.set_index("연도")

st.line_chart(
    chart_data["평균기온"],
    x_label="연도",
    y_label="연평균 기온(℃)",
    height=500
)

# -----------------------------
# 연도별 데이터
# -----------------------------
with st.expander("🔎 연도별 연평균 기온 보기"):
    display_df = annual_100.copy()
    display_df["평균기온"] = display_df["평균기온"].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

# -----------------------------
# 설명
# -----------------------------
st.info(
    "그래프는 하루 단위의 '평균기온'을 연도별로 평균 내어 "
    "서울의 연평균 기온 변화를 나타낸 것입니다."
)

st.caption("데이터 출처: 서울 기온 데이터(seoul.csv)")
