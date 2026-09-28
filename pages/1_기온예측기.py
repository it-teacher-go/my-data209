import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")

st.write(
    "서울의 과거 기온 데이터를 이용하여 연도와 연평균기온의 관계를 살펴보고, "
    "회귀 직선 위에서 원하는 연도의 평균기온을 예측합니다."
)


# --------------------------------------------------
# 데이터 주소
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()


# --------------------------------------------------
# 2025년까지만 사용
# --------------------------------------------------
df = df[
    df["연도"].notna()
    & (df["연도"] <= 2025)
].copy()


# --------------------------------------------------
# 연도별 연평균기온과 실제 관측일수 계산
#
# 평균기온 값이 있는 날짜만 관측일로 계산
# --------------------------------------------------
annual = (
    df.groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)


# --------------------------------------------------
# 관측일이 300일 이상인 연도만 사용
# --------------------------------------------------
annual = annual[
    annual["관측일수"] >= 300
].copy()

annual = annual.dropna(
    subset=["연평균기온"]
)

annual["연도"] = annual["연도"].astype(int)

annual = annual.sort_values(
    "연도"
).reset_index(drop=True)


# --------------------------------------------------
# 전체 기간 회귀분석
#
# x = 1908년부터 지난 연수
# --------------------------------------------------
annual["지난연수"] = (
    annual["연도"] - 1908
)

x = annual["지난연수"].to_numpy()
y = annual["연평균기온"].to_numpy()


# 회귀계수
slope, intercept = np.polyfit(
    x,
    y,
    1
)


# 상관계수
correlation = np.corrcoef(
    x,
    y
)[0, 1]


# 100년당 기온 변화
slope_100 = slope * 100


# --------------------------------------------------
# 최근 20년 회귀분석
# 2006 ~ 2025
# --------------------------------------------------
recent_start_year = 2006

recent = annual[
    annual["연도"] >= recent_start_year
].copy()

recent_x = recent["지난연수"].to_numpy()
recent_y = recent["연평균기온"].to_numpy()


recent_slope, recent_intercept = np.polyfit(
    recent_x,
    recent_y,
    1
)

recent_slope_100 = (
    recent_slope * 100
)


# --------------------------------------------------
# 자료 정보
# --------------------------------------------------
year_count = len(annual)

start_year = int(
    annual["연도"].min()
)

end_year = int(
    annual["연도"].max()
)


st.subheader("📌 회귀 직선에 사용한 자료")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "사용한 연도 수",
        f"{year_count}개"
    )

with col2:
    st.metric(
        "시작 연도",
        f"{start_year}년"
    )

with col3:
    st.metric(
        "끝 연도",
        f"{end_year}년"
    )


st.caption(
    "2025년 이후의 자료와 평균기온 관측일이 "
    "300일 미만인 해는 제외했습니다."
)


# --------------------------------------------------
# 기온 변화 속도
# --------------------------------------------------
st.subheader("🌡️ 기온 변화 속도 비교")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "전체 기간",
        f"{slope_100:+.2f} ℃ / 100년"
    )

    st.caption(
        f"{start_year}년 ~ {end_year}년"
    )


with col2:

    st.metric(
        "최근 20년",
        f"{recent_slope_100:+.2f} ℃ / 100년"
    )

    st.caption(
        f"{recent_start_year}년 ~ {end_year}년"
    )


st.caption(
    "최근 20년의 값은 최근 20년 동안 나타난 선형 변화 속도를 "
    "100년 단위로 환산한 값입니다."
)


# --------------------------------------------------
# 연도 선택
# --------------------------------------------------
st.divider()

st.subheader("🔮 회귀 직선으로 기온 예측")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2030,
    step=1
)


# --------------------------------------------------
# 선택 연도의 기온 예측
# --------------------------------------------------
selected_x = (
    selected_year - 1908
)

predicted_temp = (
    slope * selected_x
    + intercept
)


# 크게 표시
st.metric(
    label=f"{selected_year}년 예상 연평균기온",
    value=f"{predicted_temp:.2f} ℃"
)


# 실제 관측 범위 밖일 경우 안내
if selected_year < start_year:

    st.info(
        f"{selected_year}년은 회귀분석에 사용한 첫 연도인 "
        f"{start_year}년보다 이전입니다. "
        "회귀 직선을 과거 방향으로 연장하여 계산한 값입니다."
    )

elif selected_year > end_year:

    st.info(
        f"{selected_year}년은 실제 자료의 마지막 연도인 "
        f"{end_year}년보다 이후입니다. "
        "회귀 직선을 미래 방향으로 연장하여 계산한 예측값입니다."
    )


# --------------------------------------------------
# 그래프에 그릴 회귀선 범위 결정
#
# 선택한 연도가 2025년 이후이면
# 그 연도까지 회귀선을 연장
# --------------------------------------------------
graph_start_year = min(
    start_year,
    selected_year
)

graph_end_year = max(
    end_year,
    selected_year
)


regression_years = np.arange(
    graph_start_year,
    graph_end_year + 1
)


regression_x = (
    regression_years - 1908
)


regression_temp = (
    slope * regression_x
    + intercept
)


# --------------------------------------------------
# 산점도 + 회귀 직선 + 선택한 연도
# --------------------------------------------------
st.subheader("📈 연평균기온과 회귀 직선")

fig = go.Figure()


# 실제 연평균기온
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온",

        customdata=annual["관측일수"],

        hovertemplate=(
            "연도: %{x}년<br>"
            "실제 연평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        )
    )
)


# --------------------------------------------------
# 전체 기간으로 만든 회귀 직선
# --------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=regression_years,
        y=regression_temp,
        mode="lines",
        name="회귀 직선",

        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀선 예측기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# --------------------------------------------------
# 선택한 연도 표시
# --------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],

        mode="markers+text",

        name="선택한 연도",

        text=[
            f"{selected_year}년<br>"
            f"{predicted_temp:.2f}℃"
        ],

        textposition="top center",

        marker=dict(
            size=15
        ),

        hovertemplate=(
            f"{selected_year}년<br>"
            f"예상 연평균기온: {predicted_temp:.2f}℃"
            "<extra></extra>"
        )
    )
)


# --------------------------------------------------
# 선택 연도의 위치를 세로 점선으로 표시
# --------------------------------------------------
fig.add_vline(
    x=selected_year,
    line_dash="dot",
    opacity=0.5
)


# --------------------------------------------------
# 그래프 설정
# --------------------------------------------------
fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=600,
    hovermode="closest"
)


# 가로축에는 실제 연도 표시
fig.update_xaxes(
    tickformat="d"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# 상관계수
# --------------------------------------------------
st.subheader("🔗 상관계수")

st.metric(
    "연도와 연평균기온의 상관계수",
    f"{correlation:.4f}"
)


if correlation > 0:

    st.write(
        "상관계수가 양수이므로 시간이 지날수록 "
        "연평균기온이 높아지는 경향이 있습니다."
    )

elif correlation < 0:

    st.write(
        "상관계수가 음수이므로 시간이 지날수록 "
        "연평균기온이 낮아지는 경향이 있습니다."
    )

else:

    st.write(
        "두 변수 사이의 선형적인 관계가 거의 없습니다."
    )


# --------------------------------------------------
# 회귀식
# --------------------------------------------------
st.subheader("🧮 회귀식")

st.latex(
    rf"\hat{{y}} = "
    rf"{slope:.5f}x "
    rf"{'+' if intercept >= 0 else '-'} "
    rf"{abs(intercept):.3f}"
)


st.write(
    "**x는 1908년부터 지난 연수입니다.**"
)

st.write(
    f"선택한 {selected_year}년의 경우 "
    f"`x = {selected_year} - 1908 = {selected_x}` 입니다."
)

st.write(
    f"따라서 회귀 직선에 `x = {selected_x}`를 넣으면 "
    f"예상 연평균기온은 **{predicted_temp:.2f}℃**입니다."
)


# --------------------------------------------------
# 안내
# --------------------------------------------------
st.caption(
    "※ 이 예측은 과거 연도와 연평균기온 사이에서 나타난 "
    "선형적인 관계를 이용한 단순 회귀 예측입니다. "
    "실제 미래 기후를 정확하게 예측하는 기후 모델은 아닙니다."
)
