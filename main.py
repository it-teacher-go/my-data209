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
    "선형 회귀를 이용해 원하는 연도의 평균기온을 예측합니다."
)


# --------------------------------------------------
# 데이터
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온을 숫자로 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()


# --------------------------------------------------
# 2025년까지만 사용
# --------------------------------------------------
df = df[
    (df["연도"].notna()) &
    (df["연도"] <= 2025)
].copy()


# --------------------------------------------------
# 연도별 연평균기온 및 관측일수 계산
#
# 평균기온 값이 실제로 존재하는 날짜만
# 관측일수로 계산
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
# 관측일 300일 이상인 해만 사용
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
# 회귀분석
#
# 독립변수:
# 1908년부터 지난 연수
#
# 1908년 → 0
# 1909년 → 1
# ...
# --------------------------------------------------
annual["지난연수"] = annual["연도"] - 1908

x = annual["지난연수"].to_numpy()
y = annual["연평균기온"].to_numpy()


# 회귀계수
slope, intercept = np.polyfit(x, y, 1)


# 상관계수
correlation = np.corrcoef(x, y)[0, 1]


# 실제 자료에 대한 회귀 예측값
annual["회귀예측기온"] = (
    slope * annual["지난연수"] + intercept
)


# --------------------------------------------------
# 회귀 직선에 사용된 자료 정보
# --------------------------------------------------
year_count = len(annual)
start_year = int(annual["연도"].min())
end_year = int(annual["연도"].max())


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
# 산점도 + 회귀 직선
# --------------------------------------------------
st.subheader("📈 연도별 평균기온과 회귀 직선")

fig = go.Figure()


# 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="연평균기온",
        customdata=annual["관측일수"],
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        )
    )
)


# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["회귀예측기온"],
        mode="lines",
        name="회귀 직선",
        hovertemplate=(
            "연도: %{x}년<br>"
            "예측기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=550,
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
    "예를 들어 2025년은 "
    f"`2025 - 1908 = {2025 - 1908}`이므로 "
    f"`x = {2025 - 1908}`을 사용합니다."
)


# --------------------------------------------------
# 기온 예측
# --------------------------------------------------
st.divider()

st.subheader("🔮 연도별 기온 예측")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2030,
    step=1
)


# 1908년부터 지난 연수
selected_x = selected_year - 1908


# 예상 기온
predicted_temp = (
    slope * selected_x + intercept
)


# 크게 표시
st.metric(
    label=f"{selected_year}년 예상 연평균기온",
    value=f"{predicted_temp:.2f} ℃"
)


# --------------------------------------------------
# 자료 범위를 벗어난 경우 안내
# --------------------------------------------------
if selected_year < start_year:
    st.info(
        f"{selected_year}년은 회귀 직선을 만드는 데 사용한 "
        f"첫 연도인 {start_year}년보다 이전입니다. "
        "회귀 직선을 과거 방향으로 연장하여 계산한 값입니다."
    )

elif selected_year > end_year:
    st.info(
        f"{selected_year}년은 회귀 직선을 만드는 데 사용한 "
        f"마지막 연도인 {end_year}년보다 이후입니다. "
        "과거의 선형적인 변화가 계속된다고 가정한 예측값입니다."
    )


# --------------------------------------------------
# 선택한 연도를 회귀선 위에 표시
# --------------------------------------------------
prediction_years = np.arange(1900, 2101)

prediction_x = prediction_years - 1908

prediction_temp = (
    slope * prediction_x + intercept
)


prediction_fig = go.Figure()


prediction_fig.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_temp,
        mode="lines",
        name="회귀 직선"
    )
)


prediction_fig.add_trace(
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
        marker=dict(size=14)
    )
)


prediction_fig.update_layout(
    xaxis_title="연도",
    yaxis_title="예상 연평균기온 (℃)",
    height=400
)

prediction_fig.update_xaxes(
    range=[1900, 2100],
    tickformat="d"
)


st.plotly_chart(
    prediction_fig,
    use_container_width=True
)


# --------------------------------------------------
# 안내
# --------------------------------------------------
st.caption(
    "※ 이 예측은 과거 연도와 연평균기온 사이의 "
    "선형적인 관계를 이용한 단순 회귀 예측입니다. "
    "실제 미래 기후를 정확하게 예측하는 기후 모델은 아닙니다."
)
