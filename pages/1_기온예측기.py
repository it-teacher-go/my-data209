import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


st.set_page_config(
    page_title="기온 예측기",
    layout="wide"
)

st.title("🌡️ 서울 기온 예측기")

st.write(
    "서울의 연평균기온과 연도의 관계를 살펴보고, "
    "기간에 따라 회귀 직선의 기울기와 상관관계가 어떻게 달라지는지 비교해 봅니다."
)


# --------------------------------------------------
# 1. 데이터 불러오기
# --------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()


# --------------------------------------------------
# 2. 연도별 평균기온 계산
# --------------------------------------------------

# 수업 기준: 2025년까지
df = df[df["연도"] <= 2025].copy()


yearly = (
    df.groupby("연도")
    .agg(
        평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)


# 관측일 300일 이상인 해만 사용
yearly = yearly[
    (yearly["관측일수"] >= 300)
    & (yearly["평균기온"].notna())
].copy()


yearly = yearly.sort_values("연도").reset_index(drop=True)


# --------------------------------------------------
# 3. 회귀분석 함수
# --------------------------------------------------

def regression(data):

    # 1908년부터 지난 연수
    x = data["연도"].to_numpy() - 1908
    y = data["평균기온"].to_numpy()

    # 회귀식 y = ax + b
    slope, intercept = np.polyfit(x, y, 1)

    # 상관계수
    correlation = np.corrcoef(x, y)[0, 1]

    return slope, intercept, correlation


# --------------------------------------------------
# 4. 기간별 데이터와 회귀분석
# --------------------------------------------------

def period_regression(data, start_year, name):

    period_data = data[
        data["연도"] >= start_year
    ].copy()

    slope, intercept, correlation = regression(period_data)

    return {
        "이름": name,
        "데이터": period_data,
        "시작연도": int(period_data["연도"].min()),
        "끝연도": int(period_data["연도"].max()),
        "데이터수": len(period_data),
        "기울기": slope,
        "100년상승": slope * 100,
        "절편": intercept,
        "상관계수": correlation
    }


# 전체 기간
slope_all, intercept_all, corr_all = regression(yearly)

period_all = {
    "이름": "전체 기간",
    "데이터": yearly,
    "시작연도": int(yearly["연도"].min()),
    "끝연도": int(yearly["연도"].max()),
    "데이터수": len(yearly),
    "기울기": slope_all,
    "100년상승": slope_all * 100,
    "절편": intercept_all,
    "상관계수": corr_all
}


# 최근 50년, 30년, 20년
period_50 = period_regression(
    yearly,
    1976,
    "최근 50년"
)

period_30 = period_regression(
    yearly,
    1996,
    "최근 30년"
)

period_20 = period_regression(
    yearly,
    2006,
    "최근 20년"
)


periods = [
    period_all,
    period_50,
    period_30,
    period_20
]


# --------------------------------------------------
# 5. 학습 데이터 정보
# --------------------------------------------------

st.subheader("📌 회귀 직선에 사용한 데이터")

c1, c2, c3 = st.columns(3)

c1.metric(
    "직선을 만든 해의 개수",
    f"{len(yearly)}개년"
)

c2.metric(
    "시작 연도",
    f"{yearly['연도'].min()}년"
)

c3.metric(
    "끝 연도",
    f"{yearly['연도'].max()}년"
)

st.caption(
    "2025년까지의 자료 중 평균기온 관측일이 300일 이상인 해만 사용합니다."
)


# --------------------------------------------------
# 6. 기간별 기울기 비교
# --------------------------------------------------

st.divider()

st.subheader("📈 기간별 기온 상승 추세")

c1, c2, c3, c4 = st.columns(4)


with c1:
    st.metric(
        "전체 기간",
        f"{period_all['100년상승']:.2f} ℃ / 100년"
    )

    st.write(
        f"**상관계수 r = {period_all['상관계수']:.3f}**"
    )

    st.caption(
        f"{period_all['시작연도']}~"
        f"{period_all['끝연도']}년"
    )


with c2:
    st.metric(
        "최근 50년",
        f"{period_50['100년상승']:.2f} ℃ / 100년"
    )

    st.write(
        f"**상관계수 r = {period_50['상관계수']:.3f}**"
    )

    st.caption(
        f"{period_50['시작연도']}~"
        f"{period_50['끝연도']}년"
    )


with c3:
    st.metric(
        "최근 30년",
        f"{period_30['100년상승']:.2f} ℃ / 100년"
    )

    st.write(
        f"**상관계수 r = {period_30['상관계수']:.3f}**"
    )

    st.caption(
        f"{period_30['시작연도']}~"
        f"{period_30['끝연도']}년"
    )


with c4:
    st.metric(
        "최근 20년",
        f"{period_20['100년상승']:.2f} ℃ / 100년"
    )

    st.write(
        f"**상관계수 r = {period_20['상관계수']:.3f}**"
    )

    st.caption(
        f"{period_20['시작연도']}~"
        f"{period_20['끝연도']}년"
    )


st.info(
    "상관계수는 연도와 연평균기온이 얼마나 강하게 함께 변하는지를 나타냅니다. "
    "100년당 변화량은 각 기간에서 구한 회귀 직선의 기울기를 "
    "100년 기준으로 환산한 값입니다."
)


# --------------------------------------------------
# 7. 기간별 회귀선 시각화
# --------------------------------------------------

st.divider()

st.subheader("📊 기간에 따라 회귀 직선은 어떻게 달라질까?")


fig = go.Figure()


# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        customdata=yearly["관측일수"],
        hovertemplate=(
            "연도: %{x}년<br>"
            "평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        )
    )
)


# --------------------------------------------------
# 회귀선 추가 함수
# --------------------------------------------------

def add_regression_line(fig, period, dash_style):

    start_year = period["시작연도"]
    end_year = period["끝연도"]

    years = np.arange(
        start_year,
        end_year + 1
    )

    # 1908년부터 지난 연수
    x = years - 1908

    predicted = (
        period["기울기"] * x
        + period["절편"]
    )

    fig.add_trace(
        go.Scatter(
            x=years,
            y=predicted,
            mode="lines",

            name=(
                f"{period['이름']} "
                f"(r={period['상관계수']:.3f})"
            ),

            line=dict(
                width=3,
                dash=dash_style
            ),

            hovertemplate=(
                f"{period['이름']}<br>"
                "연도: %{x}년<br>"
                "예상기온: %{y:.2f}℃<br>"
                f"r = {period['상관계수']:.3f}<br>"
                f"100년당 변화 = "
                f"{period['100년상승']:.2f}℃"
                "<extra></extra>"
            )
        )
    )


# 네 개 회귀선
add_regression_line(
    fig,
    period_all,
    "solid"
)

add_regression_line(
    fig,
    period_50,
    "dash"
)

add_regression_line(
    fig,
    period_30,
    "dot"
)

add_regression_line(
    fig,
    period_20,
    "dashdot"
)


fig.update_layout(

    xaxis_title="연도",

    yaxis_title="연평균기온 (℃)",

    hovermode="closest",

    height=650,

    legend=dict(
        title="회귀 직선"
    )
)


fig.update_xaxes(
    tickformat="d"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


st.caption(
    "각 회귀선은 해당 기간의 데이터만 사용하여 만든 직선입니다. "
    "따라서 최근 20년 회귀선은 2006~2025년 구간에만 표시됩니다."
)


# --------------------------------------------------
# 8. 회귀식 비교
# --------------------------------------------------

st.divider()

st.subheader("🧮 기간별 회귀식")


for p in periods:

    st.write(
        f"**{p['이름']}** : "
        f"예상기온 = "
        f"{p['기울기']:.4f} × (연도 − 1908) "
        f"+ {p['절편']:.4f}"
    )


# --------------------------------------------------
# 9. 연도 선택 → 예상 기온
# --------------------------------------------------

st.divider()

st.subheader("🔮 연도를 선택해 기온을 예측해 보기")


selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


selected_x = selected_year - 1908


predicted_temp = (
    slope_all * selected_x
    + intercept_all
)


st.metric(
    f"{selected_year}년 예상 연평균기온",
    f"{predicted_temp:.2f} ℃"
)


st.caption(
    "예측에는 전체 기간의 회귀 직선을 사용합니다."
)


if selected_year < yearly["연도"].min():

    st.warning(
        "실제 회귀분석 자료보다 이전 연도입니다. "
        "회귀 직선을 과거로 연장하여 계산한 값입니다."
    )


elif selected_year > yearly["연도"].max():

    st.warning(
        "실제 관측 자료의 범위를 벗어난 미래 예측입니다. "
        "현재의 직선 추세가 계속된다고 가정한 값입니다."
    )


# --------------------------------------------------
# 10. 상세 비교표
# --------------------------------------------------

st.divider()

st.subheader("📋 기간별 회귀분석 결과")


comparison = pd.DataFrame({

    "분석 기간": [
        p["이름"]
        for p in periods
    ],

    "시작 연도": [
        p["시작연도"]
        for p in periods
    ],

    "끝 연도": [
        p["끝연도"]
        for p in periods
    ],

    "사용한 연도 수": [
        p["데이터수"]
        for p in periods
    ],

    "상관계수 r": [
        p["상관계수"]
        for p in periods
    ],

    "1년당 기울기(℃)": [
        p["기울기"]
        for p in periods
    ],

    "100년당 변화량(℃)": [
        p["100년상승"]
        for p in periods
    ]
})


st.dataframe(

    comparison.style.format({

        "상관계수 r": "{:.3f}",

        "1년당 기울기(℃)": "{:.4f}",

        "100년당 변화량(℃)": "{:.2f}"

    }),

    use_container_width=True,

    hide_index=True
)


# --------------------------------------------------
# 11. 수업용 해석
# --------------------------------------------------

st.divider()

st.subheader("💡 생각해 보기")

st.write(
    """
같은 서울 기온 자료라도 **어느 기간을 선택하여 회귀분석하느냐에 따라
회귀 직선의 기울기와 상관계수가 달라질 수 있습니다.**

따라서 회귀모델을 해석할 때는 단순히 기울기만 보는 것이 아니라
**어떤 기간의 데이터를 사용했는지**도 함께 살펴보아야 합니다.
"""
)
