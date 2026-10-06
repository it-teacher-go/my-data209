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
    "선형회귀를 이용해 연도별 예상 기온을 구해 봅니다."
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

# 수업 기준 기간: 2025년까지
df = df[df["연도"] <= 2025].copy()

# 평균기온이 실제로 관측된 날짜를 기준으로 집계
yearly = (
    df.groupby("연도")
    .agg(
        평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 관측일수가 300일 미만인 해 제외
yearly = yearly[
    (yearly["관측일수"] >= 300)
    & (yearly["평균기온"].notna())
].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)


# --------------------------------------------------
# 3. 회귀분석 함수
# --------------------------------------------------

def regression(data):
    """
    독립변수:
        1908년부터 지난 연수

    종속변수:
        연평균기온
    """

    x = data["연도"].to_numpy() - 1908
    y = data["평균기온"].to_numpy()

    # y = ax + b
    slope, intercept = np.polyfit(x, y, 1)

    # 상관계수
    correlation = np.corrcoef(x, y)[0, 1]

    return slope, intercept, correlation


# 전체 기간 회귀
slope_all, intercept_all, corr_all = regression(yearly)


# --------------------------------------------------
# 4. 기간별 기울기 계산
# --------------------------------------------------

def get_period_slope(data, years):
    """
    2025년을 마지막 해로 하여
    최근 N년 동안의 회귀 기울기를 계산
    """

    start_year = 2025 - years + 1

    period_data = data[
        data["연도"] >= start_year
    ].copy()

    slope, intercept, correlation = regression(period_data)

    return {
        "기간": f"최근 {years}년",
        "시작연도": start_year,
        "끝연도": 2025,
        "데이터수": len(period_data),
        "기울기": slope,
        "100년상승": slope * 100,
        "절편": intercept,
        "상관계수": correlation
    }


period_50 = get_period_slope(yearly, 50)
period_30 = get_period_slope(yearly, 30)
period_20 = get_period_slope(yearly, 20)


# --------------------------------------------------
# 5. 학습 데이터 정보
# --------------------------------------------------

st.subheader("📌 회귀 직선에 사용한 데이터")

col1, col2, col3 = st.columns(3)

col1.metric(
    "직선을 만든 해의 개수",
    f"{len(yearly)}개년"
)

col2.metric(
    "시작 연도",
    f"{yearly['연도'].min()}년"
)

col3.metric(
    "끝 연도",
    f"{yearly['연도'].max()}년"
)

st.caption(
    "2025년까지의 자료 중 평균기온 관측일이 300일 이상인 해만 사용했습니다."
)


# --------------------------------------------------
# 6. 상관계수
# --------------------------------------------------

st.divider()

st.subheader("🔗 연도와 연평균기온의 상관관계")

st.metric(
    "상관계수 r",
    f"{corr_all:.3f}"
)

st.write(
    "상관계수가 양수이면 시간이 지날수록 "
    "연평균기온이 높아지는 경향이 있다는 뜻입니다."
)


# --------------------------------------------------
# 7. 전체 기간 회귀 기울기
# --------------------------------------------------

st.divider()

st.subheader("📈 전체 기간의 기온 변화")

st.metric(
    "100년당 예상 기온 상승",
    f"{slope_all * 100:.2f} ℃ / 100년"
)

st.caption(
    f"회귀식의 원래 기울기는 1년에 {slope_all:.4f} ℃이며, "
    f"이를 100년 기준으로 바꾸면 {slope_all * 100:.2f} ℃입니다."
)


# --------------------------------------------------
# 8. 전체 · 최근 50년 · 30년 · 20년 비교
# --------------------------------------------------

st.divider()

st.subheader("🔥 기간에 따라 기온 상승 속도가 다를까?")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "전체 기간",
        f"{slope_all * 100:.2f} ℃",
        help="100년당 기온 변화량"
    )
    st.caption(
        f"{yearly['연도'].min()}~{yearly['연도'].max()}년"
    )

with c2:
    st.metric(
        "최근 50년",
        f"{period_50['100년상승']:.2f} ℃",
        help="최근 50년의 추세를 100년 기준으로 환산"
    )
    st.caption("1976~2025년")

with c3:
    st.metric(
        "최근 30년",
        f"{period_30['100년상승']:.2f} ℃",
        help="최근 30년의 추세를 100년 기준으로 환산"
    )
    st.caption("1996~2025년")

with c4:
    st.metric(
        "최근 20년",
        f"{period_20['100년상승']:.2f} ℃",
        help="최근 20년의 추세를 100년 기준으로 환산"
    )
    st.caption("2006~2025년")

st.info(
    "각 값은 해당 기간의 실제 기온이 100년 동안 이만큼 변했다는 뜻이 아니라, "
    "그 기간에서 나타난 회귀 직선의 기울기를 '100년당 변화량'으로 환산한 값입니다."
)


# --------------------------------------------------
# 9. 산점도 + 회귀 직선
# --------------------------------------------------

st.divider()

st.subheader("📊 연도별 평균기온과 회귀 직선")


# 회귀선용 연도
line_years = np.arange(
    yearly["연도"].min(),
    yearly["연도"].max() + 1
)

# 1908년부터 지난 연수
line_x = line_years - 1908

# 회귀식으로 예상 기온 계산
line_temp = slope_all * line_x + intercept_all


fig = go.Figure()


# 실제 연평균기온
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


# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temp,
        mode="lines",
        name="회귀 직선",
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀선 예상기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    height=600
)

# 가로축에는 실제 연도를 표시
fig.update_xaxes(
    tickformat="d"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# 10. 회귀식 설명
# --------------------------------------------------

st.caption(
    "회귀분석에서는 연도 자체가 아니라 "
    "'1908년부터 지난 연수'를 독립변수 x로 사용했습니다."
)

st.code(
    f"예상 평균기온 = {slope_all:.4f} × (연도 - 1908) "
    f"+ {intercept_all:.4f}"
)


# --------------------------------------------------
# 11. 연도별 기온 예측
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


# 관측 범위를 벗어났는지 안내
if selected_year < yearly["연도"].min():
    st.warning(
        "⚠️ 실제 회귀 직선을 만드는 데 사용한 기간보다 이전 연도입니다. "
        "회귀 직선을 뒤로 연장하여 계산한 값입니다."
    )

elif selected_year > yearly["연도"].max():
    st.warning(
        "⚠️ 실제 관측 자료의 범위를 벗어난 미래 예측입니다. "
        "현재의 직선 추세가 계속된다고 가정한 값이므로 실제 기온과 다를 수 있습니다."
    )


# --------------------------------------------------
# 12. 기간별 상세 비교
# --------------------------------------------------

st.divider()

st.subheader("📋 기간별 회귀분석 비교")

comparison = pd.DataFrame(
    {
        "분석 기간": [
            "전체 기간",
            "최근 50년",
            "최근 30년",
            "최근 20년"
        ],

        "시작 연도": [
            int(yearly["연도"].min()),
            period_50["시작연도"],
            period_30["시작연도"],
            period_20["시작연도"]
        ],

        "끝 연도": [
            int(yearly["연도"].max()),
            period_50["끝연도"],
            period_30["끝연도"],
            period_20["끝연도"]
        ],

        "사용한 연도 수": [
            len(yearly),
            period_50["데이터수"],
            period_30["데이터수"],
            period_20["데이터수"]
        ],

        "상관계수": [
            corr_all,
            period_50["상관계수"],
            period_30["상관계수"],
            period_20["상관계수"]
        ],

        "1년당 기울기(℃)": [
            slope_all,
            period_50["기울기"],
            period_30["기울기"],
            period_20["기울기"]
        ],

        "100년당 변화량(℃)": [
            slope_all * 100,
            period_50["100년상승"],
            period_30["100년상승"],
            period_20["100년상승"]
        ]
    }
)


st.dataframe(
    comparison.style.format(
        {
            "상관계수": "{:.3f}",
            "1년당 기울기(℃)": "{:.4f}",
            "100년당 변화량(℃)": "{:.2f}"
        }
    ),
    use_container_width=True,
    hide_index=True
)


st.caption(
    "최근 기간일수록 기울기가 달라질 수 있습니다. "
    "이는 회귀 직선이 어떤 기간의 데이터를 학습하느냐에 따라 "
    "달라진다는 것을 보여 줍니다."
)
