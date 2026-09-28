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

st.title("🌡️ 서울 연평균 기온 예측기")
st.write(
    "서울의 과거 기온 데이터를 이용해 연도와 연평균기온의 관계를 살펴보고, "
    "선형 회귀 직선으로 원하는 연도의 평균기온을 예측합니다."
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
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 날짜를 날짜 자료형으로 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온을 숫자로 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()


# --------------------------------------------------
# 연도별 데이터 정리
# --------------------------------------------------

# 수업 기준인 2025년까지만 사용
df_base = df[df["연도"] <= 2025].copy()

# 연도별 관측일 수
year_count = (
    df_base
    .groupby("연도")
    .size()
    .reset_index(name="관측일수")
)

# 연도별 평균기온
year_temp = (
    df_base
    .groupby("연도", as_index=False)["평균기온"]
    .mean()
    .rename(columns={"평균기온": "연평균기온"})
)

# 관측일 수와 연평균기온 합치기
annual = pd.merge(
    year_temp,
    year_count,
    on="연도",
    how="inner"
)

# 관측일이 300일 이상인 연도만 사용
annual = annual[annual["관측일수"] >= 300].copy()

# 연평균기온이 없는 경우도 제외
annual = annual.dropna(subset=["연평균기온"])

# 연도순 정렬
annual = annual.sort_values("연도").reset_index(drop=True)


# --------------------------------------------------
# 회귀분석
# --------------------------------------------------

# 1908년을 기준으로 지난 연수
annual["지난연수"] = annual["연도"] - 1908

x = annual["지난연수"].to_numpy()
y = annual["연평균기온"].to_numpy()

# y = ax + b 형태의 회귀식
slope, intercept = np.polyfit(x, y, 1)

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]

# 회귀 직선 예측값
annual["회귀예측기온"] = slope * annual["지난연수"] + intercept


# --------------------------------------------------
# 데이터 정보
# --------------------------------------------------
start_year = int(annual["연도"].min())
end_year = int(annual["연도"].max())
year_num = len(annual)

st.subheader("📊 회귀분석에 사용한 데이터")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("사용한 연도 수", f"{year_num}개")

with col2:
    st.metric("시작 연도", f"{start_year}년")

with col3:
    st.metric("끝 연도", f"{end_year}년")

st.caption(
    "※ 2025년 이후의 데이터와 관측일이 300일 미만인 연도는 "
    "회귀분석에서 제외했습니다."
)


# --------------------------------------------------
# 상관계수
# --------------------------------------------------
st.subheader("🔗 연도와 연평균기온의 상관관계")

st.metric(
    "상관계수",
    f"{correlation:.4f}"
)

if correlation > 0:
    st.write(
        "상관계수가 양수이므로, 연도가 증가할수록 "
        "연평균기온도 높아지는 경향이 나타납니다."
    )
elif correlation < 0:
    st.write(
        "상관계수가 음수이므로, 연도가 증가할수록 "
        "연평균기온이 낮아지는 경향이 나타납니다."
    )
else:
    st.write("두 변수 사이에 선형적인 상관관계가 거의 나타나지 않습니다.")


# --------------------------------------------------
# 산점도 + 회귀 직선
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
            "회귀 예측기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    height=550
)

# 가로축에 실제 연도가 표시되도록 설정
fig.update_xaxes(
    tickformat="d"
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 회귀식 표시
# --------------------------------------------------
st.subheader("🧮 만들어진 회귀식")

st.latex(
    rf"\hat{{y}} = {slope:.5f}x + {intercept:.3f}"
)

st.write(
    "여기서 **x는 1908년부터 지난 연수**입니다."
)

st.write(
    "예를 들어 2025년이라면 "
    f"`x = 2025 - 1908 = {2025 - 1908}`을 회귀식에 넣습니다."
)


# --------------------------------------------------
# 기온 예측기
# --------------------------------------------------
st.divider()

st.subheader("🔮 기온 예측기")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2030,
    step=1
)

# 선택한 연도를 1908년부터 지난 연수로 변환
selected_x = selected_year - 1908

# 회귀식으로 예측
predicted_temp = slope * selected_x + intercept


# 크게 표시
st.metric(
    label=f"{selected_year}년 예상 연평균기온",
    value=f"{predicted_temp:.2f} ℃"
)

if selected_year < start_year:
    st.info(
        f"{selected_year}년은 회귀 직선을 만드는 데 사용한 "
        f"첫 연도({start_year}년)보다 이전입니다. "
        "따라서 회귀선을 과거로 연장한 예측값입니다."
    )

elif selected_year > end_year:
    st.info(
        f"{selected_year}년은 회귀 직선을 만드는 데 사용한 "
        f"마지막 연도({end_year}년)보다 이후입니다. "
        "따라서 과거의 선형적인 경향이 계속된다고 가정한 예측값입니다."
    )


# --------------------------------------------------
# 선택한 연도를 그래프로 확인
# --------------------------------------------------
prediction_fig = go.Figure()

prediction_fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["회귀예측기온"],
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
        text=[f"{selected_year}년<br>{predicted_temp:.2f}℃"],
        textposition="top center",
        marker=dict(size=14)
    )
)

prediction_fig.update_layout(
    xaxis_title="연도",
    yaxis_title="예상 연평균기온 (℃)",
    height=400,
    showlegend=True
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
# 참고
# --------------------------------------------------
st.caption(
    "※ 이 예측값은 과거 연도와 연평균기온 사이의 선형 관계를 이용한 "
    "단순 회귀 예측입니다. 실제 미래 기온을 정확하게 예측하는 "
    "기후 예측 모델은 아닙니다."
)
