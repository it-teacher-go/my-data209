import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="기온 곡선 예측기",
    layout="wide"
)

st.title("🌡️ 서울 기온 곡선 예측기")

st.write(
    "과거 연평균기온으로 1차, 3차, 9차 회귀모델을 만들고, "
    "학습에 사용하지 않은 최근 기온으로 예측 성능을 비교합니다."
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

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()


# --------------------------------------------------
# 2. 연평균기온 계산
# --------------------------------------------------

# 수업 기준: 2025년까지
df = df[
    df["연도"] <= 2025
].copy()


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


yearly = yearly.sort_values(
    "연도"
).reset_index(drop=True)


# --------------------------------------------------
# 3. 훈련 / 테스트 데이터 분리
# --------------------------------------------------

# 2005년 이전 = 훈련
train = yearly[
    yearly["연도"] < 2005
].copy()


# 2005년부터 = 테스트
test = yearly[
    yearly["연도"] >= 2005
].copy()


# --------------------------------------------------
# 4. 데이터 개수 표시
# --------------------------------------------------

st.subheader("① 훈련 데이터와 테스트 데이터")

c1, c2 = st.columns(2)


with c1:

    st.metric(
        "훈련 데이터",
        f"{len(train)}개 연도"
    )

    st.caption(
        f"{train['연도'].min()}~{train['연도'].max()}년"
    )


with c2:

    st.metric(
        "테스트 데이터",
        f"{len(test)}개 연도"
    )

    st.caption(
        f"{test['연도'].min()}~{test['연도'].max()}년"
    )


st.info(
    "2005년 이후의 기온은 회귀곡선을 만드는 데 전혀 사용하지 않습니다. "
    "세 모델 모두 2004년까지의 자료만 학습한 뒤, "
    "2005년 이후 자료로 예측 성능을 평가합니다."
)


# --------------------------------------------------
# 5. 연도 변환 함수
# --------------------------------------------------

def transform_year(year):

    # 큰 연도 숫자를 그대로 고차식에 넣지 않도록 변환
    return (np.asarray(year) - 1908) / 100


# --------------------------------------------------
# 6. 다항회귀 모델 함수
# --------------------------------------------------

def fit_polynomial(data, degree):

    x = transform_year(
        data["연도"]
    )

    y = data["평균기온"].to_numpy()

    coefficients = np.polyfit(
        x,
        y,
        degree
    )

    return coefficients


def predict_polynomial(years, coefficients):

    x = transform_year(years)

    return np.polyval(
        coefficients,
        x
    )


def mae(y_true, y_pred):

    return np.mean(
        np.abs(
            np.asarray(y_true)
            - np.asarray(y_pred)
        )
    )


# --------------------------------------------------
# 7. 1차, 3차, 9차 모델 학습
# --------------------------------------------------

degrees = [1, 3, 9]

models = {}


for degree in degrees:

    # 반드시 훈련 데이터로만 학습
    coefficients = fit_polynomial(
        train,
        degree
    )

    # 테스트 데이터 예측
    test_pred = predict_polynomial(
        test["연도"],
        coefficients
    )

    # 테스트 데이터 MAE
    test_mae = mae(
        test["평균기온"],
        test_pred
    )

    # 2050년 예측
    pred_2050 = predict_polynomial(
        [2050],
        coefficients
    )[0]

    models[degree] = {
        "계수": coefficients,
        "테스트예측": test_pred,
        "MAE": test_mae,
        "2050예측": pred_2050
    }


# --------------------------------------------------
# 8. 테스트 데이터 평가 결과
# --------------------------------------------------

st.divider()

st.subheader("② 어떤 곡선이 최근 기온을 더 잘 예측할까?")


results = pd.DataFrame({

    "모델": [
        "1차 회귀 (직선)",
        "3차 회귀",
        "9차 회귀"
    ],

    "학습 데이터": [
        f"{train['연도'].min()}~{train['연도'].max()}",
        f"{train['연도'].min()}~{train['연도'].max()}",
        f"{train['연도'].min()}~{train['연도'].max()}"
    ],

    "평가 데이터": [
        f"{test['연도'].min()}~{test['연도'].max()}",
        f"{test['연도'].min()}~{test['연도'].max()}",
        f"{test['연도'].min()}~{test['연도'].max()}"
    ],

    "테스트 MAE (℃)": [
        models[1]["MAE"],
        models[3]["MAE"],
        models[9]["MAE"]
    ],

    "2050년 예상기온 (℃)": [
        models[1]["2050예측"],
        models[3]["2050예측"],
        models[9]["2050예측"]
    ]
})


st.dataframe(
    results.style.format({
        "테스트 MAE (℃)": "{:.3f}",
        "2050년 예상기온 (℃)": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


st.caption(
    "테스트 MAE는 2005년 이후 실제 연평균기온과 모델의 예측값이 "
    "평균적으로 몇 ℃ 차이 나는지를 나타냅니다. "
    "작을수록 테스트 데이터를 더 잘 예측한 모델입니다."
)


# --------------------------------------------------
# 9. 모델별 MAE 크게 비교
# --------------------------------------------------

c1, c2, c3 = st.columns(3)


with c1:

    st.metric(
        "1차 모델 MAE",
        f"{models[1]['MAE']:.3f} ℃"
    )


with c2:

    st.metric(
        "3차 모델 MAE",
        f"{models[3]['MAE']:.3f} ℃"
    )


with c3:

    st.metric(
        "9차 모델 MAE",
        f"{models[9]['MAE']:.3f} ℃"
    )


# --------------------------------------------------
# 10. 훈련 데이터와 테스트 데이터 시각화
# --------------------------------------------------

st.divider()

st.subheader("③ 훈련 데이터와 테스트 데이터")


fig_data = go.Figure()


# 훈련 데이터
fig_data.add_trace(
    go.Scatter(
        x=train["연도"],
        y=train["평균기온"],
        mode="markers",
        name="훈련 데이터",
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 테스트 데이터
fig_data.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["평균기온"],
        mode="markers",
        marker=dict(
            size=10,
            symbol="diamond"
        ),
        name="테스트 데이터",
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 테스트 시작 위치
fig_data.add_vline(
    x=2005,
    line_dash="dash",
    annotation_text="테스트 시작",
    annotation_position="top"
)


fig_data.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=520
)

fig_data.update_xaxes(
    tickformat="d"
)


st.plotly_chart(
    fig_data,
    use_container_width=True
)


# --------------------------------------------------
# 11. 세 회귀곡선 비교
# --------------------------------------------------

st.divider()

st.subheader("④ 1차 · 3차 · 9차 회귀곡선 비교")


fig = go.Figure()


# 훈련 데이터
fig.add_trace(
    go.Scatter(
        x=train["연도"],
        y=train["평균기온"],
        mode="markers",
        name="훈련 데이터",
        opacity=0.6
    )
)


# 테스트 데이터
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["평균기온"],
        mode="markers",
        marker=dict(
            size=9,
            symbol="diamond"
        ),
        name="테스트 데이터"
    )
)


# 회귀곡선을 2050년까지 연장
curve_years = np.linspace(
    train["연도"].min(),
    2050,
    500
)


dash_styles = {
    1: "solid",
    3: "dash",
    9: "dot"
}


for degree in degrees:

    curve_temp = predict_polynomial(
        curve_years,
        models[degree]["계수"]
    )

    fig.add_trace(
        go.Scatter(
            x=curve_years,
            y=curve_temp,
            mode="lines",

            line=dict(
                width=3,
                dash=dash_styles[degree]
            ),

            name=(
                f"{degree}차 회귀 "
                f"(MAE {models[degree]['MAE']:.2f}℃)"
            ),

            hovertemplate=(
                f"{degree}차 회귀<br>"
                "연도: %{x:.0f}년<br>"
                "예상기온: %{y:.2f}℃"
                "<extra></extra>"
            )
        )
    )


# 테스트 시작선
fig.add_vline(
    x=2005,
    line_dash="dash",
    annotation_text="테스트 시작",
    annotation_position="top"
)


# 2025 이후는 미래
fig.add_vline(
    x=2025,
    line_dash="dot",
    annotation_text="관측 자료 끝",
    annotation_position="top"
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=650,
    hovermode="closest"
)


fig.update_xaxes(
    tickformat="d"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


st.info(
    "곡선은 모두 2004년까지의 훈련 데이터로만 만들어졌습니다. "
    "2005~2025년 구간에서는 곡선을 새로 맞춘 것이 아니라, "
    "과거에 학습한 모델이 테스트 기간을 어떻게 예측하는지 보여 줍니다."
)


# --------------------------------------------------
# 12. 테스트 구간 실제값 vs 예측값
# --------------------------------------------------

st.divider()

st.subheader("⑤ 테스트 기간의 실제 기온과 예측값")


fig_test = go.Figure()


fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["평균기온"],
        mode="lines+markers",
        name="실제 기온"
    )
)


for degree in degrees:

    fig_test.add_trace(
        go.Scatter(
            x=test["연도"],
            y=models[degree]["테스트예측"],
            mode="lines+markers",
            name=f"{degree}차 회귀 예측"
        )
    )


fig_test.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=550
)


fig_test.update_xaxes(
    tickformat="d"
)


st.plotly_chart(
    fig_test,
    use_container_width=True
)


# --------------------------------------------------
# 13. 테스트 데이터 상세표
# --------------------------------------------------

prediction_table = pd.DataFrame({

    "연도":
        test["연도"].to_numpy(),

    "실제 평균기온":
        test["평균기온"].to_numpy(),

    "1차 예측":
        models[1]["테스트예측"],

    "3차 예측":
        models[3]["테스트예측"],

    "9차 예측":
        models[9]["테스트예측"]
})


prediction_table["1차 오차"] = abs(
    prediction_table["실제 평균기온"]
    - prediction_table["1차 예측"]
)

prediction_table["3차 오차"] = abs(
    prediction_table["실제 평균기온"]
    - prediction_table["3차 예측"]
)

prediction_table["9차 오차"] = abs(
    prediction_table["실제 평균기온"]
    - prediction_table["9차 예측"]
)


st.dataframe(
    prediction_table.style.format({

        "실제 평균기온": "{:.2f}",

        "1차 예측": "{:.2f}",
        "3차 예측": "{:.2f}",
        "9차 예측": "{:.2f}",

        "1차 오차": "{:.2f}",
        "3차 오차": "{:.2f}",
        "9차 오차": "{:.2f}"

    }),

    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 14. 2050년 예측 비교
# --------------------------------------------------

st.divider()

st.subheader("⑥ 2050년을 예측하면?")


c1, c2, c3 = st.columns(3)


c1.metric(
    "1차 모델",
    f"{models[1]['2050예측']:.2f} ℃"
)


c2.metric(
    "3차 모델",
    f"{models[3]['2050예측']:.2f} ℃"
)


c3.metric(
    "9차 모델",
    f"{models[9]['2050예측']:.2f} ℃"
)


st.warning(
    "2050년은 훈련 데이터 범위를 크게 벗어난 미래입니다. "
    "특히 9차와 같은 고차 다항회귀는 관측 범위를 벗어나면 "
    "곡선이 급격하게 휘어 비현실적인 값이 나올 수 있습니다. "
    "복잡한 모델이 항상 미래 예측을 더 잘하는 것은 아닙니다."
)


# --------------------------------------------------
# 15. 수업용 정리
# --------------------------------------------------

st.divider()

st.subheader("🤔 생각해 보기")


st.markdown(
    """
### 1. 곡선을 복잡하게 만들면 훈련 데이터를 더 잘 따라갈 수 있습니다.

9차 회귀는 1차 회귀보다 훨씬 자유롭게 휘어질 수 있습니다.

하지만 중요한 것은

**훈련 데이터를 얼마나 잘 따라갔는가가 아니라  
처음 보는 테스트 데이터를 얼마나 잘 예측했는가**

입니다.


### 2. 테스트 MAE를 비교해 보세요.

MAE는

**실제 기온과 예상 기온이 평균적으로 몇 ℃ 차이 나는가**

를 의미합니다.

따라서 MAE가 작을수록 새로운 데이터에 대한 예측이 더 정확합니다.


### 3. 가장 복잡한 모델이 가장 좋은 모델일까요?

9차 모델이 훈련 데이터의 모양을 매우 자세하게 따라가더라도  
2005년 이후의 테스트 데이터나 2050년 예측에서는 오히려
큰 오차가 나타날 수 있습니다.

이를 통해 **과대적합**에 대해 생각해 볼 수 있습니다.
"""
)
