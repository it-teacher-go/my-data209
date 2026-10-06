import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="기온 예측기",
    layout="wide"
)

st.title("🌡️ 서울 기온 선형회귀 예측기")

st.write(
    "서울의 연평균기온으로 선형회귀 모델을 만들고, "
    "과거 자료로 학습한 모델이 최근 기온을 얼마나 잘 예측하는지 평가합니다."
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

# 수업 기준은 2025년까지
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


# 관측일이 300일 이상인 해만 사용
yearly = yearly[
    (yearly["관측일수"] >= 300)
    & (yearly["평균기온"].notna())
].copy()


yearly = yearly.sort_values(
    "연도"
).reset_index(drop=True)


# --------------------------------------------------
# 3. 회귀 및 평가 함수
# --------------------------------------------------

def fit_model(data):

    # 1908년부터 지난 연수
    x = data["연도"].to_numpy() - 1908
    y = data["평균기온"].to_numpy()

    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    return slope, intercept


def predict(data, slope, intercept):

    x = data["연도"].to_numpy() - 1908

    return (
        slope * x
        + intercept
    )


def evaluate(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    errors = y_true - y_pred

    mae = np.mean(
        np.abs(errors)
    )

    mse = np.mean(
        errors ** 2
    )

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    r2 = 1 - ss_res / ss_tot

    return mae, mse, r2


# --------------------------------------------------
# 4. 전체 데이터로 만든 모델
# --------------------------------------------------

all_slope, all_intercept = fit_model(yearly)

all_pred = predict(
    yearly,
    all_slope,
    all_intercept
)

all_mae, all_mse, all_r2 = evaluate(
    yearly["평균기온"],
    all_pred
)


# 상관계수
all_corr = np.corrcoef(
    yearly["연도"],
    yearly["평균기온"]
)[0, 1]


# --------------------------------------------------
# 5. 훈련 / 테스트 데이터 분리
# --------------------------------------------------

# 최근 20년을 테스트 데이터로 사용
TEST_START = 2006
TEST_END = 2025


test = yearly[
    (yearly["연도"] >= TEST_START)
    & (yearly["연도"] <= TEST_END)
].copy()


# 최근 50년 학습:
# 테스트 직전 50년
train_50 = yearly[
    (yearly["연도"] >= 1956)
    & (yearly["연도"] <= 2005)
].copy()


# 최근 100년 학습:
# 테스트 직전 최대 100년
train_100 = yearly[
    (yearly["연도"] >= 1906)
    & (yearly["연도"] <= 2005)
].copy()


# --------------------------------------------------
# 6. 최근 50년 모델
# --------------------------------------------------

slope_50, intercept_50 = fit_model(
    train_50
)


pred_test_50 = predict(
    test,
    slope_50,
    intercept_50
)


mae_50, mse_50, r2_50 = evaluate(
    test["평균기온"],
    pred_test_50
)


corr_50 = np.corrcoef(
    train_50["연도"],
    train_50["평균기온"]
)[0, 1]


# --------------------------------------------------
# 7. 최근 100년 모델
# --------------------------------------------------

slope_100, intercept_100 = fit_model(
    train_100
)


pred_test_100 = predict(
    test,
    slope_100,
    intercept_100
)


mae_100, mse_100, r2_100 = evaluate(
    test["평균기온"],
    pred_test_100
)


corr_100 = np.corrcoef(
    train_100["연도"],
    train_100["평균기온"]
)[0, 1]


# --------------------------------------------------
# 8. 데이터 구분 설명
# --------------------------------------------------

st.divider()

st.subheader("① 훈련 데이터와 테스트 데이터")

c1, c2, c3 = st.columns(3)


with c1:

    st.metric(
        "테스트 데이터",
        f"{len(test)}개년"
    )

    st.caption(
        f"{TEST_START}~{TEST_END}년"
    )


with c2:

    st.metric(
        "50년 학습 데이터",
        f"{len(train_50)}개년"
    )

    st.caption(
        "1956~2005년 중 조건을 만족한 자료"
    )


with c3:

    st.metric(
        "100년 학습 데이터",
        f"{len(train_100)}개년"
    )

    st.caption(
        "1906~2005년 중 조건을 만족한 자료"
    )


st.info(
    "두 모델 모두 같은 최근 20년(2006~2025년)을 테스트 데이터로 사용합니다. "
    "따라서 50년을 학습한 모델과 100년을 학습한 모델의 예측 성능을 "
    "공정하게 비교할 수 있습니다."
)


# --------------------------------------------------
# 9. 전체 데이터 모델
# --------------------------------------------------

st.divider()

st.subheader("② 전체 데이터로 만든 회귀모델")

c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "100년당 기온 변화",
    f"{all_slope * 100:.2f} ℃"
)

c2.metric(
    "상관계수 r",
    f"{all_corr:.3f}"
)

c3.metric(
    "MAE",
    f"{all_mae:.3f} ℃"
)

c4.metric(
    "R²",
    f"{all_r2:.3f}"
)


st.caption(
    f"MSE = {all_mse:.3f}"
)


st.warning(
    "이 평가는 전체 데이터를 이용해 회귀선을 만든 뒤 "
    "같은 데이터를 다시 평가한 결과입니다. "
    "따라서 새로운 데이터에 대한 실제 예측 성능을 평가한 것은 아닙니다."
)


# --------------------------------------------------
# 10. 전체 데이터 회귀선 그래프
# --------------------------------------------------

line_years = np.arange(
    yearly["연도"].min(),
    yearly["연도"].max() + 1
)

line_temp = (
    all_slope * (line_years - 1908)
    + all_intercept
)


fig_all = go.Figure()


fig_all.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="실제 연평균기온"
    )
)


fig_all.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temp,
        mode="lines",
        name="전체 데이터 회귀선"
    )
)


fig_all.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=500
)


fig_all.update_xaxes(
    tickformat="d"
)


st.plotly_chart(
    fig_all,
    use_container_width=True
)


# --------------------------------------------------
# 11. 50년 vs 100년 학습 비교
# --------------------------------------------------

st.divider()

st.subheader("③ 50년 학습과 100년 학습 비교")


c1, c2 = st.columns(2)


with c1:

    st.markdown("### 최근 50년 학습")

    st.metric(
        "100년당 기온 변화",
        f"{slope_50 * 100:.2f} ℃"
    )

    st.metric(
        "훈련 데이터 상관계수",
        f"{corr_50:.3f}"
    )


with c2:

    st.markdown("### 최근 100년 학습")

    st.metric(
        "100년당 기온 변화",
        f"{slope_100 * 100:.2f} ℃"
    )

    st.metric(
        "훈련 데이터 상관계수",
        f"{corr_100:.3f}"
    )


# --------------------------------------------------
# 12. 훈련 데이터 + 테스트 데이터 + 회귀선
# --------------------------------------------------

st.subheader("📈 학습 기간에 따른 회귀선 비교")


fig = go.Figure()


# 전체 연평균기온
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="전체 연평균기온",
        opacity=0.35
    )
)


# 테스트 데이터
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["평균기온"],
        mode="markers",
        marker=dict(
            size=10,
            symbol="diamond"
        ),
        name="테스트 데이터 (2006~2025)"
    )
)


# 50년 회귀선
years_50 = np.arange(
    train_50["연도"].min(),
    TEST_END + 1
)


line_50 = (
    slope_50 * (years_50 - 1908)
    + intercept_50
)


fig.add_trace(
    go.Scatter(
        x=years_50,
        y=line_50,
        mode="lines",
        name=(
            f"50년 학습 회귀선 "
            f"(100년당 {slope_50 * 100:.2f}℃)"
        )
    )
)


# 100년 회귀선
years_100 = np.arange(
    train_100["연도"].min(),
    TEST_END + 1
)


line_100 = (
    slope_100 * (years_100 - 1908)
    + intercept_100
)


fig.add_trace(
    go.Scatter(
        x=years_100,
        y=line_100,
        mode="lines",
        name=(
            f"100년 학습 회귀선 "
            f"(100년당 {slope_100 * 100:.2f}℃)"
        )
    )
)


# 테스트 시작선
fig.add_vline(
    x=2006,
    line_dash="dash",
    annotation_text="테스트 시작",
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


st.caption(
    "회귀선의 실선 구간 중 2006년 이후는 학습에 사용되지 않은 테스트 기간입니다. "
    "즉, 과거 데이터로 만든 직선을 미래 방향으로 연장하여 최근 20년을 예측한 것입니다."
)


# --------------------------------------------------
# 13. 테스트 데이터 예측 성능
# --------------------------------------------------

st.divider()

st.subheader("④ 테스트 데이터 예측 성능")


c1, c2 = st.columns(2)


with c1:

    st.markdown("### 🔹 50년 학습 모델")

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "MAE",
        f"{mae_50:.3f} ℃"
    )

    m2.metric(
        "MSE",
        f"{mse_50:.3f}"
    )

    m3.metric(
        "R²",
        f"{r2_50:.3f}"
    )


with c2:

    st.markdown("### 🔸 100년 학습 모델")

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "MAE",
        f"{mae_100:.3f} ℃"
    )

    m2.metric(
        "MSE",
        f"{mse_100:.3f}"
    )

    m3.metric(
        "R²",
        f"{r2_100:.3f}"
    )


# --------------------------------------------------
# 14. 비교표
# --------------------------------------------------

comparison = pd.DataFrame({

    "모델": [
        "전체 데이터",
        "최근 50년 학습",
        "최근 100년 학습"
    ],

    "학습 기간": [
        f"{yearly['연도'].min()}~{yearly['연도'].max()}",
        "1956~2005",
        "1906~2005"
    ],

    "평가 데이터": [
        "학습 데이터와 동일",
        "2006~2025",
        "2006~2025"
    ],

    "학습 연도 수": [
        len(yearly),
        len(train_50),
        len(train_100)
    ],

    "상관계수 r": [
        all_corr,
        corr_50,
        corr_100
    ],

    "100년당 기온 변화(℃)": [
        all_slope * 100,
        slope_50 * 100,
        slope_100 * 100
    ],

    "MAE": [
        all_mae,
        mae_50,
        mae_100
    ],

    "MSE": [
        all_mse,
        mse_50,
        mse_100
    ],

    "R²": [
        all_r2,
        r2_50,
        r2_100
    ]
})


st.dataframe(
    comparison.style.format({
        "상관계수 r": "{:.3f}",
        "100년당 기온 변화(℃)": "{:.2f}",
        "MAE": "{:.3f}",
        "MSE": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 15. 테스트 데이터 실제값 vs 예측값
# --------------------------------------------------

st.divider()

st.subheader("⑤ 최근 20년의 실제 기온과 예측 기온")


prediction_table = pd.DataFrame({

    "연도": test["연도"],

    "실제 평균기온": test["평균기온"],

    "50년 학습 모델 예측": pred_test_50,

    "100년 학습 모델 예측": pred_test_100
})


prediction_table["50년 모델 오차"] = (
    prediction_table["실제 평균기온"]
    - prediction_table["50년 학습 모델 예측"]
)


prediction_table["100년 모델 오차"] = (
    prediction_table["실제 평균기온"]
    - prediction_table["100년 학습 모델 예측"]
)


st.dataframe(
    prediction_table.style.format({

        "실제 평균기온": "{:.2f}",

        "50년 학습 모델 예측": "{:.2f}",

        "100년 학습 모델 예측": "{:.2f}",

        "50년 모델 오차": "{:.2f}",

        "100년 모델 오차": "{:.2f}"

    }),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 16. 실제값과 예측값 그래프
# --------------------------------------------------

fig_pred = go.Figure()


fig_pred.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["평균기온"],
        mode="lines+markers",
        name="실제 기온"
    )
)


fig_pred.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_test_50,
        mode="lines+markers",
        name="50년 학습 모델 예측"
    )
)


fig_pred.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_test_100,
        mode="lines+markers",
        name="100년 학습 모델 예측"
    )
)


fig_pred.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=550
)


fig_pred.update_xaxes(
    tickformat="d"
)


st.plotly_chart(
    fig_pred,
    use_container_width=True
)


# --------------------------------------------------
# 17. 지표 설명
# --------------------------------------------------

st.divider()

st.subheader("💡 평가 지표는 어떻게 읽을까?")


st.markdown(
    """
### MAE
실제 기온과 예상 기온의 **차이의 절댓값을 평균**한 값입니다.

→ **작을수록 좋습니다.**  
→ 단위가 실제 기온과 같은 **℃**라서 해석하기 쉽습니다.

### MSE
예측 오차를 **제곱한 뒤 평균**한 값입니다.

→ **작을수록 좋습니다.**  
→ 크게 틀린 예측에 더 큰 벌점을 줍니다.

### R²
회귀모델이 실제 데이터의 변화를 얼마나 설명하거나 예측했는지를 나타냅니다.

→ 일반적으로 **1에 가까울수록 좋습니다.**  
→ 테스트 데이터에서는 **0보다 작게 나올 수도 있습니다.**

R²가 음수라면 테스트 데이터에서는  
단순히 테스트 데이터의 평균값으로 예측하는 것보다도
회귀모델의 예측이 좋지 않았다는 의미입니다.
"""
)


# --------------------------------------------------
# 18. 수업용 생각해 보기
# --------------------------------------------------

st.divider()

st.subheader("🤔 생각해 보기")


st.markdown(
    """
같은 서울 기온 데이터라도 **어느 기간을 학습하느냐에 따라
회귀선의 기울기가 달라질 수 있습니다.**

- 최근 50년만 학습한 모델
- 최근 100년을 학습한 모델

중 어느 모델의 회귀선이 더 가파른지 확인해 보세요.

그리고 **회귀선의 기울기가 더 최근의 경향을 잘 반영한다고 해서
반드시 테스트 데이터의 MAE, MSE, R²가 더 좋아지는지도 확인**해 보세요.

즉,

> **회귀선의 기울기와 예측 성능은 같은 개념이 아닙니다.**

라는 점을 비교할 수 있습니다.
"""
)
