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

st.title("🌡️ 기온 곡선 예측기")

st.write(
    "서울의 과거 연평균기온으로 여러 차수의 회귀곡선을 만들고, "
    "모델이 복잡해질수록 훈련 데이터와 테스트 데이터의 오차가 "
    "어떻게 달라지는지 비교합니다."
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


# 관측일이 300일 이상인 해만 사용
yearly = yearly[
    (yearly["관측일수"] >= 300)
    & (yearly["평균기온"].notna())
].copy()


yearly = yearly.sort_values(
    "연도"
).reset_index(drop=True)


# --------------------------------------------------
# 3. 훈련 / 테스트 분리
# --------------------------------------------------

# 2005년 이전은 훈련
train = yearly[
    yearly["연도"] < 2005
].copy()


# 2005년부터는 테스트
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
        f"{train['연도'].min()}~"
        f"{train['연도'].max()}년"
    )


with c2:

    st.metric(
        "테스트 데이터",
        f"{len(test)}개 연도"
    )

    st.caption(
        f"{test['연도'].min()}~"
        f"{test['연도'].max()}년"
    )


st.info(
    "2005년 이후 자료는 회귀곡선을 만드는 데 사용하지 않습니다. "
    "1차, 3차, 9차 모델 모두 2004년까지의 자료만 학습하고, "
    "2005년 이후 자료는 모델의 예측 성능을 평가하는 데만 사용합니다."
)


# --------------------------------------------------
# 5. 연도 변환
# --------------------------------------------------

def transform_year(year):

    # 고차 다항식 계산을 안정적으로 하기 위해
    # 연도 숫자를 작게 변환
    return (
        np.asarray(year, dtype=float)
        - 1908
    ) / 100


# --------------------------------------------------
# 6. 다항회귀 함수
# --------------------------------------------------

def fit_polynomial(data, degree):

    x = transform_year(
        data["연도"]
    )

    y = data[
        "평균기온"
    ].to_numpy()

    coefficients = np.polyfit(
        x,
        y,
        degree
    )

    return coefficients


def predict_polynomial(
    years,
    coefficients
):

    x = transform_year(years)

    return np.polyval(
        coefficients,
        x
    )


def calculate_mae(
    y_true,
    y_pred
):

    return np.mean(
        np.abs(
            np.asarray(y_true)
            - np.asarray(y_pred)
        )
    )


# --------------------------------------------------
# 7. 1차 · 3차 · 9차 모델 학습
# --------------------------------------------------

degrees = [
    1,
    3,
    9
]


models = {}


for degree in degrees:

    # 훈련 데이터만 사용하여 곡선 생성
    coefficients = fit_polynomial(
        train,
        degree
    )


    # 훈련 데이터 예측
    train_pred = predict_polynomial(
        train["연도"],
        coefficients
    )


    # 테스트 데이터 예측
    test_pred = predict_polynomial(
        test["연도"],
        coefficients
    )


    # 훈련 MAE
    train_mae = calculate_mae(
        train["평균기온"],
        train_pred
    )


    # 테스트 MAE
    test_mae = calculate_mae(
        test["평균기온"],
        test_pred
    )


    # 2050년 예측
    pred_2050 = predict_polynomial(
        [2050],
        coefficients
    )[0]


    models[degree] = {

        "coefficients":
            coefficients,

        "train_pred":
            train_pred,

        "test_pred":
            test_pred,

        "train_mae":
            train_mae,

        "test_mae":
            test_mae,

        "pred_2050":
            pred_2050
    }


# --------------------------------------------------
# 8. 세 곡선 한 번에 비교
# --------------------------------------------------

st.divider()

st.subheader(
    "② 1차 · 3차 · 9차 회귀곡선 비교"
)


fig = go.Figure()


# 훈련 데이터
fig.add_trace(
    go.Scatter(

        x=train["연도"],

        y=train["평균기온"],

        mode="markers",

        name="훈련 데이터",

        marker=dict(
            size=7
        ),

        hovertemplate=(
            "훈련 데이터<br>"
            "연도: %{x}년<br>"
            "평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
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

        name="테스트 데이터",

        hovertemplate=(
            "테스트 데이터<br>"
            "연도: %{x}년<br>"
            "평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 곡선은 2050년까지
curve_years = np.linspace(
    train["연도"].min(),
    2050,
    600
)


dash_styles = {

    1: "solid",

    3: "dash",

    9: "dot"
}


for degree in degrees:

    curve_temp = predict_polynomial(
        curve_years,
        models[degree]["coefficients"]
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
                f"(테스트 MAE "
                f"{models[degree]['test_mae']:.2f}℃)"
            ),

            hovertemplate=(
                f"{degree}차 회귀<br>"
                "연도: %{x:.0f}년<br>"
                "예측기온: %{y:.2f}℃"
                "<extra></extra>"
            )
        )
    )


# 테스트 시작
fig.add_vline(
    x=2005,
    line_dash="dash",
    annotation_text="테스트 시작",
    annotation_position="top"
)


# 실제 데이터 끝
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


st.caption(
    "세 곡선 모두 2004년까지의 훈련 데이터만 보고 만들어졌습니다. "
    "2005년 이후 구간은 모델이 학습하지 않은 구간입니다."
)


# --------------------------------------------------
# 9. 차수 선택 슬라이더
# --------------------------------------------------

st.divider()

st.subheader(
    "③ 곡선의 차수를 바꿔 오차를 비교해 보기"
)


selected_degree = st.select_slider(

    "회귀곡선의 차수를 선택하세요.",

    options=[
        1,
        3,
        9
    ],

    value=1,

    format_func=lambda x: f"{x}차"
)


selected = models[
    selected_degree
]


# --------------------------------------------------
# 10. 선택한 차수의 오차
# --------------------------------------------------

c1, c2, c3, c4 = st.columns(4)


c1.metric(

    f"{selected_degree}차 훈련 MAE",

    f"{selected['train_mae']:.3f} ℃"
)


c2.metric(

    f"{selected_degree}차 테스트 MAE",

    f"{selected['test_mae']:.3f} ℃"
)


mae_gap = (
    selected["test_mae"]
    - selected["train_mae"]
)


c3.metric(

    "테스트 − 훈련 오차",

    f"{mae_gap:.3f} ℃"
)


c4.metric(

    "2050년 예상 기온",

    f"{selected['pred_2050']:.2f} ℃"
)


# --------------------------------------------------
# 11. 선택한 모델 해석
# --------------------------------------------------

if selected["test_mae"] > selected["train_mae"]:

    st.write(
        f"**{selected_degree}차 모델은 "
        f"훈련 데이터에서는 평균 "
        f"{selected['train_mae']:.3f}℃ 빗나가지만, "
        f"처음 보는 테스트 데이터에서는 평균 "
        f"{selected['test_mae']:.3f}℃ 빗나갑니다.**"
    )

else:

    st.write(
        f"**{selected_degree}차 모델의 "
        f"훈련 MAE는 "
        f"{selected['train_mae']:.3f}℃, "
        f"테스트 MAE는 "
        f"{selected['test_mae']:.3f}℃입니다.**"
    )


# --------------------------------------------------
# 12. 선택한 차수의 곡선만 강조
# --------------------------------------------------

st.subheader(
    f"📈 현재 선택: {selected_degree}차 회귀"
)


fig_selected = go.Figure()


# 훈련 데이터
fig_selected.add_trace(
    go.Scatter(

        x=train["연도"],

        y=train["평균기온"],

        mode="markers",

        name="훈련 데이터"
    )
)


# 테스트 데이터
fig_selected.add_trace(
    go.Scatter(

        x=test["연도"],

        y=test["평균기온"],

        mode="markers",

        marker=dict(
            size=10,
            symbol="diamond"
        ),

        name="테스트 데이터"
    )
)


selected_curve = predict_polynomial(

    curve_years,

    selected["coefficients"]
)


fig_selected.add_trace(
    go.Scatter(

        x=curve_years,

        y=selected_curve,

        mode="lines",

        line=dict(
            width=4
        ),

        name=f"{selected_degree}차 회귀곡선"
    )
)


fig_selected.add_vline(

    x=2005,

    line_dash="dash",

    annotation_text="테스트 시작"
)


fig_selected.add_vline(

    x=2025,

    line_dash="dot",

    annotation_text="관측 자료 끝"
)


fig_selected.update_layout(

    xaxis_title="연도",

    yaxis_title="연평균기온 (℃)",

    height=600
)


fig_selected.update_xaxes(
    tickformat="d"
)


st.plotly_chart(
    fig_selected,
    use_container_width=True
)


# --------------------------------------------------
# 13. 차수가 높아지면서 오차가 어떻게 변하는지
# --------------------------------------------------

st.divider()

st.subheader(
    "④ 차수가 높아지면 오차는 어떻게 변할까?"
)


error_table = pd.DataFrame({

    "차수": [
        1,
        3,
        9
    ],

    "훈련 MAE": [
        models[1]["train_mae"],
        models[3]["train_mae"],
        models[9]["train_mae"]
    ],

    "테스트 MAE": [
        models[1]["test_mae"],
        models[3]["test_mae"],
        models[9]["test_mae"]
    ]
})


fig_error = go.Figure()


fig_error.add_trace(
    go.Scatter(

        x=error_table["차수"],

        y=error_table["훈련 MAE"],

        mode="lines+markers",

        name="훈련 MAE"
    )
)


fig_error.add_trace(
    go.Scatter(

        x=error_table["차수"],

        y=error_table["테스트 MAE"],

        mode="lines+markers",

        name="테스트 MAE"
    )
)


fig_error.update_layout(

    xaxis_title="회귀곡선 차수",

    yaxis_title="MAE (℃)",

    height=500
)


fig_error.update_xaxes(

    tickmode="array",

    tickvals=[
        1,
        3,
        9
    ],

    ticktext=[
        "1차",
        "3차",
        "9차"
    ]
)


st.plotly_chart(
    fig_error,
    use_container_width=True
)


st.caption(
    "모델이 복잡해질수록 훈련 오차는 줄어드는 경향이 있지만, "
    "테스트 오차까지 반드시 줄어드는 것은 아닙니다."
)


# --------------------------------------------------
# 14. 결과 표
# --------------------------------------------------

st.subheader(
    "📋 차수별 결과 비교"
)


result_table = pd.DataFrame({

    "모델": [
        "1차",
        "3차",
        "9차"
    ],

    "훈련 MAE (℃)": [
        models[1]["train_mae"],
        models[3]["train_mae"],
        models[9]["train_mae"]
    ],

    "테스트 MAE (℃)": [
        models[1]["test_mae"],
        models[3]["test_mae"],
        models[9]["test_mae"]
    ],

    "테스트 - 훈련 MAE": [
        models[1]["test_mae"]
        - models[1]["train_mae"],

        models[3]["test_mae"]
        - models[3]["train_mae"],

        models[9]["test_mae"]
        - models[9]["train_mae"]
    ],

    "2050년 예상기온 (℃)": [
        models[1]["pred_2050"],
        models[3]["pred_2050"],
        models[9]["pred_2050"]
    ]
})


st.dataframe(

    result_table.style.format({

        "훈련 MAE (℃)": "{:.3f}",

        "테스트 MAE (℃)": "{:.3f}",

        "테스트 - 훈련 MAE": "{:.3f}",

        "2050년 예상기온 (℃)": "{:.2f}"
    }),

    use_container_width=True,

    hide_index=True
)


# --------------------------------------------------
# 15. 테스트 기간 실제값 비교
# --------------------------------------------------

st.divider()

st.subheader(
    "⑤ 테스트 기간에서 실제값과 예측값 비교"
)


fig_test = go.Figure()


fig_test.add_trace(
    go.Scatter(

        x=test["연도"],

        y=test["평균기온"],

        mode="lines+markers",

        name="실제 연평균기온"
    )
)


for degree in degrees:

    fig_test.add_trace(
        go.Scatter(

            x=test["연도"],

            y=models[degree]["test_pred"],

            mode="lines+markers",

            name=f"{degree}차 예측"
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
# 16. 2050년 예측
# --------------------------------------------------

st.divider()

st.subheader(
    "⑥ 2050년 예측 비교"
)


c1, c2, c3 = st.columns(3)


c1.metric(
    "1차",
    f"{models[1]['pred_2050']:.2f} ℃"
)


c2.metric(
    "3차",
    f"{models[3]['pred_2050']:.2f} ℃"
)


c3.metric(
    "9차",
    f"{models[9]['pred_2050']:.2f} ℃"
)


st.warning(
    "2050년은 학습 데이터의 범위를 크게 벗어난 외삽입니다. "
    "특히 고차 다항식은 학습 범위 밖에서 급격하게 휘어질 수 있으므로 "
    "2050년 예측값이 현실적으로 타당한지도 함께 살펴봐야 합니다."
)


# --------------------------------------------------
# 17. 수업용 정리
# --------------------------------------------------

st.divider()

st.subheader(
    "💡 무엇을 확인해야 할까?"
)


st.markdown(
    """
슬라이더를 **1차 → 3차 → 9차**로 움직여 보세요.

### 훈련 MAE는 어떻게 변하나요?

곡선을 자유롭게 휘게 할수록 훈련 데이터를 더 세밀하게 따라갈 수 있기 때문에  
일반적으로 **훈련 오차는 작아집니다.**

### 그런데 테스트 MAE도 계속 작아질까요?

반드시 그렇지는 않습니다.

어느 순간부터는 훈련 데이터의 작은 변화까지 지나치게 따라가면서  
처음 보는 데이터에 대한 예측력이 떨어질 수 있습니다.

즉,

**훈련 오차 ↓**

라고 해서 항상

**테스트 오차 ↓**

인 것은 아닙니다.

이 현상이 바로 **과대적합**을 이해하는 핵심입니다.
"""
)
