import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    plt = None
    warnings.warn(f"matplotlib 미설치: {exc}. 시각화가 비활성화됩니다.")

try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
except ImportError as exc:
    plot_acf = None
    plot_pacf = None
    warnings.warn(f"statsmodels.graphics.tsaplots 미설치: {exc}. ACF/PACF가 비활성화됩니다.")

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
except ImportError:
    seasonal_decompose = None

try:
    import seaborn as sns
except ImportError:
    sns = None

warnings.filterwarnings("ignore")


# -----------------------------
# 공용 상수 및 유틸 함수
# -----------------------------
DATA_DIR = Path("./data")
REF_DATE = pd.Timestamp("2024-10-24")
MAX_PRICE = 1.0
MID_PRICE = 0.6
LIGHT_PRICE = 0.4


def adjust_hour(dt: pd.Timestamp) -> float:
    if pd.isna(dt):
        return np.nan
    return (dt.hour - 1) % 24 if dt.minute == 0 else dt.hour


def get_tou_relative_price(month: int, hour: float, period_flag: int) -> float:
    if period_flag == 1:
        if month in [7, 8]:
            if (10 <= hour < 12) or (13 <= hour < 17):
                return MAX_PRICE
            if (9 <= hour < 10) or (12 <= hour < 13) or (17 <= hour < 22):
                return MID_PRICE
            return LIGHT_PRICE
        if month in [12, 1, 2]:
            if (9 <= hour < 12) or (17 <= hour < 22):
                return MAX_PRICE
            if (12 <= hour < 17) or (22 <= hour < 23):
                return MID_PRICE
            return LIGHT_PRICE
        if 9 <= hour < 23:
            return MID_PRICE
        return LIGHT_PRICE
    if month in [7, 8]:
        if (10 <= hour < 12) or (13 <= hour < 17):
            return MAX_PRICE
        if (9 <= hour < 10) or (12 <= hour < 13) or (17 <= hour < 22):
            return MID_PRICE
        return LIGHT_PRICE
    if month in [12, 1, 2]:
        if (9 <= hour < 12) or (17 <= hour < 22):
            return MAX_PRICE
        if (12 <= hour < 17) or (22 <= hour < 23):
            return MID_PRICE
        return LIGHT_PRICE
    if 9 <= hour < 23:
        return MID_PRICE
    return LIGHT_PRICE


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["측정일시"] = pd.to_datetime(df["측정일시"], errors="coerce")
    df["월"] = df["측정일시"].dt.month
    df["일"] = df["측정일시"].dt.day
    df["요일"] = df["측정일시"].dt.dayofweek
    df["날짜"] = df["측정일시"].dt.date
    df["시간"] = df["측정일시"].apply(adjust_hour)
    df["주말여부"] = (df["요일"] >= 5).astype(int)
    df["겨울여부"] = df["월"].isin([11, 12, 1, 2]).astype(int)
    df["period_flag"] = (df["측정일시"] >= REF_DATE).astype(int)
    df["sin_time"] = np.sin(2 * np.pi * df["시간"] / 24)
    df["cos_time"] = np.cos(2 * np.pi * df["시간"] / 24)
    df["tou_relative_price"] = df.apply(
        lambda row: get_tou_relative_price(row["월"], row["시간"], row["period_flag"]), axis=1
    )
    df["tou_load_index"] = df["tou_relative_price"].map(
        {MAX_PRICE: 3, MID_PRICE: 2, LIGHT_PRICE: 1}
    )
    df["tou_price_code"] = df["period_flag"].astype(str) + "_" + df["tou_load_index"].astype(str)
    df["sin_day"] = np.sin(2 * np.pi * df["일"] / 31)
    df["cos_day"] = np.cos(2 * np.pi * df["일"] / 31)
    df["sin_month"] = np.sin(2 * np.pi * df["월"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["월"] / 12)
    return df


def calculate_demand_charge_true(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_cols = {"전력사용량(kWh)", "지상무효전력량(kVarh)"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        # 테스트 데이터에는 Stage1 예측 전력 특성이 없어 생략
        return df

    df["피상전력_sim"] = np.sqrt(df["전력사용량(kWh)"] ** 2 + df["지상무효전력량(kVarh)"] ** 2)
    df["요금적용전력_kW_true"] = 0.0
    demand_months = [12, 1, 2, 7, 8, 9]

    for idx in df.index:
        current_date = df.loc[idx, "측정일시"]
        start_date = current_date - pd.DateOffset(months=12)
        history_df = df.loc[
            (df["측정일시"] >= start_date)
            & (df["측정일시"] < current_date)
            & (df["월"].isin(demand_months))
        ]

        current_max_demand = 0.0
        if not history_df.empty:
            current_max_demand = max(current_max_demand, history_df["피상전력_sim"].max())

        if current_date.month in demand_months:
            current_max_demand = max(current_max_demand, df.loc[idx, "피상전력_sim"])

        df.loc[idx, "요금적용전력_kW_true"] = current_max_demand

    df.fillna(method="bfill", inplace=True)
    return df.fillna(0)


def post_process_stage1(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_cols = {"전력사용량(kWh)", "지상무효전력량(kVarh)", "지상역률(%)"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        return df

    P = df["전력사용량(kWh)"]
    Q = df["지상무효전력량(kVarh)"]
    safe_denominator = np.sqrt(P**2 + Q**2) + 1e-6
    df["PF_recalc"] = 100 * P / safe_denominator
    df["PF_recalc"] = df["PF_recalc"].clip(upper=100.0)
    df["PF_diff"] = df["PF_recalc"] - df["지상역률(%)"]
    is_low_kwh = df["전력사용량(kWh)"] < 0.5
    df.loc[is_low_kwh, "PF_recalc"] = 95.0
    df.loc[is_low_kwh, "PF_diff"] = 0.0
    return df


def prepare_datasets(force_reload: bool = False):
    global train_raw, test_raw, train_enriched, test_enriched
    if (
        (not force_reload)
        and "train_enriched" in globals()
        and "test_enriched" in globals()
    ):
        return train_raw, test_raw, train_enriched, test_enriched

    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")

    train_enriched = enrich(train_raw).sort_values("측정일시").reset_index(drop=True)
    test_enriched = enrich(test_raw).sort_values("측정일시").reset_index(drop=True)

    train_enriched = calculate_demand_charge_true(train_enriched)
    train_enriched = post_process_stage1(train_enriched)
    test_enriched = post_process_stage1(test_enriched)

    return train_raw, test_raw, train_enriched, test_enriched


train_raw, test_raw, train_enriched, test_enriched = prepare_datasets()


def plot_acf_pacf_series(
    series: pd.Series,
    lags: int = 96,
    title: str | None = None,
    save_dir: Path | None = None,
    highlight_lags: list[int] | int | None = None,
):
    if plt is None or plot_acf is None or plot_pacf is None:
        print("⚠️ matplotlib/statsmodels를 찾을 수 없어 ACF/PACF를 건너뜁니다.")
        return

    valid_series = series.dropna()
    if valid_series.empty:
        print("⚠️ 시계열 데이터가 비어 있어 ACF/PACF를 생성할 수 없습니다.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(valid_series, lags=lags, ax=axes[0])
    plot_pacf(valid_series, lags=lags, ax=axes[1])
    axes[0].set_title(f"{title} - ACF" if title else "ACF")
    axes[1].set_title(f"{title} - PACF" if title else "PACF")
    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)

    if highlight_lags is not None:
        lag_list = [highlight_lags] if isinstance(highlight_lags, int) else list(highlight_lags)
        for lag in lag_list:
            if lag <= 0:
                continue
            for ax in axes:
                ax.axvline(lag, color="#D0021B", linestyle="--", linewidth=1.4, alpha=0.8)
                ax.axvline(-lag, color="#D0021B", linestyle="--", linewidth=1.0, alpha=0.4)
            axes[0].annotate(
                f"Lag {lag}",
                xy=(lag, 0),
                xytext=(lag, 0.05),
                textcoords="data",
                arrowprops=dict(arrowstyle="->", color="#D0021B"),
                ha="center",
            )

    plt.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{(title or 'series').replace(' ', '_')}_acf_pacf.png"
        save_path = save_dir / file_name
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"📁 ACF/PACF 그래프 저장 완료 -> {save_path}")

    plt.show()
    plt.close(fig)


def plot_lag_scatter(series: pd.Series, lag: int, title: str, save_dir: Path | None = None):
    if plt is None:
        print("⚠️ matplotlib를 찾을 수 없어 Lag Scatter를 건너뜁니다.")
        return

    aligned = pd.concat([series, series.shift(lag)], axis=1, keys=["current", f"lag_{lag}"]).dropna()
    if aligned.empty:
        print("⚠️ 시계열 길이가 부족하여 Lag Scatter를 생성할 수 없습니다.")
        return

    corr = aligned["current"].corr(aligned[f"lag_{lag}"])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(aligned[f"lag_{lag}"], aligned["current"], s=10, alpha=0.4, color="#4A90E2")
    ax.set_xlabel(f"Lag {lag}")
    ax.set_ylabel("현재값")
    ax.set_title(f"{title}\nLag {lag} 상관계수={corr:.3f}")
    ax.grid(alpha=0.3)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{title.replace(' ', '_')}_lag{lag}_scatter.png"
        path = save_dir / file_name
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print(f"📁 Lag Scatter 저장 완료 -> {path}")

    plt.show()
    plt.close(fig)


def plot_hourly_profile(series: pd.Series, title: str, save_dir: Path | None = None):
    if plt is None or sns is None:
        print("⚠️ matplotlib 또는 seaborn을 찾을 수 없어 시간대별 프로파일을 건너뜁니다.")
        return

    df = series.to_frame("value").dropna()
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek

    hourly_mean = df.groupby("hour")["value"].mean()
    dow_hour = df.groupby(["dow", "hour"])["value"].mean().unstack()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(hourly_mean.index, hourly_mean.values, marker="o", color="#417505")
    axes[0].set_title(f"{title} - 시간대별 평균")
    axes[0].set_xlabel("시간")
    axes[0].set_ylabel("평균 값")
    axes[0].grid(alpha=0.3)

    sns.heatmap(
        dow_hour,
        cmap="YlOrRd",
        ax=axes[1],
        cbar_kws={"label": "평균 값"},
    )
    axes[1].set_title(f"{title} - 요일/시간 Heatmap")
    axes[1].set_xlabel("시간")
    axes[1].set_ylabel("요일 (0=월)")

    plt.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{title.replace(' ', '_')}_hourly_profile.png"
        path = save_dir / file_name
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print(f"📁 시간대별 패턴 저장 완료 -> {path}")

    plt.show()
    plt.close(fig)


def plot_seasonal_decomposition(series: pd.Series, period: int, title: str, save_dir: Path | None = None):
    if plt is None or seasonal_decompose is None:
        print("⚠️ matplotlib/statsmodels seasonal_decompose를 사용할 수 없어 계절 분해를 건너뜁니다.")
        return

    valid_series = series.dropna()
    if len(valid_series) < period * 2:
        print("⚠️ 데이터 길이가 짧아 계절 분해를 건너뜁니다.")
        return

    decomposition = seasonal_decompose(valid_series, model="additive", period=period)
    fig = decomposition.plot()
    fig.set_size_inches(12, 9)
    fig.suptitle(f"{title} - 계절 분해 (Period={period})", y=0.95)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{title.replace(' ', '_')}_seasonal_decompose.png"
        path = save_dir / file_name
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print(f"📁 계절 분해 그래프 저장 완료 -> {path}")

    plt.show()
    plt.close(fig)


def get_regular_series(df: pd.DataFrame, column: str, freq: str = "15T") -> pd.Series:
    if column not in df.columns:
        raise KeyError(f"{column} 컬럼이 데이터프레임에 존재하지 않습니다.")
    return (
        df.sort_values("측정일시")
        .set_index("측정일시")[column]
        .asfreq(freq)
        .interpolate(limit_direction="both")
    )


# -----------------------------
# ↓↓↓ 사용자 작성 코드 유지 (필요한 경우 아래에 배치) ↓↓↓
# -----------------------------

if __name__ == "__main__":
    analysis_columns = {
        "전력사용량(kWh)": "전력사용량(kWh)",
        "전기요금(원)": "전기요금(원)",
    }
    output_root = Path("model_insights") / "time_series"
    output_root.mkdir(parents=True, exist_ok=True)

    for col, label in analysis_columns.items():
        if col not in train_enriched.columns:
            print(f"⚠️ {col} 컬럼을 찾을 수 없어 분석을 건너뜁니다.")
            continue

        print(f"\n🔍 '{label}' 시계열 분석을 시작합니다.")
        try:
            series = get_regular_series(train_enriched, col)
        except KeyError as exc:
            print(f"⚠️ {exc}")
            continue

        day_lag = 96  # 15분 간격 데이터에서 24시간 주기를 의미
        plot_acf_pacf_series(
            series,
            lags=day_lag * 3,
            title=label,
            save_dir=output_root,
            highlight_lags=[day_lag, day_lag * 2],
        )
        plot_lag_scatter(series, lag=day_lag, title=label, save_dir=output_root)
        plot_hourly_profile(series, title=label, save_dir=output_root)
        plot_seasonal_decomposition(series, period=day_lag, title=label, save_dir=output_root)
