import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import HuberRegressor # ⭐ HuberRegressor 튜닝 대상
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

# -----------------------------
# 0) Load
# -----------------------------
# 파일 경로를 사용자 환경에 맞게 조정하세요.
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

# -----------------------------
# 1) 전역 상수 및 시간 피처 정의 
# -----------------------------
REF_DATE = pd.Timestamp("2024-10-24")
MAX_PRICE = 1.0
MID_PRICE = 0.6
LIGHT_PRICE = 0.4

def adjust_hour(dt):
    if pd.isna(dt): return np.nan
    return (dt.hour - 1) % 24 if dt.minute == 0 else dt.hour

def get_tou_relative_price(m, h, period_flag):
    if period_flag == 1: 
        if m in [7, 8]:  # Summer
            if (10 <= h < 12) or (13 <= h < 17): return MAX_PRICE
            if (9 <= h < 10) or (12 <= h < 13) or (17 <= h < 22): return MID_PRICE
            return LIGHT_PRICE
        elif m in [12, 1, 2]:  # Winter
            if (9 <= h < 12) or (17 <= h < 22): return MAX_PRICE
            if (12 <= h < 17) or (22 <= h < 23): return MID_PRICE
            return LIGHT_PRICE
        else:  # Spring/Fall
            if (9 <= h < 23): return MID_PRICE
            return LIGHT_PRICE
    else: 
        if m in [7, 8]:  # Summer
            if (10 <= h < 12) or (13 <= h < 17): return MAX_PRICE
            if (9 <= h < 10) or (12 <= h < 13) or (17 <= h < 22): return MID_PRICE
            return LIGHT_PRICE
        elif m in [12, 1, 2]:  # Winter
            if (9 <= h < 12) or (17 <= h < 22): return MAX_PRICE
            if (12 <= h < 17) or (22 <= h < 23): return MID_PRICE
            return LIGHT_PRICE
        else:  # Spring/Fall
            if (9 <= h < 23): return MID_PRICE
            return LIGHT_PRICE

def enrich(df):
    df["측정일시"] = pd.to_datetime(df["측정일시"], errors="coerce")
    df["월"] = df["측정일시"].dt.month
    df["일"] = df["측정일시"].dt.day
    df["요일"] = df["측정일시"].dt.dayofweek
    df["날짜"] = df['측정일시'].dt.date 
    df["시간"] = df["측정일시"].apply(adjust_hour)
    df["주말여부"] = (df["요일"] >= 5).astype(int)
    df["겨울여부"] = df["월"].isin([11, 12, 1, 2]).astype(int) 
    df["period_flag"] = (df["측정일시"] >= REF_DATE).astype(int)
    df["sin_time"] = np.sin(2 * np.pi * df["시간"] / 24)
    df["cos_time"] = np.cos(2 * np.pi * df["시간"] / 24)
    df["tou_relative_price"] = df.apply(lambda row: get_tou_relative_price(row["월"], row["시간"], row["period_flag"]), axis=1)
    df["tou_load_index"] = df.apply(lambda row: 3 if row["tou_relative_price"] == MAX_PRICE else (2 if row["tou_relative_price"] == MID_PRICE else 1), axis=1)
    df["tou_price_code"] = df["period_flag"].astype(str) + "_" + df["tou_load_index"].astype(str)
    df["sin_day"] = np.sin(2 * np.pi * df["일"] / 31)
    df["cos_day"] = np.cos(2 * np.pi * df["일"] / 31)
    df["sin_month"] = np.sin(2 * np.pi * df["월"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["월"] / 12)
    return df

train = enrich(train).sort_values("측정일시").reset_index(drop=True)
test = enrich(test).sort_values("측정일시").reset_index(drop=True)

# -----------------------------
# 2) 인코딩 
# -----------------------------
le_job = LabelEncoder()
train["작업유형_encoded"] = le_job.fit_transform(train["작업유형"].astype(str))
def safe_transform(le, series, mode_val):
    series_mapped = series.astype(str).map(lambda s: '-1' if s not in le.classes_ else s)
    return le.transform(series_mapped.replace('-1', mode_val))

test["작업유형_encoded"] = safe_transform(le_job, test["작업유형"], train["작업유형"].mode()[0])

le_tou = LabelEncoder()
train["tou_price_code_encoded"] = le_tou.fit_transform(train["tou_price_code"].astype(str))
test["tou_price_code_encoded"] = safe_transform(le_tou, test["tou_price_code"], train["tou_price_code"].mode()[0])

train["시간_작업유형"] = train["시간"].astype(str) + "_" + train["작업유형_encoded"].astype(str)
test["시간_작업유형"] = test["시간"].astype(str) + "_" + test["작업유형_encoded"].astype(str)
le_tj = LabelEncoder()
train["시간_작업유형_encoded"] = le_tj.fit_transform(train["시간_작업유형"])
test["시간_작업유형_encoded"] = safe_transform(le_tj, test["시간_작업유형"], train["시간_작업유형"].mode()[0])


# -----------------------------
# 2.5) 요금적용전력 (Demand Charge) 실제값 계산
# -----------------------------
def calculate_demand_charge_true(df):
    df["피상전력_sim"] = np.sqrt(df["전력사용량(kWh)"]**2 + df["지상무효전력량(kVarh)"]**2)
    df["요금적용전력_kW_true"] = 0.0
    demand_months = [12, 1, 2, 7, 8, 9] 
    
    for idx in df.index:
        current_date = df.loc[idx, "측정일시"]
        start_date = current_date - pd.DateOffset(months=12)
        history_df = df.loc[(df["측정일시"] >= start_date) & 
                            (df["측정일시"] < current_date) & 
                            (df["월"].isin(demand_months))]
        
        current_max_demand = 0.0
        if not history_df.empty:
            max_demand = history_df["피상전력_sim"].max()
            current_max_demand = max(current_max_demand, max_demand)

        if current_date.month in demand_months:
             current_max_demand = max(current_max_demand, df.loc[idx, "피상전력_sim"])

        df.loc[idx, "요금적용전력_kW_true"] = current_max_demand

    df.fillna(method='bfill', inplace=True)
    return df.fillna(0)

train = calculate_demand_charge_true(train)

# -----------------------------
# 3) Stage1: 전력특성 및 요금적용전력 예측 
# -----------------------------
targets_s1 = ["전력사용량(kWh)", "지상무효전력량(kVarh)", "진상무효전력량(kVarh)", 
              "지상역률(%)", "진상역률(%)", "요금적용전력_kW_true", "탄소배출량(tCO2)"] 
feat_s1 = ["월","일","요일","시간","주말여부","겨울여부","period_flag",
           "sin_time","cos_time","sin_day", "cos_day", "sin_month", "cos_month",
           "작업유형_encoded", "tou_relative_price", "tou_price_code_encoded", "시간_작업유형_encoded"] 

stage1_models = {
    "전력사용량(kWh)": LGBMRegressor(n_estimators=2500, learning_rate=0.012, num_leaves=128, random_state=42), 
    "지상무효전력량(kVarh)": CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=7, verbose=0, random_seed=42), 
    "진상무효전력량(kVarh)": CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=7, verbose=0, random_seed=42), 
    "지상역률(%)": LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=96, random_state=42), 
    "진상역률(%)": LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=96, random_state=42), 
    "요금적용전력_kW_true": LGBMRegressor(n_estimators=2500, learning_rate=0.008, num_leaves=64, random_state=42, 
                                          subsample=0.8, colsample_bytree=0.8,
                                          objective='huber', metric='mae', alpha=0.9),
    "탄소배출량(tCO2)": LGBMRegressor(n_estimators=2000, learning_rate=0.012, num_leaves=64, random_state=42), 
}

tscv = TimeSeriesSplit(n_splits=5)
stage1_oof = pd.DataFrame(index=train.index)
stage1_test_pred = pd.DataFrame(index=test.index)
train_targets_true = train[targets_s1].copy()

for tgt in targets_s1:
    oof_pred = np.full(len(train), np.nan, dtype=float)
    model = stage1_models[tgt]
    
    current_target = train_targets_true[tgt].copy()
    is_demand_target = (tgt == "요금적용전력_kW_true")
    if is_demand_target:
        current_target = np.log1p(current_target)

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(train), start=1):
        fold_model = model.__class__(**model.get_params())
        fold_model.fit(train.iloc[tr_idx][feat_s1], current_target.iloc[tr_idx])
        oof_pred[va_idx] = fold_model.predict(train.iloc[va_idx][feat_s1])

    missing = np.isnan(oof_pred)
    if missing.any():
        full_model = model.__class__(**model.get_params())
        full_model.fit(train[feat_s1], current_target)
        oof_pred[missing] = full_model.predict(train.loc[missing, feat_s1])
        
    if is_demand_target:
        oof_pred = np.expm1(oof_pred).clip(min=0) 

    stage1_oof[tgt] = oof_pred
    
    final_model = model.__class__(**model.get_params())
    final_model.fit(train[feat_s1], current_target)
    test_pred = final_model.predict(test[feat_s1])
    
    if is_demand_target:
        test_pred = np.expm1(test_pred).clip(min=0) 
        
    stage1_test_pred[tgt] = test_pred

for tgt in targets_s1:
    new_col_name = "요금적용전력_kW" if tgt == "요금적용전력_kW_true" else tgt
    train[new_col_name] = stage1_oof[tgt]
    test[new_col_name] = stage1_test_pred[tgt]
    
train["피상전력_sim"] = np.sqrt(train["전력사용량(kWh)"]**2 + train["지상무효전력량(kVarh)"]**2)
test["피상전력_sim"] = np.sqrt(test["전력사용량(kWh)"]**2 + test["지상무효전력량(kVarh)"]**2)


# -----------------------------
# 3.5) Stage1 예측값 후처리 및 4-6) 피처 엔지니어링 
# -----------------------------
def post_process_stage1(df):
    P = df["전력사용량(kWh)"]
    Q = df["지상무효전력량(kVarh)"]
    safe_denominator = np.sqrt(P**2 + Q**2) + 1e-6
    df["PF_recalc"] = 100 * P / safe_denominator
    df["PF_recalc"] = df["PF_recalc"].clip(upper=100.0) 
    df["PF_diff"] = df["PF_recalc"] - df["지상역률(%)"]
    is_low_kwh = (df["전력사용량(kWh)"] < 0.5)
    df["PF_recalc"] = np.where(is_low_kwh, 95.0, df["PF_recalc"])
    df["PF_diff"] = np.where(is_low_kwh, 0.0, df["PF_diff"])
    return df

train = post_process_stage1(train)
test = post_process_stage1(test)

def add_pf_features_regulated(df):
    df["유효역률(%)"] = df[["지상역률(%)", "진상역률(%)"]].max(axis=1)
    df["역률_패널티율"] = (90 - df["유효역률(%)"]).clip(lower=0) * 0.01
    df["역률_보상율"] = (df["유효역률(%)"] - 90).clip(lower=0) * 0.005
    df["역률_조정요율"] = df["역률_보상율"] - df["역률_패널티율"]
    df["주간여부"] = df["시간"].isin(range(9, 23)).astype(int)
    df["지상역률_보정"] = df["PF_recalc"].clip(lower=60)
    df["지상역률_주간클립"] = np.where(df["주간여부"] == 1, df["지상역률_보정"].clip(upper=95), df["지상역률_보정"])
    df["역률부족폭_94"] = (94 - df["지상역률_주간클립"]).clip(lower=0) * df["주간여부"]
    df["역률부족폭_90"] = (90 - df["지상역률_주간클립"]).clip(lower=0) * df["주간여부"]
    df["역률부족폭_92"] = (92 - df["지상역률_주간클립"]).clip(lower=0) * df["주간여부"]
    df["역률우수"] = (df["지상역률_주간클립"] >= 95).astype(int) 
    df["야간여부"] = (1 - df["주간여부"]).astype(int)
    df["진상역률_페널티"] = (95 - df["진상역률(%)"]).clip(lower=0) * df["야간여부"]
    df["법적페널티"] = ((df["지상역률_주간클립"] < 90) & (df["주간여부"] == 1)).astype(int)
    df["실질위험"] = ((df["지상역률_주간클립"] < 94) & (df["주간여부"] == 1)).astype(int)
    df["극저역률"] = ((df["지상역률_주간클립"] < 85) & (df["주간여부"] == 1)).astype(int)
    return df
train = add_pf_features_regulated(train)
test = add_pf_features_regulated(test)

def add_lag_roll(df, hist_data, is_train=True):
    df["kwh_lag1"] = df["전력사용량(kWh)"].shift(1)
    df["kwh_lag24"] = df["전력사용량(kWh)"].shift(24)
    df["kwh_roll24_mean"] = df["전력사용량(kWh)"].shift(1).rolling(24).mean()
    df["kwh_roll24_std"] = df["전력사용량(kWh)"].shift(1).rolling(24).std().fillna(0)
    if is_train:
        df.fillna(method='bfill', inplace=True)
        return df.fillna(0)
    else: 
        hist_data_kwh = list(hist_data["kwh"].values.astype(float))
        for i in range(len(df)):
            df.loc[df.index[i], "kwh_lag1"] = hist_data_kwh[-1] if len(hist_data_kwh) >= 1 else 0
            df.loc[df.index[i], "kwh_lag24"] = hist_data_kwh[-24] if len(hist_data_kwh) >= 24 else 0
            arr24 = np.array(hist_data_kwh[-24:])
            df.loc[df.index[i], "kwh_roll24_mean"] = arr24.mean() if arr24.size > 0 else 0
            df.loc[df.index[i], "kwh_roll24_std"] = arr24.std() if arr24.size > 1 else 0
            hist_data_kwh.append(df.loc[df.index[i], "전력사용량(kWh)"])
        return df
hist_data_train = {"kwh": train["전력사용량(kWh)"]}
hist_data_test = {"kwh": train["전력사용량(kWh)"].copy()}
train = add_lag_roll(train, hist_data_train, is_train=True)
test = add_lag_roll(test, hist_data_test, is_train=False)

kwh_mean_day_hour = train.groupby(["요일", "시간"])["전력사용량(kWh)"].mean().reset_index()
kwh_mean_day_hour.rename(columns={"전력사용량(kWh)": "kwh_요일_시간_평균"}, inplace=True)
train = pd.merge(train, kwh_mean_day_hour, on=["요일", "시간"], how="left")
test = pd.merge(test, kwh_mean_day_hour, on=["요일", "시간"], how="left")

def add_advanced_features_hybrid(df, train_means=None):
    df["무효유효비율"] = df["지상무효전력량(kVarh)"] / (df["전력사용량(kWh)"] + 1e-6)
    df["부하역률곱"] = df["전력사용량(kWh)"] * df["역률부족폭_94"] 
    df["역률당전력"] = df["전력사용량(kWh)"] / (df["지상역률_주간클립"] + 1e-6) 
    df["가을위험"] = ((df["월"].isin([9, 10])) & (df["실질위험"] == 1)).astype(int)
    df["동절기안정"] = ((df["겨울여부"] == 1) & (df["지상역률_주간클립"] >= 94)).astype(int)
    if train_means: 
        df["역률_월평균"] = df["월"].map(train_means["역률_월평균"])
        df["역률_월평균"].fillna(train_means["역률_월평균"].mean(), inplace=True) 
    else: 
        df["역률_월평균"] = df.groupby("월")["지상역률_주간클립"].transform("mean")
    df["역률_월평균차이"] = df["지상역률_주간클립"] - df["역률_월평균"]
    df["kwh_roll24_cv"] = df["kwh_roll24_std"] / (df["kwh_roll24_mean"] + 1e-6)
    df["kwh_변화율_24h"] = ((df["전력사용량(kWh)"] - df["kwh_lag24"]) / (df["kwh_lag24"] + 1e-6))
    df["전력급등"] = (df["kwh_변화율_24h"] > 0.5).astype(int)
    df["kwh_시간대비_요일"] = df["전력사용량(kWh)"] / (df["kwh_요일_시간_평균"] + 1e-6)
    df.drop("kwh_요일_시간_평균", axis=1, inplace=True)
    df["총무효전력"] = df["지상무효전력량(kVarh)"] + df["진상무효전력량(kVarh)"]
    df["요금적용전력_차이_비율"] = (df["요금적용전력_kW"] - df["피상전력_sim"]) / (df["요금적용전력_kW"] + 1e-6)
    return df
train_means_for_test = {"역률_월평균": train.groupby("월")["지상역률_주간클립"].mean()}
train = add_advanced_features_hybrid(train)
test = add_advanced_features_hybrid(test, train_means=train_means_for_test)

def add_time_dayofweek_features(df):
    df['hour_workday'] = df['시간'] * (1 - df['주말여부'])
    df['hour_weekend'] = df['시간'] * df['주말여부']
    for d in range(7):
        df[f'hour_day_{d}'] = df['시간'] * (df['요일'] == d).astype(int)
    return df
train = add_time_dayofweek_features(train)
test = add_time_dayofweek_features(test)

def create_daily_worktype_sequence(df, is_train=True):
    daily_sequence = df.groupby('날짜')['작업유형_encoded'].apply(
        lambda x: '_'.join(x.astype(str).tolist())
    ).reset_index(name='작업유형_시퀀스')
    if is_train:
        global le_sequence
        le_sequence = LabelEncoder()
        daily_sequence['작업유형_일별_시퀀스_ID'] = le_sequence.fit_transform(daily_sequence['작업유형_시퀀스'])
    else:
        daily_sequence['작업유형_일별_시퀀스_ID'] = safe_transform(le_sequence, daily_sequence['작업유형_시퀀스'], le_sequence.classes_[0])
    df = pd.merge(df, daily_sequence[['날짜', '작업유형_일별_시퀀스_ID']], on='날짜', how='left')
    df['작업유형_일별_시퀀스_ID'] = df['작업유형_일별_시퀀스_ID'].astype(int)
    return df

train = create_daily_worktype_sequence(train, is_train=True)
test = create_daily_worktype_sequence(test, is_train=False)


# -----------------------------
# 7) Stage2 Feature Set (⭐ 불안정했던 kwh_sum_4h 피처 제거)
# -----------------------------
all_features = [
    "월","일","요일","시간","주말여부","겨울여부","period_flag", "sin_day", "sin_month", "cos_month",
    "작업유형_encoded", "tou_relative_price", "tou_price_code_encoded", "시간_작업유형_encoded",
    "전력사용량(kWh)","지상무효전력량(kVarh)","진상무효전력량(kVarh)", "진상역률(%)", "유효역률(%)", "역률_조정요율",
    "지상역률_보정", "지상역률_주간클립", "주간여부", "야간여부", "실질위험", "법적페널티", "극저역률", 
    "역률부족폭_94", "역률부족폭_92", "PF_recalc", "PF_diff", 
    "무효유효비율","부하역률곱","역률_월평균", "총무효전력", "역률당전력", "진상역률_페널티", "가을위험", "동절기안정",
    "역률_월평균차이","kwh_roll24_cv","kwh_lag1", "kwh_변화율_24h", "전력급등", 
    "kwh_lag24","kwh_roll24_mean","kwh_roll24_std", "kwh_시간대비_요일", 
    "요금적용전력_kW", "피상전력_sim", "hour_workday", "hour_weekend",
    "hour_day_0", "hour_day_1", "hour_day_2", "hour_day_3", "hour_day_4", "hour_day_5", "hour_day_6",
    "탄소배출량(tCO2)", "작업유형_일별_시퀀스_ID", "요금적용전력_차이_비율"
]
feat_s2 = all_features


# -----------------------------
# 8) Stage2 학습 (⭐ Huber Regressor epsilon 튜닝)
# -----------------------------
X_all = train[feat_s2].copy()
y_all = train["전기요금(원)"].copy()
y_all_log = np.log1p(y_all)
X_te = test[feat_s2].copy()

LGB_PARAMS = dict(n_estimators=2500, learning_rate=0.015, num_leaves=75, subsample=0.8, colsample_bytree=0.8, reg_alpha=5, reg_lambda=6, random_state=42, n_jobs=-1)
XGB_PARAMS = dict(n_estimators=2500, learning_rate=0.015, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=6, reg_alpha=3, random_state=42, n_jobs=-1)
CAT_PARAMS = dict(iterations=2000, learning_rate=0.018, depth=7, l2_leaf_reg=8, random_seed=42, verbose=0, thread_count=-1)
HGB_PARAMS = dict(max_iter=2000, learning_rate=0.018, max_leaf_nodes=63, random_state=42, loss='absolute_error')

base_models = {
    "lgb": LGBMRegressor(**LGB_PARAMS),
    "xgb": XGBRegressor(**XGB_PARAMS),
    "cat": CatBoostRegressor(**CAT_PARAMS),
    "hgb": HistGradientBoostingRegressor(**HGB_PARAMS) 
}

# ⭐ 튜닝: epsilon을 1.35에서 1.30으로 낮춰 이상치 처리를 엄격하게 함
meta_learner = HuberRegressor(epsilon=1.30) 
tscv_s2 = TimeSeriesSplit(n_splits=5) 

oof_preds_s2 = pd.DataFrame(index=X_all.index, columns=base_models.keys(), dtype=float)
test_preds_s2 = np.zeros((len(X_te), len(base_models)))

print(f"\n🚀 Stage 2 모델 학습 및 OOF 예측 생성 시작 (5-Fold, Estimator 2000-2500)...")
for fold, (tr_idx, va_idx) in enumerate(tscv_s2.split(X_all), start=1):
    print(f"--- Fold {fold} ---")
    X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
    y_tr_log = y_all_log.iloc[tr_idx]

    fold_test_preds = [] 

    for name, model in base_models.items():
        print(f"  Training {name}...")
        fold_model = model.__class__(**model.get_params())
        
        if name == 'hgb':
            X_tr_hgb = X_tr.rename(columns=lambda x: str(x).replace('[', '').replace(']', ''))
            X_va_hgb = X_va.rename(columns=lambda x: str(x).replace('[', '').replace(']', ''))
            fold_model.fit(X_tr_hgb, y_tr_log)
            oof_pred = fold_model.predict(X_va_hgb)
        else:
            fold_model.fit(X_tr, y_tr_log)
            oof_pred = fold_model.predict(X_va)
        
        oof_preds_s2.iloc[va_idx, list(base_models.keys()).index(name)] = oof_pred
        
        if name == 'hgb':
            X_te_hgb = X_te.rename(columns=lambda x: str(x).replace('[', '').replace(']', ''))
            fold_test_preds.append(fold_model.predict(X_te_hgb))
        else:
            fold_test_preds.append(fold_model.predict(X_te))

    test_preds_s2 += np.mean(fold_test_preds, axis=0)[:, np.newaxis] / tscv_s2.n_splits

print("\n✅ OOF 예측 생성 완료.")

oof_valid_idx = oof_preds_s2.dropna().index
print(f"\n🧠 Meta-Learner ({meta_learner.__class__.__name__}) 학습 시작 (데이터 {len(oof_valid_idx)}개, epsilon=1.30)...")
meta_test_input = pd.DataFrame(test_preds_s2, columns=base_models.keys(), index=X_te.index)

meta_learner.fit(oof_preds_s2.loc[oof_valid_idx], y_all_log.loc[oof_valid_idx])
print(f"✅ Meta-Learner 학습 완료.")

# 최종 Test 예측
print("\n🧪 최종 Test 예측 생성...")
pred_te_log = meta_learner.predict(meta_test_input)
pred_te = np.expm1(pred_te_log)

# OOF 검증 점수 계산
oof_pred_final_log = meta_learner.predict(oof_preds_s2.loc[oof_valid_idx])
oof_pred_final = np.expm1(oof_pred_final_log)
oof_mae = mean_absolute_error(y_all.loc[oof_valid_idx], oof_pred_final)
oof_r2 = r2_score(y_all.loc[oof_valid_idx], oof_pred_final)
print(f"\n📊 OOF 검증 (Stacking): MAE={oof_mae:.2f} | R²={oof_r2:.4f}")


# -----------------------------
# 9) 후처리 및 제출
# -----------------------------
# 예측 범위 클리핑
low, high = np.percentile(pred_te, [0.01, 99.9]) 
pred_te = np.clip(pred_te, low, high)
pred_te = np.clip(pred_te, a_min=500, a_max=450000) 

submission = pd.DataFrame({"id": test["id"], "target": pred_te})
submission.to_csv("submission_737_hubertuning_final.csv", index=False) 
print("\n💾 submission_737_hubertuning_final.csv 저장 완료! (최종 튜닝 적용)")
print(f"예측 범위: {pred_te.min():.2f} ~ {pred_te.max():.2f}")
print(f"예측 평균: {pred_te.mean():.2f}")

# -----------------------------
# 10) 모델 사후 분석 시각화
# -----------------------------
from pathlib import Path

if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.ticker import MaxNLocator
        from matplotlib import font_manager
    except ImportError as exc:
        print(f"\n⚠️ 시각화 패키지를 불러올 수 없어 그래프 생성을 건너뜁니다: {exc}")
    else:
        sns.set_style("whitegrid")
        plt.rcParams["axes.unicode_minus"] = False

        # 한글 폰트 설정 (환경에 존재하는 첫 번째 폰트 선택)
        available_fonts = {f.name for f in font_manager.fontManager.ttflist}
        for font_name in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
            if font_name in available_fonts:
                plt.rcParams["font.family"] = font_name
                break

        output_dir = Path("model_insights")
        output_dir.mkdir(parents=True, exist_ok=True)

        def save_fig(fig, path: Path):
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            print(f"📁 그래프 저장 완료 -> {path}")

        # Stage1 품질 요약 시각화
        stage1_targets_for_plot = [
            "전력사용량(kWh)",
            "지상무효전력량(kVarh)",
            "진상무효전력량(kVarh)",
            "지상역률(%)",
            "진상역률(%)",
            "요금적용전력_kW_true",
        ]

        stage1_metrics = []
        fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
        axes1 = axes1.flatten()

        for idx, tgt in enumerate(stage1_targets_for_plot):
            actual = train_targets_true[tgt]
            pred = stage1_oof[tgt]
            valid_mask = (~actual.isna()) & (~pd.Series(pred).isna())
            actual = actual[valid_mask]
            pred = pred[valid_mask]

            if actual.empty:
                axes1[idx].text(0.5, 0.5, "데이터 없음", ha="center", va="center")
                axes1[idx].set_axis_off()
                continue

            mae = mean_absolute_error(actual, pred)
            r2 = r2_score(actual, pred)
            stage1_metrics.append({"target": tgt, "mae": mae, "r2": r2})

            min_val = min(actual.min(), pred.min())
            max_val = max(actual.max(), pred.max())
            diag = np.linspace(min_val, max_val, 100)

            axes1[idx].scatter(
                actual,
                pred,
                s=10,
                alpha=0.35,
                color="#4A90E2",
                edgecolors="none",
            )
            axes1[idx].plot(diag, diag, "--", color="#D0021B", label="Perfect")
            axes1[idx].set_title(f"{tgt}\nMAE={mae:.2f}, R²={r2:.4f}")
            axes1[idx].set_xlabel("실제값")
            axes1[idx].set_ylabel("예측값")
            axes1[idx].legend(loc="upper left")
            axes1[idx].grid(alpha=0.3)

        fig1.suptitle("□ Stage 1: 6개 타겟 예측 품질 (실제값 vs 예측값)", fontsize=16, y=0.98)
        save_fig(fig1, output_dir / "stage1_target_performance.png")

        if stage1_metrics:
            stage1_summary = pd.DataFrame(stage1_metrics).sort_values("mae")
            stage1_summary.to_csv(output_dir / "stage1_performance_summary.csv", index=False)

        # PF(역률) 종합 분석 대시보드
        pf_base = pd.DataFrame(
            {
                "시간": train["시간"],
                "월": train["월"],
                "PF_recalc": train["PF_recalc"],
                "예측_역률": train["지상역률(%)"],
                "실제_역률": train_targets_true["지상역률(%)"],
                "전기요금(원)": y_all,
                "요금적용전력_true": train_targets_true["요금적용전력_kW_true"],
            }
        ).dropna(subset=["시간", "실제_역률"])

        pf_bins = pd.cut(
            pf_base["실제_역률"],
            bins=[0, 80, 85, 90, 94, 110],
            labels=["위험(<80)", "경고(80-85)", "주의(85-90)", "양호(90-94)", "우수(94+)"],
            right=False,
        )
        pf_base["역률구간"] = pf_bins

        hourly_pf = (
            pf_base.groupby("시간")[["PF_recalc", "실제_역률", "예측_역률"]]
            .mean()
            .rename(columns={"PF_recalc": "PF 재계산"})
        )

        pf_group = pf_base.groupby("역률구간").agg(
            데이터수=("실제_역률", "count"),
            평균요금=("전기요금(원)", "mean"),
            평균PF=("실제_역률", "mean"),
        )

        fig2, axes2 = plt.subplots(2, 2, figsize=(18, 12))

        hourly_pf.plot(ax=axes2[0, 0], marker="o")
        axes2[0, 0].axhline(94, color="#F5A623", linestyle="--", linewidth=1.4, label="기준 94%")
        axes2[0, 0].axhline(90, color="#D0021B", linestyle="--", linewidth=1.4, label="법적 90%")
        axes2[0, 0].set_title("시간대별 평균 역률 비교")
        axes2[0, 0].set_xlabel("시간")
        axes2[0, 0].set_ylabel("역률 (%)")
        axes2[0, 0].legend(loc="lower right")
        axes2[0, 0].grid(alpha=0.3)

        axes2[0, 1].bar(pf_group.index.astype(str), pf_group["데이터수"], color="#4A90E2", alpha=0.7, label="데이터 수")
        axes2[0, 1].set_ylabel("데이터 수")
        axes2[0, 1].set_xlabel("역률 구간")
        axes2[0, 1].tick_params(axis="x", rotation=20)
        axes2[0, 1].set_title("역률 구간별 분포 & 평균 요금")
        ax2_twin = axes2[0, 1].twinx()
        ax2_twin.plot(
            pf_group.index.astype(str),
            pf_group["평균요금"],
            color="#F5A623",
            marker="o",
            label="평균 요금",
        )
        ax2_twin.set_ylabel("평균 요금 (원)")

        scatter = axes2[1, 0].scatter(
            pf_base["실제_역률"],
            pf_base["PF_recalc"],
            c=pf_base["전기요금(원)"],
            cmap="viridis",
            alpha=0.35,
            s=10,
        )
        axes2[1, 0].plot([50, 110], [50, 110], "--", color="#D0021B", linewidth=1.2)
        axes2[1, 0].set_xlim(50, 110)
        axes2[1, 0].set_ylim(50, 110)
        axes2[1, 0].set_title("PF 재계산 vs 실제 지상역률 (색상: 요금)")
        axes2[1, 0].set_xlabel("실제 지상역률 (%)")
        axes2[1, 0].set_ylabel("PF 재계산 (%)")
        cbar = plt.colorbar(scatter, ax=axes2[1, 0])
        cbar.set_label("전기요금(원)")

        pf_group["평균요금"].plot.barh(
            ax=axes2[1, 1],
            color=["#D0021B", "#F5A623", "#F8E71C", "#7ED321", "#417505"],
            alpha=0.8,
        )
        axes2[1, 1].set_title("역률 부족폭별 평균 요금")
        axes2[1, 1].set_xlabel("평균 전기요금 (원)")
        axes2[1, 1].set_ylabel("역률 구간")

        fig2.suptitle("□ 역률(PF) 종합 분석", fontsize=16, y=0.98)
        save_fig(fig2, output_dir / "pf_overview.png")

        # Stage2 예측 품질 및 오차 패턴 분석
        stage2_idx = oof_valid_idx
        stage2_eval = pd.DataFrame(
            {
                "실제_전기요금": y_all.loc[stage2_idx],
                "예측_전기요금": oof_pred_final,
            },
            index=stage2_idx,
        )
        stage2_eval["오차"] = stage2_eval["예측_전기요금"] - stage2_eval["실제_전기요금"]
        stage2_eval["절대오차"] = stage2_eval["오차"].abs()

        stage2_context = stage2_eval.join(
            train.loc[
                stage2_idx,
                [
                    "월",
                    "시간",
                    "요일",
                    "주말여부",
                    "tou_relative_price",
                    "tou_load_index",
                    "PF_recalc",
                    "요금적용전력_kW",
                ],
            ]
        )
        stage2_context["실제_지상역률"] = train_targets_true.loc[stage2_idx, "지상역률(%)"]
        stage2_context["역률구간"] = pf_base.loc[stage2_idx, "역률구간"]

        hourly_fee = stage2_context.groupby("시간")[["실제_전기요금", "예측_전기요금"]].mean()
        hourly_fee["TOU 가격"] = stage2_context.groupby("시간")["tou_relative_price"].mean()

        tou_mae = (
            stage2_context.groupby("tou_load_index")
            .apply(lambda df: mean_absolute_error(df["실제_전기요금"], df["예측_전기요금"]))
            .rename({1: "LIGHT", 2: "MID", 3: "MAX"})
        )

        pf_mae = (
            stage2_context.dropna(subset=["역률구간"])
            .groupby("역률구간")
            .apply(lambda df: mean_absolute_error(df["실제_전기요금"], df["예측_전기요금"]))
            .reindex(pf_group.index)
        )

        month_hour_mae = (
            stage2_context.groupby(["월", "시간"])["절대오차"].mean().unstack(fill_value=np.nan)
        )

        top_errors = stage2_context.sort_values("절대오차", ascending=False).head(8)

        fig3 = plt.figure(figsize=(18, 14))
        gs = fig3.add_gridspec(3, 2, height_ratios=[1, 1, 1.1])

        ax3_1 = fig3.add_subplot(gs[0, 0])
        hourly_fee[["실제_전기요금", "예측_전기요금"]].plot(ax=ax3_1, marker="o")
        ax3_1.set_title("시간대별 평균 전기요금 (실제 vs 예측)")
        ax3_1.set_xlabel("시간")
        ax3_1.set_ylabel("평균 전기요금 (원)")
        ax3_1.grid(alpha=0.3)
        ax3_1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax3_1_2 = ax3_1.twinx()
        ax3_1_2.bar(
            hourly_fee.index,
            hourly_fee["TOU 가격"],
            alpha=0.2,
            color="#F5A623",
            label="평균 TOU 상대가격",
        )
        ax3_1_2.set_ylabel("TOU 상대 가격")

        ax3_2 = fig3.add_subplot(gs[0, 1])
        tou_mae.plot.bar(color=["#50E3C2", "#F8E71C", "#D0021B"], ax=ax3_2)
        ax3_2.set_title("TOU 부하 구간별 예측 오차 (MAE)")
        ax3_2.set_xlabel("TOU 구간")
        ax3_2.set_ylabel("MAE (원)")

        ax3_3 = fig3.add_subplot(gs[1, 0])
        stage2_eval["오차"].plot.hist(
            bins=60,
            ax=ax3_3,
            color="#4A90E2",
            alpha=0.75,
            edgecolor="white",
        )
        ax3_3.axvline(stage2_eval["오차"].mean(), color="#D0021B", linestyle="--", label="평균 오차")
        ax3_3.axvline(0, color="#417505", linestyle="-.", label="오차=0")
        ax3_3.set_title(f"예측 오차 분포 (Std={stage2_eval['오차'].std():.0f})")
        ax3_3.set_xlabel("예측 오차 (예측 - 실제)")
        ax3_3.set_ylabel("빈도")
        ax3_3.legend()

        ax3_4 = fig3.add_subplot(gs[1, 1])
        sns.heatmap(
            month_hour_mae,
            cmap="YlOrRd",
            ax=ax3_4,
            cbar_kws={"label": "MAE (원)"},
        )
        ax3_4.set_title("월-시간대별 평균 MAE Heatmap")
        ax3_4.set_xlabel("시간")
        ax3_4.set_ylabel("월")

        ax3_5 = fig3.add_subplot(gs[2, 0])
        pf_mae.plot.barh(color="#BD10E0", ax=ax3_5)
        ax3_5.set_title("역률 구간별 전기요금 MAE")
        ax3_5.set_xlabel("MAE (원)")
        ax3_5.set_ylabel("역률 구간")

        ax3_6 = fig3.add_subplot(gs[2, 1])
        ax3_6.axis("off")
        table_data = top_errors[
            ["실제_전기요금", "예측_전기요금", "오차", "월", "시간", "tou_load_index", "주말여부"]
        ].copy()
        table_data["오차"] = table_data["오차"].round(0).astype(int)
        table_data["실제_전기요금"] = table_data["실제_전기요금"].round(0).astype(int)
        table_data["예측_전기요금"] = table_data["예측_전기요금"].round(0).astype(int)
        table = ax3_6.table(
            cellText=table_data.values,
            colLabels=[
                "실제",
                "예측",
                "오차",
                "월",
                "시간",
                "TOU",
                "주말",
            ],
            loc="center",
            cellLoc="center",
        )
        table.scale(1, 1.4)
        ax3_6.set_title("상위 예측 오차 샘플", pad=20)

        fig3.suptitle("□ 오차 패턴 종합 분석", fontsize=16, y=0.99)
        save_fig(fig3, output_dir / "stage2_error_dashboard.png")

        # Stage2 변수 중요도 계산
        stage2_full_models = {}
        importance_frames = []
        X_all_hgb = X_all.rename(columns=lambda x: str(x).replace("[", "").replace("]", ""))

        for name, base in base_models.items():
            model = base.__class__(**base.get_params())
            if name == "hgb":
                model.fit(X_all_hgb, y_all_log)
            else:
                model.fit(X_all, y_all_log)
            stage2_full_models[name] = model

            if hasattr(model, "feature_importances_"):
                importance_frames.append(
                    pd.DataFrame(
                        {
                            "feature": X_all.columns,
                            "importance": model.feature_importances_,
                            "model": name.upper(),
                        }
                    )
                )

        if importance_frames:
            importance_df = pd.concat(importance_frames, ignore_index=True)
            agg_importance = (
                importance_df.groupby("feature")["importance"].mean().sort_values(ascending=False)
            )

            fig4, ax4 = plt.subplots(figsize=(12, 10))
            top_imp = agg_importance.head(20).sort_values()
            sns.barplot(x=top_imp.values, y=top_imp.index, ax=ax4, palette="Blues_d")
            ax4.set_title("Stage2 평균 변수 중요도 Top 20 (Tree 기반 모델 평균)")
            ax4.set_xlabel("평균 중요도")
            ax4.set_ylabel("Feature")
            save_fig(fig4, output_dir / "stage2_feature_importance.png")

            importance_df.to_csv(output_dir / "stage2_feature_importance_raw.csv", index=False)

        # SHAP 분석 (가능할 경우)
        try:
            import shap

            shap_sample = X_all.sample(n=min(3000, len(X_all)), random_state=42)
            shap_model = stage2_full_models.get("lgb")

            if shap_model is not None:
                explainer = shap.TreeExplainer(shap_model)
                shap_values = explainer.shap_values(shap_sample, check_additivity=False)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]

                shap.summary_plot(shap_values, shap_sample, show=False, plot_type="bar")
                fig = plt.gcf()
                fig.set_size_inches(10, 8)
                save_fig(fig, output_dir / "stage2_shap_summary_bar.png")

                shap.summary_plot(shap_values, shap_sample, show=False)
                fig = plt.gcf()
                fig.set_size_inches(12, 8)
                save_fig(fig, output_dir / "stage2_shap_summary_beeswarm.png")
        except ImportError as exc:
            print(f"⚠️ SHAP 패키지 미설치로 SHAP 그래프를 건너뜁니다: {exc}")

        # 학습된 모델 및 전처리 객체 피클 저장
        try:
            import pickle
        except ImportError as exc:
            print(f"⚠️ pickle 모듈을 불러올 수 없어 모델 아티팩트를 저장하지 못했습니다: {exc}")
        else:
            artifacts_dir = Path("models") / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Stage1 전체 데이터 재학습 후 저장 (실제 예측용)
            stage1_trained = {}
            for tgt, base_model in stage1_models.items():
                full_model = base_model.__class__(**base_model.get_params())
                target_series = train_targets_true[tgt].copy()
                use_log1p = tgt == "요금적용전력_kW_true"
                y_train = np.log1p(target_series) if use_log1p else target_series
                full_model.fit(train[feat_s1], y_train)
                stage1_trained[tgt] = {
                    "estimator": full_model,
                    "use_log1p": use_log1p,
                }

            stage1_payload = {
                "models": stage1_trained,
                "feature_names": feat_s1,
                "targets": targets_s1,
            }
            with open(artifacts_dir / "stage1_models.pkl", "wb") as f:
                pickle.dump(stage1_payload, f)

            # Stage2 스태킹 모델 및 메타 러너 저장
            stage2_payload = {
                "base_models": stage2_full_models,
                "meta_model": meta_learner,
                "feature_names": feat_s2,
                "hgb_feature_names": list(X_all_hgb.columns),
                "base_model_order": list(stage2_full_models.keys()),
                "target_transform": "log1p",
            }
            with open(artifacts_dir / "stage2_ensemble.pkl", "wb") as f:
                pickle.dump(stage2_payload, f)

            # 인코더 및 기타 전처리 자원 저장
            preprocess_payload = {
                "label_encoders": {
                    "작업유형": le_job,
                    "tou_price_code": le_tou,
                    "시간_작업유형": le_tj,
                },
                "constants": {
                    "REF_DATE": REF_DATE,
                    "MAX_PRICE": MAX_PRICE,
                    "MID_PRICE": MID_PRICE,
                    "LIGHT_PRICE": LIGHT_PRICE,
                },
                "feature_sets": {
                    "stage1": feat_s1,
                    "stage2": feat_s2,
                },
            }
            with open(artifacts_dir / "preprocess_assets.pkl", "wb") as f:
                pickle.dump(preprocess_payload, f)

            print(f"💾 모델 및 전처리 아티팩트를 '{artifacts_dir}' 경로에 저장했습니다.")
