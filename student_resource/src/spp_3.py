import os, re, math
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optional: Use HistGradientBoosting for faster runs
try:
    from xgboost import XGBRegressor
    USE_XGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    USE_XGB = False

# --- File Paths ---
DATASET_FOLDER = "dataset"
TRAIN_CSV = os.path.join(DATASET_FOLDER, "train.csv")
TEST_CSV  = os.path.join(DATASET_FOLDER, "test.csv")
OUT_CSV   = os.path.join(DATASET_FOLDER, "test_out.csv")
CV_LOG_CSV= os.path.join(DATASET_FOLDER, "cv_metrics_log.csv")

# --- Regex Functions ---
_pack_re = re.compile(r"[Pp]ack\s*of\s*(\d+)")
def _extract_pack_of(s: str) -> float:
    if not isinstance(s, str): return 1.0
    m = _pack_re.search(s)
    return float(m.group(1)) if m else 1.0

_val_re  = re.compile(r"^Value:\s*([\-+]?\d+(\.\d+)?)\s*$", re.M)
_unit_re = re.compile(r"^Unit:\s*([A-Za-z %/]+)\s*$", re.M)
def _extract_value_unit(s: str) -> Tuple[float, str]:
    if not isinstance(s, str): return (np.nan, "UNK")
    vm = _val_re.search(s); um = _unit_re.search(s)
    v = float(vm.group(1)) if vm else np.nan
    u = um.group(1).strip() if um else "UNK"
    return (v, u)

# --- Structured Feature Builder ---
def build_structured(df: pd.DataFrame) -> pd.DataFrame:
    s = pd.Series(df["catalog_content"].fillna(""), index=df.index)
    out = pd.DataFrame(index=df.index)
    out["pack_of"] = s.apply(_extract_pack_of).astype(float)
    vu = s.apply(_extract_value_unit)
    out["value_num"] = vu.apply(lambda t: t[0]).astype(float)
    out["unit"] = vu.apply(lambda t: t[1])
    # Derived features
    out["log_pack_of"] = np.log1p(out["pack_of"])
    out["log_value_num"] = np.log1p(out["value_num"].fillna(0.0))
    vpp = out["value_num"] / out["pack_of"].replace(0, np.nan)
    vpp = vpp.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["value_per_pack"] = vpp
    out["log_value_per_pack"] = np.log1p(vpp)
    # Text features
    out["txt_len"] = s.str.len().astype(float)
    out["txt_words"] = s.str.split().apply(len).astype(float)
    out["txt_digits"] = s.str.count(r"\d").astype(float)
    out["txt_upper"] = s.str.count(r"[A-Z]").astype(float)
    out["txt_digit_ratio"] = (out["txt_digits"] / out["txt_len"].replace(0, np.nan)).fillna(0.0)
    out["txt_upper_ratio"] = (out["txt_upper"] / out["txt_len"].replace(0, np.nan)).fillna(0.0)
    return out[["log_pack_of","log_value_num","log_value_per_pack",
                "txt_len","txt_words","txt_digits","txt_upper",
                "txt_digit_ratio","txt_upper_ratio","unit"]]

# --- SMAPE Metric ---
def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1.0
    return 100.0 * np.mean(np.abs(y_true - y_pred) / denom)

# --- Regressor Builder ---
def make_regressor():
    if USE_XGB:
        return "XGBoost", XGBRegressor(
            n_estimators=800, learning_rate=0.05, max_depth=7,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, tree_method="hist", n_jobs=-1
        )
    else:
        return "HGBRegressor", HistGradientBoostingRegressor(
            max_depth=10, learning_rate=0.06, max_iter=800,
            l2_regularization=0.0, random_state=42
        )

# --- Main Pipeline ---
def main():
    np.random.seed(42)
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    train["catalog_content"].fillna("", inplace=True)
    test["catalog_content"].fillna("", inplace=True)
    
    # --- Target Log Transform (stabilizes regression) ---
    y = np.log1p(train["price"].astype(float))
    
    # --- Structured Features ---
    Xs_tr = build_structured(train)
    Xs_te = build_structured(test)
    
    # --- Text Features: smaller TF-IDF + SVD for speed ---
    word_tfidf = TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                 ngram_range=(1,2), min_df=5, max_features=50_000,
                                 sublinear_tf=True, stop_words="english")
    char_tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5),
                                 min_df=5, max_features=25_000, lowercase=True)
    text_pipe = Pipeline([
        ("tfidf", ColumnTransformer([
            ("word", word_tfidf, "catalog_content"),
            ("char", char_tfidf, "catalog_content")
        ], remainder="drop")),
        ("svd", TruncatedSVD(n_components=150, random_state=42))
    ])
    
    # --- Structured pipeline ---
    num_cols = ["log_pack_of","log_value_num","log_value_per_pack",
                "txt_len","txt_words","txt_digits","txt_upper",
                "txt_digit_ratio","txt_upper_ratio"]
    cat_cols = ["unit"]
    struct_pipe = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat_cols)
    ], remainder="drop")
    
    # --- Combined Features ---
    features = ColumnTransformer([
        ("text", text_pipe, ["catalog_content"]),
        ("struct", struct_pipe, num_cols + cat_cols)
    ], remainder="drop")
    
    # --- Final pipeline ---
    model_name, regressor = make_regressor()
    pipe = Pipeline([("features", features), ("model", regressor)])
    
    # --- Repeated KFold CV (fewer folds for speed) ---
    n_splits, n_repeats = 3, 1
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    rows: list[Dict[str, Any]] = []
    
    for run, (tr, va) in enumerate(rkf.split(train), start=1):
        X_tr = pd.concat([train[["catalog_content"]].iloc[tr], Xs_tr.iloc[tr]], axis=1)
        X_va = pd.concat([train[["catalog_content"]].iloc[va], Xs_tr.iloc[va]], axis=1)
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        pipe.fit(X_tr, y_tr)
        pred_va = np.clip(np.expm1(pipe.predict(X_va)), 0.01, None)  # invert log
        rows.append(dict(Model=model_name, Run=run,
                         SMAPE=smape(np.expm1(y_va), pred_va),
                         RMSE=math.sqrt(mean_squared_error(np.expm1(y_va), pred_va)),
                         MAE=mean_absolute_error(np.expm1(y_va), pred_va),
                         R2=r2_score(np.expm1(y_va), pred_va)))
    
    df_cv = pd.DataFrame(rows)
    df_cv.to_csv(CV_LOG_CSV, index=False)
    
    # --- Train on full data ---
    X_full = pd.concat([train[["catalog_content"]], Xs_tr], axis=1)
    pipe.fit(X_full, y)
    
    X_te = pd.concat([test[["catalog_content"]], Xs_te], axis=1)
    pred_test = np.clip(np.expm1(pipe.predict(X_te)), 0.01, None)
    
    # --- Save submission ---
    pd.DataFrame({"sample_id": test["sample_id"], "price": pred_test.astype(float)}).to_csv(OUT_CSV, index=False)
    print(f"Predictions saved to {OUT_CSV}")

if __name__ == "__main__":
    main()