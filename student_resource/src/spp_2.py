import os, re, math, time, logging, random, pickle
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

from xgboost import XGBRegressor

SPEED_MODE = os.environ.get("SPEED_MODE", "fast").lower()

DO_QUICK_CV = False

MAX_CHARS = 500

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("runner")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(BASE_DIR, "dataset")

TRAIN_CSV = os.path.join(DATASET_FOLDER, "train.csv")
TEST_CSV  = os.path.join(DATASET_FOLDER, "test.csv")
OUT_CSV   = os.path.join(DATASET_FOLDER, "test_out.csv")
CV_LOG_CSV= os.path.join(DATASET_FOLDER, "cv_metrics_log.csv")

SEED = 42
random.seed(SEED); np.random.seed(SEED)

WORD_FEATS = 2**15  

XGB_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.06,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.7,
    min_child_weight=3.0,
    reg_alpha=0.0,
    reg_lambda=1.3,
    tree_method="hist",
    random_state=SEED,
    n_jobs=-1,
    verbosity=0,
    eval_metric="rmse",
)

_pack_re = re.compile(r"[Pp]ack\s*of\s*(\d+)")
_val_re  = re.compile(r"^Value:\s*([\-+]?\d+(\.\d+)?)\s*$", re.M)
_unit_re = re.compile(r"^Unit:\s*([A-Za-z %/]+)\s*$", re.M)

def _extract_pack_of(s: str) -> float:
    if not isinstance(s, str): return 1.0
    m = _pack_re.search(s)
    return float(m.group(1)) if m else 1.0

def _extract_value_unit(s: str) -> Tuple[float, str]:
    if not isinstance(s, str): return (np.nan, "UNK")
    vm = _val_re.search(s); um = _unit_re.search(s)
    v = float(vm.group(1)) if vm else np.nan
    u = um.group(1).strip() if um else "UNK"
    return (v, u)

def build_structured(df: pd.DataFrame) -> pd.DataFrame:
    s = pd.Series(df["catalog_content"].fillna(""), index=df.index)

    out = pd.DataFrame(index=df.index)
    out["pack_of"] = s.apply(_extract_pack_of).astype(float)
    vu = s.apply(_extract_value_unit)
    out["value_num"] = vu.apply(lambda t: t[0]).astype(float)
    out["unit"] = vu.apply(lambda t: t[1])

    out["log_pack_of"] = np.log1p(out["pack_of"])
    out["log_value_num"] = np.log1p(out["value_num"].fillna(0.0))

    vpp = out["value_num"] / out["pack_of"].replace(0, np.nan)
    out["value_per_pack"] = vpp
    out["log_value_per_pack"] = np.log1p(vpp.replace([np.inf, -np.inf], np.nan).fillna(0))

    out["txt_len"] = s.str.len().astype(float)
    out["txt_words"] = s.str.split().apply(len).astype(float)
    out["txt_digits"] = s.str.count(r"\d").astype(float)
    out["txt_upper"] = s.str.count(r"[A-Z]").astype(float)
    out["txt_digit_ratio"] = (out["txt_digits"] / out["txt_len"].replace(0, np.nan)).fillna(0.0)
    out["txt_upper_ratio"] = (out["txt_upper"] / out["txt_len"].replace(0, np.nan)).fillna(0.0)

    return out[
        [
            "log_pack_of","log_value_num","log_value_per_pack",
            "txt_len","txt_words","txt_digits","txt_upper",
            "txt_digit_ratio","txt_upper_ratio","unit"
        ]
    ]

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1.0
    return 100.0 * np.mean(np.abs(y_true - y_pred) / denom)

def _safe_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse=True)

def _sanitize_xgb(params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params)
    try:
        _ = XGBRegressor(**p)
        return p
    except TypeError:
        p.pop("verbosity", None)
    return p

def _prepare_text_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.slice(0, MAX_CHARS)

def _cache_paths(tag: str):
    return (
        os.path.join(DATASET_FOLDER, f"{tag}_Xtr.npz"),
        os.path.join(DATASET_FOLDER, f"{tag}_Xte.npz"),
        os.path.join(DATASET_FOLDER, f"{tag}_meta.pkl"),
    )

def _save_sparse(path, mat):
    sparse.save_npz(path, mat, compressed=True)

def _load_sparse(path):
    return sparse.load_npz(path)

def main():
    t0 = time.time()
    log.info("Reading dataset from %s", DATASET_FOLDER)

    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)

    assert "price" in train.columns, "Missing target column 'price'"
    assert "catalog_content" in train.columns and "catalog_content" in test.columns, "Missing 'catalog_content'"

    y = train["price"].astype(float)
    y_log = np.log1p(y)

    # Structured features
    Xs_tr = build_structured(train)
    Xs_te = build_structured(test)

    num_cols = [
        "log_pack_of","log_value_num","log_value_per_pack",
        "txt_len","txt_words","txt_digits","txt_upper",
        "txt_digit_ratio","txt_upper_ratio"
    ]
    cat_col = ["unit"]

    num_scaler = StandardScaler(with_mean=False)
    num_scaler.fit(Xs_tr[num_cols].fillna(0.0))

    ohe = _safe_ohe()
    ohe.fit(Xs_tr[cat_col])

    tag = f"hash_w{WORD_FEATS}_mc{MAX_CHARS}_v1"
    tr_path, te_path, meta_path = _cache_paths(tag)

    if all(os.path.exists(p) for p in [tr_path, te_path, meta_path]):
        log.info("Loading cached hashed features …")
        X_tr_sparse = _load_sparse(tr_path)
        X_te_sparse = _load_sparse(te_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        log.info("Cache info: %s", meta)
    else:
        log.info("Hashing text (single pass, word-only) …")
        hv_word = HashingVectorizer(
            n_features=WORD_FEATS,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1,2),
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
        )

        tr_text = _prepare_text_series(train["catalog_content"])
        te_text = _prepare_text_series(test["catalog_content"])

        W_tr = hv_word.transform(tr_text)
        W_te = hv_word.transform(te_text)

        N_tr = num_scaler.transform(Xs_tr[num_cols].fillna(0.0))
        U_tr = ohe.transform(Xs_tr[cat_col])
        N_te = num_scaler.transform(Xs_te[num_cols].fillna(0.0))
        U_te = ohe.transform(Xs_te[cat_col])

        X_tr_sparse = sparse.hstack([W_tr, N_tr, U_tr], format="csr")
        X_te_sparse = sparse.hstack([W_te, N_te, U_te], format="csr")

        _save_sparse(tr_path, X_tr_sparse)
        _save_sparse(te_path, X_te_sparse)
        with open(meta_path, "wb") as f:
            pickle.dump({"WORD_FEATS": WORD_FEATS, "MAX_CHARS": MAX_CHARS}, f)
        log.info("Cached features to dataset/")

    if DO_QUICK_CV:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
        rows = []
        for run, (tr_idx, va_idx) in enumerate(kf.split(X_tr_sparse), 1):
            Fe_tr, Fe_va = X_tr_sparse[tr_idx], X_tr_sparse[va_idx]
            ytr_log, yva_log = y_log.iloc[tr_idx], y_log.iloc[va_idx]
            yva = np.expm1(yva_log)

            if SPEED_MODE == "ultra":
                model = Ridge(alpha=1.0, random_state=SEED)
            else:
                params = _sanitize_xgb(XGB_PARAMS)
                model = XGBRegressor(**params)
            model.fit(Fe_tr, ytr_log)

            pred_va = np.expm1(model.predict(Fe_va))
            pred_va = np.clip(pred_va, 0.0, None)
            rows.append(dict(
                Run=run,
                SMAPE=smape(yva, pred_va),
                RMSE=math.sqrt(mean_squared_error(yva, pred_va)),
                MAE=mean_absolute_error(yva, pred_va),
                R2=r2_score(yva, pred_va),
            ))
            log.info("CV %d | SMAPE=%.3f RMSE=%.3f R2=%.4f",
                     run, rows[-1]["SMAPE"], rows[-1]["RMSE"], rows[-1]["R2"])
        pd.DataFrame(rows).to_csv(CV_LOG_CSV, index=False)
        log.info("Saved CV log to %s", CV_LOG_CSV)

    # ----------------------------- final fit + predict -----------------------------
    ss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    tr_final, va_final = next(ss.split(X_tr_sparse))

    Fe_tr_f, Fe_va_f = X_tr_sparse[tr_final], X_tr_sparse[va_final]
    y_tr_f, y_va_f = y_log.iloc[tr_final], y_log.iloc[va_final]

    if SPEED_MODE == "ultra":
        final = Ridge(alpha=1.0, random_state=SEED)
    else:
        params = _sanitize_xgb(XGB_PARAMS)
        final = XGBRegressor(**params)

    t_fit = time.time()
    final.fit(Fe_tr_f, y_tr_f)
    log.info("Final fit done in %.2fs", time.time() - t_fit)

    pred_test = np.expm1(final.predict(X_te_sparse))
    pred_test = np.clip(pred_test, 0.01, None)

    out = pd.DataFrame({"sample_id": test["sample_id"], "price": pred_test.astype(float)})
    out.to_csv(OUT_CSV, index=False)

    log.info("Saved predictions to %s", OUT_CSV)
    log.info("Total runtime: %.2fs", time.time() - t0)

if __name__ == "__main__":
    main()