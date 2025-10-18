import os, re, math, time, logging, random, pickle, json, warnings
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.utils import Bunch

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

SPEED_MODE = os.environ.get("SPEED_MODE", "fast").lower()
DO_QUICK_CV = os.environ.get("DO_QUICK_CV", "0") == "1"
MAX_CHARS = int(os.environ.get("MAX_CHARS", "600"))

SEED = int(os.environ.get("SEED", "42"))
random.seed(SEED); np.random.seed(SEED)

WORD_FEATS = int(os.environ.get("WORD_FEATS", str(2**15)))
TFIDF_FEATS = int(os.environ.get("TFIDF_FEATS", str(2**16)))
NGRAMS = (1, 2)

XGB_PARAMS = dict(
    n_estimators=900,
    learning_rate=0.07,
    max_depth=8,
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

_pack_re = re.compile(r"[Pp]ack\s*of\s*(\d+)")
_val_re  = re.compile(r"^Value:\s*([\-\+]?\d+(?:\.\d+)?)\s*$", re.M)
_unit_re = re.compile(r"^Unit:\s*([A-Za-z %/]+)\s*$", re.M)

UNIT_MAP = {
    "gram": ("g", 1.0), "grams": ("g", 1.0), "g": ("g", 1.0),
    "kilogram": ("g", 1000.0), "kg": ("g", 1000.0),
    "ounce": ("g", 28.3495), "ounces": ("g", 28.3495), "oz": ("g", 28.3495), "Oz": ("g", 28.3495), "Ounce": ("g", 28.3495),
    "pound": ("g", 453.592), "Pound": ("g", 453.592), "lb": ("g", 453.592), "LB": ("g", 453.592),
    "Fl Oz": ("ml", 29.5735), "fl oz": ("ml", 29.5735), "FL Oz": ("ml", 29.5735),
    "Fluid Ounce": ("ml", 29.5735), "Fluid Ounces": ("ml", 29.5735), "fluid ounces": ("ml", 29.5735),
    "Liter": ("ml", 1000.0), "Liters": ("ml", 1000.0), "l": ("ml", 1000.0), "L": ("ml", 1000.0),
    "milliliter": ("ml", 1.0), "milliliters": ("ml", 1.0), "ml": ("ml", 1.0), "mL": ("ml", 1.0),
    "Count": ("count", 1.0), "count": ("count", 1.0), "Each": ("count", 1.0), "ct": ("count", 1.0),
    "None": ("UNK", 1.0), "UNK": ("UNK", 1.0), "Pack": ("count", 1.0),
}

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

def _norm_unit(unit: str) -> Tuple[str, float]:
    """Return (base_unit, multiplier) to convert to 'g', 'ml', or 'count' when possible."""
    if unit in UNIT_MAP:
        return UNIT_MAP[unit]
    key = unit.strip()
    low = key.lower().replace(".", "").replace("fluid", "").strip()
    if low in UNIT_MAP:
        return UNIT_MAP[low]
    return ("UNK", 1.0)

def _prepare_text_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.slice(0, MAX_CHARS)

def build_structured(df: pd.DataFrame) -> pd.DataFrame:
    s = pd.Series(df["catalog_content"].fillna(""), index=df.index)

    out = pd.DataFrame(index=df.index)
    out["pack_of"] = s.apply(_extract_pack_of).astype(float)
    v_u = s.apply(_extract_value_unit)
    out["value_num"] = v_u.apply(lambda t: t[0]).astype(float)
    out["unit_raw"] = v_u.apply(lambda t: t[1])

    nu = out["unit_raw"].apply(_norm_unit)
    out["unit_base"] = nu.apply(lambda t: t[0])
    out["unit_mult"] = nu.apply(lambda t: t[1]).astype(float)

    out["value_base"] = out["value_num"] * out["unit_mult"]
    out["value_per_pack_base"] = out["value_base"] / out["pack_of"].replace(0, np.nan)

    out["log_pack_of"] = np.log1p(out["pack_of"])
    out["log_value_base"] = np.log1p(out["value_base"].fillna(0.0))
    out["log_value_per_pack_base"] = np.log1p(out["value_per_pack_base"].replace([np.inf, -np.inf], np.nan).fillna(0))

    s_len = s.str.len().astype(float)
    out["txt_len"] = s_len
    out["txt_words"] = s.str.split().apply(len).astype(float)
    out["txt_digits"] = s.str.count(r"\d").astype(float)
    out["txt_upper"] = s.str.count(r"[A-Z]").astype(float)
    out["txt_digit_ratio"] = (out["txt_digits"] / s_len.replace(0, np.nan)).fillna(0.0)
    out["txt_upper_ratio"] = (out["txt_upper"] / s_len.replace(0, np.nan)).fillna(0.0)

    il = df.get("image_link", pd.Series("", index=df.index)).fillna("")
    out["img_ext"] = il.str.extract(r"\.([A-Za-z0-9]+)$", expand=False).str.lower().fillna("unk")
    out["img_has_dim"] = il.str.contains(r"\d{2,4}x\d{2,4}", regex=True).astype(int)
    out["img_host"] = il.str.extract(r"https?://([^/]+)/", expand=False).str.lower().fillna("unk")
    out["img_len"] = il.str.len().astype(float)

    return out

def _safe_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse=True)

def _sanitize_xgb(params: Dict[str, Any]) -> Dict[str, Any]:
    if not _HAS_XGB:
        return {}
    p = dict(params)
    try:
        _ = XGBRegressor(**p)
        return p
    except TypeError:
        p.pop("verbosity", None)
    return p

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1.0
    return 100.0 * np.mean(np.abs(y_true - y_pred) / denom)

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

def _fit_predict_sparse_model(model, Xtr, ytr, Xva=None, yva=None) -> Bunch:
    info = {}
    try:
        if hasattr(model, "set_params"):
            if _HAS_XGB and isinstance(model, XGBRegressor) and Xva is not None and yva is not None:
                try:
                    model.set_params(early_stopping_rounds=50)
                    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
                    info["early_stopping_used"] = True
                except TypeError:
                    pars = {k: v for k, v in model.get_params().items() if k != "early_stopping_rounds"}
                    model.set_params(**pars)
                    model.fit(Xtr, ytr)
                    info["early_stopping_used"] = False
            else:
                model.fit(Xtr, ytr)
        else:
            model.fit(Xtr, ytr)
    except TypeError:
        model.fit(Xtr, ytr)
        info["early_stopping_used"] = False
    return Bunch(model=model, info=info)

def main():
    t0 = time.time()
    log.info("Reading dataset from %s", DATASET_FOLDER)

    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)

    assert "price" in train.columns, "Missing target column 'price'"
    assert "catalog_content" in train.columns and "catalog_content" in test.columns, "Missing 'catalog_content'"

    y = train["price"].astype(float)
    y_log = np.log1p(y)

    Xs_tr = build_structured(train)
    Xs_te = build_structured(test)

    num_cols = [
        "log_pack_of","log_value_base","log_value_per_pack_base",
        "txt_len","txt_words","txt_digits","txt_upper",
        "txt_digit_ratio","txt_upper_ratio","img_len"
    ]
    cat_cols = ["unit_base","img_ext","img_host"]

    num_scaler = StandardScaler(with_mean=False)
    num_scaler.fit(Xs_tr[num_cols].fillna(0.0))

    ohe = _safe_ohe()
    ohe.fit(Xs_tr[cat_cols].astype(str))

    tr_text = _prepare_text_series(train["catalog_content"])
    te_text = _prepare_text_series(test["catalog_content"])

    tag = f"feats_w{WORD_FEATS}_t{TFIDF_FEATS}_mc{MAX_CHARS}_v2"
    tr_path, te_path, meta_path = _cache_paths(tag)

    if all(os.path.exists(p) for p in [tr_path, te_path, meta_path]):
        log.info("Loading cached features …")
        X_tr_sparse = _load_sparse(tr_path)
        X_te_sparse = _load_sparse(te_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        log.info("Cache info: %s", meta)
    else:
        log.info("Vectorizing text … (Hashing + TF-IDF)")
        hv = HashingVectorizer(
            n_features=WORD_FEATS, alternate_sign=False, norm="l2",
            ngram_range=NGRAMS, lowercase=True, strip_accents="unicode",
            stop_words="english",
        )
        tv = TfidfVectorizer(
            max_features=TFIDF_FEATS, ngram_range=NGRAMS, lowercase=True,
            strip_accents="unicode", stop_words="english", dtype=np.float32,
        )

        W_tr = hv.transform(tr_text)
        W_te = hv.transform(te_text)

        T_tr = tv.fit_transform(tr_text)
        T_te = tv.transform(te_text)

        N_tr = num_scaler.transform(Xs_tr[num_cols].fillna(0.0))
        U_tr = ohe.transform(Xs_tr[cat_cols].astype(str))
        N_te = num_scaler.transform(Xs_te[num_cols].fillna(0.0))
        U_te = ohe.transform(Xs_te[cat_cols].astype(str))

        X_tr_sparse = sparse.hstack([W_tr, T_tr, N_tr, U_tr], format="csr")
        X_te_sparse = sparse.hstack([W_te, T_te, N_te, U_te], format="csr")

        _save_sparse(tr_path, X_tr_sparse)
        _save_sparse(te_path, X_te_sparse)
        with open(meta_path, "wb") as f:
            pickle.dump(
                {"WORD_FEATS": WORD_FEATS, "TFIDF_FEATS": TFIDF_FEATS, "MAX_CHARS": MAX_CHARS},
                f
            )
        log.info("Cached features to dataset/")

    if DO_QUICK_CV:
        kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
        rows = []
        for run, (tr_idx, va_idx) in enumerate(kf.split(X_tr_sparse), 1):
            Fe_tr, Fe_va = X_tr_sparse[tr_idx], X_tr_sparse[va_idx]
            ytr_log, yva_log = y_log.iloc[tr_idx], y_log.iloc[va_idx]
            yva = np.expm1(yva_log)

            models = []
            models.append(("ridge", Ridge(alpha=1.0, random_state=SEED)))
            if _HAS_XGB and SPEED_MODE != "ultra":
                params = _sanitize_xgb(XGB_PARAMS)
                models.append(("xgb", XGBRegressor(**params)))

            preds = []
            for name, m in models:
                bundle = _fit_predict_sparse_model(m, Fe_tr, ytr_log, Fe_va, yva_log)
                p = np.expm1(bundle.model.predict(Fe_va))
                preds.append(p)

            pred_va = np.mean(preds, axis=0)
            pred_va = np.clip(pred_va, 0.0, None)

            rows.append(dict(
                Run=run,
                SMAPE=smape(yva, pred_va),
                RMSE=math.sqrt(mean_squared_error(yva, pred_va)),
                MAE=mean_absolute_error(yva, pred_va),
                R2=r2_score(yva, pred_va),
            ))
            log.info("CV %d | SMAPE=%.3f RMSE=%.3f MAE=%.3f R2=%.4f",
                     run, rows[-1]["SMAPE"], rows[-1]["RMSE"], rows[-1]["MAE"], rows[-1]["R2"])
        pd.DataFrame(rows).to_csv(CV_LOG_CSV, index=False)
        log.info("Saved CV log to %s", CV_LOG_CSV)

    ss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    tr_final, va_final = next(ss.split(X_tr_sparse))

    Fe_tr_f, Fe_va_f = X_tr_sparse[tr_final], X_tr_sparse[va_final]
    y_tr_f, y_va_f = y_log.iloc[tr_final], y_log.iloc[va_final]

    models = []
    models.append(("ridge", Ridge(alpha=1.0, random_state=SEED)))
    if _HAS_XGB and SPEED_MODE != "ultra":
        params = _sanitize_xgb(XGB_PARAMS)
        models.append(("xgb", XGBRegressor(**params)))

    preds_va = []
    preds_te = []
    for name, m in models:
        bundle = _fit_predict_sparse_model(m, Fe_tr_f, y_tr_f, Fe_va_f, y_va_f)
        pv = np.expm1(bundle.model.predict(Fe_va_f))
        pt = np.expm1(bundle.model.predict(X_te_sparse))
        preds_va.append(pv); preds_te.append(pt)
        log.info("Model %-5s | early_stopping=%s", name, getattr(bundle, "info", {}).get("early_stopping_used", False))

    pred_va = np.mean(preds_va, axis=0)
    pred_test = np.mean(preds_te, axis=0)

    pred_va = np.clip(pred_va, 0.01, np.quantile(pred_va, 0.997))
    pred_test = np.clip(pred_test, 0.01, np.quantile(pred_test, 0.997))

    yva = np.expm1(y_va_f)
    log.info("Holdout | SMAPE=%.3f RMSE=%.3f MAE=%.3f R2=%.4f",
             smape(yva, pred_va),
             math.sqrt(mean_squared_error(yva, pred_va)),
             mean_absolute_error(yva, pred_va),
             r2_score(yva, pred_va))

    out = pd.DataFrame({"sample_id": test["sample_id"], "price": pred_test.astype(float)})
    out.to_csv(OUT_CSV, index=False)

    log.info("Saved predictions to %s", OUT_CSV)
    log.info("Total runtime: %.2fs", time.time() - t0)

if __name__ == "__main__":
    main()
