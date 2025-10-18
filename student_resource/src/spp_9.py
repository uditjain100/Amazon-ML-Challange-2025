import os, re, math, time, logging, random, pickle, json, warnings
from typing import Tuple, Dict, Any, Optional, List

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

# ----------------------------- config / env -----------------------------
SPEED_MODE = os.environ.get("SPEED_MODE", "fast").lower()
DO_QUICK_CV = os.environ.get("DO_QUICK_CV", "0") == "1"
MAX_CHARS = int(os.environ.get("MAX_CHARS", "600"))

SEED = int(os.environ.get("SEED", "42"))
random.seed(SEED); np.random.seed(SEED)

WORD_FEATS = int(os.environ.get("WORD_FEATS", str(2**15)))
TFIDF_FEATS = int(os.environ.get("TFIDF_FEATS", str(120_000)))
NGRAMS = (1, 3)  # richer word context

XGB_PARAMS = dict(
    n_estimators=1500,
    learning_rate=0.05,
    max_depth=9,
    subsample=0.85,
    colsample_bytree=0.75,
    min_child_weight=2.5,
    reg_alpha=0.1,
    reg_lambda=1.1,
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
MODEL_PKL = os.path.join(DATASET_FOLDER, "final_model.pkl")

# ----------------------------- regex utils -----------------------------
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

# ----------------------------- feature engineering -----------------------------
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

    # text stats (richer)
    s_len = s.str.len().astype(float)
    words = s.str.split()
    out["txt_len"] = s_len
    out["txt_words"] = words.apply(len).astype(float)
    out["txt_digits"] = s.str.count(r"\d").astype(float)
    out["txt_upper"] = s.str.count(r"[A-Z]").astype(float)
    out["txt_punct"] = s.str.count(r"[^\w\s]").astype(float)
    out["has_number"] = (out["txt_digits"] > 0).astype(int)
    out["avg_word_len"] = words.apply(lambda ws: float(np.mean([len(w) for w in ws])) if len(ws) else 0.0)
    out["unique_words"] = words.apply(lambda ws: float(len(set(ws))) if len(ws) else 0.0)
    out["txt_digit_ratio"] = (out["txt_digits"] / s_len.replace(0, np.nan)).fillna(0.0)
    out["txt_upper_ratio"] = (out["txt_upper"] / s_len.replace(0, np.nan)).fillna(0.0)
    out["txt_punct_ratio"] = (out["txt_punct"] / s_len.replace(0, np.nan)).fillna(0.0)
    out["value_log_ratio"] = np.log1p(out["value_base"] / out["value_num"].replace(0, np.nan)).fillna(0.0)

    # image_link derived (optional)
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

# ----------------------------- metrics -----------------------------
def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1.0
    return 100.0 * np.mean(np.abs(y_true - y_pred) / denom)

# ----------------------------- caching utils -----------------------------
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

# ----------------------------- training utils -----------------------------
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

def _inverse_rmse_weights(y_true: np.ndarray, preds: List[np.ndarray]) -> np.ndarray:
    """Compute blend weights proportional to 1/RMSE^2, normalized."""
    rmses = [math.sqrt(mean_squared_error(y_true, p)) + 1e-9 for p in preds]
    inv = np.array([1.0/(r*r) for r in rmses], dtype=float)
    w = inv / inv.sum()
    return w

# ----------------------------- main -----------------------------
def main():
    t0 = time.time()
    log.info("Reading dataset from %s", DATASET_FOLDER)

    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)

    # quick schema log
    log.info("Train shape=%s | Test shape=%s", train.shape, test.shape)
    log.info("Train cols=%s", list(train.columns))
    log.info("Test  cols=%s", list(test.columns))

    assert "price" in train.columns, "Missing target column 'price'"
    assert "catalog_content" in train.columns and "catalog_content" in test.columns, "Missing 'catalog_content'"

    y = train["price"].astype(float)
    y_log = np.log1p(y)

    # structured
    Xs_tr = build_structured(train)
    Xs_te = build_structured(test)

    # numerical / categorical partitions
    num_cols = [
        "log_pack_of","log_value_base","log_value_per_pack_base",
        "txt_len","txt_words","txt_digits","txt_upper","txt_punct",
        "txt_digit_ratio","txt_upper_ratio","txt_punct_ratio",
        "avg_word_len","unique_words","has_number","value_log_ratio","img_len"
    ]
    cat_cols = ["unit_base","img_ext","img_host"]

    num_scaler = StandardScaler(with_mean=False)
    num_scaler.fit(Xs_tr[num_cols].fillna(0.0))

    ohe = _safe_ohe()
    ohe.fit(Xs_tr[cat_cols].astype(str))

    tr_text = _prepare_text_series(train["catalog_content"])
    te_text = _prepare_text_series(test["catalog_content"])

    tag = f"feats_w{WORD_FEATS}_t{TFIDF_FEATS}_mc{MAX_CHARS}_ngr{NGRAMS[0]}{NGRAMS[1]}_v3"
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
            ngram_range=(1, 2), lowercase=True, strip_accents="unicode",
            stop_words="english",
        )
        tv = TfidfVectorizer(
            max_features=TFIDF_FEATS, ngram_range=NGRAMS, lowercase=True,
            strip_accents="unicode", stop_words="english", dtype=np.float32,
            sublinear_tf=True, analyzer="word"
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
                {"WORD_FEATS": WORD_FEATS, "TFIDF_FEATS": TFIDF_FEATS, "MAX_CHARS": MAX_CHARS, "NGRAMS": NGRAMS},
                f
            )
        log.info("Cached features to dataset/")

    # ----------------- model reuse (skip training) -----------------
    if os.path.exists(MODEL_PKL):
        log.info("Found cached model at %s — reusing it for test predictions.", MODEL_PKL)
        with open(MODEL_PKL, "rb") as f:
            model_bundle = pickle.load(f)
        models = model_bundle["models"]          # List[Tuple[name, estimator]]
        weights = model_bundle["weights"]        # np.ndarray
        log.info("Loaded models: %s | weights=%s", [n for n,_ in models], weights)

        preds_te = []
        for name, m in models:
            pt = np.expm1(m.predict(X_te_sparse))
            preds_te.append(pt)
        pred_test = np.average(np.stack(preds_te, axis=1), axis=1, weights=weights)
        pred_test = np.clip(pred_test, 0.01, np.quantile(pred_test, 0.997))

        out = pd.DataFrame({"sample_id": test["sample_id"], "price": pred_test.astype(float)})
        out.to_csv(OUT_CSV, index=False)
        log.info("Saved predictions to %s", OUT_CSV)
        log.info("Total runtime: %.2fs", time.time() - t0)
        return

    # ----------------- CV (optional quick) -----------------
    if DO_QUICK_CV:
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        rows = []
        for run, (tr_idx, va_idx) in enumerate(kf.split(X_tr_sparse), 1):
            Fe_tr, Fe_va = X_tr_sparse[tr_idx], X_tr_sparse[va_idx]
            ytr_log, yva_log = y_log.iloc[tr_idx], y_log.iloc[va_idx]
            yva = np.expm1(yva_log)

            models_cv = []
            models_cv.append(("ridge", Ridge(alpha=0.3, solver="lsqr", random_state=SEED)))
            if _HAS_XGB and SPEED_MODE != "ultra":
                params = _sanitize_xgb(XGB_PARAMS)
                models_cv.append(("xgb", XGBRegressor(**params)))

            preds = []
            indiv = {}
            for name, m in models_cv:
                bundle = _fit_predict_sparse_model(m, Fe_tr, ytr_log, Fe_va, yva_log)
                p = np.expm1(bundle.model.predict(Fe_va))
                preds.append(p)
                indiv[name] = dict(
                    SMAPE=smape(yva, p),
                    RMSE=math.sqrt(mean_squared_error(yva, p)),
                    MAE=mean_absolute_error(yva, p),
                    R2=r2_score(yva, p),
                )
                log.info("CV %d | %-5s | early_stopping=%s | SMAPE=%.3f RMSE=%.3f MAE=%.3f R2=%.4f",
                         run, name, getattr(bundle, "info", {}).get("early_stopping_used", False),
                         indiv[name]["SMAPE"], indiv[name]["RMSE"], indiv[name]["MAE"], indiv[name]["R2"])

            # inverse-RMSE weighting for blend
            w = _inverse_rmse_weights(yva, preds)
            pred_va = np.average(np.stack(preds, axis=1), axis=1, weights=w)
            pred_va = np.clip(pred_va, 0.01, np.quantile(pred_va, 0.997))

            rows.append(dict(
                Run=run,
                BlendWeights=json.dumps(w.tolist()),
                SMAPE=smape(yva, pred_va),
                RMSE=math.sqrt(mean_squared_error(yva, pred_va)),
                MAE=mean_absolute_error(yva, pred_va),
                R2=r2_score(yva, pred_va),
                **{f"{k}_RMSE": v["RMSE"] for k,v in indiv.items()},
                **{f"{k}_SMAPE": v["SMAPE"] for k,v in indiv.items()},
            ))
            log.info("CV %d | BLEND | w=%s | SMAPE=%.3f RMSE=%.3f MAE=%.3f R2=%.4f",
                     run, np.round(w,4).tolist(), rows[-1]["SMAPE"], rows[-1]["RMSE"], rows[-1]["MAE"], rows[-1]["R2"])
        pd.DataFrame(rows).to_csv(CV_LOG_CSV, index=False)
        log.info("Saved CV log to %s", CV_LOG_CSV)

    # ----------------- final split + training -----------------
    ss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    tr_final, va_final = next(ss.split(X_tr_sparse))

    Fe_tr_f, Fe_va_f = X_tr_sparse[tr_final], X_tr_sparse[va_final]
    y_tr_f, y_va_f = y_log.iloc[tr_final], y_log.iloc[va_final]
    yva = np.expm1(y_va_f)

    models: List[Tuple[str, Any]] = []
    models.append(("ridge", Ridge(alpha=0.3, solver="lsqr", random_state=SEED)))
    if _HAS_XGB and SPEED_MODE != "ultra":
        params = _sanitize_xgb(XGB_PARAMS)
        models.append(("xgb", XGBRegressor(**params)))

    preds_va = []
    preds_te = []
    indiv_logs = []
    for name, m in models:
        bundle = _fit_predict_sparse_model(m, Fe_tr_f, y_tr_f, Fe_va_f, y_va_f)
        pv = np.expm1(bundle.model.predict(Fe_va_f))
        pt = np.expm1(bundle.model.predict(X_te_sparse))
        preds_va.append(pv); preds_te.append(pt)
        log.info("FINAL %-5s | early_stopping=%s", name, getattr(bundle, "info", {}).get("early_stopping_used", False))
        indiv_logs.append(dict(
            name=name,
            SMAPE=smape(yva, pv),
            RMSE=math.sqrt(mean_squared_error(yva, pv)),
            MAE=mean_absolute_error(yva, pv),
            R2=r2_score(yva, pv),
        ))
        log.info("FINAL %-5s | SMAPE=%.3f RMSE=%.3f MAE=%.3f R2=%.4f",
                 name, indiv_logs[-1]["SMAPE"], indiv_logs[-1]["RMSE"], indiv_logs[-1]["MAE"], indiv_logs[-1]["R2"])

    # data-driven weights (inverse RMSE) on holdout
    w = _inverse_rmse_weights(yva, preds_va)
    log.info("Blend weights (inverse-RMSE): %s", np.round(w, 4).tolist())

    pred_va = np.average(np.stack(preds_va, axis=1), axis=1, weights=w)
    pred_test = np.average(np.stack(preds_te, axis=1), axis=1, weights=w)

    # gentle clipping
    pred_va = np.clip(pred_va, 0.01, np.quantile(pred_va, 0.997))
    pred_test = np.clip(pred_test, 0.01, np.quantile(pred_test, 0.997))

    # holdout report
    log.info("Holdout BLEND | SMAPE=%.3f RMSE=%.3f MAE=%.3f R2=%.4f",
             smape(yva, pred_va),
             math.sqrt(mean_squared_error(yva, pred_va)),
             mean_absolute_error(yva, pred_va),
             r2_score(yva, pred_va))

    # save submission
    out = pd.DataFrame({"sample_id": test["sample_id"], "price": pred_test.astype(float)})
    out.to_csv(OUT_CSV, index=False)
    log.info("Saved predictions to %s", OUT_CSV)

    # --------------- persist final models for reuse ---------------
    try:
        with open(MODEL_PKL, "wb") as f:
            pickle.dump({"models": models, "weights": w}, f)
        log.info("Saved trained models & weights to %s", MODEL_PKL)
    except Exception as e:
        log.warning("Could not save model pickle: %s", e)

    log.info("Total runtime: %.2fs", time.time() - t0)

if __name__ == "__main__":
    main()
