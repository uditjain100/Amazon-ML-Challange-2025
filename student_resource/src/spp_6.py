import os, re, math, time, logging, random, pickle, warnings
from typing import Tuple, Dict, Any
import numpy as np, pandas as pd
from scipy import sparse

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.utils import Bunch

warnings.filterwarnings("ignore")

_HAS_XGB = False
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

SPEED_MODE   = os.environ.get("SPEED_MODE", "fast").lower()   
MAX_CHARS    = int(os.environ.get("MAX_CHARS", "1200"))
WORD_TFIDF   = int(os.environ.get("WORD_TFIDF", str(1<<16)))  
CHAR_TFIDF   = int(os.environ.get("CHAR_TFIDF", str(1<<14)))  
USE_CHAR     = os.environ.get("USE_CHAR", "0") == "1"         
OOF_FOLDS    = int(os.environ.get("OOF_FOLDS", "5"))
SEED         = int(os.environ.get("SEED", "42"))
USE_XGB      = os.environ.get("USE_XGB", "1") == "1"
USE_SGD      = os.environ.get("USE_SGD", "1") == "1"
USE_RIDGE    = os.environ.get("USE_RIDGE", "0") == "1"        
OHE_MIN_FREQ = int(os.environ.get("OHE_MIN_FREQ", "50"))
IMG_HOST_TOPK= int(os.environ.get("IMG_HOST_TOPK", "500"))

random.seed(SEED); np.random.seed(SEED)

if SPEED_MODE == "ultra":
    MAX_CHARS  = min(MAX_CHARS, 800)
    WORD_TFIDF = min(WORD_TFIDF, 1<<15)
    CHAR_TFIDF = min(CHAR_TFIDF, 1<<13)
    OOF_FOLDS  = min(OOF_FOLDS, 3)
elif SPEED_MODE == "fast":
    MAX_CHARS  = min(MAX_CHARS, 1200)
    WORD_TFIDF = min(WORD_TFIDF, 1<<16)
    CHAR_TFIDF = min(CHAR_TFIDF, 1<<14)

try:
    import psutil
    total_gb = psutil.virtual_memory().total / (1024**3)
    if total_gb < 16:
        WORD_TFIDF = min(WORD_TFIDF, 1<<15)
        CHAR_TFIDF = min(CHAR_TFIDF, 1<<13)
        OOF_FOLDS  = min(OOF_FOLDS, 4)
        if os.environ.get("FORCE_HEAVY","0") != "1":
            USE_CHAR = False
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(BASE_DIR, "dataset")
TRAIN_CSV  = os.path.join(DATASET_FOLDER, "train.csv")
TEST_CSV   = os.path.join(DATASET_FOLDER, "test.csv")
OUT_CSV    = os.path.join(DATASET_FOLDER, "test_out.csv")
CV_LOG_CSV = os.path.join(DATASET_FOLDER, "cv_metrics_log.csv")
OOF_PKL    = os.path.join(DATASET_FOLDER, "oof_preds.pkl")

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("smart_pricing")

_pack_re = re.compile(r"[Pp]ack\s*of\s*(\d+)")
_val_re  = re.compile(r"^Value:\s*([\-+]?\d+(?:\.\d+)?)\s*$", re.M)
_unit_re = re.compile(r"^Unit:\s*([A-Za-z %/]+)\s*$", re.M)

UNIT_MAP = {
    "gram": ("g", 1.0), "grams": ("g", 1.0), "g": ("g", 1.0),
    "kilogram": ("g", 1000.0), "kg": ("g", 1000.0),
    "ounce": ("g", 28.3495), "ounces": ("g", 28.3495), "oz": ("g", 28.3495),
    "pound": ("g", 453.592), "lb": ("g", 453.592),
    "Fl Oz": ("ml", 29.5735), "fl oz": ("ml", 29.5735),
    "Fluid Ounce": ("ml", 29.5735), "Fluid Ounces": ("ml", 29.5735),
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
    if unit in UNIT_MAP: return UNIT_MAP[unit]
    key = str(unit).strip()
    low = key.lower().replace(".", "").replace("fluid", "").strip()
    return UNIT_MAP.get(low, ("UNK", 1.0))

def _prepare_text_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.slice(0, MAX_CHARS)

def _finite_or_none(a):
    a = np.asarray(a, dtype=float)
    if a.ndim != 1: a = a.ravel()
    bad = ~np.isfinite(a)
    if bad.any():
        a = a.copy()
        a[bad] = np.nan
    return a

def _nanmedian_fallback(a, default_val):
    a = np.asarray(a, dtype=float)
    if np.all(~np.isfinite(a)):
        return float(default_val)
    return float(np.nanmedian(a))

def _rowwise_nanmean(P):
    with np.errstate(invalid="ignore"):
        m = np.nanmean(P, axis=1)
    return m

def _model_path(tag: str, model_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_=-]", "_", tag)
    return os.path.join(DATASET_FOLDER, f"{safe}_{model_name}_model.pkl")

def _save_model(path: str, model) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)

def _load_model_if_exists(path: str):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                mdl = pickle.load(f)
            log.info("Loaded cached model: %s", os.path.basename(path))
            return mdl
        except Exception as e:
            log.warning("Failed to load cached model (%s). Will retrain. Error: %s", path, e)
    return None

def _clone_without_es(mdl):
    if hasattr(mdl, "get_params"):
        p = mdl.get_params().copy()
        p.pop("early_stopping_rounds", None)
        return mdl.__class__(**p)
    return mdl

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

    out["log_pack_of"] = np.log1p(out["pack_of"])
    out["log_value_base"] = np.log1p(out["value_base"].fillna(0.0))
    out["log_value_per_pack_base"] = np.log1p(out["value_per_pack_base"].replace([np.inf,-np.inf], np.nan).fillna(0.0))
    return out

def _safe_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=OHE_MIN_FREQ, sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=OHE_MIN_FREQ, sparse=True)

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1.0
    return 100.0 * np.mean(np.abs(y_true - y_pred) / denom)

def _sanitize_xgb(params: Dict[str, Any]) -> Dict[str, Any]:
    if not _HAS_XGB: return {}
    p = dict(params)
    p.pop("early_stopping_rounds", None)
    try:
        _ = XGBRegressor(**p); return p
    except TypeError:
        p.pop("verbosity", None)
    return p

XGB_PARAMS = dict(
    n_estimators=900, learning_rate=0.06, max_depth=8, subsample=0.8, colsample_bytree=0.7,
    min_child_weight=3.0, reg_alpha=0.0, reg_lambda=1.2, tree_method="hist",
    n_jobs=-1, random_state=SEED, verbosity=0, eval_metric="rmse",
)

def _cache_paths(tag: str):
    return (
        os.path.join(DATASET_FOLDER, f"{tag}_Xtr.npz"),
        os.path.join(DATASET_FOLDER, f"{tag}_Xte.npz"),
        os.path.join(DATASET_FOLDER, f"{tag}_meta.pkl"),
    )

def _save_sparse(path, mat): sparse.save_npz(path, mat, compressed=True)
def _load_sparse(path): return sparse.load_npz(path)

def build_features(train: pd.DataFrame, test: pd.DataFrame):
    Xs_tr = build_structured(train)
    Xs_te = build_structured(test)

    top_hosts = Xs_tr["img_host"].value_counts().nlargest(IMG_HOST_TOPK).index
    Xs_tr.loc[~Xs_tr["img_host"].isin(top_hosts), "img_host"] = "__other__"
    Xs_te.loc[~Xs_te["img_host"].isin(top_hosts), "img_host"] = "__other__"

    tr_text = _prepare_text_series(train["catalog_content"])
    te_text = _prepare_text_series(test["catalog_content"])

    tag = f"tfidf_w{WORD_TFIDF}_c{CHAR_TFIDF}_mc{MAX_CHARS}_useC{int(USE_CHAR)}_topH{IMG_HOST_TOPK}_mf{OHE_MIN_FREQ}_v4"
    tr_path, te_path, meta_path = _cache_paths(tag)

    if all(os.path.exists(p) for p in [tr_path, te_path, meta_path]):
        log.info("Loading cached features for tag=%s", tag)
        X_tr_sparse = _load_sparse(tr_path)
        X_te_sparse = _load_sparse(te_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta["tag"] = tag
        log.info("Cache meta: %s", meta)
        return X_tr_sparse, X_te_sparse, meta

    tfw = TfidfVectorizer(
        max_features=WORD_TFIDF, ngram_range=(1,2), lowercase=True,
        strip_accents="unicode", stop_words="english", dtype=np.float32
    )
    W_tr = tfw.fit_transform(tr_text)
    W_te = tfw.transform(te_text)

    if USE_CHAR:
        tfc = TfidfVectorizer(analyzer="char", ngram_range=(3,5), max_features=CHAR_TFIDF, dtype=np.float32)
        C_tr = tfc.fit_transform(tr_text)
        C_te = tfc.transform(te_text)
    else:
        C_tr = sparse.csr_matrix((W_tr.shape[0], 0), dtype=np.float32)
        C_te = sparse.csr_matrix((W_te.shape[0], 0), dtype=np.float32)

    num_cols = ["log_pack_of","log_value_base","log_value_per_pack_base","txt_len","txt_words","txt_digits","txt_upper","txt_digit_ratio","txt_upper_ratio","img_len"]
    cat_cols = ["unit_base","img_ext","img_host"]

    scaler = StandardScaler(with_mean=False).fit(Xs_tr[num_cols].fillna(0.0))
    N_tr = scaler.transform(Xs_tr[num_cols].fillna(0.0)).astype(np.float32)
    N_te = scaler.transform(Xs_te[num_cols].fillna(0.0)).astype(np.float32)

    ohe = _safe_ohe().fit(Xs_tr[cat_cols].astype(str))
    U_tr = ohe.transform(Xs_tr[cat_cols].astype(str)).astype(np.float32)
    U_te = ohe.transform(Xs_te[cat_cols].astype(str)).astype(np.float32)

    X_tr_sparse = sparse.hstack([W_tr, C_tr, N_tr, U_tr], format="csr", dtype=np.float32)
    X_te_sparse = sparse.hstack([W_te, C_te, N_te, U_te], format="csr", dtype=np.float32)

    _save_sparse(tr_path, X_tr_sparse)
    _save_sparse(te_path, X_te_sparse)
    meta = {"WORD_TFIDF": WORD_TFIDF, "CHAR_TFIDF": CHAR_TFIDF, "MAX_CHARS": MAX_CHARS,
            "USE_CHAR": USE_CHAR, "IMG_HOST_TOPK": IMG_HOST_TOPK, "OHE_MIN_FREQ": OHE_MIN_FREQ, "tag": tag}
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    log.info("Cached features to dataset/ (tag=%s)", tag)
    return X_tr_sparse, X_te_sparse, meta

def fit_model(name: str, model, Xtr, ytr, Xva=None, yva=None) -> Bunch:
    """Never mutate the original model; only enable ES when a val set exists."""
    info = {"early_stopping_used": False}
    mdl = _clone_without_es(model)

    if name == "xgb" and _HAS_XGB and Xva is not None and yva is not None:
        try:
            mdl.set_params(early_stopping_rounds=50)
            mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            info["early_stopping_used"] = True
        except TypeError:
            params = mdl.get_params()
            params.pop("early_stopping_rounds", None)
            mdl = mdl.__class__(**params)
            mdl.fit(Xtr, ytr)
    else:
        mdl.fit(Xtr, ytr)

    return Bunch(model=mdl, info=info)

def blend_weights_nnls(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    try:
        w, *_ = np.linalg.lstsq(P, y, rcond=None)
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s <= 1e-9: raise ValueError
        return w / s
    except Exception:
        return np.ones(P.shape[1]) / P.shape[1]

def main():
    t0 = time.time()
    log.info("Loading data from %s", DATASET_FOLDER)
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    assert "price" in train.columns and "catalog_content" in train.columns
    assert "catalog_content" in test.columns

    y = train["price"].astype(float)
    y_log = np.log1p(y)

    X_tr_sparse, X_te_sparse, meta = build_features(train, test)
    tag = meta.get("tag", "default")

    kf = KFold(n_splits=OOF_FOLDS, shuffle=True, random_state=SEED)
    metrics_rows = []
    oof_meta_preds, oof_targets_nat = [], []

    model_defs = []
    if USE_SGD:
        model_defs.append(("sgd_sqr",  SGDRegressor(loss="squared_error", alpha=1e-5, penalty="l2", random_state=SEED, max_iter=5000)))
        model_defs.append(("sgd_huber", SGDRegressor(loss="huber",          alpha=5e-5, penalty="l2", random_state=SEED, max_iter=5000)))
    if USE_RIDGE:
        model_defs.append(("ridge", Ridge(alpha=1.0, solver="lsqr")))
    if USE_XGB and _HAS_XGB and SPEED_MODE != "ultra":
        params = _sanitize_xgb(XGB_PARAMS)
        model_defs.append(("xgb", XGBRegressor(**params)))

    oof_preds_per_model = {name: np.zeros(train.shape[0], dtype=float) for name, _ in model_defs}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr_sparse), 1):
        Xtr, Xva = X_tr_sparse[tr_idx], X_tr_sparse[va_idx]
        ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]

        fold_preds = []
        for name, mdl in model_defs:
            bundle = fit_model(name, mdl, Xtr, ytr, Xva, yva)
            pv_log = bundle.model.predict(Xva)
            pv = np.expm1(pv_log)

            pv = _finite_or_none(pv)
            if np.all(np.isnan(pv)):
                pv = np.full_like(pv, fill_value=float(np.expm1(np.nanmedian(yva.values))), dtype=float)

            oof_preds_per_model[name][va_idx] = np.nan_to_num(pv, nan=float(np.nanmedian(pv)))
            fold_preds.append(pv)
            log.info("Fold %d | %-10s early_stopping=%s", fold, name, getattr(bundle, "info", {}).get("early_stopping_used", False))

        P = np.vstack(fold_preds).T
        y_nat = np.expm1(yva.values)

        oof_meta_preds.append(P)
        oof_targets_nat.append(y_nat)

        pred_blend = _rowwise_nanmean(P)
        if np.any(~np.isfinite(pred_blend)):
            fill_val = _nanmedian_fallback(y_nat, default_val=np.nanmedian(y_nat))
            pred_blend = np.where(np.isfinite(pred_blend), pred_blend, fill_val)

        finite_idx = np.isfinite(pred_blend)
        if finite_idx.any():
            hi = np.quantile(pred_blend[finite_idx], 0.997)
            pred_blend[finite_idx] = np.clip(pred_blend[finite_idx], 0.0, hi)
        else:
            pred_blend[:] = _nanmedian_fallback(y_nat, default_val=1.0)
    
        metrics_rows.append(dict(
            Fold=fold,
            SMAPE=smape(y_nat, pred_blend),
            RMSE=math.sqrt(mean_squared_error(y_nat, pred_blend)),
            MAE=mean_absolute_error(y_nat, pred_blend),
            R2=r2_score(y_nat, pred_blend),
        ))

    cv_df = pd.DataFrame(metrics_rows)
    log.info("OOF mean | SMAPE=%.3f RMSE=%.3f MAE=%.3f R2=%.4f",
             cv_df.SMAPE.mean(), cv_df.RMSE.mean(), cv_df.MAE.mean(), cv_df.R2.mean())
    cv_df.to_csv(CV_LOG_CSV, index=False)

    P_all = np.vstack(oof_meta_preds)
    y_all = np.concatenate(oof_targets_nat)

    row_ok = np.isfinite(y_all)
    row_ok &= np.all(np.isfinite(P_all), axis=1)
    if not np.any(row_ok):
        w = np.ones(P_all.shape[1]) / P_all.shape[1]
    else:
        w = blend_weights_nnls(P_all[row_ok], y_all[row_ok])

    blend_info = {"model_names": [n for n,_ in model_defs], "weights": w.tolist(), "tag": tag}
    with open(OOF_PKL, "wb") as f:
        pickle.dump({"oof_preds_per_model": oof_preds_per_model, "blend_info": blend_info}, f)
    log.info("Blend weights: %s", blend_info)

    preds_test_list = []
    for (name, mdl), weight in zip(model_defs, w):
        path = _model_path(tag, name)
        cached = _load_model_if_exists(path)

        if cached is None:
            log.info("Training final model: %s (no cache found)", name)
            mdl_fresh = _clone_without_es(mdl)
            bundle = fit_model(name, mdl_fresh, X_tr_sparse, y_log, None, None)
            final_model = bundle.model
            try:
                _save_model(path, final_model)
                log.info("Saved model: %s", os.path.basename(path))
            except Exception as e:
                log.warning("Failed to save model %s: %s", path, e)
        else:
            final_model = cached

        pt = np.expm1(final_model.predict(X_te_sparse))
        pt = _finite_or_none(pt)
        if np.any(~np.isfinite(pt)):
            fill_val = np.nanmedian(pt)
            if not np.isfinite(fill_val): fill_val = 0.0
            pt = np.where(np.isfinite(pt), pt, fill_val)
        preds_test_list.append(weight * pt)

    pred_test = np.sum(np.vstack(preds_test_list), axis=0)
    finite_idx = np.isfinite(pred_test)
    if finite_idx.any():
        hi = np.quantile(pred_test[finite_idx], 0.997)
        pred_test[finite_idx] = np.clip(pred_test[finite_idx], 0.01, hi)
    else:
        pred_test[:] = 0.01

    out = pd.DataFrame({"sample_id": test["sample_id"], "price": pred_test.astype(float)})
    out.to_csv(OUT_CSV, index=False)
    log.info("Saved CV log to %s", CV_LOG_CSV)
    log.info("Saved OOF/blend info to %s", OOF_PKL)
    log.info("Saved predictions to %s", OUT_CSV)
    log.info("Total runtime: %.2fs", time.time() - t0)

if __name__ == "__main__":
    main()