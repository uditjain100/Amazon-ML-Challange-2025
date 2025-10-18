import os, re, math, time, logging, random, pickle, warnings
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge

SPEED_MODE = "ultra"         
WORD_FEATS = 2**13            
NGRAMS = (1, 1)              
MAX_CHARS = 400              
SEED = 42
random.seed(SEED); np.random.seed(SEED)
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("fast")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(BASE_DIR, "dataset")
TRAIN_CSV = os.path.join(DATASET_FOLDER, "train.csv")
TEST_CSV  = os.path.join(DATASET_FOLDER, "test.csv")
OUT_CSV   = os.path.join(DATASET_FOLDER, "test_out.csv")

_pack_re = re.compile(r"[Pp]ack\s*of\s*(\d+)")
_val_re  = re.compile(r"^Value:\s*([\-\+]?\d+(?:\.\d+)?)\s*$", re.M)
_unit_re = re.compile(r"^Unit:\s*([A-Za-z %/]+)\s*$", re.M)

UNIT_MAP = {
    "gram": ("g", 1.0), "grams": ("g", 1.0), "g": ("g", 1.0),
    "kg": ("g", 1000.0), "kilogram": ("g", 1000.0),
    "Ounce": ("g", 28.3495), "oz": ("g", 28.3495), "Fl Oz": ("ml", 29.5735),
    "Count": ("count", 1.0), "count": ("count", 1.0), "Each": ("count", 1.0),
}

def _extract_pack_of(s: str) -> float:
    if not isinstance(s, str): return 1.0
    m = _pack_re.search(s)
    return float(m.group(1)) if m else 1.0

def _extract_value_unit(s: str):
    if not isinstance(s, str): return (np.nan, "UNK")
    vm = _val_re.search(s); um = _unit_re.search(s)
    v = float(vm.group(1)) if vm else np.nan
    u = um.group(1).strip() if um else "UNK"
    return (v, u)

def _norm_unit(u: str):
    if u in UNIT_MAP: return UNIT_MAP[u]
    l = u.lower().replace(".", "")
    return UNIT_MAP.get(l, ("UNK", 1.0))

def build_structured(df: pd.DataFrame) -> pd.DataFrame:
    s = df["catalog_content"].fillna("")
    out = pd.DataFrame(index=df.index)
    out["pack_of"] = s.apply(_extract_pack_of)
    vu = s.apply(_extract_value_unit)
    out["value_num"] = vu.apply(lambda t: t[0])
    out["unit"] = vu.apply(lambda t: t[1])
    nb = out["unit"].apply(_norm_unit)
    out["unit_base"] = nb.apply(lambda t: t[0])
    out["unit_mult"] = nb.apply(lambda t: t[1])
    out["value_base"] = out["value_num"] * out["unit_mult"]
    out["value_per_pack"] = out["value_base"] / out["pack_of"].replace(0, np.nan)
    out["log_value_base"] = np.log1p(out["value_base"].fillna(0))
    out["log_value_per_pack"] = np.log1p(out["value_per_pack"].fillna(0))
    out["txt_len"] = s.str.len().astype(float)
    out["txt_words"] = s.str.split().apply(len).astype(float)
    out["txt_digits"] = s.str.count(r"\d").astype(float)
    return out

def main():
    t0 = time.time()
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    y = np.log1p(train["price"].astype(float))

    Xs_tr = build_structured(train)
    Xs_te = build_structured(test)

    num_cols = ["log_value_base","log_value_per_pack","txt_len","txt_words","txt_digits"]
    cat_cols = ["unit_base"]

    num_scaler = StandardScaler(with_mean=False).fit(Xs_tr[num_cols].fillna(0))
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=10)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
    ohe.fit(Xs_tr[cat_cols].astype(str))
    hv = HashingVectorizer(n_features=WORD_FEATS, ngram_range=NGRAMS, alternate_sign=False, stop_words="english")
    tr_text = train["catalog_content"].fillna("").astype(str).str.slice(0, MAX_CHARS)
    te_text = test["catalog_content"].fillna("").astype(str).str.slice(0, MAX_CHARS)
    W_tr = hv.transform(tr_text)
    W_te = hv.transform(te_text)

    N_tr = num_scaler.transform(Xs_tr[num_cols].fillna(0))
    N_te = num_scaler.transform(Xs_te[num_cols].fillna(0))
    U_tr = ohe.transform(Xs_tr[cat_cols].astype(str))
    U_te = ohe.transform(Xs_te[cat_cols].astype(str))

    X_tr = sparse.hstack([W_tr, N_tr, U_tr], format="csr")
    X_te = sparse.hstack([W_te, N_te, U_te], format="csr")

    model = Ridge(alpha=1.0, solver="sag", random_state=SEED)
    model.fit(X_tr, y)
    pred = np.expm1(model.predict(X_te))
    pred = np.clip(pred, 0.01, np.quantile(pred, 0.997))
    pd.DataFrame({"sample_id": test["sample_id"], "price": pred}).to_csv(OUT_CSV, index=False)
    print(f"✅ Done. Saved {OUT_CSV} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()