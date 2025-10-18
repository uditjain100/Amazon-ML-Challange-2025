import re, numpy as np, pandas as pd, gc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
import lightgbm as lgb

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / np.maximum(denom, 1e-8)
    return np.mean(diff) * 100

smape_scorer = make_scorer(smape, greater_is_better=False)

def extract_structured_features(df):
    df = df.copy()
    df['pack_of'] = df['catalog_content'].str.extract(r'[Pp]ack\s*of\s*(\d+)').astype(float)
    df['value_num'] = df['catalog_content'].str.extract(r'(\d+(?:\.\d+)?)\s*(?:ml|g|kg|l|L|pcs|units|tabs|m)')[[0]].astype(float)
    df['text_len'] = df['catalog_content'].str.len()
    df['digit_ratio'] = df['catalog_content'].str.count(r'\d') / df['text_len'].clip(lower=1)
    df['upper_ratio'] = df['catalog_content'].apply(lambda x: sum(1 for c in x if c.isupper())/len(x) if len(x)>0 else 0)
    df['word_count'] = df['catalog_content'].apply(lambda x: len(str(x).split()))
    return df

train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

train = extract_structured_features(train)
test = extract_structured_features(test)

text_col = 'catalog_content'
num_cols = ['pack_of', 'value_num', 'text_len', 'digit_ratio', 'upper_ratio', 'word_count']

print("Extracting text features...")
tfidf = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    stop_words='english'
)
svd = TruncatedSVD(n_components=100, random_state=42)

X_text = tfidf.fit_transform(train[text_col])
X_text = svd.fit_transform(X_text)
print(f"Text shape after SVD: {X_text.shape}")

X_num = train[num_cols].fillna(0).values
scaler = StandardScaler()
X_num = scaler.fit_transform(X_num)

X = np.hstack([X_text, X_num])
y = train['price'].values

model = lgb.LGBMRegressor(
    n_estimators=700,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=4   
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
smape_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    score = smape(y_val, preds)
    print(f"SMAPE: {score:.4f}")
    smape_scores.append(score)
    gc.collect()

print(f"\nMean SMAPE: {np.mean(smape_scores):.4f}, Std: {np.std(smape_scores):.4f}")

X_test_text = svd.transform(tfidf.transform(test[text_col]))
X_test_num = scaler.transform(test[num_cols].fillna(0).values)
X_test = np.hstack([X_test_text, X_test_num])

model.fit(X, y)
test_pred = model.predict(X_test)

if 'id' not in test.columns:
    test['id'] = np.arange(len(test))
pd.DataFrame({'id': test['id'], 'price': test_pred}).to_csv('test_out.csv', index=False)
print("test_out.csv saved successfully!")