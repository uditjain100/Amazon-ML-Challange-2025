import argparse
import os
import re
import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

_pack_re = re.compile(r"[Pp]ack\s*of\s*(\d+)")
_val_re = re.compile(r"^Value:\s*([\-\+]?\d+(?:\.\d+)?)\s*$", re.M)
_unit_re = re.compile(r"^Unit:\s*([A-Za-z %/]+)\s*$", re.M)

UNIT_MAP: Dict[str, Tuple[str, float]] = {
    "gram": ("g", 1.0), "grams": ("g", 1.0), "g": ("g", 1.0),
    "kilogram": ("g", 1000.0), "kg": ("g", 1000.0),
    "ounce": ("g", 28.3495), "ounces": ("g", 28.3495), "oz": ("g", 28.3495), "Oz": ("g", 28.3495), "Ounce": ("g", 28.3495),
    "pound": ("g", 453.592), "Pound": ("g", 453.592), "lb": ("g", 453.592), "LB": ("g", 453.592),
    "Fl Oz": ("ml", 29.5735), "fl oz": ("ml", 29.5735), "FL Oz": ("ml", 29.5735),
    "Fluid Ounce": ("ml", 29.5735), "Fluid Ounces": ("ml", 29.5735), "fluid ounces": ("ml", 29.5735),
    "Liter": ("ml", 1000.0), "Liters": ("ml", 1000.0), "l": ("ml", 1000.0), "L": ("ml", 1000.0),
    "milliliter": ("ml", 1.0), "milliliters": ("ml", 1.0), "ml": ("ml", 1.0), "mL": ("ml", 1.0),
    "Count": ("count", 1.0), "count": ("count", 1.0), "Each": ("count", 1.0), "ct": ("count", 1.0),
    "Pack": ("count", 1.0),
    "None": ("UNK", 1.0), "UNK": ("UNK", 1.0),
}


def _extract_pack_of(s: str) -> float:
    if not isinstance(s, str):
        return 1.0
    m = _pack_re.search(s)
    return float(m.group(1)) if m else 1.0


def _extract_value_unit(s: str) -> Tuple[float, str]:
    if not isinstance(s, str):
        return (np.nan, "UNK")
    vm = _val_re.search(s)
    um = _unit_re.search(s)
    v = float(vm.group(1)) if vm else np.nan
    u = um.group(1).strip() if um else "UNK"
    return (v, u)


def _norm_unit(unit: str) -> Tuple[str, float]:
    if unit in UNIT_MAP:
        return UNIT_MAP[unit]
    key = unit.strip().lower().replace(".", "").replace("fluid", "").strip()
    return UNIT_MAP.get(key, ("UNK", 1.0))


def build_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    s = df["catalog_content"].fillna("").astype(str)
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
    out["log_value_per_pack_base"] = np.log1p(
        out["value_per_pack_base"].replace([np.inf, -np.inf], np.nan).fillna(0)
    )
    s_len = s.str.len().astype(float)
    out["txt_len"] = s_len
    out["txt_words"] = s.str.split().apply(len).astype(float)
    out["txt_digits"] = s.str.count(r"\d").astype(float)
    out["txt_upper"] = s.str.count(r"[A-Z]").astype(float)
    out["txt_digit_ratio"] = (out["txt_digits"] / s_len.replace(0, np.nan)).fillna(0.0)
    out["txt_upper_ratio"] = (out["txt_upper"] / s_len.replace(0, np.nan)).fillna(0.0)
    return out

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1.0
    return 100.0 * np.mean(np.abs(y_true - y_pred) / denom)


class TextStructuredDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        structured_df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        is_train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.structured = structured_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

        self.struct_array = self.structured.values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["catalog_content"])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        struct_feats = torch.tensor(self.struct_array[idx], dtype=torch.float32)
        if self.is_train:
            target = torch.tensor(float(row["price"]), dtype=torch.float32)
            return input_ids, attention_mask, struct_feats, target
        else:
            return input_ids, attention_mask, struct_feats


class TextStructuredRegressor(nn.Module):
    def __init__(
        self,
        text_model_name: str,
        struct_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        fine_tune_text: bool = False,
    ):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.token_dim = self.text_model.config.hidden_size
        if not fine_tune_text:
            for p in self.text_model.parameters():
                p.requires_grad = False
        self.struct_encoder = nn.Sequential(
            nn.Linear(struct_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.struct_dim = hidden_dim // 2
        fusion_input = self.token_dim + self.struct_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, input_ids, attention_mask, struct_feats):
        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = text_out.last_hidden_state[:, 0, :]
        struct_emb = self.struct_encoder(struct_feats)
        fused = torch.cat([text_emb, struct_emb], dim=1)
        log_price = self.fusion(fused).squeeze(1)
        price = torch.exp(log_price)
        return price


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    early_stopping_patience: int = 3,
):
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_smape = float("inf")
    patience_counter = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: List[float] = []
        start_time = time.time()
        for batch in train_loader:
            input_ids, attention_mask, struct_feats, targets = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            struct_feats = struct_feats.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                preds = model(input_ids, attention_mask, struct_feats)
                loss = F.mse_loss(torch.log1p(preds), torch.log1p(targets))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
        if scheduler is not None:
            scheduler.step()
        model.eval()
        val_targets: List[float] = []
        val_preds: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, struct_feats, targets = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                struct_feats = struct_feats.to(device)
                preds = model(input_ids, attention_mask, struct_feats)
                val_targets.extend(targets.cpu().numpy().tolist())
                val_preds.extend(preds.cpu().numpy().tolist())
        val_smape = smape(np.array(val_targets), np.array(val_preds))
        duration = time.time() - start_time
        avg_train_loss = np.mean(train_losses)
        print(
            f"Epoch {epoch}/{epochs} | Time: {duration:.1f}s | Train Loss: {avg_train_loss:.4f} | Val SMAPE: {val_smape:.3f}"
        )
        if val_smape + 1e-4 < best_smape:
            best_smape = val_smape
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_text.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch}. Best Val SMAPE: {best_smape:.3f}."
                )
                model.load_state_dict(torch.load("best_model_text.pth", map_location=device))
                break


def main():
    parser = argparse.ArgumentParser(description="Text‑only price prediction")
    parser.add_argument("--train_csv", type=str, required=True, help="Training CSV path")
    parser.add_argument("--test_csv", type=str, required=True, help="Test CSV path")
    parser.add_argument("--output_csv", type=str, default="test_out.csv", help="Output CSV path")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--fine_tune_text", type=int, default=0, help="Fine‑tune text encoder (1=yes, 0=no)"
    )
    parser.add_argument("--max_length", type=int, default=128, help="Max tokens for text")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("Loading data…")
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    if "catalog_content" not in train_df.columns or "catalog_content" not in test_df.columns:
        raise ValueError("Both CSV files must contain 'catalog_content' column")
    if "price" not in train_df.columns:
        raise ValueError("Training CSV must include 'price' column")

    struct_train = build_structured_features(train_df)
    struct_test = build_structured_features(test_df)

    numeric_cols = [c for c in struct_train.columns if c not in {"unit_base", "unit_raw", "unit_mult"}]
    scaler = StandardScaler(with_mean=False)
    scaler.fit(struct_train[numeric_cols].fillna(0.0))
    def encode_struct(df_struct: pd.DataFrame) -> pd.DataFrame:
        df_num = pd.DataFrame(
            scaler.transform(df_struct[numeric_cols].fillna(0.0)),
            index=df_struct.index,
            columns=[f"scaled_{c}" for c in numeric_cols],
        )

        vc = df_struct["unit_base"].fillna("unk").astype(str).value_counts()
        keep = vc[vc >= 20].index
        cat = df_struct["unit_base"].fillna("unk").astype(str)
        cat = cat.where(cat.isin(keep), other="other")
        dummies = pd.get_dummies(cat, prefix="unit_base")
        return pd.concat([df_num, dummies], axis=1)
    encoded_train = encode_struct(struct_train)
    encoded_test = encode_struct(struct_test)
    encoded_test = encoded_test.reindex(columns=encoded_train.columns, fill_value=0.0)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)), test_size=args.val_size, random_state=args.seed, shuffle=True
    )
    train_loader = DataLoader(
        TextStructuredDataset(
            train_df.iloc[train_idx],
            encoded_train.iloc[train_idx],
            tokenizer=tokenizer,
            max_length=args.max_length,
            is_train=True,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TextStructuredDataset(
            train_df.iloc[val_idx],
            encoded_train.iloc[val_idx],
            tokenizer=tokenizer,
            max_length=args.max_length,
            is_train=True,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TextStructuredRegressor(
        text_model_name="distilbert-base-uncased",
        struct_dim=encoded_train.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=0.2,
        fine_tune_text=bool(args.fine_tune_text),
    )
    model.to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("Training model…")
    train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        early_stopping_patience=2,
    )

    if os.path.exists("best_model_text.pth"):
        model.load_state_dict(torch.load("best_model_text.pth", map_location=device))

    test_loader = DataLoader(
        TextStructuredDataset(
            test_df,
            encoded_test,
            tokenizer=tokenizer,
            max_length=args.max_length,
            is_train=False,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print("Predicting on test set…")
    model.eval()
    preds: List[float] = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, struct_feats = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            struct_feats = struct_feats.to(device)
            outputs = model(input_ids, attention_mask, struct_feats)
            preds.extend(outputs.cpu().numpy().tolist())
    pred_arr = np.maximum(np.array(preds), 1e-3)

    high_clip = np.quantile(pred_arr, 0.997)
    pred_arr = np.clip(pred_arr, 0.01, high_clip)

    if "sample_id" in test_df.columns:
        submission = pd.DataFrame({"sample_id": test_df["sample_id"], "price": pred_arr})
    else:
        submission = pd.DataFrame({"price": pred_arr})
    submission.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()