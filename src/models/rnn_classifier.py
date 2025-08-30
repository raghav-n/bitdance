import json
import re
from typing import List, Dict, Callable, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from . import LABELS


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


class Vocab:
    def __init__(self, min_freq=2):
        self.stoi = {"<pad>": 0, "<unk>": 1}
        self.itos = ["<pad>", "<unk>"]
        self.freq = {}
        self.min_freq = min_freq

    def build(self, texts: List[str]):
        for t in texts:
            for tok in tokenize(t):
                self.freq[tok] = self.freq.get(tok, 0) + 1
        for tok, f in sorted(self.freq.items()):
            if f >= self.min_freq:
                self.stoi.setdefault(tok, len(self.itos))
                self.itos.append(tok)

    def encode(self, text: str, max_len=256):
        ids = [self.stoi.get(tok, 1) for tok in tokenize(text)][:max_len]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids


def build_text(r: Dict) -> str:
    return (
        f"BUSINESS: {r.get('business_name','')}\n"
        f"AUTHOR: {r.get('author_name','')}\n"
        f"RATING: {r.get('rating','')}\n"
        f"TEXT: {r.get('text','')}"
    )


class ReviewDataset(Dataset):
    def __init__(self, rows: List[Dict], vocab: Vocab, max_len=256):
        self.rows = rows
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        x = torch.tensor(self.vocab.encode(build_text(r), self.max_len), dtype=torch.long)
        y = torch.tensor([float(bool(r.get(name, False))) for name in LABELS], dtype=torch.float32)
        return x, y


class TextRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden=256, layers=1, dropout=0.2, num_labels=len(LABELS)):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            emb_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, num_labels)
        )

    def forward(self, x):
        e = self.emb(x)
        h, _ = self.rnn(e)
        pooled, _ = torch.max(h, dim=1)
        return self.head(pooled)


def _load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def train_rnn_jsonl(
    jsonl_train: str,
    jsonl_val: str,
    out_path: str,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    early_stopping_patience: int = 2,
    early_stopping_min_delta: float = 0.0,
):
    train_rows = _load_jsonl(jsonl_train)
    val_rows = _load_jsonl(jsonl_val)
    vocab = Vocab(min_freq=2)
    vocab.build([build_text(r) for r in train_rows])
    train_ds = ReviewDataset(train_rows, vocab)
    val_ds = ReviewDataset(val_rows, vocab)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = TextRNN(vocab_size=len(vocab.itos))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_left = early_stopping_patience
    from sklearn.metrics import f1_score
    history = []

    for _epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for xb, yb in train_dl:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)

        # validation
        model.eval()
        y_true, y_pred = [], []
        val_running_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                logits = model(xb)
                val_loss = loss_fn(logits, yb)
                val_running_loss += float(val_loss.item())
                val_batches += 1
                probs = torch.sigmoid(logits)
                y_true.append(yb.numpy())
                y_pred.append((probs >= 0.5).float().numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        val_loss_avg = val_running_loss / max(1, val_batches)

        history.append({
            "epoch": int(_epoch + 1),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss_avg),
            "f1_macro": float(f1),
        })

        # Early stopping on validation loss
        if val_loss_avg + early_stopping_min_delta < best_val_loss:
            best_val_loss = val_loss_avg
            patience_left = early_stopping_patience
            torch.save({"model": model.state_dict(), "vocab": vocab.__dict__}, out_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    # Save training history
    import json as _json
    from pathlib import Path as _Path
    hist_path = _Path(out_path).parent / "training_history.json"
    hist_path.write_text(_json.dumps(history, indent=2), encoding="utf-8")


def load_rnn(model_path: str) -> TextRNN:
    ckpt = torch.load(model_path, map_location="cpu")
    vocab = Vocab()
    vocab.__dict__.update(ckpt["vocab"])  # restore vocab
    model = TextRNN(vocab_size=len(vocab.itos))
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.vocab = vocab
    return model


@torch.no_grad()
def infer_rnn(model: TextRNN, example: Dict, threshold: float = 0.5) -> Dict:
    x = torch.tensor(model.vocab.encode(build_text(example)), dtype=torch.long).unsqueeze(0)
    probs = torch.sigmoid(model(x)).squeeze(0).numpy()
    return {LABELS[i]: bool(probs[i] >= threshold) for i in range(len(LABELS))}


@torch.no_grad()
def predict_probs_rnn(
    model: TextRNN,
    jsonl_path: str,
    batch_size: int = 128,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Batch inference for RNN model returning probabilities array (N, L)."""
    rows = _load_jsonl(jsonl_path)
    total = len(rows)
    if progress_callback:
        progress_callback(0, total)

    class _TmpDS(Dataset):
        def __init__(self, rows, vocab):
            self.rows = rows
            self.vocab = vocab
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, idx):
            r = self.rows[idx]
            return torch.tensor(self.vocab.encode(build_text(r)), dtype=torch.long)

    ds = _TmpDS(rows, model.vocab)
    dl = DataLoader(ds, batch_size=max(1, int(batch_size)))
    out: List[np.ndarray] = []
    done = 0
    for xb in dl:
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
        out.append(probs)
        done += xb.shape[0]
        if progress_callback:
            progress_callback(done, total)
    return np.vstack(out)
