"""Advanced Flash Crash Early Warning System.

Phase-1 baseline upgrades included:
- Reproducibility controls (global seed)
- Train/validation/test split
- Imbalance-aware training via BCEWithLogitsLoss(pos_weight)
- Validation threshold tuning for F1
- Metrics + config logging and checkpoint saving
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GATConv


NIFTY_50_BANK_CORE = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "KOTAKBANK.NS",
    "AXISBANK.NS",
    "TCS.NS",
    "INFY.NS",
    "LT.NS",
    "ITC.NS",
]


@dataclass
class Config:
    start: str = "2020-01-01"
    end: str = "2024-01-01"
    interval: str = "5m"
    seq_len: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 10
    edge_corr_threshold: float = 0.60
    precrash_horizon_steps: int = 2
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    seed: int = 42
    threshold_min: float = 0.10
    threshold_max: float = 0.90
    threshold_steps: int = 17
    output_root: str = "results"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MarketSimulator:
    """Agent-based market simulator for realistic synthetic crash dynamics."""

    def __init__(self, n_agents: int = 200, steps: int = 2000, seed: int = 42):
        self.n_agents = n_agents
        self.steps = steps
        self.rng = np.random.default_rng(seed)

    def _fundamental_trader(self) -> float:
        return self.rng.normal(0, 0.4)

    def _momentum_trader(self, last_return: float) -> float:
        return 12.0 * last_return + self.rng.normal(0, 0.2)

    def _liquidity_trader(self) -> float:
        return self.rng.normal(0, 1.2)

    def _hft_trader(self, short_vol: float, drawdown: float) -> float:
        if short_vol > 0.02 or drawdown < -0.03:
            return -self.rng.uniform(4.0, 10.0)
        return self.rng.normal(0, 1.0)

    def simulate(self, initial_price: float = 100.0) -> pd.DataFrame:
        prices = []
        price = initial_price
        last_return = 0.0

        for t in range(self.steps):
            hist = np.array(prices[-20:]) if len(prices) >= 20 else np.array([price])
            short_vol = np.std(np.diff(np.log(hist))) if len(hist) > 2 else 0.0
            drawdown = (price / np.max(hist)) - 1.0 if len(hist) > 1 else 0.0

            orders = []
            for _ in range(self.n_agents):
                kind = self.rng.choice(["f", "m", "l", "h"], p=[0.3, 0.2, 0.35, 0.15])
                if kind == "f":
                    orders.append(self._fundamental_trader())
                elif kind == "m":
                    orders.append(self._momentum_trader(last_return))
                elif kind == "l":
                    orders.append(self._liquidity_trader())
                else:
                    orders.append(self._hft_trader(short_vol, drawdown))

            net_order = np.sum(orders)
            impact = 0.0045 * net_order
            price = max(1.0, price * (1.0 + impact / 100.0))
            last_return = impact / 100.0
            prices.append(price)

            if t > 50 and short_vol > 0.03 and self.rng.random() < 0.02:
                price *= 1.0 - self.rng.uniform(0.04, 0.10)
                prices[-1] = price

        df = pd.DataFrame({"Close": prices})
        df["Volume"] = np.abs(self.rng.normal(1_000_000, 250_000, len(df))).astype(int)
        df["High"] = df["Close"] * (1 + np.abs(self.rng.normal(0.0008, 0.0003, len(df))))
        df["Low"] = df["Close"] * (1 - np.abs(self.rng.normal(0.0008, 0.0003, len(df))))
        df["Open"] = df["Close"].shift(1).fillna(df["Close"]) * (
            1 + self.rng.normal(0, 0.0005, len(df))
        )
        return df


def download_multi_stock_data(
    tickers: List[str], start: str, end: str, interval: str
) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if not df.empty:
            data[ticker] = df.dropna().copy()
    return data


def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["return"] = out["Close"].pct_change()
    out["volatility"] = out["return"].rolling(20).std()
    out["price_acceleration"] = out["return"].diff()
    out["bid"] = out["Close"] * (1 - 0.0005)
    out["ask"] = out["Close"] * (1 + 0.0005)
    out["bid_ask_spread"] = out["ask"] - out["bid"]
    out["market_depth"] = out["Volume"].rolling(10).mean()
    out["depth_collapse"] = out["market_depth"].pct_change()
    out["volume_change"] = out["Volume"].pct_change()
    out["crash"] = (out["Close"].pct_change(5) < -0.05).astype(int)
    return out.dropna()


def add_precrash_labels(df: pd.DataFrame, horizon_steps: int) -> pd.DataFrame:
    out = df.copy()
    out["precrash"] = 0
    crash_idx = np.where(out["crash"].values == 1)[0]
    for idx in crash_idx:
        start = max(0, idx - horizon_steps)
        out.iloc[start : idx + 1, out.columns.get_loc("precrash")] = 1
    return out


def build_stock_edge_index(stock_feature_map: Dict[str, pd.DataFrame], threshold: float) -> torch.Tensor:
    returns_df = pd.DataFrame({k: v["return"].values for k, v in stock_feature_map.items()}).dropna()
    corr = returns_df.corr().values

    edges: List[List[int]] = []
    n = corr.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and corr[i, j] >= threshold:
                edges.append([i, j])

    if not edges:
        edges = [[i, i] for i in range(n)]

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_sequences(
    feature_map: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    stocks = list(feature_map.keys())
    min_len = min(len(feature_map[s]) for s in stocks)
    trimmed = {s: feature_map[s].iloc[-min_len:].reset_index(drop=True) for s in stocks}

    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for t in range(seq_len, min_len):
        window_stack = []
        labels = []
        for s in stocks:
            win = trimmed[s].iloc[t - seq_len : t][feature_cols].values
            window_stack.append(win)
            labels.append(trimmed[s].iloc[t]["precrash"])
        x_list.append(np.stack(window_stack, axis=0))
        y_list.append(np.array(labels, dtype=np.float32))

    return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 32,
        nhead: int = 4,
        layers: int = 2,
    ):
        super().__init__()

        if model_dim % nhead != 0:
            raise ValueError("model_dim must be divisible by nhead")

        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=model_dim * 2,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        encoded = self.encoder(x)
        return encoded[:, -1, :]


class GraphTemporalCrashModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, temporal_dim: int = 32):
        super().__init__()
        self.temporal = TemporalTransformer(input_dim=in_dim, model_dim=temporal_dim, nhead=4)
        self.gat1 = GATConv(in_channels=temporal_dim, out_channels=hidden_dim, heads=1)
        self.gat2 = GATConv(in_channels=hidden_dim, out_channels=hidden_dim, heads=1)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, num_stocks, seq_len, in_dim]
        batch_size, num_stocks, seq_len, in_dim = x.shape
        reshaped = x.reshape(batch_size * num_stocks, seq_len, in_dim)
        temporal_emb = self.temporal(reshaped)
        temporal_emb = temporal_emb.reshape(batch_size, num_stocks, -1)

        outputs = []
        for b in range(batch_size):
            node_feats = temporal_emb[b]
            g = torch.relu(self.gat1(node_feats, edge_index))
            g = torch.relu(self.gat2(g, edge_index))
            outputs.append(self.out(g).squeeze(-1))
        return torch.stack(outputs, dim=0)


def split_indices(n: int, train_ratio: float, val_ratio: float) -> Tuple[slice, slice, slice]:
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def compute_pos_weight(y_train: torch.Tensor) -> torch.Tensor:
    positives = y_train.sum().item()
    negatives = y_train.numel() - positives
    if positives <= 0:
        return torch.tensor(1.0)
    return torch.tensor(max(negatives / positives, 1.0), dtype=torch.float32)


def collect_predictions(model: nn.Module, loader: DataLoader, edge_index: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb, edge_index)
            probs = torch.sigmoid(logits)
            y_true.extend(yb.numpy().ravel().tolist())
            y_prob.extend(probs.numpy().ravel().tolist())
    return np.array(y_true).astype(int), np.array(y_prob)


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics: Dict[str, float] = {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    return metrics


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray, cfg: Config) -> Tuple[float, Dict[str, float]]:
    grid = np.linspace(cfg.threshold_min, cfg.threshold_max, cfg.threshold_steps)
    best_thr = 0.5
    best = {"f1": -1.0}
    for thr in grid:
        cur = metrics_at_threshold(y_true, y_prob, float(thr))
        if cur["f1"] > best["f1"]:
            best_thr, best = float(thr), cur
    return best_thr, best


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    edge_index: torch.Tensor,
    cfg: Config,
    output_dir: Path,
) -> Dict[str, object]:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    y_train_full = np.concatenate([yb.numpy().ravel() for _, yb in train_loader])
    pos_weight = compute_pos_weight(torch.tensor(y_train_full, dtype=torch.float32))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best = {"epoch": -1, "f1": -1.0, "threshold": 0.5}
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb, edge_index)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total += float(loss.item())

        y_val_true, y_val_prob = collect_predictions(model, val_loader, edge_index)
        tuned_thr, val_metrics = tune_threshold(y_val_true, y_val_prob, cfg)
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": total,
            "val_metrics": val_metrics,
        }
        history.append(epoch_log)
        print(
            f"epoch={epoch + 1} loss={total:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_thr={tuned_thr:.2f}"
        )

        if val_metrics["f1"] > best["f1"]:
            best = {"epoch": epoch + 1, "f1": val_metrics["f1"], "threshold": tuned_thr}
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    return {"history": history, "best": best, "pos_weight": float(pos_weight.item())}


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    edge_index: torch.Tensor,
    threshold: float,
) -> Dict[str, object]:
    y_true, y_prob = collect_predictions(model, test_loader, edge_index)
    y_pred = (y_prob >= threshold).astype(int)

    print(classification_report(y_true, y_pred, zero_division=0))
    metrics = metrics_at_threshold(y_true, y_prob, threshold)
    if "roc_auc" in metrics:
        print("ROC-AUC:", metrics["roc_auc"])
        print("PR-AUC:", metrics["pr_auc"])
    return {
        "metrics": metrics,
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }


def run_pipeline(use_simulator: bool = True) -> None:
    cfg = Config()
    set_global_seed(cfg.seed)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(cfg.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [
        "return",
        "volatility",
        "price_acceleration",
        "bid_ask_spread",
        "market_depth",
        "depth_collapse",
        "volume_change",
    ]

    if use_simulator:
        simulator = MarketSimulator(seed=cfg.seed)
        raw_map = {
            ticker: simulator.simulate(initial_price=100 + i * 25)
            for i, ticker in enumerate(NIFTY_50_BANK_CORE)
        }
    else:
        raw_map = download_multi_stock_data(
            NIFTY_50_BANK_CORE,
            start=cfg.start,
            end=cfg.end,
            interval=cfg.interval,
        )

    if not raw_map:
        raise RuntimeError("No market data available. Check symbols/date range or network access.")

    feature_map: Dict[str, pd.DataFrame] = {}
    for ticker, df in raw_map.items():
        feat = compute_microstructure_features(df)
        feat = add_precrash_labels(feat, horizon_steps=cfg.precrash_horizon_steps)
        if not feat.empty:
            feature_map[ticker] = feat

    if len(feature_map) < 2:
        raise RuntimeError("Need at least 2 non-empty stock series to build a cross-stock graph.")

    x, y = build_sequences(feature_map, feature_cols, seq_len=cfg.seq_len)
    if len(x) < 10:
        raise RuntimeError("Not enough sequence samples after feature engineering.")

    scaler = StandardScaler()
    b, n, t, f = x.shape
    x_scaled = scaler.fit_transform(x.reshape(-1, f)).reshape(b, n, t, f)

    edge_index = build_stock_edge_index(feature_map, threshold=cfg.edge_corr_threshold)
    xb = torch.tensor(x_scaled, dtype=torch.float32)
    yb = torch.tensor(y, dtype=torch.float32)

    tr, va, te = split_indices(len(xb), cfg.train_ratio, cfg.val_ratio)
    train_set = TensorDataset(xb[tr], yb[tr])
    val_set = TensorDataset(xb[va], yb[va])
    test_set = TensorDataset(xb[te], yb[te])

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)

    model = GraphTemporalCrashModel(in_dim=f, hidden_dim=64)
    train_artifacts = train_model(model, train_loader, val_loader, edge_index, cfg, output_dir)

    best_path = output_dir / "best_model.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location="cpu"))

    evaluation = evaluate_model(
        model,
        test_loader,
        edge_index,
        threshold=float(train_artifacts["best"]["threshold"]),
    )

    summary = {
        "config": asdict(cfg),
        "num_stocks": len(feature_map),
        "num_sequences": int(len(x)),
        "train_artifacts": train_artifacts,
        "test": evaluation,
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f_out:
        json.dump(summary, f_out, indent=2)

    print(f"Saved artifacts to: {output_dir}")


if __name__ == "__main__":
    run_pipeline(use_simulator=True)
