"""Advanced Flash Crash Early Warning System.

Includes:
- Agent-based flash crash simulation
- Multi-stock data ingestion (NIFTY-oriented tickers)
- Order-book style microstructure features
- Cross-stock contagion graph with GAT
- Temporal Transformer for pre-crash prediction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.metrics import classification_report, roc_auc_score
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
                price *= (1.0 - self.rng.uniform(0.04, 0.10))
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
        self.out = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

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


def train_model(model: nn.Module, loader: DataLoader, edge_index: torch.Tensor, cfg: Config) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(cfg.epochs):
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb, edge_index)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        print(f"epoch={epoch + 1} loss={total:.4f}")


def evaluate_model(model: nn.Module, loader: DataLoader, edge_index: torch.Tensor) -> None:
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for xb, yb in loader:
            prob = model(xb, edge_index)
            y_true.extend(yb.numpy().ravel().tolist())
            y_prob.extend(prob.numpy().ravel().tolist())

    y_true_np = np.array(y_true).astype(int)
    y_prob_np = np.array(y_prob)
    y_pred_np = (y_prob_np > 0.70).astype(int)

    print(classification_report(y_true_np, y_pred_np, zero_division=0))
    if len(np.unique(y_true_np)) > 1:
        print("ROC-AUC:", roc_auc_score(y_true_np, y_prob_np))


def run_pipeline(use_simulator: bool = True) -> None:
    cfg = Config()
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
        simulator = MarketSimulator()
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

    feature_map: Dict[str, pd.DataFrame] = {}
    for ticker, df in raw_map.items():
        feat = compute_microstructure_features(df)
        feat = add_precrash_labels(feat, horizon_steps=cfg.precrash_horizon_steps)
        feature_map[ticker] = feat

    stocks = list(feature_map.keys())
    x, y = build_sequences(feature_map, feature_cols, seq_len=cfg.seq_len)

    scaler = StandardScaler()
    b, n, t, f = x.shape
    x_scaled = scaler.fit_transform(x.reshape(-1, f)).reshape(b, n, t, f)

    edge_index = build_stock_edge_index(feature_map, threshold=cfg.edge_corr_threshold)
    xb = torch.tensor(x_scaled, dtype=torch.float32)
    yb = torch.tensor(y, dtype=torch.float32)

    split = int(0.8 * len(xb))
    train_set = TensorDataset(xb[:split], yb[:split])
    test_set = TensorDataset(xb[split:], yb[split:])

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)

    model = GraphTemporalCrashModel(in_dim=f, hidden_dim=64)
    train_model(model, train_loader, edge_index, cfg)
    evaluate_model(model, test_loader, edge_index)


if __name__ == "__main__":
    run_pipeline(use_simulator=True)
