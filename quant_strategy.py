"""
ML Quantitative Equity Strategy
=================================
Full pipeline: feature engineering, walk-forward validation, backtesting,
Monte Carlo simulation, Sharpe optimization, and live signal generation.

Usage:
    # Full backtest and analysis
    python quant_strategy.py

    # Get today's trading signal
    python quant_strategy.py --signal

pip install yfinance scikit-learn pandas numpy matplotlib seaborn joblib
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from joblib import dump, load
from datetime import datetime, timedelta

import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ── Config ─────────────────────────────────────────────────────────────────────
TICKER         = "SPY"
START_DATE     = "2015-01-01"
TRAIN_END      = "2021-12-31"
VAL_END        = "2023-06-30"
PROB_THRESHOLD = 0.55
TCOST_BPS      = 5
RANDOM_STATE   = 42
MODEL_PATH     = Path("trained_model.joblib")
SCALER_PATH    = Path("trained_scaler.joblib")

plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor": "#141414",
    "axes.edgecolor": "#2a2a2a",
    "axes.labelcolor": "#e8e8e8",
    "text.color": "#e8e8e8",
    "xtick.color": "#888888",
    "ytick.color": "#888888",
    "grid.color": "#1a1a1a",
    "grid.linestyle": "--",
    "font.family": "monospace",
})

COLORS = {"green": "#00ff87", "blue": "#00b4d8", "red": "#ff4757",
          "yellow": "#ffd166", "gray": "#555555"}

# ── 1. Data Download ───────────────────────────────────────────────────────────
def download_data(ticker=TICKER, start=START_DATE):
    print(f"Downloading {ticker} from {start}...")
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    print(f"Downloaded {len(df)} rows through {df.index[-1].date()}")
    return df

# ── 2. Feature Engineering ─────────────────────────────────────────────────────
def build_features(df):
    px = df["close"].squeeze()
    hi = df["high"].squeeze()
    lo = df["low"].squeeze()
    vol = df["volume"].squeeze()
    feat = pd.DataFrame(index=df.index)

    # Returns
    for k in [1, 3, 5, 10, 21]:
        feat[f"ret_{k}d"] = px.pct_change(k)

    # Volatility
    feat["vol_10d"]  = feat["ret_1d"].rolling(10).std()
    feat["vol_21d"]  = feat["ret_1d"].rolling(21).std()
    feat["vol_ratio"] = feat["vol_5d"] = feat["ret_1d"].rolling(5).std()

    # Moving averages
    for w in [5, 10, 20, 50]:
        feat[f"ma_{w}"] = px.rolling(w).mean() / px - 1

    # RSI
    delta = px.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feat["rsi_14"] = 100 - (100 / (1 + rs))
    feat["rsi_norm"] = (feat["rsi_14"] - 50) / 50

    # Bollinger Bands
    ma20 = px.rolling(20).mean()
    std20 = px.rolling(20).std()
    feat["bb_upper"] = (px - (ma20 + 2 * std20)) / px
    feat["bb_lower"] = (px - (ma20 - 2 * std20)) / px
    feat["bb_width"] = (4 * std20) / ma20

    # MACD
    ema12 = px.ewm(span=12, adjust=False).mean()
    ema26 = px.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    feat["macd_hist"] = (macd - signal_line) / px

    # Volume features
    feat["vol_ratio_20d"] = vol / vol.rolling(20).mean()
    feat["vol_trend"] = vol.pct_change(5)

    # ATR (volatility regime)
    tr = pd.concat([
        hi - lo,
        (hi - px.shift()).abs(),
        (lo - px.shift()).abs()
    ], axis=1).max(axis=1)
    feat["atr_14"] = tr.rolling(14).mean() / px

    # Rolling beta vs market (SPY IS the market, but useful if you extend to other tickers)
    spy_ret = px.pct_change()
    cov = spy_ret.rolling(60).cov(spy_ret)
    var = spy_ret.rolling(60).var()
    feat["beta_60d"] = cov / var.replace(0, np.nan)

    # Target: next day up or down
    feat["target"] = (px.pct_change(1).shift(-1) > 0).astype(int)
    feat["next_ret"] = px.pct_change(1).shift(-1)
    feat["close"] = px

    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
    return feat

# ── 3. Walk-Forward Validation ─────────────────────────────────────────────────
def walk_forward_backtest(feat, n_splits=5):
    """
    Walk-forward validation: train on expanding window, test on next period.
    More realistic than a single train/test split for time series.
    """
    print("\nRunning walk-forward validation...")

    total_len = len(feat)
    split_size = total_len // (n_splits + 1)
    min_train = split_size * 2

    all_probas = []
    all_actuals = []
    all_dates = []
    aucs = []

    feature_cols = [c for c in feat.columns if c not in ["target", "next_ret", "close"]]

    for i in range(n_splits):
        train_end_idx = min_train + i * split_size
        test_end_idx = min(train_end_idx + split_size, total_len - 1)

        train = feat.iloc[:train_end_idx]
        test = feat.iloc[train_end_idx:test_end_idx]

        if len(test) < 20:
            continue

        X_train = train[feature_cols]
        y_train = train["target"]
        X_test = test[feature_cols]
        y_test = test["target"]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20,
            random_state=RANDOM_STATE, n_jobs=-1
        )
        model.fit(X_train_s, y_train)
        proba = model.predict_proba(X_test_s)[:, 1]

        auc = roc_auc_score(y_test, proba)
        aucs.append(auc)
        all_probas.extend(proba)
        all_actuals.extend(y_test.values)
        all_dates.extend(test.index)

        print(f"  Fold {i+1}: train={len(train)} test={len(test)} ROC-AUC={auc:.4f}")

    print(f"\nMean walk-forward ROC-AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")
    return pd.Series(all_probas, index=all_dates), np.mean(aucs)

# ── 4. Train Final Model ───────────────────────────────────────────────────────
def train_final_model(feat):
    feature_cols = [c for c in feat.columns if c not in ["target", "next_ret", "close"]]

    train = feat[feat.index <= TRAIN_END]
    X_train = train[feature_cols]
    y_train = train["target"]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=500, max_depth=6, min_samples_leaf=20,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train_s, y_train)

    dump(model, MODEL_PATH)
    dump(scaler, SCALER_PATH)
    print(f"\nFinal model saved: {MODEL_PATH}")
    return model, scaler

# ── 5. Backtest ────────────────────────────────────────────────────────────────
def run_backtest(feat, model, scaler):
    feature_cols = [c for c in feat.columns if c not in ["target", "next_ret", "close"]]
    test = feat[feat.index > VAL_END]

    if len(test) == 0:
        test = feat[feat.index > TRAIN_END]

    X_test = scaler.transform(test[feature_cols])
    proba = model.predict_proba(X_test)[:, 1]
    signal = (proba > PROB_THRESHOLD).astype(int)

    next_ret = test["next_ret"].fillna(0)
    turnover = np.abs(np.diff(signal, prepend=signal[0]))
    tcost = turnover * (TCOST_BPS / 10000)

    strat_ret = signal * next_ret - tcost
    bh_ret = next_ret

    results = pd.DataFrame({
        "proba": proba,
        "signal": signal,
        "next_ret": next_ret,
        "strat_ret": strat_ret,
        "bh_ret": bh_ret,
        "strat_curve": (1 + strat_ret).cumprod(),
        "bh_curve": (1 + bh_ret).cumprod(),
    }, index=test.index)

    return results

# ── 6. Performance Metrics ─────────────────────────────────────────────────────
def compute_metrics(results):
    strat = results["strat_ret"]
    bh = results["bh_ret"]
    trading_days = 252

    def sharpe(r): return (r.mean() / r.std()) * np.sqrt(trading_days) if r.std() > 0 else 0
    def max_dd(curve):
        roll_max = curve.cummax()
        dd = (curve - roll_max) / roll_max
        return dd.min()
    def ann_return(r): return (1 + r.mean()) ** trading_days - 1
    def calmar(r, curve): return ann_return(r) / abs(max_dd(curve)) if max_dd(curve) != 0 else 0

    strat_curve = results["strat_curve"]
    bh_curve = results["bh_curve"]

    metrics = {
        "Strategy Annual Return": f"{ann_return(strat)*100:.1f}%",
        "Buy & Hold Annual Return": f"{ann_return(bh)*100:.1f}%",
        "Strategy Sharpe Ratio": f"{sharpe(strat):.3f}",
        "Buy & Hold Sharpe Ratio": f"{sharpe(bh):.3f}",
        "Strategy Max Drawdown": f"{max_dd(strat_curve)*100:.1f}%",
        "Buy & Hold Max Drawdown": f"{max_dd(bh_curve)*100:.1f}%",
        "Strategy Calmar Ratio": f"{calmar(strat, strat_curve):.3f}",
        "Win Rate": f"{(strat > 0).mean()*100:.1f}%",
        "Days in Market": f"{(results['signal'] == 1).mean()*100:.1f}%",
        "Total Trades": str(int(np.abs(np.diff(results['signal'].values)).sum())),
    }

    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    for k, v in metrics.items():
        print(f"{k:<35} {v}")
    return metrics

# ── 7. Monte Carlo Simulation ──────────────────────────────────────────────────
def monte_carlo(results, n_sims=1000):
    print("\nRunning Monte Carlo simulation...")
    daily_returns = results["strat_ret"].values
    n_days = len(daily_returns)
    sims = np.zeros((n_sims, n_days))

    for i in range(n_sims):
        resampled = np.random.choice(daily_returns, size=n_days, replace=True)
        sims[i] = (1 + resampled).cumprod()

    return sims

# ── 8. Generate All Plots ──────────────────────────────────────────────────────
def generate_plots(results, wf_probas, mc_sims, feat):
    print("\nGenerating plots...")

    # ── Plot 1: Equity Curve ──
    fig, ax = plt.subplots(figsize=(14, 7))
    results["strat_curve"].plot(ax=ax, color=COLORS["green"], lw=2, label="ML Strategy")
    results["bh_curve"].plot(ax=ax, color=COLORS["blue"], lw=2, label="Buy & Hold SPY")
    ax.set_title("Equity Curve — ML Strategy vs Buy & Hold", color=COLORS["green"], fontsize=14)
    ax.set_ylabel("Portfolio Value ($1 start)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("equity_curve.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close()

    # ── Plot 2: Drawdown ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Drawdown Analysis", color=COLORS["green"], fontsize=14)

    for curve, color, label in [
        (results["strat_curve"], COLORS["green"], "ML Strategy"),
        (results["bh_curve"], COLORS["blue"], "Buy & Hold")
    ]:
        roll_max = curve.cummax()
        dd = (curve - roll_max) / roll_max
        ax1.fill_between(dd.index, dd, 0, alpha=0.4, color=color, label=label)
        ax1.plot(dd.index, dd, color=color, lw=1)

    ax1.set_ylabel("Drawdown")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    results["signal"].plot(ax=ax2, color=COLORS["yellow"], alpha=0.7, drawstyle="steps-post")
    ax2.set_ylabel("Signal (1=Long, 0=Cash)")
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("drawdown_signal.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close()

    # ── Plot 3: Monte Carlo ──
    fig, ax = plt.subplots(figsize=(14, 7))
    for i in range(min(200, len(mc_sims))):
        ax.plot(mc_sims[i], color=COLORS["green"], alpha=0.03, lw=0.5)

    percentiles = np.percentile(mc_sims, [5, 25, 50, 75, 95], axis=0)
    labels = ["5th pct", "25th pct", "Median", "75th pct", "95th pct"]
    colors_p = [COLORS["red"], COLORS["yellow"], COLORS["green"], COLORS["blue"], "#c77dff"]

    for pct, label, color in zip(percentiles, labels, colors_p):
        ax.plot(pct, color=color, lw=2, label=label)

    ax.axhline(y=1.0, color=COLORS["gray"], linestyle="--", lw=1)
    ax.set_title("Monte Carlo Simulation — 1,000 Resampled Paths", color=COLORS["green"], fontsize=14)
    ax.set_ylabel("Portfolio Value")
    ax.set_xlabel("Trading Days")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("monte_carlo.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close()

    # ── Plot 4: Feature Importance ──
    feature_cols = [c for c in feat.columns if c not in ["target", "next_ret", "close"]]
    train = feat[feat.index <= TRAIN_END]
    scaler = load(SCALER_PATH)
    model = load(MODEL_PATH)

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(importances.index, importances.values, color=COLORS["green"], alpha=0.85)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top 15 Feature Importances — Random Forest", color=COLORS["green"], fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close()

    # ── Plot 5: Rolling Sharpe ──
    fig, ax = plt.subplots(figsize=(14, 5))
    rolling_sharpe = results["strat_ret"].rolling(63).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() > 0 else 0
    )
    rolling_sharpe_bh = results["bh_ret"].rolling(63).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() > 0 else 0
    )
    rolling_sharpe.plot(ax=ax, color=COLORS["green"], lw=1.5, label="ML Strategy")
    rolling_sharpe_bh.plot(ax=ax, color=COLORS["blue"], lw=1.5, label="Buy & Hold")
    ax.axhline(y=0, color=COLORS["gray"], linestyle="--", lw=1)
    ax.axhline(y=1, color=COLORS["yellow"], linestyle="--", lw=1, alpha=0.5, label="Sharpe=1")
    ax.set_title("Rolling 63-Day Sharpe Ratio", color=COLORS["green"], fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("rolling_sharpe.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close()

    print("Saved: equity_curve.png, drawdown_signal.png, monte_carlo.png, feature_importance.png, rolling_sharpe.png")

# ── 9. Live Signal Generator ───────────────────────────────────────────────────
def get_live_signal():
    """
    Downloads the latest data and outputs today's trading signal.
    Run this each morning before market open.
    """
    if not MODEL_PATH.exists():
        print("No trained model found. Run the full pipeline first (without --signal flag).")
        return

    print(f"\n{'='*50}")
    print("LIVE TRADING SIGNAL")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    df = download_data(start="2020-01-01")
    feat = build_features(df)

    model = load(MODEL_PATH)
    scaler = load(SCALER_PATH)

    feature_cols = [c for c in feat.columns if c not in ["target", "next_ret", "close"]]
    latest = feat[feature_cols].iloc[-1:]
    latest_scaled = scaler.transform(latest)

    proba = model.predict_proba(latest_scaled)[0][1]
    signal = "LONG SPY" if proba > PROB_THRESHOLD else "CASH / FLAT"
    confidence = "HIGH" if abs(proba - 0.5) > 0.15 else "MODERATE" if abs(proba - 0.5) > 0.07 else "LOW"

    print(f"\nTicker:       {TICKER}")
    print(f"Date:         {feat.index[-1].date()}")
    print(f"Close:        ${feat['close'].iloc[-1]:.2f}")
    print(f"Up Prob:      {proba:.3f}")
    print(f"Threshold:    {PROB_THRESHOLD}")
    print(f"Signal:       {signal}")
    print(f"Confidence:   {confidence}")
    print(f"\nNote: This is a systematic signal, not financial advice.")
    print(f"Always apply your own risk management.\n")

    # Recent feature snapshot
    print("Recent feature snapshot:")
    snapshot_features = ["ret_1d", "ret_5d", "rsi_norm", "ma_5", "ma_20", "bb_width", "vol_10d"]
    for f in snapshot_features:
        if f in feat.columns:
            print(f"  {f:<15} {feat[f].iloc[-1]:.4f}")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", action="store_true", help="Get today's live trading signal")
    args = parser.parse_args()

    if args.signal:
        get_live_signal()
        return

    # Full pipeline
    df = download_data()
    feat = build_features(df)
    print(f"Features built: {feat.shape[1]-3} features, {len(feat)} rows")

    wf_probas, mean_auc = walk_forward_backtest(feat)

    model, scaler = train_final_model(feat)

    results = run_backtest(feat, model, scaler)
    metrics = compute_metrics(results)

    mc_sims = monte_carlo(results)

    generate_plots(results, wf_probas, mc_sims, feat)

    results.to_csv("backtest_results.csv")
    print("\nBacktest results saved: backtest_results.csv")
    print(f"\nTo get today's signal: python quant_strategy.py --signal")

if __name__ == "__main__":
    main()
