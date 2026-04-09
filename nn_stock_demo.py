"""
=============================================================================
  Neural Network Algorithms — Stock Market Analysis Demo
  Ticker: AAPL  |  Period: 2018–2024
=============================================================================
  Algorithms (all implemented from scratch — no Keras / TensorFlow):
    1. Perceptron              — price direction classification (up/down)
    2. ADALINE                 — adaptive linear prediction (LMS rule)
    3. Backpropagation MLP     — non-linear pattern learning & prediction
    4. Kohonen SOM             — market regime clustering (visualised)
    5. MaxNet (Competitive)    — winner-take-all regime selection
=============================================================================
"""

import numpy as np
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


# =============================================================================
#  SECTION 0 — DATA LOADING & FEATURE ENGINEERING
# =============================================================================

def _synthetic_prices(n=1510, seed=42):
    """
    Generates realistic synthetic AAPL-like prices using geometric Brownian
    motion when yfinance is unavailable (e.g. offline / sandbox environment).
    """
    rng    = np.random.default_rng(seed)
    dt     = 1/252
    mu     = 0.15       # ~15% annual drift
    sigma  = 0.25       # ~25% annual volatility
    S0     = 160.0
    log_r  = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*rng.standard_normal(n)
    prices = S0 * np.exp(np.cumsum(log_r))
    return prices


def load_and_prepare(ticker="AAPL", start="2018-01-01", end="2024-01-01"):
    """
    Download stock data and engineer features for each algorithm.
    Features:
        f0 — 5-day return  (short momentum)
        f1 — 10-day return (medium momentum)
        f2 — 20-day MA ratio (trend: price / 20-day moving avg - 1)
        f3 — 10-day volatility (normalised std of returns)
    Label:
        y = 1 if next day's close > today's close, else 0
    """
    print(separator('='))
    print("  STOCK DATA LOADING")
    print(separator('='))
    print(f"\n  Ticker : {ticker}")
    print(f"  Period : {start}  →  {end}")

    try:
        df = yf.download(ticker, start=start, end=end,
                         progress=False, auto_adjust=True)
        prices = df['Close'].values.flatten().astype(float)
        if len(prices) < 100:
            raise ValueError("Insufficient data")
        print(f"  Loaded : {len(prices)} trading days  [live data]\n")
    except Exception:
        prices = _synthetic_prices(n=1510)
        print(f"  [Note] yfinance unavailable — using synthetic GBM price series.")
        print(f"  Loaded : {len(prices)} trading days  [synthetic data]\n")

    X_raw, y_raw = [], []
    for i in range(20, len(prices) - 1):
        r5   = prices[i] / prices[i-5]  - 1
        r10  = prices[i] / prices[i-10] - 1
        ma20 = prices[i] / np.mean(prices[i-20:i]) - 1
        vol  = np.std(prices[i-10:i] / prices[i-11:i-1] - 1)

        X_raw.append([r5, r10, ma20, vol])
        label = 1 if prices[i+1] > prices[i] else 0
        y_raw.append(label)

    X_raw = np.array(X_raw)
    y_raw = np.array(y_raw)

    # Normalise features to [0, 1]
    X_min = X_raw.min(axis=0)
    X_max = X_raw.max(axis=0)
    X_norm = (X_raw - X_min) / (X_max - X_min + 1e-8)

    # Bipolar version {-1, +1} for Perceptron
    X_bip = X_norm * 2 - 1
    y_bip = np.where(y_raw == 1, 1, -1)

    # Train / test split (80 / 20, no shuffle — time series)
    split = int(0.8 * len(X_norm))
    data = {
        "prices"  : prices,
        "X_norm"  : X_norm,
        "X_bip"   : X_bip,
        "y_bin"   : y_raw,
        "y_bip"   : y_bip,
        "split"   : split,
        "X_train" : X_norm[:split],
        "X_test"  : X_norm[split:],
        "y_train" : y_raw[:split],
        "y_test"  : y_raw[split:],
        "Xb_train": X_bip[:split],
        "Xb_test" : X_bip[split:],
        "yb_train": y_bip[:split],
        "yb_test" : y_bip[split:],
    }

    up_pct = 100 * y_raw.mean()
    print(f"  Samples  : {len(X_norm)} feature vectors (train={split}, test={len(X_norm)-split})")
    print(f"  Up days  : {up_pct:.1f}%   Down days: {100-up_pct:.1f}%")
    print(f"  Features : 5d-return, 10d-return, MA20-ratio, 10d-volatility\n")
    return data


# =============================================================================
#  SECTION 1 — PERCEPTRON
# =============================================================================

class Perceptron:
    """
    Classic Rosenblatt Perceptron (1958).
    Binary linear classifier using a step activation.

    Update rule (bipolar, {-1, +1}):
        if prediction != target:
            w = w + lr * target * x
            b = b + lr * target
    """
    def __init__(self, n_features, lr=0.01):
        self.w  = np.zeros(n_features)
        self.b  = 0.0
        self.lr = lr

    def _activate(self, x):
        return 1 if np.dot(self.w, x) + self.b >= 0 else -1

    def train(self, X, y, epochs=50):
        errors_per_epoch = []
        for _ in range(epochs):
            errs = 0
            for xi, yi in zip(X, y):
                pred = self._activate(xi)
                if pred != yi:
                    self.w += self.lr * yi * xi
                    self.b += self.lr * yi
                    errs   += 1
            errors_per_epoch.append(errs)
        return errors_per_epoch

    def predict(self, X):
        return np.array([self._activate(xi) for xi in X])

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)


def run_perceptron(data):
    print(separator('='))
    print("  ALGORITHM 1 — PERCEPTRON")
    print(separator('='))
    print("  Learning rule : Rosenblatt update  (w += lr * y * x  if misclassified)")
    print("  Task          : Classify next-day price direction  (+1 = UP, -1 = DOWN)\n")

    p = Perceptron(n_features=4, lr=0.01)
    errors = p.train(data["Xb_train"], data["yb_train"], epochs=60)

    train_acc = p.accuracy(data["Xb_train"], data["yb_train"])
    test_acc  = p.accuracy(data["Xb_test"],  data["yb_test"])

    preds = p.predict(data["Xb_test"])

    print(f"  Training accuracy : {train_acc*100:.1f}%")
    print(f"  Test accuracy     : {test_acc*100:.1f}%")
    print(f"  Final weights     : [{', '.join(f'{w:.4f}' for w in p.w)}]")
    print(f"  Bias              : {p.b:.4f}")

    last5 = preds[-5:]
    print(f"\n  Last 5 test predictions : {['UP' if v==1 else 'DOWN' for v in last5]}")
    print()
    return {"errors": errors, "preds": preds, "test_acc": test_acc}


# =============================================================================
#  SECTION 2 — ADALINE
# =============================================================================

class ADALINE:
    """
    ADAptive LInear NEuron — Widrow & Hoff, 1960.
    Uses the LMS (Least Mean Squares) / delta rule.

    Unlike the Perceptron, ADALINE updates weights on the LINEAR output
    (before thresholding), minimising MSE continuously.

    Update rule:
        error = target - net_output    (net = w·x + b)
        w = w + lr * error * x
        b = b + lr * error
    """
    def __init__(self, n_features, lr=0.001):
        self.w  = np.random.randn(n_features) * 0.01
        self.b  = 0.0
        self.lr = lr

    def _net(self, x):
        return np.dot(self.w, x) + self.b

    def train(self, X, y, epochs=100):
        mse_history = []
        for _ in range(epochs):
            epoch_loss = 0
            for xi, yi in zip(X, y):
                net   = self._net(xi)
                err   = yi - net
                self.w += self.lr * err * xi
                self.b += self.lr * err
                epoch_loss += err ** 2
            mse_history.append(epoch_loss / len(X))
        return mse_history

    def predict(self, X):
        nets = np.array([self._net(xi) for xi in X])
        return np.where(nets >= 0.5, 1, 0)    # threshold at 0.5 (binary labels)

    def predict_raw(self, X):
        return np.array([self._net(xi) for xi in X])

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


def run_adaline(data):
    print(separator('='))
    print("  ALGORITHM 2 — ADALINE  (Adaptive Linear Neuron)")
    print(separator('='))
    print("  Learning rule : LMS / Widrow-Hoff delta rule")
    print("  Task          : Continuous price-direction estimation\n")

    ada = ADALINE(n_features=4, lr=0.005)
    mse = ada.train(data["X_train"], data["y_train"], epochs=150)

    train_acc = ada.accuracy(data["X_train"], data["y_train"])
    test_acc  = ada.accuracy(data["X_test"],  data["y_test"])
    raw_out   = ada.predict_raw(data["X_test"])

    print(f"  Initial MSE       : {mse[0]:.6f}")
    print(f"  Final MSE         : {mse[-1]:.6f}")
    print(f"  Training accuracy : {train_acc*100:.1f}%")
    print(f"  Test accuracy     : {test_acc*100:.1f}%")
    print(f"  Weights           : [{', '.join(f'{w:.4f}' for w in ada.w)}]")
    print(f"\n  Confidence on last 5 test samples (raw net output):")
    for v in raw_out[-5:]:
        bar = "▓" * int(min(max(v, 0), 1) * 20)
        print(f"    {v:.4f}  {bar}")
    print()
    return {"mse": mse, "test_acc": test_acc, "raw_out": raw_out}


# =============================================================================
#  SECTION 3 — BACKPROPAGATION MLP
# =============================================================================

class BackpropMLP:
    """
    Multi-Layer Perceptron trained by standard Backpropagation.
    Architecture: 4 → 8 → 4 → 1  (sigmoid hidden, sigmoid output)

    Forward pass:
        z1 = sigmoid(W1·x + b1)
        z2 = sigmoid(W2·z1 + b2)
        o  = sigmoid(W3·z2 + b3)

    Backward pass (chain rule):
        δ_out = (target - o) * σ'(o)
        δ_h2  = (W3^T · δ_out) * σ'(z2)
        δ_h1  = (W2^T · δ_h2)  * σ'(z1)
        W += lr * δ * activation^T
    """
    def __init__(self, lr=0.01):
        # Xavier initialisation
        self.W1 = np.random.randn(8, 4) * np.sqrt(2/4)
        self.b1 = np.zeros((8, 1))
        self.W2 = np.random.randn(4, 8) * np.sqrt(2/8)
        self.b2 = np.zeros((4, 1))
        self.W3 = np.random.randn(1, 4) * np.sqrt(2/4)
        self.b3 = np.zeros((1, 1))
        self.lr = lr

    @staticmethod
    def _sig(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def _sig_d(z):
        s = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        return s * (1 - s)

    def _forward(self, x):
        x  = x.reshape(-1, 1)
        n1 = self.W1 @ x  + self.b1;   z1 = self._sig(n1)
        n2 = self.W2 @ z1 + self.b2;   z2 = self._sig(n2)
        n3 = self.W3 @ z2 + self.b3;   o  = self._sig(n3)
        return x, n1, z1, n2, z2, n3, o

    def _backward(self, x, n1, z1, n2, z2, n3, o, target):
        t      = np.array([[target]])
        d_out  = (t - o)  * self._sig_d(n3)
        d_h2   = (self.W3.T @ d_out) * self._sig_d(n2)
        d_h1   = (self.W2.T @ d_h2)  * self._sig_d(n1)

        self.W3 += self.lr * d_out @ z2.T;  self.b3 += self.lr * d_out
        self.W2 += self.lr * d_h2  @ z1.T;  self.b2 += self.lr * d_h2
        self.W1 += self.lr * d_h1  @ x.T;   self.b1 += self.lr * d_h1

        return float(np.sum((t - o) ** 2))

    def train(self, X, y, epochs=200):
        loss_history = []
        for ep in range(epochs):
            total_loss = 0
            idx = np.random.permutation(len(X))
            for i in idx:
                fwd  = self._forward(X[i])
                loss = self._backward(*fwd, y[i])
                total_loss += loss
            loss_history.append(total_loss / len(X))
            if (ep + 1) % 50 == 0:
                acc = self.accuracy(X, y)
                print(f"    Epoch {ep+1:>4}  |  Loss: {loss_history[-1]:.6f}  |  Acc: {acc*100:.1f}%")
        return loss_history

    def predict_raw(self, X):
        return np.array([self._forward(xi)[-1].flatten()[0] for xi in X])

    def predict(self, X):
        return (self.predict_raw(X) >= 0.5).astype(int)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


def run_backprop(data):
    print(separator('='))
    print("  ALGORITHM 3 — BACKPROPAGATION MLP")
    print(separator('='))
    print("  Architecture  : 4 → 8 → 4 → 1  (sigmoid activations)")
    print("  Learning rule : Gradient descent via chain rule")
    print("  Task          : Predict next-day price direction\n")

    mlp  = BackpropMLP(lr=0.01)
    loss = mlp.train(data["X_train"], data["y_train"], epochs=200)

    train_acc = mlp.accuracy(data["X_train"], data["y_train"])
    test_acc  = mlp.accuracy(data["X_test"],  data["y_test"])
    raw_out   = mlp.predict_raw(data["X_test"])

    print(f"\n  Training accuracy : {train_acc*100:.1f}%")
    print(f"  Test accuracy     : {test_acc*100:.1f}%")
    print(f"  Initial loss      : {loss[0]:.6f}")
    print(f"  Final loss        : {loss[-1]:.6f}")

    print(f"\n  Last 5 test outputs (raw sigmoid)  →  prediction:")
    for v in raw_out[-5:]:
        label = "UP  ↑" if v >= 0.5 else "DOWN ↓"
        bar   = "▓" * int(v * 20)
        print(f"    {v:.4f}  {bar:<20}  {label}")
    print()
    return {"loss": loss, "test_acc": test_acc, "raw_out": raw_out}


# =============================================================================
#  SECTION 4 — KOHONEN SELF-ORGANISING MAP (SOM)
# =============================================================================

class KohonenSOM:
    """
    Kohonen Self-Organising Map — Teuvo Kohonen, 1982.
    Unsupervised competitive learning that clusters feature vectors
    onto a 2D grid while preserving topological relationships.

    Algorithm per training step:
        1. Find Best Matching Unit (BMU): neuron with minimum Euclidean distance to x
        2. Update BMU and its neighbours:
               w_i += lr * h(i, BMU) * (x - w_i)
           where h is a Gaussian neighbourhood function that shrinks over time.
    """
    def __init__(self, grid_h=4, grid_w=4, n_features=4):
        self.grid_h    = grid_h
        self.grid_w    = grid_w
        self.n         = grid_h * grid_w
        self.weights   = np.random.rand(self.n, n_features)
        # 2D positions of each neuron on the grid
        self.positions = np.array([[i, j]
                                   for i in range(grid_h)
                                   for j in range(grid_w)], dtype=float)

    def _bmu(self, x):
        dists = np.linalg.norm(self.weights - x, axis=1)
        return np.argmin(dists)

    def _neighbourhood(self, bmu_idx, sigma):
        bmu_pos = self.positions[bmu_idx]
        dists   = np.linalg.norm(self.positions - bmu_pos, axis=1)
        return np.exp(-(dists ** 2) / (2 * sigma ** 2))

    def train(self, X, epochs=300, lr0=0.5, sigma0=2.0):
        quant_errors = []
        for ep in range(epochs):
            # Decay learning rate and neighbourhood radius
            t     = ep / epochs
            lr    = lr0 * np.exp(-t * 3)
            sigma = max(sigma0 * np.exp(-t * 3), 0.5)

            epoch_err = 0
            for x in X:
                bmu  = self._bmu(x)
                h    = self._neighbourhood(bmu, sigma)
                self.weights += lr * h[:, np.newaxis] * (x - self.weights)
                epoch_err    += np.linalg.norm(x - self.weights[bmu])
            quant_errors.append(epoch_err / len(X))
        return quant_errors

    def map_data(self, X):
        """Return BMU index for each sample."""
        return np.array([self._bmu(x) for x in X])

    def cluster_labels(self, X, n_clusters=3):
        """
        Map each sample to a market regime cluster.
        Uses SOM neuron weights clustered by k-means (manual).
        Returns: cluster ID per sample (0=bearish, 1=neutral, 2=bullish)
        """
        bmus     = self.map_data(X)
        bmu_weights = self.weights[bmus]

        # Simple k-means on the mapped weights (3 clusters = 3 market regimes)
        centres = bmu_weights[np.random.choice(len(bmu_weights), 3, replace=False)]
        for _ in range(50):
            dists    = np.linalg.norm(bmu_weights[:, np.newaxis] - centres[np.newaxis], axis=2)
            clusters = np.argmin(dists, axis=1)
            new_c    = np.array([bmu_weights[clusters == k].mean(axis=0)
                                 if (clusters == k).any() else centres[k]
                                 for k in range(3)])
            if np.allclose(new_c, centres, atol=1e-6):
                break
            centres = new_c

        # Sort clusters by first feature (5d-return) → 0=bear, 1=neutral, 2=bull
        order    = np.argsort(centres[:, 0])
        remap    = {old: new for new, old in enumerate(order)}
        clusters = np.array([remap[c] for c in clusters])
        return clusters, centres[order]


def run_som(data):
    print(separator('='))
    print("  ALGORITHM 4 — KOHONEN SELF-ORGANISING MAP (SOM)")
    print(separator('='))
    print("  Type          : Unsupervised competitive learning")
    print("  Task          : Discover market regimes (Bear / Neutral / Bull)")
    print("  Grid          : 4×4 neurons  |  Features: 4\n")

    som    = KohonenSOM(grid_h=4, grid_w=4, n_features=4)
    errors = som.train(data["X_norm"], epochs=300, lr0=0.5, sigma0=2.0)

    clusters, centres = som.cluster_labels(data["X_norm"], n_clusters=3)
    names             = ["Bearish ↓", "Neutral →", "Bullish ↑"]

    print(f"  Final quantisation error : {errors[-1]:.6f}")
    print(f"\n  Cluster centres (feature space):")
    print(f"  {'Regime':<12} {'5d-ret':>8} {'10d-ret':>9} {'MA-ratio':>10} {'Volatility':>12}")
    print(f"  {'-'*52}")
    for i, (name, c) in enumerate(zip(names, centres)):
        print(f"  {name:<12} {c[0]:>8.4f} {c[1]:>9.4f} {c[2]:>10.4f} {c[3]:>12.4f}")

    for i, name in enumerate(names):
        count = (clusters == i).sum()
        pct   = 100 * count / len(clusters)
        bar   = "█" * int(pct / 2)
        print(f"\n  {name:<12}: {count:>4} days  ({pct:>5.1f}%)  {bar}")
    print()
    return {"clusters": clusters, "centres": centres,
            "errors": errors, "som": som}


# =============================================================================
#  SECTION 5 — MAXNET (COMPETITIVE NETWORK)
# =============================================================================

class MaxNet:
    """
    MaxNet — Lippmann, 1987.
    A simple competitive network that performs winner-take-all (WTA)
    selection: given a set of activations, iteratively suppresses all
    but the strongest until only one neuron remains active.

    Recurrent update:
        y_i(t+1) = max(0, y_i(t) - ε * Σ_{j≠i} y_j(t))
    where ε < 1/n  ensures convergence.
    """
    def __init__(self, n_neurons):
        self.n   = n_neurons
        self.eps = 1.0 / (n_neurons + 1)   # stability condition: ε < 1/n

    def compete(self, activations, max_iter=100):
        y = np.array(activations, dtype=float)
        y = np.maximum(y, 0)     # clamp negatives before starting
        for _ in range(max_iter):
            y_new = np.maximum(0, y - self.eps * (y.sum() - y))
            if np.allclose(y_new, y, atol=1e-8):
                break
            y = y_new
        winner = int(np.argmax(y))
        return winner, y


def run_maxnet(data, som_result):
    """
    Uses MaxNet to select the dominant market regime for any given day.
    The SOM cluster memberships act as initial activations.
    """
    print(separator('='))
    print("  ALGORITHM 5 — MAXNET  (Competitive Network)")
    print(separator('='))
    print("  Type          : Winner-Take-All competitive suppression")
    print("  Task          : Select dominant market regime from SOM cluster scores\n")

    clusters = som_result["clusters"]
    names    = ["Bearish ↓", "Neutral →", "Bullish ↑"]
    net      = MaxNet(n_neurons=3)

    # For demonstration: compute cluster membership scores (fraction in each)
    # then run MaxNet to pick the dominant regime for the full dataset
    # and for the last 30 days (recent market state)
    def regime_scores(mask):
        counts = np.array([(clusters[mask] == i).sum() for i in range(3)], dtype=float)
        return counts / counts.sum()

    # Overall
    all_scores   = regime_scores(np.ones(len(clusters), dtype=bool))
    winner_all, final_all = net.compete(all_scores)

    # Recent 30 days
    recent_mask  = np.zeros(len(clusters), dtype=bool)
    recent_mask[-30:] = True
    rec_scores   = regime_scores(recent_mask)
    winner_rec, final_rec = net.compete(rec_scores)

    print("  Overall market regime scores (all data):")
    for i, (name, s) in enumerate(zip(names, all_scores)):
        bar = "▓" * int(s * 40)
        print(f"    {name:<12}: {s:.4f}  {bar}")
    print(f"  MaxNet winner → {names[winner_all]}\n")

    print("  Recent 30-day regime scores:")
    for i, (name, s) in enumerate(zip(names, rec_scores)):
        bar = "▓" * int(s * 40)
        print(f"    {name:<12}: {s:.4f}  {bar}")
    print(f"  MaxNet winner → {names[winner_rec]}\n")

    # Day-by-day: run MaxNet on a sliding 30-day window
    daily_winners = []
    for i in range(len(clusters)):
        start = max(0, i - 29)
        window_scores = regime_scores(np.arange(start, i+1))
        w, _ = net.compete(window_scores)
        daily_winners.append(w)

    return {"daily_winners": np.array(daily_winners),
            "winner_overall": winner_all,
            "winner_recent": winner_rec}


# =============================================================================
#  SECTION 6 — VISUALISATION
# =============================================================================

def plot_results(data, perc_result, ada_result, bp_result, som_result, max_result):
    prices   = data["prices"]
    split    = data["split"] + 20      # offset for feature window

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0f0f1a')
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.55, wspace=0.38,
                            left=0.07, right=0.97,
                            top=0.93, bottom=0.06)

    regime_colors = ['#e74c3c', '#f39c12', '#2ecc71']
    regime_names  = ['Bearish', 'Neutral', 'Bullish']

    # ── 1. AAPL Price + Market Regimes ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor('#12122b')
    price_slice = prices[20:]      # align with feature window

    clusters    = som_result["clusters"]
    for i, (color, name) in enumerate(zip(regime_colors, regime_names)):
        mask = clusters == i
        idx  = np.where(mask)[0]
        if len(idx):
            ax1.scatter(idx, price_slice[idx], c=color, s=3,
                        alpha=0.5, label=name, zorder=2)

    ax1.plot(range(len(price_slice)), price_slice,
             color='#a0a8d0', linewidth=0.7, alpha=0.6, zorder=1)
    ax1.axvline(data["split"], color='#ffffff', linewidth=1,
                linestyle='--', alpha=0.4, label='Train/Test split')
    ax1.set_title('AAPL Closing Price — SOM Market Regimes', color='white', fontsize=11)
    ax1.set_xlabel('Trading days', color='#aaaacc')
    ax1.set_ylabel('Price (USD)', color='#aaaacc')
    ax1.tick_params(colors='#888899')
    ax1.legend(fontsize=8, facecolor='#1a1a3a', labelcolor='white',
               markerscale=3, loc='upper left')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#333355')

    # ── 2. Accuracy Comparison Bar Chart ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor('#12122b')
    algo_names = ['Perceptron', 'ADALINE', 'Backprop\nMLP']
    accs       = [perc_result["test_acc"]*100,
                  ada_result["test_acc"]*100,
                  bp_result["test_acc"]*100]
    colors_bar = ['#9b59b6', '#3498db', '#1abc9c']
    bars = ax2.bar(algo_names, accs, color=colors_bar, width=0.5, alpha=0.85)
    ax2.set_ylim(0, 100)
    ax2.axhline(50, color='#ff6b6b', linewidth=1, linestyle='--',
                alpha=0.6, label='Baseline (50%)')
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', va='bottom',
                 color='white', fontsize=9)
    ax2.set_title('Test Accuracy Comparison', color='white', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', color='#aaaacc')
    ax2.tick_params(colors='#888899')
    ax2.legend(fontsize=8, facecolor='#1a1a3a', labelcolor='white')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#333355')

    # ── 3. Perceptron Training Errors ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#12122b')
    ax3.plot(perc_result["errors"], color='#9b59b6', linewidth=1.5)
    ax3.fill_between(range(len(perc_result["errors"])),
                     perc_result["errors"], alpha=0.2, color='#9b59b6')
    ax3.set_title('Perceptron — Training Errors', color='white', fontsize=10)
    ax3.set_xlabel('Epoch', color='#aaaacc')
    ax3.set_ylabel('Misclassifications', color='#aaaacc')
    ax3.tick_params(colors='#888899')
    for spine in ax3.spines.values():
        spine.set_edgecolor('#333355')

    # ── 4. ADALINE MSE Curve ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#12122b')
    ax4.plot(ada_result["mse"], color='#3498db', linewidth=1.5)
    ax4.fill_between(range(len(ada_result["mse"])),
                     ada_result["mse"], alpha=0.2, color='#3498db')
    ax4.set_title('ADALINE — MSE Loss Curve', color='white', fontsize=10)
    ax4.set_xlabel('Epoch', color='#aaaacc')
    ax4.set_ylabel('MSE', color='#aaaacc')
    ax4.tick_params(colors='#888899')
    for spine in ax4.spines.values():
        spine.set_edgecolor('#333355')

    # ── 5. Backprop Loss Curve ────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor('#12122b')
    ax5.plot(bp_result["loss"], color='#1abc9c', linewidth=1.5)
    ax5.fill_between(range(len(bp_result["loss"])),
                     bp_result["loss"], alpha=0.2, color='#1abc9c')
    ax5.set_title('Backprop MLP — Loss Curve', color='white', fontsize=10)
    ax5.set_xlabel('Epoch', color='#aaaacc')
    ax5.set_ylabel('MSE Loss', color='#aaaacc')
    ax5.tick_params(colors='#888899')
    for spine in ax5.spines.values():
        spine.set_edgecolor('#333355')

    # ── 6. SOM Quantisation Error ─────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.set_facecolor('#12122b')
    ax6.plot(som_result["errors"], color='#e67e22', linewidth=1.5)
    ax6.fill_between(range(len(som_result["errors"])),
                     som_result["errors"], alpha=0.2, color='#e67e22')
    ax6.set_title('SOM — Quantisation Error', color='white', fontsize=10)
    ax6.set_xlabel('Epoch', color='#aaaacc')
    ax6.set_ylabel('Avg Distance to BMU', color='#aaaacc')
    ax6.tick_params(colors='#888899')
    for spine in ax6.spines.values():
        spine.set_edgecolor('#333355')

    # ── 7. SOM Regime Distribution (Pie) ─────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.set_facecolor('#12122b')
    counts = [(clusters == i).sum() for i in range(3)]
    wedges, texts, autotexts = ax7.pie(
        counts, labels=regime_names, colors=regime_colors,
        autopct='%1.1f%%', startangle=90,
        textprops={'color': 'white', 'fontsize': 9},
        wedgeprops={'edgecolor': '#0f0f1a', 'linewidth': 1.5}
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(8)
    ax7.set_title('SOM Regime Distribution', color='white', fontsize=10)

    # ── 8. MaxNet Daily Regime (recent 200 days) ──────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.set_facecolor('#12122b')
    dw = max_result["daily_winners"][-200:]
    for i, (color, name) in enumerate(zip(regime_colors, regime_names)):
        mask = dw == i
        ax8.fill_between(range(len(dw)), i, i+0.8,
                         where=mask, color=color, alpha=0.7, label=name)
    ax8.set_yticks([0.4, 1.4, 2.4])
    ax8.set_yticklabels(regime_names, color='white', fontsize=8)
    ax8.set_title('MaxNet — Daily Regime (last 200 days)', color='white', fontsize=10)
    ax8.set_xlabel('Recent trading days', color='#aaaacc')
    ax8.tick_params(colors='#888899', left=False)
    for spine in ax8.spines.values():
        spine.set_edgecolor('#333355')

    # Title
    fig.suptitle('Neural Network Stock Analysis — AAPL  (2018–2024)',
                 color='white', fontsize=14, fontweight='bold', y=0.97)

    out = "nn_stock_results.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    print(f"  Chart saved → {out}\n")
    plt.close()


# =============================================================================
#  SECTION 7 — SUMMARY
# =============================================================================

def print_summary(perc, ada, bp, som, maxr):
    names   = ["Bearish ↓", "Neutral →", "Bullish ↑"]
    print(separator('='))
    print("  FINAL SUMMARY")
    print(separator('='))
    print(f"\n  {'Algorithm':<28} {'Test Acc':>10}  {'Notes'}")
    print(f"  {'-'*65}")
    print(f"  {'1. Perceptron':<28} {perc['test_acc']*100:>9.1f}%  Linear classifier, bipolar update")
    print(f"  {'2. ADALINE':<28} {ada['test_acc']*100:>9.1f}%  LMS rule, continuous output")
    print(f"  {'3. Backpropagation MLP':<28} {bp['test_acc']*100:>9.1f}%  4→8→4→1, gradient descent")
    print(f"  {'4. Kohonen SOM':<28} {'unsup':>10}  3 market regimes discovered")
    print(f"  {'5. MaxNet':<28} {'WTA':>10}  Recent regime: {names[maxr['winner_recent']]}")
    print(f"\n  Overall market state (MaxNet): {names[maxr['winner_overall']]}")
    print(f"  Recent  market state (MaxNet): {names[maxr['winner_recent']]}")
    print()


# =============================================================================
#  HELPERS
# =============================================================================

def separator(c='-', n=60):
    return f"\n  {c * n}"


# =============================================================================
#  MAIN
# =============================================================================

def get_user_inputs():
    """
    Interactively prompt the user for ticker, start date, and end date.
    Validates format and logical order before returning.
    """
    from datetime import datetime

    print("\n" + "=" * 62)
    print("  NEURAL NETWORK STOCK ANALYSIS")
    print("  Algorithms: Perceptron | ADALINE | Backprop | SOM | MaxNet")
    print("=" * 62)
    print("\n  Configure your analysis below.")
    print("  Press Enter to accept the default shown in [brackets].\n")

    # Ticker
    while True:
        ticker = input("  Stock ticker  [AAPL] : ").strip().upper()
        if ticker == "":
            ticker = "AAPL"
        if ticker.isalpha() and 1 <= len(ticker) <= 6:
            break
        print("  [!] Enter a valid ticker symbol (e.g. AAPL, TSLA, MSFT).")

    # Start date
    while True:
        raw = input("  Start date    [2018-01-01] (YYYY-MM-DD) : ").strip()
        if raw == "":
            raw = "2018-01-01"
        try:
            start_dt = datetime.strptime(raw, "%Y-%m-%d")
            start = raw
            break
        except ValueError:
            print("  [!] Use YYYY-MM-DD format, e.g. 2015-06-01.")

    # End date
    while True:
        raw = input("  End date      [2024-01-01] (YYYY-MM-DD) : ").strip()
        if raw == "":
            raw = "2024-01-01"
        try:
            end_dt = datetime.strptime(raw, "%Y-%m-%d")
            end = raw
            if end_dt <= start_dt:
                print("  [!] End date must be after start date.")
                continue
            if (end_dt - start_dt).days / 365 < 0.5:
                print("  [!] Warning: Range < 6 months — results may be unreliable.")
            break
        except ValueError:
            print("  [!] Use YYYY-MM-DD format, e.g. 2024-06-01.")

    print(f"\n  Running: {ticker}  |  {start}  to  {end}\n")
    return ticker, start, end


if __name__ == "__main__":
    ticker, start, end = get_user_inputs()

    # Load data
    data = load_and_prepare(ticker, start, end)

    # ── Run each algorithm ────────────────────────────────────────────────────
    perc_result = run_perceptron(data)
    ada_result  = run_adaline(data)
    bp_result   = run_backprop(data)
    som_result  = run_som(data)
    max_result  = run_maxnet(data, som_result)

    # ── Summary + Plot ────────────────────────────────────────────────────────
    print_summary(perc_result, ada_result, bp_result, som_result, max_result)

    print(separator('='))
    print("  GENERATING CHARTS...")
    print(separator('=') + "\n")
    plot_results(data, perc_result, ada_result, bp_result, som_result, max_result)

    print("  Done! All algorithms complete.\n")