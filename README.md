# Neural Network Stock Analysis

> A project demonstrating **5 classical neural network algorithms** applied to real stock market data - build entirely from scratch, no Keras or TensorFlow.

---

## Overview

This program downloads real stock price data (default: **AAPL, 2018-2024**) and runs it through five neural network algorithms to:

- **Predict** next-day price direction (UP ↑ / DOWN ↓)
- **Cluster** market conditions into regimes (Bearish / Neutral /Bullish)
- **Visualise** all results in a single dashboard chart

All algorithms are implemented from scratch using only `NumPy` - no deep learning libraries.

---

## Algorithms Used

| # | Algorithm | Type | Task |
|---|-----------|------|------|
| 1 | **Perceptron** | Supervised | Binary UP/DOWN classification using Rosenblatt update rule |
| 2 | **ADALINE** | Supervised | Continuous price-direction estimation via LMS / delta rule |
| 3 | **Backpropagation MLP** | Supervised | 4→8→4→1 network with sigmoid activation and gradient descent |
| 4 | **Kohonen SOM** | Unsupervised | 4×4 self-organising map to discover market regimes |
| 5 | **MaxxNet** | Competitive | Winner-take-all lateral inhibition to select dominant regime |

---

## Project Structure

```
nn-stock-demo/
|
├── nn_stock_demo.py      # Main script - all algoritms + visualisation
├── nn_stock_results.png  # Output chart (generated on run)
└── README.md
```

---

## Requirements

- Python 3.8+
- Dependencies:

```bash
pip install yfinance nump matplotlib
```

---

## How to Run

```bash
python nn_stock_demo.py
```

The program will prompt you for three inputs:

```
  Stock ticker  [AAPL] : TSLA
  Start date    [2018-01-01] (YYYY-MM-DD) : 2020-01-01
  End date      [2024-01-01] (YYYY-MM-DD) : 2023-06-01
```

Press **Enter** on any field to use the default value shown in brackets.

### Input Validation
- Wrong date format → re-prompted automatically
- End date before start date → re-prompted with error message
- Range under 6 months → warning shown, still prceeds
- Invalid ticket format → re-prompted

> **Offline fallback:** If `yfinance` cannot reach Yahoo Finance, the program automatically switches to a synthetic price series generated via Geometric Brownian Motion so all algorithms still run.

---

## Output

### Terminal
Each algorithm prints its own section with learning rule, accuracy, weights, and samle predictions:

```
  ============================================================
  ALGORITHM 3 - BACKPROPAGATION MLP
  ============================================================
  Architecture  : 4 → 8 → 4 → 1  (sigmoid activations)
  Learning rule : Gradient descent via chain rule

     Epoch   50  |  Loss: 0.249136  |  Acc: 53.4%
     Epoch  100  |  Loss: 0.249127  |  Acc: 53.4%
     ```

  Training accuracy  : 53.4%
  Test accuracy      : 53.7%
```

### Chart (`nn_stock_result.png`)
An 8-panel dashboard saved in the same folder as the script:

| Panel | Conent |
|-------|--------|
| Top-left (wide) | AAPL price coloured by SOM-detected market regime |
| Top-right | Test accuracy comparison bar chart (all 3 supervised models) |
| Mid-left | Perceptron training error curve |
| Mid-centre | ADALINE MSE lose curve |
| Mid-right | Backprop MLP loss curve |
| Bottom-left | SOM quantisation error over epochs |
| Bottom-centre | SOM regime distribution (pie chart) |
| Bottom-right | MaxNet daily regime - last 200 days |

---

## Feature Engineering

Four features are derived from the raw closing price series:

| Feature | Description |
|---------|-------------|
| `5-day return` | `price[t] / price[t-5] - 1` - short-term momentum |
| `10-day return` | `price[t] / price[t-10] - 1` - medium-term momentum |
| `MA20 ratio` | `price[t] / mean(price[t-20:t]) - 1` - trend indicator |
| `10-day volatility` | STD dev 10-day returns - uncertainty proxy |

All features are normalised to `[0, 1]` before tarining. A bipolar `{-1, +1}` version is used for the Perceptron.

**Label:** `1` if `price[t+1] > price[t]`, else `0`

---

## Results (AAPL 2018-2024)

| Algorithm | Test Accuracy | Notes |
|-----------|---------------|-------|
| Perceptron | ~51% | Linear boundary - cannot capture non-linear stock patterns |
| ADALINE | ~50.7% | Smooth convergence, continuous confidence output |
| Backprop MLP | ~53.7% | Best result - hidden layers find non-linear relationships |
| Kohonen SOM | - (unsupervised) | Discovered 3 regimes: Bear 21%, Neutral 43%, Bull 36% |
| MaxNet | - (competitive) | Selected **Neutral** as overall and recent dominant regime |

> **Why ~50%?** Stock price direction is close to a random walk. Any consistent resukt above 50% represents genuine learned signal. The Backprop MLP's 53-54% accross both train and test is meaningful.

---

## Algorithm Details

### 1. Perceptron
The simple linear classifier. Updates weights only when prediction is wrong:
```
if predicted ≠ target:
    w = w + lr × target × x
    b = b + lr × target
```

### 2. ADALINE (Adaptive Linear Neuron)
Uses the **LMS / Widrow-Hoff delta rule** - updates on the continuous net output before thresholding, giving nore stable convergence:
```
error = target - net_output
w = w + lr × error × x
```

### 3. Backpropagation MLP
A 3-layer network `(4 → 8 → 4 → 1)` trained by the chain rule. Hidden layers use sigmoid activation to model non-linear feature interactions:
```
δ_out = (target - output) × σ'(net_out)
δ_h   = (Wᵀ × δ_out) × σ'(net_h)
W    += lr × δ × activation
```

### 4. Kohonen SOM
A 4×4 grid of neurons that organises itself to reflect the topology of the input space. Each day is mapped to a Best Matching Unit (BMU), and three market regime clusters are extracted:
```
BMU   = argmin ||x - w_i||
w_i  += lr × h(i,BMU) × (x - w_i)
```
where `h` is a Gaussian neighbourhood function that shrinks over epochs.

### 5. MaxNet
Arecurrent competitive network that performs **winner-take-all** selection from the SOM cluster scores:
```
y_i(t+1) = max(0, y_i(t) - ε × Σ_{j≠i} y_j(t))
```
Converges when only one neuron remains active (ε < 1/n).

---

## License

This project is for academic and educational purposes.
