# 🧠 Reinforcement Learning for GME Trading (TensorFlow + Stable-Baselines)

This project demonstrates how to apply **Reinforcement Learning (RL)** to **GameStop (GME)** stock trading using `gym-anytrading` and **Stable-Baselines (DQN / A2C)** built on **TensorFlow 1.15**.

The notebook trains an RL agent to learn profitable buy/sell actions based on historical stock price data and technical indicators.

---

## 📚 Table of Contents
1. [Overview](#overview)
2. [Environment Setup](#environment-setup)
3. [Dataset](#dataset)
4. [Preprocessing](#preprocessing)
5. [Environment Setup](#environment)
6. [Model Training](#model-training)
7. [Evaluation](#evaluation)
8. [Visualization](#visualization)
9. [Flow Diagram](#flow-diagram)
10. [Extensions](#extensions)
11. [Disclaimer](#disclaimer)

---

## 🧩 Overview

This project explores algorithmic trading through **reinforcement learning** using **Deep Q-Networks (DQN)** and **Actor-Critic (A2C)**.  
The agent interacts with a simulated stock market environment (`gym-anytrading`) to learn optimal trading policies.

The workflow:
1. Load and clean GME stock data.
2. Add technical indicators (SMA, RSI, OBV).
3. Train an RL agent with DQN (`MlpLstmPolicy`).
4. Evaluate performance on unseen data.
5. Visualize trading performance.

---

## ⚙️ Environment Setup

> Recommended: **Python 3.7.x**

### Install dependencies
```bash
pip install tensorflow-gpu==1.15.0 tensorflow==1.15.0 stable-baselines gym-anytrading gym
pip install gdown finta
```

> 🧠 Note: This project uses legacy TensorFlow 1.x and the original `stable-baselines` (not `stable-baselines3`).

---

## 📈 Dataset

- Download GME stock data (CSV) from [MarketWatch](https://www.marketwatch.com/investing/stock/gme/download-data)
- Example date range: **May 19, 2020 – May 19, 2021**
- Expected filename: `gme_data.csv`

Example code:
```python
import pandas as pd

df = pd.read_csv('gme_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)
```

---

## 🧮 Preprocessing

### Fix Volume
```python
df['Volume'] = df['Volume'].apply(lambda x: float(x.replace(",", "")))
```

### Add Technical Indicators
```python
from finta import TA

df['SMA'] = TA.SMA(df, 12)
df['RSI'] = TA.RSI(df)
df['OBV'] = TA.OBV(df)
df.fillna(0, inplace=True)
```

---

## 🏦 Environment

### Base Environment (`stocks-v0`)
```python
import gym
import gym_anytrading

env = gym.make('stocks-v0', df=df, frame_bound=(5,100), window_size=5)
```

- **frame_bound:** defines start & end of the trading period  
- **window_size:** size of lookback window  
- **Action Space:** `[0, 1]` = `[Sell, Buy]`

### Custom Environment with Indicators
```python
from gym_anytrading.envs import StocksEnv

def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Low', 'Volume', 'SMA', 'RSI', 'OBV']].to_numpy()[start:end]
    return prices, signal_features

class MyCustomEnv(StocksEnv):
    _process_data = add_signals

env2 = MyCustomEnv(df=df, window_size=12, frame_bound=(12,50))
```

---

## 🤖 Model Training

### Wrap Environment
```python
from stable_baselines.common.vec_env import DummyVecEnv
env_maker = lambda: gym.make('stocks-v0', df=df, frame_bound=(5,100), window_size=5)
env = DummyVecEnv([env_maker])
```

### Train DQN Agent
```python
from stable_baselines import DQN

model = DQN('MlpLstmPolicy', env, verbose=1)
model.learn(total_timesteps=50000)
```

- **Policy:** MLP + LSTM for temporal features  
- **Timesteps:** 50,000 (tunable)  
- **Alternative:** Try `A2C('MlpPolicy', env)` for Actor-Critic

---

## 🧪 Evaluation

Test on unseen timeframes:
```python
env = gym.make('stocks-v0', df=df, frame_bound=(90,100), window_size=5)
obs = env.reset()

while True:
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        break
```

---

## 📊 Visualization

Render trading performance:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(15,6))
env.render_all()
plt.show()
```

The plot shows:
- Stock price curve
- Buy/Sell points
- Profit and benchmark comparison

---

## 🧠 Flow Diagram

```mermaid
flowchart TD
    A[Start Notebook] --> B[Install Dependencies<br/>TF 1.15, Stable-Baselines, Gym, Finta]
    B --> C[Load GME Data (CSV)]
    C --> D[Preprocess Data<br/>Fix Volume, Convert Date, Sort]
    D --> E[Compute Technical Indicators<br/>SMA, RSI, OBV]
    E --> F[Create Base Env<br/>stocks-v0 (Gym AnyTrading)]
    E --> G[Define Custom Env<br/>MyCustomEnv with Indicators]
    F --> H[Wrap Env using DummyVecEnv]
    G --> H2[Wrap Custom Env (Optional)]
    H --> I[Train DQN Agent<br/>MlpLstmPolicy, 50k Steps]
    I --> J[Test Env 1<br/>(90–100)]
    I --> K[Test Env 2<br/>(120–130)]
    J --> L[Predict Actions<br/>Buy/Sell Decisions]
    K --> L
    L --> M[Evaluate Performance<br/>info: Profit, Max Profit]
    M --> N[Visualize Results<br/>env.render_all()]
    N --> O[End]
```

---

## 🚀 Extensions

- **Try other algorithms:** PPO, A2C, DDPG  
- **Add transaction cost modeling**
- **Introduce position sizing (fractional trades)**
- **Integrate news sentiment / volume spikes**
- **Migrate to TensorFlow 2.x + Stable-Baselines3**
- **Use LSTM stateful policy for long-term trend detection**

---

## ⚠️ Disclaimer

> This project is for **educational purposes only**.  
> It does **not** constitute financial advice or guarantee profit.  
> Real-world trading involves risk and should only be done with professional guidance.

---

## 🏁 Summary

✅ Demonstrated RL-based stock trading  
✅ Built custom Gym environment with indicators  
✅ Trained and evaluated DQN agent  
✅ Visualized trading decisions and profit metrics  

---

**Author:** [Your Name]  
**Frameworks:** TensorFlow 1.15 · Stable-Baselines · Gym · Finta  
**License:** MIT
