# Trading Agent with FinRL, AutoEncoders and multiprocessing 

This project documents the process of training a stock trading agent using a **Reinforcement Learning (RL)** approach with advanced methods.

---

### Tasks

- **Install Dependencies** ✅
    ```bash
    pip install git+[https://github.com/AI4Finance-Foundation/FinRL.git](https://github.com/AI4Finance-Foundation/FinRL.git)
    pip install stockstats
    ```

- **Download Data** ✅

- **Fill NaNs and Add MACD, RSI Scores for Stocks** ✅

- **Add Day of the Week, Day of the Month, etc.** ✅

- **Train the Autoencoder to Compress Huge Data into Simpler Values** ✅

    An **Autoencoder** is used to reduce the dimensionality of the extensive market data. By compressing the raw stock features into a lower-dimensional representation, the Autoencoder helps simplify the input for the reinforcement learning agent. This can lead to faster training and better generalization by capturing the most salient features of the data.

- **Switch Reward Function in FinRL TradingEnv to Sharpe Ratio Reward Function** ✅

    The standard reward function in FinRL's `TradingEnv` was replaced with a **Sharpe Ratio**-based reward. The Sharpe Ratio measures risk-adjusted return, calculated as the average return earned in excess of the risk-free rate per unit of volatility or total risk. By optimizing for Sharpe Ratio, the agent is incentivized to not only maximize returns but also to do so with minimal risk, leading to more robust and stable trading strategies.

- **Add the Optimization Process with Optuna to Find the Best Hyperparameters** ✅

    **Optuna**, an automatic hyperparameter optimization framework, was integrated into the training pipeline. This allows for systematic exploration of the hyperparameter space for the **PPO (Proximal Policy Optimization)** algorithm used in FinRL. Optuna helps identify the optimal combination of hyperparameters that maximize the trading agent's performance (e.g., higher Sharpe Ratio, lower drawdown).

- **validation**
- **GPU**

---

### Optuna Optimization Results of PPO FinRL Training

The following images illustrate the results from the Optuna hyperparameter optimization process for the PPO agent's training.

#### Empirical Distribution Function (EDF) of Optimization History

This plot shows the cumulative distribution of the objective values (e.g., Sharpe Ratio) obtained during the Optuna optimization trials. A steeper curve generally indicates that a significant number of trials achieved higher objective values, suggesting efficient exploration and convergence towards better solutions.

![Empirical Distribution Function](pipeline_checkpoint\opt_results_ppo_50\emp_dist_func.png)

#### Optimization History

This graph visualizes the objective value (e.g., Sharpe Ratio) over each trial during the Optuna optimization. It helps in understanding the search process, showing how the model's performance improved or varied as different hyperparameter combinations were explored. A clear upward trend or stabilization at a high value suggests effective optimization.

![Optimization History](pipeline_checkpoint\opt_results_ppo_50\opt_hist.png)

#### Hyperparameter Importances

This chart indicates the relative importance of each hyperparameter in influencing the objective function (e.g., Sharpe Ratio). Hyperparameters with higher importance scores had a more significant impact on the trading agent's performance during the optimization process, helping to identify which parameters are crucial for fine-tuning.

![Parameter Importances](pipeline_checkpoint\opt_results_ppo_50\params_importances.png)

## 📊 Validation Results Experiments and Results

Here’s a comparison of all models based on validation metrics:

| Model | Annual return | Sharpe ratio | Calmar ratio | Max drawdown | Trade_Perf | Description |
|---|---|---|---|---|---|---|
| optimized_non_compressed | 0.1488 | 1.1651 | 2.0604 | -0.0722 | 3.4673 | Optimization with 5 Trials, with 20_000 steps of training |
| default_non_compressed | 0.1189 | 0.9398 | 1.5720 | -0.0756 | 1.7331 | Default hiperparameters, with 20_000 steps of training |
| optimized_non_compressed_non_pca | 0.1054 | 0.8518 | 1.2520 | -0.0842 | 5.7167 | No PCA analisys in dataprocessing, Optimized with 5 Trials, with 20_000 steps of training |
| default_non_compressed_non_pca | 0.0868 | 0.8896 | 1.3477 | -0.0644 | 2.0119 | No PCA analisys in dataprocessing, Default parameters, with 20_000 steps of training |
| optimized_full_compressed_data_15e | 0.0525 | 0.5673 | 0.6486 | -0.0810 | 2.7690 | Data processing with Data one hot encodeing, PCA analisys, and compression with autoencoders (15 epochs), Optimized with 5 trials, with 20_000 steps of training |
| optimized_full_compressed_data_20e | 0.0946 | 0.9175 | 1.2235 | -0.0774 | 1.7919 | Data processing with Data one hot encodeing, PCA analisys, and compression with autoencoders (20 epochs), Optimized with 5 trials, with 20_000 steps of training |
| default_full_compressed_data_20e | 0.1509 | 1.2093 | 1.8840 | -0.0801 | 2.1637 | Data processing with Data one hot encodeing, PCA analisys, and compression with autoencoders (20 epochs), Default Parameters, with 20_000 steps of training |


### Sharpe Ratio and Trading performance metrics plots
![Sharpe Ratio and Trading performance plots](comparison_results\sharpe_trade_perf_comparison.png)


# Dependencies 

This project was build on Python 3.10.11
