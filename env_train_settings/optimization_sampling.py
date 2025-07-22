import optuna

def sample_sac_params_all(trial: optuna.Trial,
                          # fixed values from previous study
                          learning_rate=0.0103,
                          batch_size=128,
                          buffer_size=int(1e6)):

    gamma = trial.suggest_categorical("gamma", [0.94, 0.96, 0.98])
    tau = trial.suggest_categorical("tau", [0.005, 0.01, 0.02])  # SAC usually uses lower tau

    train_freq = trial.suggest_categorical("train_freq", [1, 64, 128])  # SAC updates more frequently
    gradient_steps = train_freq

    ent_coef = trial.suggest_categorical("ent_coef", ["auto", 0.1, 0.2, 0.5])  # entropy coefficient

    net_arch_choice = trial.suggest_categorical("net_arch", ["small", "big"])

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [512, 512],
    }[net_arch_choice]

    # Sweet SAC-specific hyperparams 💞
    hyperparams = {
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "gamma": gamma,
        "gradient_steps": gradient_steps,
        "learning_rate": learning_rate,
        "tau": tau,
        "train_freq": train_freq,
        "ent_coef": ent_coef,
        "policy_kwargs": dict(net_arch=net_arch)
    }

    return hyperparams

def sample_ddpg_params_all(trial:optuna.Trial,
                           # fixed values from previous study
                           learning_rate=0.0103,
                           batch_size=128,
                           buffer_size=int(1e6)):

    gamma = trial.suggest_categorical("gamma", [0.94, 0.96, 0.98])
    # Polyak coeff
    tau = trial.suggest_categorical("tau", [0.08, 0.1, 0.12])

    train_freq = trial.suggest_categorical("train_freq", [512,768,1024])
    gradient_steps = train_freq
    
    noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", None])
    noise_std = trial.suggest_categorical("noise_std", [.1,.2,.3] )

    # NOTE: Add "verybig" to net_arch when tuning HER (see TD3)
    net_arch = trial.suggest_categorical("net_arch", ["small", "big"])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [512, 512],
    }[net_arch]
  
    hyperparams = {
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "gamma": gamma,
        "gradient_steps": gradient_steps,
        "learning_rate": learning_rate,
        "tau": tau,
        "train_freq": train_freq,
        #"noise_std": noise_std,
        #"noise_type": noise_type,
        
        "policy_kwargs": dict(net_arch=net_arch)
    }
    return hyperparams

