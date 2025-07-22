import pandas as pd
import numpy as np
import optuna
import matplotlib
from os.path import join, exists # Import 'exists' for checking file existence
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.meta.data_processor import DataProcessor

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from pprint import pprint

from finrl import config
from finrl import config_tickers

import os
import json

from finrl.main import check_and_make_directories
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)
check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
import itertools
from finrl.agents.stablebaselines3.models import DRLAgent,DRLEnsembleAgent


from env_train_settings import (TradingEnvBlendSharpeRation, 
                                TradePerformanceMetric, 
                                LoggingCallback, 
                                sample_ddpg_params_all, 
                                sample_sac_params_all)
from data_processing import ( # Data processing 
                            load_data, 
                            fill_by_group_interpolation, 
                             add_date_features_and_onehot,
                             PCA_analisys,  
                             fit_transform_with_autoencoder)
import config


class Pipeline(): 
    def __init__(self,
                 tickets, # for now is available with only one tickets
                 start_date,
                 end_date,
                 start_date_trade, 
                 end_date_trade, 

                 encoder_training_kwargs: dict = {},
                 env_training_kwargs: dict = {},
                 a2c_model_kwargs: dict = {},
                 ppo_model_kwargs: dict = {},
                 ddpg_model_kwargs: dict = {},
                 sac_model_kwargs: dict = {},
                 td3_model_kwargs: dict = {},
                 timesteps_dict_kwargs: dict = {},
                 optimization_metrics:dict = {}, 
                

                 compress_data_with_autoencoder = True,
                 one_hot_date_features = True,
                 pca_analisys = True,
                 checkpoint_dir = "pipeline_checkpoint",

                 model_policy = "sac", 
                 tp_metric = 'avgwl',   # specified trade_param_metric: ratio avg value win/loss
                 use_default_env = True, 
                 ):
        """
        Initializes the Pipeline object with data, date ranges, and configurations.

        Args:
            tickets (list): A list of stock tickers.
            start_date (str): The start date for the data in 'YYYY-MM-DD' format.
            end_date (str): The end date for the data in 'YYYY-MM-DD' format.
            encoder_training_kwargs (dict, optional): Kwargs for the autoencoder training. Defaults to {}.
            env_training_kwargs (dict, optional): Kwargs for the trading environment. Defaults to {}.
            a2c_model_kwargs (dict, optional): Kwargs for the A2C model. Defaults to {}.
            ppo_model_kwargs (dict, optional): Kwargs for the PPO model. Defaults to {}.
            ddpg_model_kwargs (dict, optional): Kwargs for the DDPG model. Defaults to {}.
            sac_model_kwargs (dict, optional): Kwargs for the SAC model. Defaults to {}.
            td3_model_kwargs (dict, optional): Kwargs for the TD3 model. Defaults to {}.
            timesteps_dict_kwargs (dict, optional): Kwargs for the training timesteps. Defaults to {}.
            compress_data_with_autoencoder (bool, optional): Flag to use autoencoder. Defaults to True.
            one_hot_date_features (bool, optional): Flag to use one-hot encoding for dates. Defaults to True.
            pca_analisys (bool, optional): Flag to perform PCA analysis. Defaults to True.
            compressed_data_dir (str, optional): Directory for compressed data. Defaults to "compressed".
        """
        # data preprocessing 
        self.checkpoint_dir = checkpoint_dir
        self.tickets = tickets
        self.start_date = start_date
        self.end_date = end_date

        self.compress_data_with_autoencoder = compress_data_with_autoencoder
        self.one_hot_date_features = one_hot_date_features
        self.pca_analisys = pca_analisys

        self.compressed_data_dir = os.path.join(self.checkpoint_dir, "compressed") 
        # training 
        self.model_policy = model_policy
        self.default_env = use_default_env
        self.start_date_trade,self.end_date_trade = start_date_trade, end_date_trade
        self.tp_metric = tp_metric

        # Initialize all parameter configurations
        self.define_parameters_configs(
            encoder_training_kwargs=encoder_training_kwargs,
            env_training_kwargs=env_training_kwargs,
            a2c_model_kwargs=a2c_model_kwargs,
            ppo_model_kwargs=ppo_model_kwargs,
            ddpg_model_kwargs=ddpg_model_kwargs,
            sac_model_kwargs=sac_model_kwargs,
            td3_model_kwargs=td3_model_kwargs,
            timesteps_dict_kwargs=timesteps_dict_kwargs, 
            opt_metrics=optimization_metrics
        )

    def define_parameters_configs(self,
                                  encoder_training_kwargs: dict,
                                  env_training_kwargs: dict,
                                  a2c_model_kwargs: dict,
                                  ppo_model_kwargs: dict,
                                  ddpg_model_kwargs: dict,
                                  sac_model_kwargs: dict,
                                  td3_model_kwargs: dict,
                                  timesteps_dict_kwargs: dict, 
                                  opt_metrics):
        """
        Defines default parameters and updates them with any provided kwargs.
        """
        # --- Autoencoder Training Parameters ---
        self.auto_encoder_training_params = {
            "learning_rate": 5e-4,
            "batch_size": 8,
            "epochs": 150,
            "latent_space": 7,
            "checkpoint_dir": "autoencoder_checkpoints_7",
            "deep": True,
            "tanh": True,
        }
        # Update with user-provided values
        self.auto_encoder_training_params.update(encoder_training_kwargs)

        # --- Ensemble Agent Environment Parameters ---
        self.env_params = {
            "hmax": 100,
            "initial_amount": 1000000,
            "buy_cost_pct": 0.001,
            "sell_cost_pct": 0.001,
            # These values will be defined in the future
            # "state_space": state_space,
            # "stock_dim": stock_dimension,
            # "tech_indicator_list": INDICATORS,
            # "action_space": stock_dimension,
            "reward_scaling": 1e-4,
            "print_verbosity": 5
        }
        # Update with user-provided values
        self.env_params.update(env_training_kwargs)

        # --- A2C Model Parameters ---
        self.A2C_model_kwargs = {
            'n_steps': 5,
            'ent_coef': 0.005,
            'learning_rate': 0.0007
        }
        # Update with user-provided values
        self.A2C_model_kwargs.update(a2c_model_kwargs)

        # --- PPO Model Parameters ---
        self.PPO_model_kwargs = {
            "ent_coef": 0.01,
            "n_steps": 2048,
            "learning_rate": 0.00025,
            "batch_size": 128
        }
        # Update with user-provided values
        self.PPO_model_kwargs.update(ppo_model_kwargs)

        # --- DDPG Model Parameters ---
        self.DDPG_model_kwargs = {
            # "action_noise":"ornstein_uhlenbeck",
            "buffer_size": 10_000,
            "learning_rate": 0.0005,
            "batch_size": 64
        }
        # Update with user-provided values
        self.DDPG_model_kwargs.update(ddpg_model_kwargs)

        # --- SAC Model Parameters ---
        self.SAC_model_kwargs = {
            "batch_size": 64,
            "buffer_size": 100000,
            "learning_rate": 0.0001,
            "learning_starts": 100,
            "ent_coef": "auto_0.1",
        }
        # Update with user-provided values
        self.SAC_model_kwargs.update(sac_model_kwargs)

        # --- TD3 Model Parameters ---
        self.TD3_model_kwargs = {
            "batch_size": 100,
            "buffer_size": 1000000,
            "learning_rate": 0.0001
        }
        # Update with user-provided values
        self.TD3_model_kwargs.update(td3_model_kwargs)

        # --- Timesteps Dictionary ---
        self.timesteps_dict = {
            'a2c': 10_000,
            'ppo': 10_000,
            'ddpg': 10,
            'sac': 10,
            'td3': 10
        }
        # Update with user-provided values
        self.timesteps_dict.update(timesteps_dict_kwargs)

        self.opt_metrics = dict(n_trials = 5,  # number of HP optimization runs
                                total_timesteps = 2000, # per HP optimization run
                                ## Logging callback params
                                lc_threshold=1e-5, 
                                lc_patience=15, 
                                lc_trial_number=5)
        self.opt_metrics.update(opt_metrics)
    

    def save_config(self, ):
        """
        Saves the entire pipeline configuration to a JSON file.

        Args:
            path (str): The file path (including filename) to save the config.
        """
        # Consolidate all configuration attributes into a single dictionary
        config_data = {
            "tickets": self.tickets,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "start_date_trade": self.start_date_trade, 
            "end_date_trade": self.end_date_trade, 
            

            "compress_data_with_autoencoder": self.compress_data_with_autoencoder,
            "one_hot_date_features": self.one_hot_date_features,
            "pca_analisys": self.pca_analisys,

            "checkpoint_dir": self.checkpoint_dir, 
            "compressed_data_dir": self.compressed_data_dir,

            "auto_encoder_training_params": self.auto_encoder_training_params,
            "env_params": self.env_params,
            "A2C_model_kwargs": self.A2C_model_kwargs,
            "PPO_model_kwargs": self.PPO_model_kwargs,
            "DDPG_model_kwargs": self.DDPG_model_kwargs,
            "SAC_model_kwargs": self.SAC_model_kwargs,
            "TD3_model_kwargs": self.TD3_model_kwargs,
            "timesteps_dict": self.timesteps_dict, 
            "optimization_metric": self.opt_metrics,

            "model_policy": self.model_policy,
            "use_default_env": self.default_env, 
            "tp_metric": self.tp_metric

        }
        # Write the dictionary to a file in JSON format
        try:
            path = os.path.join(self.checkpoint_dir, "config.json")
            with open(path, 'w') as f:
                json.dump(config_data, f, indent=4)
            print(f"Configuration successfully saved to {path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")


    def data_process(self): 
        # load data 
        processed_file_dir = self.compressed_data_dir
        processed_file_path = os.path.join(processed_file_dir, f"compressed_{self.tickets}.csv")

        if exists(processed_file_path): 
            compressed_full = pd.read_csv(processed_file_path)
        else: 
            
            # Ensure the data save directory exists
            if not os.path.exists(processed_file_dir):
                os.makedirs(processed_file_dir)
            # create the folder for processed files 


            # load data 
            df = load_data(start_time=self.start_date, 
                        end_time=self.end_date, 
                        stock_names=self.tickets, 
                        )

            # adding some more technical values 
            fe = FeatureEngineer(
                        use_technical_indicator=True,
                        tech_indicator_list = INDICATORS,
                        use_vix=True,
                        use_turbulence=True,
                        user_defined_feature = False)

            processed = fe.preprocess_data(df)

            list_ticker = processed["tic"].unique().tolist()
            list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
            combination = list(itertools.product(list_date,list_ticker))

            processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
            processed_full = processed_full[processed_full['date'].isin(processed['date'])]
            processed_full = processed_full.sort_values(['date','tic'])

            processed_full = fill_by_group_interpolation(processed_full, group_col='tic')

            # Debugging 
            print(processed_full.columns)

            print("---Features added into dataset---")
            print(processed_full.sort_values(['date','tic'],ignore_index=True).head(10))
            print("---Full shape of Dataset---", processed_full.shape)

            if self.pca_analisys: 
                print("---Adding the PCA analisys values---")
                processed_full = PCA_analisys(processed_full)

            if self.one_hot_date_features: 
                processed_full = add_date_features_and_onehot(processed_full)

            print(processed_full.columns)

            if self.compress_data_with_autoencoder: 
                compressed_full, trainer = fit_transform_with_autoencoder(processed_full, 
                                                                        **self.auto_encoder_training_params)
            
            
            print(compressed_full.columns)
            print("---Saving the compressed data into---")
            compressed_full.to_csv(processed_file_path, index = False )

        print("---Ready to start training---")
        return compressed_full
    
    def run_ensemble_validation(self, data):
        """
        Runs ensemble validation of multiple DRL models to determine the optimal trading strategy.
        
        Args:
            data (pd.DataFrame): Input DataFrame containing:
                - A column 'tic' for asset identifiers
                - Feature columns (technical indicators should contain 'enc' in their names)
            output_path (str): File path to save the validation results (CSV format)

        Returns: 
            pd.DataFrame: Results DataFrame containing performance metrics for all tested strategies
        """

        # Calculate environment dimensions
        num_assets = len(data['tic'].unique())
        features = [col for col in data.columns if "enc" in col]
        state_dim = 1 + 2 * num_assets + len(features) * num_assets

        print(f"Assets: {num_assets}, State Dimension: {state_dim}")

        # Configure trading environment
        env_config = {
            "state_space": state_dim,
            "stock_dim": num_assets,
            "tech_indicator_list": features,
            "action_space": num_assets,
            **self.env_params  # Inherit base configuration
        }

        # Set training schedule
        retrain_interval = 63  # Days between model updates
        evaluation_period = 63  # Days for validation/testing

        # Initialize ensemble learner
        strategy_evaluator = DRLEnsembleAgent(
            df=data,
            train_period=(TRAIN_START_DATE, TRAIN_END_DATE),
            val_test_period=(TEST_START_DATE, TEST_END_DATE),
            rebalance_window=retrain_interval,
            validation_window=evaluation_period,
            **env_config
        )

        # Execute multi-model evaluation
        print("\n=== Starting Ensemble Evaluation ===")
        validation_results = strategy_evaluator.run_ensemble_strategy(
            self.A2C_model_kwargs,
            self.PPO_model_kwargs,
            self.DDPG_model_kwargs,
            self.SAC_model_kwargs,
            self.TD3_model_kwargs,
            self.timesteps_dict
        )

        # Persist results
        output_path = os.path.join(self.checkpoint_dir, "ensemble_run_results.csv")
        validation_results.to_csv(output_path)
        print(f"\nResults saved to: {output_path}")

        unique_trade_date = data[(data.date > TEST_START_DATE)&(data.date <= TEST_END_DATE)].date.unique()
        df_trade_date = pd.DataFrame({'datadate':unique_trade_date})

        df_account_value=pd.DataFrame()
        for i in range(retrain_interval+evaluation_period, len(unique_trade_date)+1,retrain_interval):
            temp = pd.read_csv('results/account_value_trade_{}_{}.csv'.format('ensemble',i))
            df_account_value = df_account_value.append(temp,ignore_index=True)
        sharpe=(252**0.5)*df_account_value.account_value.pct_change(1).mean()/df_account_value.account_value.pct_change(1).std()
        print('Sharpe Ratio: ',sharpe)
        df_account_value=df_account_value.join(df_trade_date[evaluation_period:].reset_index(drop=True))
        
        print(df_account_value.head())
        df_account_value.account_value.plot()

        return validation_results, strategy_evaluator

    # there is a code to find the best params for model 
    # it will save the results of optimization, the best params or something like this 
    # and after that the train fn 
    # If will load the params, found by optimization, ifs there is none of them, it will use default ones 
    def optimize(self, data, visualize = True):   
  
        # Okay, first, we split data 
        train = data_split(data, self.start_date, self.end_date)
        trade = data_split(data, self.start_date_trade,self.end_date_trade)

        print(f'Number of training samples: {len(train)}')
        print(f'Number of testing samples: {len(trade)}')


        # Calculate environment dimensions
        num_assets = len(data['tic'].unique())
        features = [col for col in data.columns if "enc" in col]
        state_dim = 1 + 2 * num_assets + len(features) * num_assets

        print(f"Assets: {num_assets}, State Dimension: {state_dim}")

        # Configure trading environment
        env_kwargs = {
            "state_space": state_dim,
            "stock_dim": num_assets,
            "tech_indicator_list": features,
            "action_space": num_assets,
            **self.env_params  # Inherit base configuration
        }

        if self.default_env: 
            #Instantiate the training gym compatible environment
            e_train_gym = StockTradingEnv(df = train, **env_kwargs)
            e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = None, **env_kwargs)
        else: 
            e_train_gym = TradingEnvBlendSharpeRation(df = train, **env_kwargs), 
            #Instantiate the trading environment
            e_trade_gym = TradingEnvBlendSharpeRation(df = trade, turbulence_threshold = None, **env_kwargs)

        #Instantiate the training environment
        # Also instantiate our training gent
        env_train, _ = e_train_gym.get_sb_env()
        #print(type(env_train))
        agent = DRLAgent(env = env_train)
        performance_metric = TradePerformanceMetric()

        if self.model_policy == "sac": 
            sampling_fn = sample_sac_params_all
        elif self.model_policy == "ddpg": 
            sampling_fn = sample_ddpg_params_all
        else: 
            print(f"Policy model {self.model_policy} choosen wrong, only sac or ddpg are available now")

        def objective(trial:optuna.Trial):
            #Trial will suggest a set of hyperparamters from the specified range

            # Optional to optimize larger set of parameters
            # hyperparameters = sample_ddpg_params_all(trial)
            
            # Optimize buffer size, batch size, learning rate
            hyperparameters = sampling_fn(trial)
            #print(f'Hyperparameters from objective: {hyperparameters.keys()}')
            policy_kwargs = None  # default
            if 'policy_kwargs' in hyperparameters.keys():
                policy_kwargs = hyperparameters['policy_kwargs']
                del hyperparameters['policy_kwargs']
                #print(f'Policy keyword arguments {policy_kwargs}')
            model_ddpg = agent.get_model(self.model_policy,
                                        policy_kwargs = policy_kwargs,
                                        model_kwargs = hyperparameters )
            #You can increase it for better comparison
            trained_ddpg = agent.train_model(model=model_ddpg,
                                            tb_log_name=self.model_policy,
                                            total_timesteps=self.opt_metrics['total_timesteps']
                                            )
            trained_ddpg.save('models/{}_{}.pth'.format(self.model_policy, trial.number))
            # clear_output(wait=True)
            #For the given hyperparamters, determine the account value in the trading period
            df_account_value, df_actions = DRLAgent.DRL_prediction(
                model=trained_ddpg, 
                environment = e_trade_gym)
            
            # Calculate trade performance metric
            # Currently ratio of average win and loss market values
            tpm = performance_metric.calc_trade_perf_metric(df_actions,trade,self.tp_metric)

            return tpm
        
                
        #Create a study object and specify the direction as 'maximize'
        #As you want to maximize sharpe
        #Pruner stops not promising iterations
        #Use a pruner, else you will get error related to divergence of model
        #You can also use Multivariate samplere
        #sampler = optuna.samplers.TPESampler(multivarite=True,seed=42)
        sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(study_name="ddpg_study",direction='maximize',
                                    sampler = sampler, pruner=optuna.pruners.HyperbandPruner())

        logging_callback = LoggingCallback(threshold=self.opt_metrics['lc_threshold'],
                                        patience=self.opt_metrics['lc_patience'],
                                        trial_number=self.opt_metrics['lc_trial_number'])
        #You can increase the n_trials for a better search space scanning
        study.optimize(objective, n_trials=self.opt_metrics['n_trials'],catch=(ValueError,),callbacks=[logging_callback])

        #Get the best hyperparamters
        print('Hyperparameters after tuning',study.best_params)

        # Save the best parameters to a JSON file
        saving_path = os.path.join(self.checkpoint_dir, "optimization_best_parameters.json")
        best_params = study.best_params
        with open(saving_path, 'w') as f:
            json.dump(best_params, f, indent=4)

        print(f"Best parameters saved to {saving_path}")

        if visualize: 
            #Certainly you can afford more number of trials for further optimization
            from optuna.visualization import plot_optimization_history
            from optuna.visualization import plot_contour
            from optuna.visualization import plot_edf
            from optuna.visualization import plot_intermediate_values
            from optuna.visualization import plot_optimization_history
            from optuna.visualization import plot_parallel_coordinate
            from optuna.visualization import plot_param_importances
            from optuna.visualization import plot_slice

            fig = plot_optimization_history(study)
            #"./"+config.RESULTS_DIR+
            fig.write_image(os.path.join(self.checkpoint_dir, "opt_hist.png"))

            try:
                fig = plot_param_importances(study)
                fig.write_image(os.path.join(self.checkpoint_dir,"params_importances.png"))
            except:
                print('Cannot calculate hyperparameter importances: no variation')
            
            fig = plot_edf(study)
            fig.write_image(os.path.join(self.checkpoint_dir, "emp_dist_func.png"))
                
    
if __name__ == "__main__": 
    print(INDICATORS)

    # initialize pipeline 
    pipe = Pipeline(tickets=config.stock_names, 
                    end_date=config.TRAIN_END_DATE, 
                    start_date=config.TRAIN_START_DATE, 

                    end_date_trade=config.TRADE_END_DATE, 
                    start_date_trade=config.TRADE_START_DATE, 
                    encoder_training_kwargs = dict(learning_rate=5e-4, 
                                                    batch_size =128, 
                                                    epochs = 15, 
                                                    latent_space=7,
                                                    checkpoint_dir="autoencoder_checkpoints_7", 
                                                    deep = True, )
                    )
    
    compressed_df = pipe.data_process()
    pipe.optimize(compressed_df)
