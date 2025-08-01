import pandas as pd
import numpy as np
import optuna
from os.path import join, exists # Import 'exists' for checking file existence
# matplotlib.use('Agg')
import datetime

from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.meta.data_processor import DataProcessor

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from pprint import pprint

from finrl import config

import os
import json

from finrl.main import check_and_make_directories
from finrl.config import (
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
)
import itertools
import multiprocessing
from finrl.agents.stablebaselines3.models import DRLAgent,DRLEnsembleAgent

import torch 

from env_train_settings import (TradingEnvBlendSharpeRation, 
                                TradePerformanceMetric, 
                                LoggingCallback, 
                                sample_ddpg_params_all, 
                                sample_sac_params_all, 
                                sample_ppo_params_all, 
                                sample_net_arch
                                )
from data_processing import ( # Data processing 
                            load_data, 
                            fill_by_group_interpolation, 
                             add_date_features_and_onehot,
                             PCA_analisys,  
                             fit_transform_with_autoencoder)
from training_utils import *
import config


class Pipeline(): 
    def __init__(self,
                 tickets, # for now is available with only one tickets
                 start_date,
                 end_date,
                 start_date_trade, 
                 end_date_trade, 

                 auto_encoder_training_params: dict = {},
                 env_params: dict = {},
                 A2C_model_kwargs: dict = {},
                 PPO_model_kwargs: dict = {},
                 DDPG_model_kwargs: dict = {},
                 SAC_model_kwargs: dict = {},
                 TD3_model_kwargs: dict = {},
                 timesteps_dict: dict = {},
                 opt_metrics:dict = {}, 
                

                 compress_data_with_autoencoder = True,
                 one_hot_date_features = True,
                 pca_analisys = True,
                 checkpoint_dir = "pipeline_checkpoint",

                 model_policy = "ppo", 
                 tp_metric = 'avgwl',   # specified trade_param_metric: ratio avg value win/loss
                 default_env = True, 
                 training_total_steps = 50_000, 
                 tickers_in_data = []
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

        # training 
        self.model_policy = model_policy
        self.default_env = default_env
        self.start_date_trade,self.end_date_trade = start_date_trade, end_date_trade
        self.tp_metric = tp_metric
        self.training_total_steps  = training_total_steps
        self.tickers_in_data =tickers_in_data 
        # Initialize all parameter configurations
        self.define_parameters_configs(
            encoder_training_kwargs=auto_encoder_training_params,
            env_training_kwargs=env_params,
            a2c_model_kwargs=A2C_model_kwargs,
            ppo_model_kwargs=PPO_model_kwargs,
            ddpg_model_kwargs=DDPG_model_kwargs,
            sac_model_kwargs=SAC_model_kwargs,
            td3_model_kwargs=TD3_model_kwargs,
            timesteps_dict_kwargs=timesteps_dict, 
            opt_metrics=opt_metrics
        )

    def save_config(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(self.checkpoint_dir, "config.json")
        config_dict = self.__dict__.copy()
        
        # Remove any non-serializable items if needed (like functions)
        for k, v in list(config_dict.items()):
            try:
                json.dumps(v)
            except:
                config_dict.pop(k)
        
        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def from_config(cls, filepath):
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

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
            "print_verbosity": 5, 
            "num_stock_shares": 2, 
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

    def actual_data_processing(self, df):
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
            # Since there supposed to the saved in checkpoint 
            # Its actually supposed to work there 

            autoencoder_checkpoint_path = os.path.join(self.checkpoint_dir, f"autoencoder_checkpoint_{self.tickets}")
            self.auto_encoder_training_params.update({"checkpoint_dir": autoencoder_checkpoint_path})

            processed_full, trainer = fit_transform_with_autoencoder(processed_full, 
                                                                    **self.auto_encoder_training_params)
        
        return processed_full

    def data_process(self): 
        # load data 
        processed_file_dir = os.path.join(self.checkpoint_dir, "compressed") 
        processed_file_path = os.path.join(processed_file_dir, f"compressed_{self.tickets}.csv")

        if exists(processed_file_path): 
            processed_full = pd.read_csv(processed_file_path)
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
            
            processed_full = self.actual_data_processing(df)

            print(processed_full.columns)
            print("---Saving the compressed data into---")
            processed_full.to_csv(processed_file_path, index = False )

        print("---Ready to start training---")
        
        # Sometimes, where we are you using the data from, 2010
        # some companies may not have any data of stocks, because they didnt had stocks 
        # So, to know which tickers exactly is in data where for the next time - we gonna save it
        self.tickers_in_data = list(processed_full['tic'].unique())
        return processed_full, processed_file_path
    
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
        
    def optimize(self, data_path, visualize = True):   
        env_parameters = define_env(
            data_path,
            start_date=self.start_date,
            end_date=self.end_date,
            start_date_trade=self.start_date_trade,
            end_date_trade=self.end_date_trade,
            env_params=self.env_params,
            default_env=self.default_env
        )

        e_train_gym, e_trade_gym = env_parameters['train_env'], env_parameters['trade_env']
        train, trade = env_parameters['train_df'], env_parameters['trade_df']

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
        elif self.model_policy == "ppo": 
            sampling_fn = sample_ppo_params_all    
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

            hyperparameters.update({"device": 'cuda' if torch.cuda.is_available() else 'cpu'})
            model_ddpg = agent.get_model(self.model_policy,
                                        policy_kwargs = policy_kwargs,
                                        model_kwargs = hyperparameters
                                          )
            #You can increase it for better comparison
            trained_ddpg = agent.train_model(model=model_ddpg,
                                            tb_log_name=self.model_policy,
                                            total_timesteps=self.opt_metrics['total_timesteps']
                                            )
            trained_ddpg.save(os.path.join(self.checkpoint_dir, 'models/{}_{}.pth'.format(self.model_policy, trial.number)))
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
        study = optuna.create_study(study_name=f"{self.model_policy}_study",direction='maximize',
                                    sampler = sampler, pruner=optuna.pruners.HyperbandPruner())

        logging_callback = LoggingCallback(threshold=self.opt_metrics['lc_threshold'],
                                        patience=self.opt_metrics['lc_patience'],
                                        trial_number=self.opt_metrics['lc_trial_number'])
        #You can increase the n_trials for a better search space scanning
        study.optimize(objective, n_trials=self.opt_metrics['n_trials'],catch=(ValueError,),callbacks=[logging_callback])


        #Get the best hyperparamters
        print('Hyperparameters after tuning',study.best_params)

        optimization_results_folder = join(self.checkpoint_dir, "opt_results_{}_{}".format(self.model_policy, self.opt_metrics['n_trials']))
        os.makedirs(optimization_results_folder, exist_ok=True)
        # Save the best parameters to a JSON file
        saving_path = os.path.join(optimization_results_folder, "optimization_best_parameters.json")
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
            fig.write_image(os.path.join(optimization_results_folder, "opt_hist.png"))

            try:
                fig = plot_param_importances(study)
                fig.write_image(os.path.join(optimization_results_folder,"params_importances.png"))
            except:
                print('Cannot calculate hyperparameter importances: no variation')
            
            fig = plot_edf(study)
            fig.write_image(os.path.join(optimization_results_folder, "emp_dist_func.png"))

    def train(self, data_path, compare_to_default=True):
        
        if compare_to_default: 
            print("---First, training with default parameters---")
            log_dir_default = join(self.checkpoint_dir, "training_logs", '{}_{}'.format(self.model_policy, self.training_total_steps))
            os.makedirs(log_dir_default, exist_ok=True)

            info = {
            "data_path": data_path,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "start_date_trade": self.start_date_trade,
            "end_date_trade": self.end_date,
            "env_params": self.env_params,
            "default_env": self.default_env,
            "model_policy": self.model_policy,
            "training_total_steps": self.training_total_steps,
            "tpm_metric": self.tp_metric,
            "dir": '', 
            "checkpoint_dir": self.checkpoint_dir}

            tmp_default, model_save_path = train_from_params_path(info)
            print("---For comparison, the default training was over with results---")
            print("---Default Model with score performance {} was saved into {}---".format(tmp_default, model_save_path))


        optimization_result_dirs = [
            join(self.checkpoint_dir, d) for d in os.listdir(self.checkpoint_dir)
            if os.path.isdir(os.path.join(self.checkpoint_dir, d)) and
               d.startswith("opt_results") and
               self.model_policy in d
        ]
        infoss = [{
            "data_path": data_path,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "start_date_trade": self.start_date_trade,
            "end_date_trade": self.end_date,
            "env_params": self.env_params,
            "default_env": self.default_env,
            "model_policy": self.model_policy,
            "training_total_steps": self.training_total_steps,
            "tpm_metric": self.tp_metric,
            "dir": dir, 
            "checkpoint_dir": self.checkpoint_dir
            } for dir in optimization_result_dirs]

        num_workers = 5 # multiprocessing.cpu_count()
        print("---{} Processes will work in paralell---".format(num_workers))
        with multiprocessing.Pool(num_workers) as pool:
            training_results = pool.map(train_from_params_path, infoss)

        sorted_training_results = sorted(training_results, key=lambda i: i['score'])
        print("---MultiProcessedTraining was over with this results---")
        pprint(sorted_training_results)
        
        return sorted_training_results

    def validate_saved_models(self, data_path):
        # 1️⃣ Locate all your saved model files
        model_dir = os.path.join(self.checkpoint_dir, "models")
        model_files = [
            f for f in os.listdir(model_dir)
            if f.endswith(".pth") or f.endswith(".zip")
        ]
        if not model_files:
            print("No saved models found in", model_dir)
            return

        # 2️⃣ Build a list of `info` dicts—one per model
        info_list = []
        for fname in model_files:
            info = {
                **{
                    "data_path":        data_path,
                    "start_date":       self.start_date,
                    "end_date":         self.end_date,
                    "start_date_trade": self.start_date_trade,
                    "end_date_trade":   self.end_date_trade,
                    "env_params":       self.env_params,
                    "default_env":      self.default_env,
                    "model_policy":     self.model_policy,
                    "tpm_metric":       self.tp_metric,
                },
                "model_path":      os.path.join(model_dir, fname),
                "checkpoint_path": self.checkpoint_dir,
            }
            info_list.append(info)

        # 3️⃣ Fire up a pool to validate them in parallel!
        print(f"Starting validation of {len(info_list)} models…")
        with multiprocessing.Pool(processes=min(len(info_list), os.cpu_count())) as pool:
            pool.map(validate_model_by_path, info_list)

        print("All validations complete, sweetie~ 💖")

    def predict(self, data=None, data_is_ready=False, remove_temp_data = True): 
        """
            Arguments : 
                data - Dataframe of stocks prices and other teachnical indicators 
                data
            Returns 
        """
        # Since we using the multiprocessing and cant send the classes like DataFrame into the Pool 
        # We gonna create a tamporarely folder to save all temporarell data 
        temp_dir = join(self.checkpoint_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True )

        # Process data if its not processed already 
        if data_is_ready==False: 
            
            if "tic" not in data.columns:
                raise ValueError("Input data must contain a 'tic' column for ticker symbols.")

            # Keep only rows with tickers that are in self.tickers_in_data
            data = data[data["tic"].isin(self.tickers_in_data)].copy()

            # Optionally sort by date + ticker to keep it organized
            data.sort_values(by=["date", "tic"], inplace=True)


            processed_data = self.actual_data_processing(data)
        else: 
            processed_data = data 
        processed_data_path = join(temp_dir, "data.csv")
        processed_data.to_csv(processed_data_path, index=False)

        # Locate all saved model files
        model_dir = os.path.join(self.checkpoint_dir, "models")
        model_files = [
            f for f in os.listdir(model_dir)
            if f.endswith(".pth") or f.endswith(".zip")
        ]
        if not model_files:
            print("No saved models found in", model_dir)
            return

        info_list = []
        for fname in model_files:
            info = {
                **{
                    "env_params":       self.env_params,
                    "default_env":      self.default_env,
                    "model_policy":     self.model_policy,
                },
                "data_path": processed_data_path, 
                "model_path": os.path.join(model_dir, fname),
                "output_saving_dir": join(temp_dir, os.path.basename(fname)),
            }
            info_list.append(info)

        print(f"Starting prediction with {len(info_list)} models…")
        with multiprocessing.Pool(processes=min(len(info_list), os.cpu_count())) as pool:
            outputs = pool.map(predict_model_path_data_by_path, info_list)


        if remove_temp_data: 
            import shutil
            shutil.rmtree(temp_dir)
        
        return outputs

def compare_validation_results(validation_tables_path: list):
    """
    Arguments: 
        validation_tables_path must be like
        [
            {"name": "run_number1", 
             "path": "run_number1_results.csv", 
             "description": "What trained with compression"},
            ...
        ]

    Returns: 
        Saves the summary table and comparison plots with results
        Returns dict with paths to saved objects
    """
    results_df = pd.DataFrame()

    for result in validation_tables_path:
        df = pd.read_csv(result["path"], index_col=0)
        df = df.rename(columns={"Value": result["name"]})
        results_df = pd.concat([results_df, df[result["name"]]], axis=1)

    # Transpose for easier plotting
    results_df_T = results_df.T
    results_df_T["Description"] = [x["description"] for x in validation_tables_path]

    os.makedirs("comparison_results", exist_ok=True)
    summary_path = "comparison_results/summary_table.csv"
    results_df_T.to_csv(summary_path, index=False)

    # Plot Sharpe ratio & Trade performance comparison
    plot_path = "comparison_results/sharpe_trade_perf_comparison.png"
    fig, ax1 = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    index = range(len(results_df_T))
    
    ax1.bar([i - bar_width/2 for i in index], results_df_T["Sharpe ratio"], 
            bar_width, label='Sharpe Ratio', color='skyblue')
    ax1.bar([i + bar_width/2 for i in index], results_df_T["Trade_Perf"], 
            bar_width, label='Trade Performance', color='salmon')

    ax1.set_xticks(index)
    ax1.set_xticklabels(results_df_T.index, rotation=45, ha='right')
    ax1.set_ylabel("Metrics")
    ax1.set_title("Sharpe Ratio & Trade Performance Comparison")
    ax1.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return {
        "summary_table": summary_path,
        "plot": plot_path
    }



if __name__ == "__main__":
    # Compare the validation results 
    # models_data = [
    #         dict(
    #             path = r"non_compressed_checkpoint\validation_results\perf_stats_opt_results_ppo_5_20000.pth.csv", 
    #             description = "Optimization with 5 Trials, with 20_000 steps of training", 
    #             name = "optimized_non_compressed", 
    #              ), 
    #         dict(
    #             path = r"non_compressed_checkpoint\validation_results\perf_stats_ppo_20000_20000.pth.csv", 
    #             description = "Default hiperparameters, with 20_000 steps of training",
    #             name = "default_non_compressed", 
    #              ), 
    #         dict(
    #             path = r"non_pca_compressed_checkpoint\validation_results\perf_stats_opt_results_ppo_5_20000.pth.csv", 
    #             description = "No PCA analisys in dataprocessing, Optimized with 5 Trials, with 20_000 steps of training",
    #             name = "optimized_non_compressed_non_pca", 
    #              ), 
    #         dict(
    #             path = r"non_pca_compressed_checkpoint\validation_results\perf_stats_ppo_20000_20000.pth.csv", 
    #             description = "No PCA analisys in dataprocessing, Default parameters, with 20_000 steps of training",
    #             name = "default_non_compressed_non_pca", 
    #              ), 

    #         dict(
    #             path = r"pipeline_checkpoint\validation_results\perf_stats_opt_results_ppo_5_20000.pth.csv", 
    #             description = "Data processing with Data one hot encodeing, PCA analisys, and compression with autoencoders (15 epochs), Optimized with 5 trials, with 20_000 steps of training",
    #             name = "optimized_full_compressed_data_15e", 
    #              ), 

    #         dict(
    #             path = r"compressed_checkpoint_20e\validation_results\perf_stats_opt_results_ppo_5_20000.pth.csv", 
    #             description = "Data processing with Data one hot encodeing, PCA analisys, and compression with autoencoders (20 epochs), Optimized with 5 trials, with 20_000 steps of training",
    #             name = "optimized_full_compressed_data_20e", 
    #              ), 
    #         dict(
    #             path = r"compressed_checkpoint_20e\validation_results\perf_stats_ppo_20000_20000.pth.csv", 
    #             description = "Data processing with Data one hot encodeing, PCA analisys, and compression with autoencoders (20 epochs), Default Parameters, with 20_000 steps of training",
    #             name = "default_full_compressed_data_20e", 
    #              ), 
    #     ]
    
    pipe = Pipeline.from_config("non_compressed_checkpoint\config.json")
    data = load_data(start_time='2023-01-01', end_time='2025-07-31', stock_names=None)
    r = pipe.predict(data, remove_temp_data=False)
    print(r)