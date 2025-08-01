import pandas as pd
import numpy as np
import optuna
import matplotlib
from os.path import join, exists # Import 'exists' for checking file existence
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime

from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.meta.data_processor import DataProcessor

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

import os
import json

from finrl.main import check_and_make_directories
from finrl.config import (
    INDICATORS,
)
from finrl.agents.stablebaselines3.models import DRLAgent,DRLEnsembleAgent
import torch 

from env_train_settings import (TradingEnvBlendSharpeRation, 
                                TradePerformanceMetric,
                                sample_net_arch
                                )

def define_env(data_path, 
               start_date, 
               end_date, 
               start_date_trade, 
               end_date_trade, 
               env_params, 
               default_env=True, 
               trade_only = False ):  
    data  = pd.read_csv(data_path)
    results = {} 

    if trade_only: 
        
        data = data.sort_values(['date', "tic"], ignore_index=True)
        data.index = data['date'].factorize()[0]
        
        trade = data 
        results['trade_df'] = trade
         
        print(f'Number of samples: {len(trade)}')
        
    else: 
        train = data_split(data, start_date, end_date)
        results['train_df'] = train

        trade = data_split(data, start_date_trade, end_date_trade)
        results['trade_df'] = trade 

        print(f'Number of training samples: {len(train)}')
        print(f'Number of testing samples: {len(trade)}')

    # Calculate environment dimensions
    num_assets = len(data['tic'].unique())


    features = [col for col in data.columns if col.startswith("enc")]
    if len(features) > 0: 
        print("---Found Autoencoder compressed columns, using them intead of indicators---")
        state_dim = 1 + 2 * num_assets + len(features) * num_assets
    else: 
        print("---Using the defult finrl indicator---")
        features = INDICATORS
        state_dim = 1 + 2 * num_assets + len(features) * num_assets

    print(f"Assets: {num_assets}, State Dimension: {state_dim}")
    print("IMPORTANT!!!!", data.columns )

    # Configure trading environment
    env_kwargs = {
        "state_space": state_dim,
        "stock_dim": num_assets,
        "tech_indicator_list": features,
        "action_space": num_assets,
        **env_params  # Inherit base configuration
    }
        
    env_kwargs.update({
        "num_stock_shares": [env_params['num_stock_shares']] * num_assets, 
        "sell_cost_pct": [env_params['sell_cost_pct']] * num_assets, 
        "buy_cost_pct": [env_params['buy_cost_pct']] * num_assets, 
    })

    if default_env: 
        if trade_only == False: 
            results['train_env']= StockTradingEnv(df=train, **env_kwargs)

        results['trade_env'] = StockTradingEnv(df=trade, turbulence_threshold=None, **env_kwargs)
    else: 
        if trade_only == False: 
            results['train_env']= TradingEnvBlendSharpeRation(df=train, **env_kwargs)

        results['trade_env'] = TradingEnvBlendSharpeRation(df=trade.reset_index(drop=True), turbulence_threshold=None, **env_kwargs)

    return results



def train_from_params(model_policy, 
                        training_total_steps, 
                        tpm_metric, 
                        agent:DRLAgent, 
                        trade_env, # Enviroment to do validation 
                        trade, # Data to do validation 
                        **kwargs): 
    
    model = agent.get_model(model_policy,
                            **kwargs)
            # Start training 
    # You can increase it for better comparison
    model = agent.train_model(model=model,
                                    tb_log_name=model_policy,
                                    total_timesteps=training_total_steps
                                    )
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=model, 
        environment = trade_env)
    
    performance_metric = TradePerformanceMetric() 
    # Calculate trade performance metric
    # Currently ratio of average win and loss market values
    tpm = performance_metric.calc_trade_perf_metric(df_actions,trade,tpm_metric)
    return tpm, model

def train_from_params_path(info, ):  
    env_parameters = define_env(
        data_path=info['data_path'],
        start_date=info['start_date'],
        end_date=info['end_date'],
        start_date_trade=info['start_date_trade'],
        end_date_trade=info['end_date_trade'],
        env_params=info['env_params'],
        default_env=info.get('default_env', True)
    )

    train_env, trade_env = env_parameters['train_env'], env_parameters['trade_env']
    train, trade = env_parameters['train_df'], env_parameters['trade_df']

    # Instantiate the training environment and the agent
    env_train, _ = train_env.get_sb_env()
    agent = DRLAgent(env=env_train)

    unique_name = os.path.basename(info['dir'])
    best_params_path = os.path.join(info['dir'], "optimization_best_parameters.json")
    
    print(f"Loading saved best parameters from {best_params_path}")

    if len(info['dir']) == 0 : 
        policy_kwargs = {}
        hyperparameters = {}
        log_dir = join(info['checkpoint_dir'], "training_logs", 'default_{}_{}'.format(info['model_policy'], info["training_total_steps"]))
        unique_name = "{}_{}".format(info['model_policy'], info['training_total_steps'])

    else:
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)


        log_dir = join(info['checkpoint_dir'], "training_logs", f'{unique_name}_{info["training_total_steps"]}')
        os.makedirs(log_dir, exist_ok=True)
        print(f"Logs will be saved into {log_dir}")

        net_arch = best_params.pop("net_arch")
        net_arch = sample_net_arch(net_arch)
        policy_kwargs = dict(net_arch=net_arch)
        hyperparameters = best_params

    hyperparameters.update({"device": 'cuda' if torch.cuda.is_available() else 'cpu'})
    tpm, model = train_from_params(
        agent=agent,
        model_policy=info['model_policy'],
        training_total_steps=info['training_total_steps'],
        tpm_metric=info['tpm_metric'],
        trade_env=trade_env,
        trade=trade,
        policy_kwargs=policy_kwargs,
        model_kwargs=hyperparameters,
        tensorboard_log=log_dir
    )

    model_save_name = '{}_{}.pth'.format(unique_name, info['training_total_steps'])
    model_save_path = join(info['checkpoint_dir'],"models", model_save_name)
    model.save(model_save_path)

    return {
        "score": tpm,
        "saved_path": model_save_path
    }


def add_trade_perf_metric(df_actions, 
                          perf_stats_all,
                          trade, 
                          tp_metric):
  metric = TradePerformanceMetric()

  tpm = metric.calc_trade_perf_metric(df_actions,trade,tp_metric)
  trp_metric = {'Value':tpm}
  df2 = pd.DataFrame(trp_metric,index=['Trade_Perf'])
  perf_stats_all = pd.concat([perf_stats_all, df2])
  return perf_stats_all

def validate_model_by_path(info): 
    from stable_baselines3 import PPO, SAC, DDPG

    env_parameters = define_env(
        data_path=info['data_path'],
        start_date=info['start_date'],
        end_date=info['end_date'],
        start_date_trade=info['start_date_trade'],
        end_date_trade=info['end_date_trade'],
        env_params=info['env_params'],
        default_env=info.get('default_env', True)
    )

    _, trade_env = env_parameters['train_env'], env_parameters['trade_env']
    _, trade = env_parameters['train_df'], env_parameters['trade_df']
    
    if info['model_policy'] == "ppo": 
        tuned_model = PPO
    elif info['model_policy'] == "sac": 
        tuned_model = SAC
    else: 
        print("didnt find model policy available")

    tuned_model = tuned_model.load(info['model_path'])

    # load the save PPO model from info ['model_dir']
    df_account_value_tuned, df_actions_tuned = DRLAgent.DRL_prediction(
        model=tuned_model, 
        environment = trade_env)
    
        #Backtesting with our pruned model
    print("==============Get Backtest Results===========")
    print("==============Pruned Model===========")
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

    perf_stats_all_tuned = backtest_stats(account_value=df_account_value_tuned)
    perf_stats_all_tuned = pd.DataFrame(perf_stats_all_tuned)
    perf_stats_all_tuned.columns = ['Value']
    # add trade performance metric
    perf_stats_all_tuned = \
    add_trade_perf_metric(df_actions_tuned,
                            perf_stats_all_tuned,
                            trade,
                            info['tpm_metric'])
    

    result_dir = join(info['checkpoint_path'], 'validation_results')
    results_path = join(result_dir, "perf_stats_" + os.path.basename(info['model_path']) + '.csv')
    os.makedirs(result_dir, exist_ok=True)

    perf_stats_all_tuned.to_csv(results_path)
    print("---Performanse Stats saved into {}---".format(results_path))
    
 
def predict_model_path_data_by_path(info:dict = {}, # must contain 
            # data_path, 
            # model_path, 
            # model_policy, 
            # output_saving_dir, 
            # env_params = {}, 
            # default_env = True, 
            *args, 
            **kwargs
            ): 
    from stable_baselines3 import PPO, SAC, DDPG
    """
    Arguments if info Dictionary: 
        data_path - path to saved data 
        model_path - path to saved model 
        model_policy - the name of models policy, the available model_policy variants now is ["ppo", "sac", and "ddgp"]
        output_saving_dir - folder where the df_account_value, df_actions files will be saved 

        env_params = {}, setting for FinRls Trading Enviroment 
        default_env = True, If True uses the standart FinRLs TradingEnviroment, if False, uses Experimental one
    Returns: 
        the path to saved df_account_value, df_actions results 
    """
    data_path = info.get('data_path')
    model_path = info.get('model_path')
    model_policy = info.get('model_policy')
    output_saving_dir = info.get('output_saving_dir')
    env_params = info.get('env_params', {})
    default_env = info.get('default_env', True)
    # Data processing and envs 
    data = pd.read_csv (data_path)
    env_params = define_env(data_path, 
               start_date = '', 
               end_date = '', 
               start_date_trade = '', 
               end_date_trade = '', 
               env_params = env_params, 
               default_env=default_env, 
               trade_only = True)

    trade_env = env_params['trade_env']

    # Loading model 
    if model_policy == "ppo": 
        tuned_model = PPO
    elif model_policy == "sac": 
        tuned_model = SAC
    else: 
        print("didnt find model policy available")

    tuned_model = tuned_model.load(model_path)

    # Predicting 
    df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=tuned_model,
    environment=trade_env)
    
    # Saving 
    os.makedirs(output_saving_dir, exist_ok=True )
    df_account_value_path = join(output_saving_dir, "account_value.csv")
    df_account_value.to_csv(df_account_value_path, index = False )

    df_actions_path = join(output_saving_dir, "actions.csv")
    df_actions.to_csv(df_actions_path, index=False )
    
    return {"accaunt_value_path": df_account_value_path, 
            "actions_path": df_actions_path, 
            "model_name": os.path.basename(model_path)}
    