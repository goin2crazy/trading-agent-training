# From https://github.com/AI4Finance-Foundation/FinRL-Tutorials/blob/master/4-Optimization/FinRL_HyperparameterTuning_Optuna.ipynb

#Main method
# Calculates Trade Performance for Objective
# Called from objective method
# Returns selected trade perf metric(s)
# Requires actions and associated prices
import pandas as pd
import numpy as np


class TradePerformanceMetric(): 
    def __init__(self):
        self.tpm_hist = {}
        
    def calc_trade_perf_metric(self, df_actions, 
                            df_prices_trade,
                            tp_metric,
                            dbg=False):
    
        df_actions_p, df_prices_p, tics = prep_data(df_actions.copy(),
                                                    df_prices_trade.copy())
        # actions predicted by trained model on trade data
        df_actions_p.to_csv('df_actions.csv') 

        
        # Confirms that actions, prices and tics are consistent
        df_actions_s, df_prices_s, tics_prtfl = \
            sync_tickers(df_actions_p.copy(),df_prices_p.copy(),tics)
        
        # copy to ensure that tics from portfolio remains unchanged
        tics = tics_prtfl.copy()
        
        # Analysis is performed on each portfolio ticker
        perf_data= collect_performance_data(df_actions_s, df_prices_s, tics)
        # profit/loss for each ticker
        pnl_all = calc_pnl_all(perf_data, tics)
        # values for trade performance metrics
        perf_results = calc_trade_perf(pnl_all)
        df = pd.DataFrame.from_dict(perf_results, orient='index')
        
        # calculate and return trade metric value as objective
        m = self.calc_trade_metric(df,tp_metric)
        print(f'Ratio Avg Win/Avg Loss: {m}')
        k = str(len(self.tpm_hist)+1)
        # save metric value
        self.tpm_hist[k] = m
        return m

    # Supporting methods
    def calc_trade_metric(self, df,metric='avgwl'):
        '''# trades', '# wins', '# losses', 'wins total value', 'wins avg value',
        'losses total value', 'losses avg value'''
        # For this tutorial, the only metric available is the ratio of 
        #  average values of winning to losing trades. Others are in development.
        
        # some test cases produce no losing trades.
        # The code below assigns a value as a multiple of the highest value during
        # previous hp optimization runs. If the first run experiences no losses,
        # a fixed value is assigned for the ratio
        tpm_mult = 1.0
        avgwl_no_losses = 25
        if metric == 'avgwl':
            if sum(df['# losses']) == 0:
                try:
                    return max(self.tpm_hist.values())*tpm_mult
                except ValueError:
                    return avgwl_no_losses
        avg_w = sum(df['wins total value'])/sum(df['# wins'])
        avg_l = sum(df['losses total value'])/sum(df['# losses'])
        m = abs(avg_w/avg_l)

        return m


def prep_data(df_actions,
              df_prices_trade):
    
    df=df_prices_trade[['date','close','tic']]
    df['Date'] = pd.to_datetime(df['date'])
    df = df.set_index('Date')
    # set indices on both df to datetime
    idx = pd.to_datetime(df_actions.index, infer_datetime_format=True)
    df_actions.index=idx
    tics = np.unique(df.tic)
    n_tics = len(tics)
    print(f'Number of tickers: {n_tics}')
    print(f'Tickers: {tics}')
    dategr = df.groupby('tic')
    p_d={t:dategr.get_group(t).loc[:,'close'] for t in tics}
    df_prices = pd.DataFrame.from_dict(p_d)
    df_prices.index = df_prices.index.normalize()
    return df_actions, df_prices, tics


# prepares for integrating action and price files
def link_prices_actions(df_a,
                        df_p):
    cols_a = [t + '_a' for t in df_a.columns]
    df_a.columns = cols_a
    cols_p = [t + '_p' for t in df_p.columns]
    df_p.columns = cols_p
    return df_a, df_p


def sync_tickers(df_actions,df_tickers_p,tickers):
    # Some DOW30 components may not be included in portfolio
    # passed tickers includes all DOW30 components
    # actions and ticker files may have different length indices
    if len(df_actions) != len(df_tickers_p):
      msng_dates = set(df_actions.index)^set(df_tickers_p.index)
      try:
        #assumption is prices has one additional timestamp (row)
        df_tickers_p.drop(msng_dates,inplace=True)
      except:
        df_actions.drop(msng_dates,inplace=True)
    df_actions, df_tickers_p = link_prices_actions(df_actions,df_tickers_p)
    # identify any DOW components not in portfolio
    t_not_in_a = [t for t in tickers if t + '_a' not in list(df_actions.columns)]
  
    # remove t_not_in_a from df_tickers_p
    drop_cols = [t + '_p' for t in t_not_in_a]
    df_tickers_p.drop(columns=drop_cols,inplace=True)
    
    # Tickers in portfolio
    tickers_prtfl = [c.split('_')[0] for c in df_actions.columns]
    return df_actions,df_tickers_p, tickers_prtfl

def collect_performance_data(dfa,dfp,tics, dbg=False):
    
    perf_data = {}
    # In current version, files columns include secondary identifier
    for t in tics:
        # actions: purchase/sale of DOW equities
        acts = dfa['_'.join([t,'a'])].values
        # ticker prices
        prices = dfp['_'.join([t,'p'])].values
        # market value of purchases/sales
        tvals_init = np.multiply(acts,prices)
        d={'actions':acts, 'prices':prices,'init_values':tvals_init}
        perf_data[t]=d

    return perf_data


def calc_pnl_all(perf_dict, tics_all):
    # calculate profit/loss for each ticker
    print(f'Calculating profit/loss for each ticker')
    pnl_all = {}
    for tic in tics_all:
        pnl_t = []
        tic_data = perf_dict[tic]
        init_values = tic_data['init_values']
        acts = tic_data['actions']
        prices = tic_data['prices']
        cs = np.cumsum(acts)
        args_s = [i + 1 for i in range(len(cs) - 1) if cs[i + 1] < cs[i]]
        # tic actions with no sales
        if not args_s:
            pnl = complete_calc_buyonly(acts, prices, init_values)
            pnl_all[tic] = pnl
            continue
        # copy acts: acts_rev will be revised based on closing/reducing init positions
        pnl_all = execute_position_sales(tic,acts,prices,args_s,pnl_all)

    return pnl_all


def complete_calc_buyonly(actions, prices, init_values):
    # calculate final pnl for each ticker assuming no sales
    fnl_price = prices[-1]
    final_values = np.multiply(fnl_price, actions)
    pnl = np.subtract(final_values, init_values)
    return pnl

def execute_position_sales(tic, acts, prices, args_s, pnl_all):
    pnl_t = []
    acts_rev = acts.copy()

    for s in args_s:
        act_s = [acts_rev[s]]
        args_b = [i for i in range(s) if acts_rev[i] > 0]  # prior buys

        # 🩷 Add this check:
        if not args_b:
            print(f"[WARNING] No prior buy action for {tic} at step {s}, skipping this sell.")
            continue  # skip this sale if no buys to match

        prcs_init_trades = prices[args_b]
        acts_init_trades = acts_rev[args_b]

        # 🩷 safe now:
        arg_sel = min(args_b)

        acts_shares = min(abs(act_s.pop()), acts_rev[arg_sel])

        mv_p = abs(acts_shares * prices[arg_sel])
        mv_s = abs(acts_shares * prices[s])

        pnl = mv_s - mv_p

        acts_rev[arg_sel] -= acts_shares
        acts_rev[s] += acts_shares

        pnl_t.append(pnl)

    pnl_op = calc_pnl_for_open_positions(acts_rev, prices)
    pnl_t.extend(pnl_op)
    pnl_all[tic] = np.array(pnl_t)
    return pnl_all


def calc_pnl_for_open_positions(acts,prices):
    # identify any positive share values after accounting for sales
    pnl = []
    fp = prices[-1] # last price
    open_pos_arg = np.argwhere(acts>0)
    if len(open_pos_arg)==0:return pnl # no open positions

    mkt_vals_open = np.multiply(acts[open_pos_arg], prices[open_pos_arg])
    # mkt val at end of testing period
    # treat as trades for purposes of calculating pnl at end of testing period
    mkt_vals_final = np.multiply(fp, acts[open_pos_arg])
    pnl_a = np.subtract(mkt_vals_final, mkt_vals_open)
    #convert to list
    pnl = [i[0] for i in pnl_a.tolist()]
    #print(f'Market value of open positions at end of testing {pnl}')
    return pnl


def calc_trade_perf(pnl_d):
    # calculate trade performance metrics
    perf_results = {}
    for t,pnl in pnl_d.items():
        wins = pnl[pnl>0]  # total val
        losses = pnl[pnl<0]
        n_wins = len(wins)
        n_losses = len(losses)
        n_trades = n_wins + n_losses
        wins_val = np.sum(wins)
        losses_val = np.sum(losses)
        wins_avg = 0 if n_wins==0 else np.mean(wins)
        #print(f'{t} n_wins: {n_wins} n_losses: {n_losses}')
        losses_avg = 0 if n_losses==0 else np.mean(losses)
        d = {'# trades':n_trades,'# wins':n_wins,'# losses':n_losses,
             'wins total value':wins_val, 'wins avg value':wins_avg,
             'losses total value':losses_val, 'losses avg value':losses_avg,}
        perf_results[t] = d
    return perf_results