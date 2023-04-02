import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import json
import os
import sys
import time
import warnings
import math
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm


#  1.  Preproccessing


#  1.1 Mean_center


def apply_mean_center(x):
    # import numpy as np
    mu = np.mean(x)
    xm = x / mu
    return xm, mu


def mean_center_transform(df, cols):
    '''
    returns:
    mean-centered df
    scaler, dict
    '''
    df_new = pd.DataFrame()
    sc = {}
    for col in cols:
        x = df[col].values
        df_new[col], mu = apply_mean_center(x)
        sc[col] = mu
    return df_new, sc


def get_mean_std(df, columns):
    '''
    columns : list of columns
    example : means_stds = get_mean_std(df, ['mdip_dm'])
    '''
    resp = {}
    for col in columns:
        mean = np.mean(df[col])
        std = np.std(df[col])
        resp[col] = {'mean': mean, 'std': std}
    return resp


def standardize(df, columns):
    '''
    example: standardize(df, ['mdip_dm'])
    '''
    for col in columns:
        df[col + '_std'] = (df[col] - np.mean(df[col])) / np.std(df[col])


def revert_standardize(x, mean, std):
    '''
    revert - for calculating "real" costs after model prediction (in model preprocessed data goes)
    zi = (x - mean) / std
    x - mean = std * zi
    x = std * zi + mean
    example : revert_standardize(df['mdip_dm_std'], means_stds['mdip_dm']['mean'], means_stds['mdip_dm']['std'])
    '''
    return std * x + mean


#  1.2 Log-n


def log1p(df, columns):
    '''
    log with np.log1p (for escape NaN values)
    '''
    for col in columns:
        df[col + '_log1p'] = df[col].apply(np.log1p, axis=0)


def revert_log1p(x):
    '''
    revert - for calculating "real" costs after model prediction (in model preprocessed data goes)
    '''
    return np.expm1(x)


#  1.3 Add seasonal dummy-vars


def dummy_seas(df, date_column):
    # functionn adds 12 columns (for each month) filled with 0 or 1 to df
    for m in range(1, 13, 1):
        df['seas_' + format(m, '02')] = 0
    for i in range(len(df)):
        for m in range(1, 13, 1):
            if df[date_column][i].strftime('%m') == format(m, '02'):
                df['seas_' + format(m, '02')][i] = 1


#  2.  EDA


#  3.  S-shape Functions


#  3.1 S_shape


def s_shape(x, alpha=0.91, beta=1.0):
    return (beta / 10 ** 10) ** (alpha ** ((x / np.max(x)) * 100))


#  3.2 S_origin


def s_origin(x, alpha=0.91, beta=1.0):
    return (beta / 10 ** 9) ** (alpha ** ((x / np.max(x)) * 100)) - (beta / 10 ** 9)


#  3.3 Negative Exponential


def neg_exp(x, alpha=5):
    return 1 - np.exp(-x / alpha)


#  3.4 Indexed Exponential


def ind_exp(x, alpha=1):
    return 1 - np.exp(-0.1 * alpha * x)


#  3.5 Hill Function


def apply_hill(x, k=0.95, s=3.0):
    # Classic Hill-Function
    return 1 / (1 + (k / x) ** (s))


def apply_beta_hill(beta, x, k=0.95, s=3.0):
    # Beta Hill = classic Hill * beta
    return beta / (1 + (k / x) ** (s))


def hill_transform(df, md_cols):
    for col in md_cols:
        df[col + '_shaped'] = apply_hill(df[col])


def plot_shapes(md_cols):
    fig = plt.figure(figsize=(24, 12))
    fig.subplots_adjust(wspace=0.2)
    for i, col in zip(range(len(md_cols)), md_cols):
        ax = fig.add_subplot(3, 4, i + 1)
        ax.scatter(df[col], df[col + '_shaped'])
        ax.set_title(col.split('_')[1])


#  4.  Adstock Transformation


#  5.  OLS Model Creation


def fit_ols(df, x_cols, y_cols):
    return sm.OLS(df[y_cols], sm.add_constant(df[x_cols])).fit()


def extract_ols(fit_model):
    return pd.read_html(fit_model.summary().tables[1].as_html(), header=0, index_col=0)[0]


def create_hill_model_data(df, media):
    y = df['ctrb_' + media].values
    x = df['mdsp_' + media].values
    # centralize
    mu_x, mu_y = x.mean(), y.mean()
    sc = {'x': mu_x, 'y': mu_y}
    x = x / mu_x
    y = y / mu_y

    model_data = {
        'N': len(y),
        'y': y,
        'X': x
    }
    return model_data, sc


def train_hill_model(df, media, sm):
    '''
    params:
    media: 'dm', 'inst', 'nsp', 'auddig', 'audtr', 'vidtr', 'viddig', 'so', 'on', 'sem'
    sm: stan model object
    returns:
    a dict of model data, scaler, parameters
    '''
    data, sc = create_hill_model_data(df, media)
    fit = sm.sampling(data=data, iter=2000, chains=4)
    fit_result = fit.extract()
    hill_model = {
        'beta_hill_list': fit_result['beta_hill'].tolist(),
        'ec_list': fit_result['ec'].tolist(),
        'slope_list': fit_result['slope'].tolist(),
        'sc': sc,
        'data': {
            'X': data['X'].tolist(),
            'y': data['y'].tolist(),
        }
    }
    return hill_model


def extract_hill_model_params(hill_model, method='mean'):
    if method == 'mean':
        hill_model_params = {
            'beta_hill': np.mean(hill_model['beta_hill_list']),
            'ec': np.mean(hill_model['ec_list']),
            'slope': np.mean(hill_model['slope_list'])
        }
    elif method == 'median':
        hill_model_params = {
            'beta_hill': np.median(hill_model['beta_hill_list']),
            'ec': np.median(hill_model['ec_list']),
            'slope': np.median(hill_model['slope_list'])
        }
    return hill_model_params


def hill_model_predict(hill_model_params, x):
    beta_hill, ec, slope = hill_model_params['beta_hill'], hill_model_params['ec'], hill_model_params['slope']
    y_pred = beta_hill * hill_transform(x, ec, slope)
    return y_pred


def evaluate_hill_model(hill_model, hill_model_params):
    x = np.asarray(hill_model['data']['X'])
    y_true = np.asarray(hill_model['data']['y']) * hill_model['sc']['y']
    y_pred = hill_model_predict(hill_model_params, x) * hill_model['sc']['y']
    print('mape on original data: ',
          mean_absolute_percentage_error(y_true, y_pred))
    return y_true, y_pred


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


def apply_adstock(x, L=2, P=0, D=0.7):
    '''

    Create new array with transformed (adstocked) data x.
    Can be used for single series of data independently.

    params:
    x: original media variable, array
    L: length
    P: peak, delay in effect
    D: decay, retain rate
    returns:
    array, adstocked media variable
    '''
    x = np.append(np.zeros(L - 1), x)

    weights = np.zeros(L)
    for l in range(L):
        weight = D ** ((l - P) ** 2)  # decay powered by
        weights[L - 1 - l] = weight

    adstocked_x = []
    for i in range(L - 1, len(x)):
        x_array = x[i - L + 1: i + 1]  # through the window
        xi = sum(x_array * weights) / sum(weights)  # calculation
        adstocked_x.append(xi)
    adstocked_x = np.array(adstocked_x)
    return adstocked_x


def calc_adstock_params(df, mdsp_cols, sales_cols):
    L = np.arange(2, 9, 1)
    P = np.arange(0.1, 2.1, 0.1)
    D = np.arange(0.1, 1.91, 0.01)
    ad_params = pd.DataFrame(list(product(L, P, D)), columns=['lag', 'peak', 'decay'])
    sales_arr = df[sales_cols[0]].values
    adstock_params = {}
    for ad in mdsp_cols:
        ad_arr = np.array(df[ad])
        ad_corr = []
        for i in range(len(ad_params)):
            ad_corr.append(np.corrcoef(sales_arr, apply_adstock(ad_arr, ad_params['lag'][i], ad_params['peak'][i],
                                                                ad_params['decay'][i]))[1, 0])
        ad_params[ad + '_corr'] = ad_corr
        adstock_params[ad.split('_')[1]] = {
            'L': ad_params.loc[ad_params[ad + '_corr'] == ad_params[ad + '_corr'].max(), 'lag'].values[0],
            'P': ad_params.loc[ad_params[ad + '_corr'] == ad_params[ad + '_corr'].max(), 'peak'].values[0],
            'D': ad_params.loc[ad_params[ad + '_corr'] == ad_params[ad + '_corr'].max(), 'decay'].values[0]
        }
    return adstock_params


def plot_adstocked(md_cols, ad_cols, x, y):
    fig = plt.figure(figsize=(24, 30))
    fig.subplots_adjust(wspace=0.1, hspace=0.2)
    for i in range(len(md_cols)):
        ax = fig.add_subplot(x, y, i + 1)
        ax.plot(df.index, df[md_cols[i]], label='raw', linestyle=":")
        ax.plot(df.index, df[ad_cols[i]], label='adstocked')
        ax.legend()
        ax.set_title(md_cols[i].split('_')[1])


def adstock_transform(df, md_cols, adstock_params):
    '''
    params:
    df: original data
    md_cols: list, media variables to be transformed
    adstock_params: dict,
        e.g., {'sem': {'L': 8, 'P': 0, 'D': 0.1}, 'dm': {'L': 4, 'P': 1, 'D': 0.7}}
    returns:
    adstocked df
    '''
    for md_col in md_cols:
        md = md_col.split('_')[1]
        L, P, D = adstock_params[md]['L'], adstock_params[md]['P'], adstock_params[md]['D']
        xa = apply_adstock(df[md_col].values, L, P, D)
        df[md_col + '_adstocked'] = xa