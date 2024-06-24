import numpy as np
import pandas as pd
import time
import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


matrix = np.array([
    [2, 5, -10, 3, 7, 1, 8, -5, 9, -2, 4, 6, -1, 8, -3, 7, 2, 5, 1, -6],
    [1, 2, -8, 3, 5, 0, 6, -3, 7, -1, 2, 4, -2, 5, -1, 6, 1, 3, 0, -5],
    [4, 6, -9, 2, 8, 3, 7, -4, 10, -3, 5, 7, -1, 9, -2, 8, 3, 6, 2, -4],
    [3, 7, -11, 4, 9, 2, 9, -6, 11, -4, 6, 8, -3, 10, -5, 9, 4, 7, 3, -7],
    [0, 3, -6, 1, 4, -1, 5, -2, 6, -3, 1, 4, -1, 7, -2, 5, 2, 3, 0, -8],
    [5, 8, -7, 6, 10, 4, 11, -5, 12, -1, 7, 9, -4, 13, -6, 10, 6, 8, 4, -9],
    [3, 7, -11, 4, 9, 2, 9, -6, 1, -4, 6, 8, -3, 1, -5, 9, 42, 7, 3, -7],
])


df = pd.DataFrame(matrix.T, columns=[f'Series {i+1}' for i in range(matrix.shape[0])])
print("dataset ; ")
print(df)

def check_trend(timeseries):
    n = len(timeseries)
    x = np.arange(n)
    A = np.vstack([x, np.ones(n)]).T # [index, 1]
    m, c = np.linalg.lstsq(A, timeseries, rcond=None)[0] # 최소자승법 
    
    y_pred = m * x + c # trend pred
    residuals = timeseries - y_pred # residual(org timeseries - pred_trend)
    sse = np.sum(residuals ** 2) # Sum of Squared Errors
    sst = np.sum((timeseries - np.mean(timeseries)) ** 2) # Total Sum of Squares
    r2 = 1 - sse / sst # det coeff (R^2) = 추세 모델 설명력
    
    # m t-stat
    se_m = np.sqrt(np.sum((x - np.mean(x))**2))
    t_stat = m / se_m
    
    critical_t = 2.086  # df = n-2 
    
    trend_exists = abs(t_stat) > critical_t # t-stat > crit -> trend exists
    
    return trend_exists, y_pred, m


def remove_trend(timeseries, trend, m):
    detrended = timeseries - trend
    return detrended, m


def ols_regression(X, y):
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    residuals = y - X @ beta
    
    model = sm.OLS(y, X).fit()
    beta1 = model.params
    residuals1 = model.resid
    print("answ ; ", beta1, residuals1)
    print("my ; ", beta, residuals)
    
    return beta, residuals


def adf_test(X, max_lag=1):
    delta_X = np.diff(X, n=1) 
    y = delta_X[max_lag:] 
    
    X_lagged = X[:-1] # t-1 
    X_lagged = X_lagged[max_lag:]

    # model에 constant, trend, lagged, additional lagged diff 포함
    constant = np.ones_like(y)
    time_trend = np.arange(1, len(y) + 1)
    X_design = np.vstack([constant, time_trend, X_lagged]).T
    # X_design = np.vstack([constant, X_lagged]).T

    for i in range(1, max_lag + 1):
        delta_X_lag = delta_X[max_lag - i:-i]
        X_design = np.column_stack((X_design, delta_X_lag))
    # for i in range(1, max_lag + 1):
    #     delta_X_lag = delta_X[max_lag - i:-i] if i != 0 else delta_X[max_lag:]
    #     X_design = np.column_stack((X_design, delta_X_lag))

    print("matrix X:")
    print(X_design)

    # 최소자승법 -> coeff estimation
    beta, residuals = ols_regression(X_design, y)

    # reg coeff
    # a0, delta = beta[:2]
    # additional_betas = beta[2:]
    a0, trend, delta = beta[:3]
    additional_betas = beta[3:]

    # coeff std err 
    sigma2 = np.var(residuals, ddof=len(beta))
    # sigma2 = np.var(residuals, ddof=X_design.shape[1] - 1)
    # sigma2 = np.var(residuals, ddof=len(y) - 1)
    XTX_inv = np.linalg.pinv(X_design.T @ X_design)
    var_beta = sigma2 * np.diag(XTX_inv)
    # s_delta = np.sqrt(var_beta[1])  
    s_delta = np.sqrt(var_beta[2])  # delta std err (trend 추가로 1 -> 2)

    df_statistic = delta / s_delta

    critical_values = {
        '1%': -3.43,
        '5%': -2.86,
        '10%': -2.57
    }

    p_value = 0.01 if df_statistic < critical_values['1%'] else \
              0.05 if df_statistic < critical_values['5%'] else \
              0.10 if df_statistic < critical_values['10%'] else 0.50

    return df_statistic, p_value, critical_values, a0, delta

# adf test
for i in range(matrix.shape[0]):  # matrix.shape[0] = 시계열 수
    timeseries = matrix[i]
    
    trend_exists, trend, m = check_trend(timeseries)
    
    if trend_exists:
        print(f"Series {i + 1} trend m: {m}")
        detrended, _ = remove_trend(timeseries, trend, m)
    else:
        detrended = timeseries
    
    start = time.time()

    df_statistic, p_value, critical_values, a0, delta = adf_test(detrended, max_lag=1) # rm trend
    # df_statistic, p_value, critical_values, a0, delta = adf_test(timeseries, max_lag=1)

    time_taken = time.time() - start
    times = str(datetime.timedelta(seconds=time_taken))
    short = times.split(".")[0]
    
    print(f"\nSeries {i + 1} adf_test res ;")
    print(f"  ADF Statistic: {df_statistic}")
    print(f"  p-value: {p_value}")
    for key, value in critical_values.items():
        print(f"  Critical Value ({key}): {value}")
    print(f"  a0: {a0}")
    print(f"  delta: {delta}")
    print(f"  time: {short} sec")



# adfuller 
for i in range(matrix.shape[0]):
    timeseries = matrix[i]
    start = time.time()
    result = adfuller(timeseries, regression='ct', maxlag=1, autolag=None)
    
    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]
    
    time_taken = time.time() - start
    times = str(datetime.timedelta(seconds=time_taken))
    short = times.split(".")[0]
    
    print(f"\nSeries {i + 1} ADF fuller res (statsmodels):")
    print(f"  ADF Statistic: {adf_statistic}")
    print(f"  p-value: {p_value}")
    for key, value in critical_values.items():
        print(f"  Critical Value ({key}): {value}")
    print(f"  time: {short} sec")
