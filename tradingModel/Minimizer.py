import pickle
import os
from taq.MyDirectories import *
from taq.TAQTradesReader import *
from vwapUtils.MergebyTime import *
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tradingModel.Deviation import *
from scipy.optimize import Bounds, minimize
import random
from multiprocessing import Pool
import time
import warnings

class TookTooLong(Warning):
    pass

class Minimizer:
    def __init__(self, timeout, maxfun, maxiter):
        self.timeout = timeout
        self.maxfun = maxfun
        self.maxiter = maxiter

    def minimize(self, ticker, arb_dates=None, mode=None):
        def deviation_func(params, ticker, dates, lmbda=10 ** (-6)):
            beta = params[0]
            deviation = []
            for date in dates:
                dev = Deviation(ticker, date)
                dev.calc_deviation(beta, mode=mode)
                deviation.append(dev.deviation)
            deviation = np.array(deviation)

            mean = np.mean(deviation)
            std = np.std(deviation)


            return mean + lmbda * std

        print(f"Working on {ticker}")
        dates = np.load('../data/dates.npy')
        holdout = int(len(dates) * 0.3)

        train_dates = dates[:-holdout]

        if arb_dates is not None:
            train_dates = arb_dates

        initial_guess = np.array([0.01])
        bounds = Bounds([1E-10], [np.inf])

        self.start_time = time.time()
        # minimize
        res = minimize(deviation_func,
                       x0       = initial_guess,
                       method   ="L-BFGS-B",
                       bounds   = bounds,
                       tol      = 10**(-6),
                       options  = {"maxfun":self.maxfun,
                                   "maxiter":self.maxiter,
                                   "ftol":10**(-6)},
                       args     = (ticker, train_dates,),
                       callback = self.callback
                       )

        elapsed = time.time() - self.start_time
        print(f"{ticker} Finished. Beta: {res.x[0]}. Elapsed: %.3f sec" % elapsed)

        return ticker, res.x[0]

    def callback(self, x):
        # callback to terminate if max_sec exceeded
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout:
            warnings.warn("Terminating optimization: time limit reached",
                          TookTooLong)
            return True
        else:
            pass

if __name__ == "__main__":
    # Load filtered dictionary
    file_path = '../data/filtered_ticker_dates_dict.pkl'

    with open(file_path, 'rb') as f:
        filtered_ticker_dates_dict = pickle.load(f)

    minimizer = Minimizer(300, 20, 40)

    with Pool() as pool:
        # call the same function with different data in parallel
        result = pool.map(minimizer.minimize, list(filtered_ticker_dates_dict.keys()))

    print(result)

    res_series = pd.DataFrame(columns=['beta'])
    for ticker, beta in result:
        res_series.loc[ticker, 'beta'] = beta

    res_series.to_csv('../data/fixed_kappa.csv')
    print("Done.")