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

class TradingStatsAnalysis(object):
    def __init__(self):
        beta_df = pd.read_csv('../data/beta.csv')
        beta_df.set_index(beta_df.columns[0], inplace=True)
        self.beta_df = beta_df

        fixed_kappa_df = pd.read_csv('../data/fixed_kappa.csv')
        fixed_kappa_df.set_index(fixed_kappa_df.columns[0], inplace=True)
        self.fixed_kappa_df = fixed_kappa_df

        data = np.load('../data/1_min_data.npy')
        self.data = data

        ticker_key = np.load('../data/1_min_ticker_key.npy')
        self.ticker_key = ticker_key

        self.lmbda = 10 ** (-6)

        self.stats_df = pd.DataFrame()

    def train_test(self, test_ratio=0.3):
        dates = np.load('../data/dates.npy')
        holdout = int(len(dates) * test_ratio)

        train_dates = dates[:-holdout]
        test_dates = dates[-holdout:]

        self.train_dates = train_dates
        self.test_dates = test_dates

    def no_adjustment(self, dates):
        for ticker in tqdm(self.ticker_key):
            deviation = []

            for date in dates:
                dev = Deviation(ticker, date)
                dev.calc_deviation(0, "no_adjustment")
                deviation.append(dev.deviation)

            E = np.mean(deviation)
            V = np.std(deviation)
            L = self.lmbda

            self.stats_df.loc[ticker, "no_adj"] = E + L * V

    def fixed_kappa(self, dates):
        for ticker in tqdm(self.ticker_key):
            deviation = []

            for date in dates:
                dev = Deviation(ticker, date)
                kappa = self.fixed_kappa_df.loc[ticker, 'beta']
                dev.calc_deviation(kappa, "fixed_kappa")
                deviation.append(dev.deviation)

            E = np.mean(deviation)
            V = np.std(deviation)
            L = self.lmbda

            self.stats_df.loc[ticker, "fixed_kappa"] = E + L * V

    def optimized_beta(self, dates):
        for ticker in tqdm(self.ticker_key):
            deviation = []

            for date in dates:
                dev = Deviation(ticker, date)
                beta = self.beta_df.loc[ticker, 'beta']
                dev.calc_deviation(beta)
                deviation.append(dev.deviation)

            E = np.mean(deviation)
            V = np.std(deviation)
            L = self.lmbda

            self.stats_df.loc[ticker, "opt_beta"] = E + L * V

    def save(self, title):
        self.stats_df.to_csv(f'../data/{title}.csv')

if __name__ == "__main__":
    df = pd.read_csv('../data/stats_df.csv')
    df.set_index(df.columns[0], inplace=True)

    # Define the position of bars
    x = np.arange(len(df))
    width = 0.35  # the width of the bars

    # Create bar chart
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width / 2, df['fixed_kappa'], width, label='Fixed Kappa')
    bar2 = ax.bar(x + width / 2, df['opt_beta'], width, label='Stochastic Kappa')

    # Add labels, title, and custom x-axis tick labels
    ax.set_xlabel('Stocks')
    ax.set_ylabel('E+LV')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.legend()

    plt.show()
