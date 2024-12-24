import pickle
import os
from taq.MyDirectories import *
from taq.TAQTradesReader import *
from vwapUtils.MergebyTime import *
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Merge Volume Data for average volume data for each 30 mins bucket
class VolumeStatsAnalysis(object):
    def __init__(self, ticker_dates_dict):
        self.ticker_dates_dict = ticker_dates_dict

    def filter(self, desired_length = 64):
        self.desired_length = desired_length
        # Filter if there are less than a desired length
        filtered_dict = {key: value for key, value in self.ticker_dates_dict.items() \
                         if len(value) >= desired_length}

        # Specify the file path
        file_path = '../data/filtered_ticker_dates_dict.pkl'

        # Save the dictionary to a pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(filtered_dict, f)

        self.filtered_ticker_dates_dict = filtered_dict

    def volume_data(self):
        d = self.filtered_ticker_dates_dict
        tickers = d.keys()

        start_ts = 19 * 60 * 60 * 1000 / 2
        end_ts = 16 * 60 * 60 * 1000 - 30 * 60 * 1000
        thirty_mins = 30 * 60 * 1000

        iter_num = int((end_ts - start_ts) / thirty_mins)

        for ticker in tqdm(tickers):
            dates = d[ticker]

            volume_arr = np.zeros((self.desired_length, iter_num))

            j = 0
            for date in dates:
                path = os.path.join(getTradesDir(), date + '/' + ticker + '_trades.binRT')
                reader = TAQTradesReader(path)
                merged_reader = MergebyTime(reader)
                merged_reader.merge()

                ts = merged_reader.ts
                s = merged_reader.s

                volume = 0
                thresh = start_ts + thirty_mins
                interval = 0
                for i in range(len(ts)):
                    if ts[i] < thresh:
                        volume += s[i]
                    else:
                        volume_arr[j, interval] = volume
                        volume = s[i]
                        thresh += thirty_mins
                        interval += 1
                j += 1

            np.save(f'../data/volumes/{ticker}_volume.npy', volume_arr)


if __name__ == "__main__":
    # Load dictionary
    file_path = '../data/ticker_dates_dict.pkl'

    with open(file_path, 'rb') as f:
        ticker_dates_dict = pickle.load(f)

    print("Number of volumes before filtering:", len(ticker_dates_dict.keys()))

    # Input dictionary
    vsa = VolumeStatsAnalysis(ticker_dates_dict)

    # Filter volumes
    vsa.filter(64)

    print("Number of volumes after filtering:", len(vsa.filtered_ticker_dates_dict.keys()))


    # Plot the volumes
    tickers = vsa.filtered_ticker_dates_dict.keys()


    # Total Volume estimation via lagging total volumes
    y = []
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []

    for ticker in tickers:
        loaded_array = np.load(f'../data/volumes/{ticker}_volume.npy')
        volume_sum = np.sum(loaded_array, axis=1)

        for i in range(5, len(volume_sum)):
            y.append(volume_sum[i])
            x1.append(volume_sum[i - 1])
            x2.append(volume_sum[i - 2])
            x3.append(volume_sum[i - 3])
            x4.append(volume_sum[i - 4])
            x5.append(volume_sum[i - 5])

    # With 5 lags
    X = np.zeros((len(x1), 5))
    X[:, 0] = x1
    X[:, 1] = x2
    X[:, 2] = x3
    X[:, 3] = x4
    X[:, 4] = x5

    model = sm.OLS(y, X)
    result = model.fit()

    print(result.summary()) # r-squared 0.864

    # With 4 lags
    X = np.zeros((len(x1), 4))
    X[:, 0] = x1
    X[:, 1] = x2
    X[:, 2] = x4
    X[:, 3] = x5

    model = sm.OLS(y, X)
    result = model.fit()

    print(result.summary()) # r-squared 0.864

    # Choose lag1, lag2, lag4, lag5 total volumes as features

    # np.save('../data/volume_model_params.npy', np.array(result.params))