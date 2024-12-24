from taq.TAQTradesReader import TAQTradesReader
from taq.MyDirectories import *
import numpy as np
import time
import pandas as pd
import pickle
from tqdm import tqdm

class MergebyTime(object):
    def __init__(self, taq_trades_reader):
        self.taq = taq_trades_reader

        self.p = taq_trades_reader.getTotalPrice()
        self.s = taq_trades_reader.getTotalSize()
        self.ts = taq_trades_reader.getTotalTimestamp()

    def merge(self):
        self.p = []
        self.s = []
        self.ts = []

        self.p.append(self.taq.getPrice(0))
        self.s.append(self.taq.getSize(0))
        self.ts.append(self.taq.getTimestamp(0))

        i = 0
        j = 1
        while j < self.taq.getN():
            if self.ts[i] == self.taq.getTimestamp(j):
                self.p[i] = (self.p[i] * self.s[i] + self.taq.getPrice(j) * self.taq.getSize(j))\
                            / (self.s[i] + self.taq.getSize(j)) # VWAP
                self.s[i] = self.s[i] + self.taq.getSize(j)
            else:
                self.p.append(self.taq.getPrice(j))
                self.s.append(self.taq.getSize(j))
                self.ts.append(self.taq.getTimestamp(j))
                i += 1
            j += 1

    def merge_by_one_min(self):
        start_ts = 9 * 60 * 60 * 1000 + 30 * 60 * 1000
        end_ts = 16 * 60 * 60 * 1000 - 30 * 60 * 1000
        one_min = 1 * 60 * 1000


        ts = [x for x in range(start_ts, end_ts, one_min)]
        p = np.zeros(len(ts))
        s = np.zeros(len(ts))

        starting_point = self.ts[0]
        starting_idx = 0

        for i in range(len(ts)):
            if (starting_point >= ts[i]) & (starting_point < ts[i + 1]):
                starting_idx = i
                if i > 0:
                    for j in range(starting_idx):
                        p[i] = np.nan
                        s[i] = 0
                break

        idx = starting_idx

        thresh = ts[idx]

        p_temp = 0
        s_temp = 0
        for i in range(len(self.ts)):
            if self.ts[i] < thresh:
                p_temp += self.p[i] * self.s[i]
                s_temp += self.s[i]
            else:
                if s_temp == 0:
                    p[idx] = np.nan
                else:
                    p[idx] = p_temp / s_temp
                s[idx] = s_temp

                p_temp = 0
                s_temp = 0
                # print(p)
                # print(s)
                idx += 1
                if idx >= 360:
                    break
                thresh = ts[idx]

        self.p_1m = pd.Series(p).bfill().ffill().values
        self.s_1m = s
        self.ts_1m = ts

    def getN(self):
        return len(self.p)

    def getPrice(self):
        return self.p

    def getSize(self):
        return self.s

    def getTimestamp(self):
        return self.ts

    def liquidity_test(self):
        start_ts = 19 * 60 * 60 * 1000 / 2
        end_ts = 16 * 60 * 60 * 1000 - 30 * 60 * 1000
        thirty_mins = 30 * 60 * 1000

        iter_num = int((end_ts - start_ts) / thirty_mins)

        result = True

        t = start_ts

        for i in range(iter_num):
            # Define the range boundaries
            lower_bound = t
            upper_bound = t + thirty_mins

            # is_within_range = any(lower_bound <= ts < upper_bound for ts in self.ts)

            temp_li = sum([1 for ts in self.ts if lower_bound <= ts < upper_bound])

            if temp_li >= 100: # At least 100 trading times in each 30 min bucket
                more_than_100_trades = True
            else:
                more_than_100_trades = False

            result &= more_than_100_trades

            t += thirty_mins

        return result

if __name__ == "__main__":
    # Load filtered dictionary
    file_path = '../data/filtered_ticker_dates_dict.pkl'

    with open(file_path, 'rb') as f:
        filtered_ticker_dates_dict = pickle.load(f)

    tickers = np.array(list(filtered_ticker_dates_dict.keys()))
    np.save('../data/1_min_ticker_key.npy', tickers)

    dates = filtered_ticker_dates_dict['AAPL']
    np.save('../data/dates.npy', dates)

    data = np.zeros((len(tickers), 2, len(dates), 360))
    for t in tqdm(range(len(tickers))):
        for d, date in enumerate(dates):
            reader = TAQTradesReader(f'../data/trades/{date}/{tickers[t]}_trades.binRT')
            merged_reader = MergebyTime(reader)
            merged_reader.merge_by_one_min()
            p = merged_reader.p_1m
            s = merged_reader.s_1m
            data[t, 0, d, :] = p
            data[t, 1, d, :] = s

        # print(data[t,0,:,:])
        # print(data[t,0,:,:])

    np.save('../data/1_min_data.npy', data)

    data = np.load('../data/1_min_data.npy')
    print(data[1, 0, 0, :])
    print(data[1, 1, 0, :])