from vwapUtils.MergebyTime import *


class Deviation(object):
    def __init__(self, ticker, date):
        self.ticker = ticker
        self.date = date

        self.tickers = np.load('../data/1_min_ticker_key.npy')
        self.dates = np.load('../data/dates.npy')

        self.ticker_idx = np.argwhere(self.tickers == ticker)[0][0]
        self.date_idx = np.argwhere(self.dates == date)[0][0]

    def calc_deviation(self, beta, mode=None):
        # Load price, size, timestamp
        data = np.load('../data/1_min_data.npy')

        price = data[self.ticker_idx, 0, self.date_idx, :]
        size = data[self.ticker_idx, 0, self.date_idx, :]

        # Load volume data
        volume_path = '../data/volumes/' + self.ticker + '_volume.npy'
        volume_data = np.load(volume_path)

        # Get params
        params = np.load('../data/volume_model_params.npy')

        # Estimate total volume
        total_volume = np.sum(volume_data, axis=1)

        estimated_total_volume = params[0] * total_volume[self.date_idx - 1]+ \
                                 params[1] * total_volume[self.date_idx - 2]+ \
                                 params[2] * total_volume[self.date_idx - 3]+ \
                                 params[3] * total_volume[self.date_idx - 4]

        # Get volume proportion
        volume_proportion = np.mean(volume_data, axis=0) / np.sum(np.mean(volume_data, axis=0))

        # Requested volume
        requested_volume = np.sum(size) * 0.05 # 5% of total volumes

        # sigma
        sigma = price.std() * np.sqrt(30)
        self.sigma = sigma

        def h_func(sigma, X, VT):
            eta = 0.142
            beta = 0.5

            return sigma * eta * np.sign(X) * np.abs(X / VT)**beta

        # Calculate volume
        iter_num = 0

        interval_stop = 30
        volume_proportion_model = np.copy(volume_proportion)
        cum_h = 0
        g = 0
        g_decay = 0
        size_real = 0
        vwap_real = 0
        vwap_model = 0
        X = requested_volume * volume_proportion_model[iter_num]
        VT = np.sum(size[:interval_stop])

        volume_proportion_real = np.zeros(len(volume_proportion_model))

        h = h_func(sigma, X, VT)

        model_volume = requested_volume * volume_proportion_model[iter_num] / 30

        for ts in range(len(price)):

            # Get p, cum_h, g
            p = price[ts]
            cum_h += h / 30
            g -= g_decay

            real_price = p + cum_h + g

            real_volume = model_volume + size[ts]

            vwap_model += real_price * model_volume
            vwap_real += real_price * real_volume
            size_real += real_volume

            if ts < interval_stop:
                pass
            else:
                if mode == "no_adjustment":
                    continue

                # Volume proportion gap
                volume_proportion_real[iter_num] = VT / estimated_total_volume
                gap = np.sum(volume_proportion_real[:iter_num+1]) - sum(volume_proportion_model[:iter_num+1])
                gap_sign = np.sign(gap)

                # Adjust volume_proportion_model
                kappa = beta * np.abs(gap) # beta is a coefficient for kappa

                if mode == "fixed_kappa":
                    kappa = beta

                tk = np.linspace(0, 1, 12 - iter_num)
                cumul = 1 - np.sinh(kappa * (1 - tk)) / np.sinh(kappa * 1)

                weights = cumul[1:] - cumul[:-1]

                if gap_sign > 0:
                    # gap > 0 => front-weight
                    leftover = np.array(volume_proportion_model[iter_num + 1:])
                    min_val = min(np.min(volume_proportion_model), 0.05)
                    temp_arr = leftover - min_val + weights * (min_val * len(leftover))
                    volume_proportion_model = np.append(np.copy(volume_proportion_model[:iter_num + 1]), temp_arr)
                else:
                    # gap < 0 => back-weight
                    leftover = np.array(volume_proportion_model[iter_num + 1:])
                    min_val = min(np.min(volume_proportion_model), 0.05)
                    temp_arr = leftover - min_val + weights[::-1] * (min_val * len(leftover))
                    volume_proportion_model = np.append(np.copy(volume_proportion_model[:iter_num + 1]), temp_arr)

                # Reset variables
                g = cum_h
                g_decay = g / 30

                cum_h = 0

                iter_num += 1
                interval_stop += 30

                X = requested_volume * volume_proportion_model[iter_num]
                VT = size[interval_stop - 30:interval_stop].sum()
                model_volume = requested_volume * volume_proportion_model[iter_num] / 30
                h = h_func(sigma, X, VT)

        vwap_model /= requested_volume
        vwap_real /= size_real

        # Save outputs
        self.vwap_real = vwap_real
        self.vwap_model = vwap_model

        # Deviation
        deviation = np.abs(vwap_real - vwap_model)
        self.deviation = deviation

        # print(deviation)

        return deviation


if __name__=="__main__":
    dev = Deviation('AACC', '20070626')
    dev.calc_deviation(1)







