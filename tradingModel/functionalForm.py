import matplotlib.pyplot as plt

from tradingModel.Deviation import *
from tradingModel.Minimizer import *
import pandas as pd
from scipy.optimize import curve_fit

class functionalForm(object):
    def __init__(self):
        self.data = np.load('../data/1_min_data.npy')
        self.dates = np.load('../data/dates.npy')
        self.ticker_key = np.load('../data/1_min_ticker_key.npy')

        multi_index = pd.MultiIndex.from_product([self.ticker_key, self.dates], names=['ticker', 'date'])
        self.df = pd.DataFrame(index=multi_index, columns=['liquidity', 'E', 'V', 'volume_proportion_error', 'total_volume_error'])


        self.lmbda = 10 ** (-6)
        kappa_df = pd.read_csv('../data/fixed_kappa.csv')
        kappa_df.set_index(kappa_df.columns[0], inplace=True)
        self.fixed_kappa_df = kappa_df

    def liquidity_proxy(self):
        liquidity = 0

        for ticker in self.ticker_key:
            ticker_idx = np.argwhere(self.ticker_key == ticker)[0][0]

            for date in self.dates:
                date_idx = np.argwhere(self.dates == date)[0][0]

                price = self.data[ticker_idx, 0, date_idx, :]
                size = self.data[ticker_idx, 0, date_idx, :]

                sigma = price.std() * np.sqrt(30 * 12)

                V = np.sum(size)

                liquidity = np.sqrt(sigma**2 / V)
                self.df.loc[(ticker, date), 'liquidity'] = liquidity

    def E_LV(self):
        for ticker in tqdm(self.ticker_key):
            deviation = []

            for date in self.dates:
                dev = Deviation(ticker, date)
                kappa = self.fixed_kappa_df.loc[ticker, 'beta']
                dev.calc_deviation(kappa, "fixed_kappa")
                deviation.append(dev.deviation)

                self.df.loc[(ticker, date), 'E'] = dev.deviation

            self.df.loc[ticker, 'V'] = np.std(deviation)

        self.df['E_LV'] = self.df['E'] + self.lmbda * self.df['V']

    def volume_vol(self):
        for ticker in self.ticker_key:
            ticker_idx = np.argwhere(self.ticker_key == ticker)[0][0]
            volume = np.load(f'../data/volumes/{ticker}_volume.npy')
            volume_proportion = np.mean(volume, axis=0) / np.sum(np.mean(volume, axis=0))
            for date in self.dates:
                date_idx = np.argwhere(self.dates == date)[0][0]


                date_volume = volume[date_idx, :]

                date_volume_proportion = date_volume / np.sum(date_volume)

                volume_error = np.sum(date_volume_proportion - volume_proportion)

                self.df.loc[(ticker, date), 'volume_proportion_error'] = volume_error

    def volume_error(self):
        # self.df['total_volume_error'] = []

        for ticker in self.ticker_key:
            # Get params
            params = np.load('../data/volume_model_params.npy')
            volume = np.load(f'../data/volumes/{ticker}_volume.npy')
            # Estimate total volume
            total_volume = np.sum(volume, axis=1)

            for date in self.dates:
                date_idx = np.argwhere(self.dates == date)[0][0]
                estimated_total_volume = params[0] * total_volume[date_idx - 1]+ \
                                         params[1] * total_volume[date_idx - 2]+ \
                                         params[2] * total_volume[date_idx - 3]+ \
                                         params[3] * total_volume[date_idx - 4]
                volume_error = (estimated_total_volume - total_volume[date_idx]) / total_volume[date_idx]
                self.df.loc[(ticker, date), 'total_volume_error'] = volume_error

    def volume_error_volatility(self):

        for ticker in self.ticker_key:
            # Get params
            params = np.load('../data/volume_model_params.npy')
            volume = np.load(f'../data/volumes/{ticker}_volume.npy')
            # Estimate total volume
            total_volume = np.sum(volume, axis=1)
            volume_error_li = []

            for date in self.dates:
                date_idx = np.argwhere(self.dates == date)[0][0]
                estimated_total_volume = params[0] * total_volume[date_idx - 1]+ \
                                         params[1] * total_volume[date_idx - 2]+ \
                                         params[2] * total_volume[date_idx - 3]+ \
                                         params[3] * total_volume[date_idx - 4]
                volume_error = (estimated_total_volume - total_volume[date_idx]) / total_volume[date_idx]
                volume_error_li.append(volume_error)

            self.df.loc[ticker, 'volume_error_volatility'] = np.std(volume_error_li)

    def plot(self):
        plt.figure(figsize=(10,8))
        plt.scatter(self.df['liquidity'].values, self.df['E_LV'])
        plt.xlabel('Liquidity')
        plt.ylabel('E+LV')
        plt.title('Liquidity vs E+LV')
        plt.savefig('../data/Liquidity vs E+LV.png')

        plt.figure(figsize=(10, 8))
        plt.scatter(self.df['volume_proportion_error'].values, self.df['E_LV'])
        plt.xlabel('Volume Proportion Error')
        plt.ylabel('E+LV')
        plt.title('Volume Proportion Error vs E+LV')
        plt.savefig('../data/Volume Proportion Error vs E+LV.png')

        plt.figure(figsize=(10, 8))
        plt.scatter(self.df['total_volume_error'].values, self.df['E_LV'])
        plt.xlabel('Volume Error')
        plt.ylabel('E+LV')
        plt.title('Total Volume Error vs E+LV')
        plt.savefig('../data/Total Volume Error vs E+LV.png')

        plt.figure(figsize=(10, 8))
        plt.scatter(self.df['volume_error_volatility'].values, self.df['E_LV'])
        plt.xlabel('Volume Error Volatility')
        plt.ylabel('E+LV')
        plt.title('Volume Error Volatility vs E+LV')
        plt.savefig('../data/Volume Error Volatility vs E+LV.png')

    def save(self):
        self.df.to_csv('../data/functional_form.csv')

    def get_params(self):
        liquidity = self.df['liquidity']
        cost = self.df['E_LV']

        def objective_function(x, a, b, c):
            return a * x ** b + c / x

        initial_guess = [1, 1, 1]
        popt, pcov = curve_fit(objective_function, liquidity, cost, p0=initial_guess)

        # Extract fitted parameters
        a_fit, b_fit, c_fit = popt

        # Print the results
        print(f"Fitted parameters: a = {a_fit}, b = {b_fit}, c = {c_fit}")

        self.a_fit = a_fit
        self.b_fit = b_fit
        self.c_fit = c_fit


if __name__ == "__main__":
    # ff = functionalForm()
    # ff.volume_error()
    # ff.volume_vol()
    # ff.liquidity_proxy()
    # ff.E_LV()
    # ff.volume_error_volatility()
    # ff.save()
    # ff.plot()


    df = pd.read_csv('../data/functional_form.csv')
    df.set_index(['ticker', 'date'], inplace=True)

    liquidity = df['liquidity']
    cost = df['E_LV']

    def objective_function(x, a, b, c):
        return a * x**b + c / x

    initial_guess = [1, 1, 1]
    popt, pcov = curve_fit(objective_function, liquidity, cost, p0=initial_guess)

    # Extract fitted parameters
    a_fit, b_fit, c_fit = popt

    # Print the results
    print(f"Fitted parameters: a = {a_fit}, b = {b_fit}, c = {c_fit}")

    x_arr = np.linspace(0.0005, 1.6, 1000)
    plt.figure(figsize=(10, 8))
    plt.scatter(liquidity, cost)
    plt.plot(x_arr, objective_function(x_arr, a_fit, b_fit, c_fit), color='red', lw=4)
    plt.xlabel('Liquidity')
    plt.ylabel('E+LV')
    plt.title('Functional Form')
    plt.savefig('../data/Functional Form.png')

    x_arr = np.linspace(0.0005, 1.6, 1000)
    plt.figure(figsize=(10, 8))
    plt.scatter(liquidity, objective_function(liquidity, a_fit, b_fit, c_fit) - cost)
    plt.xlabel('Liquidity')
    plt.ylabel('Error')
    plt.title('Error of the Formula')
    plt.savefig('../data/Error of Functional Form.png')
