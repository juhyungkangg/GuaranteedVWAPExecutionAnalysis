import matplotlib.pyplot as plt
from tradingModel.Deviation import *


if __name__ == "__main__":
    df = pd.read_csv('../data/stats_df.csv')
    df.set_index(df.columns[0], inplace=True)

    val = -(df['opt_beta'] - df['fixed_kappa']) / df['fixed_kappa']

    plt.figure(figsize=(12, 8))
    plt.title("Improvement by Stochastic Kappa")
    plt.bar(df.index, val)
    plt.xlabel('Stocks')
    plt.xticks(np.arange(0, len(df), 30))
    plt.ylabel('Improvement')
    plt.savefig('../data/model_comparison.png')
