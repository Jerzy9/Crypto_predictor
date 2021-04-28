import pandas as pd
import numpy as np
import statsmodels as statsmodels
import tqdm as tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller


class SarimaModel:

    @staticmethod
    def load_data_from_csv():
        bc = pd.read_csv("crypto.csv")
        # setting the index as dates
        bc.set_index('date', inplace=True)
        # bc = bc.asfreq(pd.infer_freq(bc.index))

        return bc

    @staticmethod
    def remove_trend(bc):
        first_diff = bc.diff()[1:]
        plt.figure(figsize=(10, 4))
        plt.plot(first_diff)
        plt.title('bitcoin', fontsize=20)
        plt.ylabel('price', fontsize=16)
        # for year in range(start_date.year, end_date.year):
        #     plt.axvline(pd.to_datetime(str(year) + '-01-01'), color='k', linestyle='--', alpha=0.2)
        plt.axhline(0, color='k', linestyle='--', alpha=0.2)
        plt.show()

        return first_diff

    @staticmethod
    def acf(first_diff):
        acf_vals = acf(first_diff)
        num_lags = 20
        plt.bar(range(num_lags), acf_vals[:num_lags])
        plt.show()

        return acf_vals

    @staticmethod
    def pacf(first_diff):
        pacf_vals = pacf(first_diff)
        num_lags = 15
        plt.bar(range(num_lags), pacf_vals[:num_lags])
        plt.show()

        return pacf_vals

    @staticmethod
    def prepare_data(bc):
        # convert to the logarithmic
        bc_log = pd.DataFrame(np.log(bc.bitcoin))

        bc_log_diff = bc_log.diff().dropna()

        # bc_log.hist()
        # pyplot.show()
        # print(bc_log_diff)

        # .view(sim=system)
        res = adfuller(bc_log_diff['bitcoin'])
        print(f"P-value: {res[1]}")
        return bc_log_diff
        # print()

    @staticmethod
    def best_param(model, data, pdq, pdqs):
        """
        Loops through each possible combo for pdq and pdqs
        Runs the model for each combo
        Retrieves the model with lowest AIC score
        """
        ans = []
        for comb in tqdm(pdq):
            for combs in tqdm(pdqs):
                try:
                    mod = model(data,
                                order=comb,
                                seasonal_order=combs,
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                                freq='D')

                    output = mod.fit()
                    ans.append([comb, combs, output.aic])

                except:
                    continue

        ans_df = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'aic'])
        return ans_df.loc[ans_df.aic.idxmin()]

    @staticmethod
    def train_data(bc_log):
        index = round(len(bc) * .80)
        train = bc_log.iloc[:index]
        test = bc_log.iloc[index:]

        # Fitting the model to the training set
        model = SARIMAX(train,
                        order=(1, 0, 0),
                        seasonal_order=(0, 0, 0, 0),
                        freq='D',
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        output = model.fit()
        return output

    @staticmethod
    def train_data_2(bc):
        train_end = datetime(1999, 7, 1)
        test_end = datetime(2000, 1, 1)

        train_data = bc[:train_end]
        test_data = bc[train_end + timedelta(days=1):test_end]

    @staticmethod
    def show_bc_chart(bc):
        plt.figure(figsize=(10, 4))
        plt.plot(bc)
        plt.title('bitcoin', fontsize=20)
        plt.show()


bc = SarimaModel.load_data_from_csv()
# SarimaModel.show_bc_chart(bc)
# log = SarimaModel.prepare_data(bc)
first_diff = SarimaModel.remove_trend(bc)
SarimaModel.acf(first_diff)
# acff = acf(bc)
# print(acff)

# SarimaModel.best_param()
