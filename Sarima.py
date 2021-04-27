import pandas as pd
import numpy as np
import statsmodels as statsmodels
import tqdm as tqdm
from matplotlib import pyplot
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf

from statsmodels.tsa.stattools import adfuller


class SarimaModel:

    @staticmethod
    def load_data():
        bc = pd.read_csv("crypto.csv")
        # setting the index as dates
        bc.set_index('date', inplace=True)

        return bc

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


bc = SarimaModel.load_data()
log = SarimaModel.prepare_data(bc)

acff = acf(bc)
print(acff)

# SarimaModel.best_param()
