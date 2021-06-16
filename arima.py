import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import create_csv

pd.options.mode.chained_assignment = None


def plot_stat(data_set):
    fig, axes = plt.subplots(3, 2)
    fig.tight_layout(pad=0.05)

    # # Original Series
    d0 = adfuller(data_set.dropna())
    axes[0, 0].plot(data_set.value)
    axes[0, 0].set_title('Original Series  ' + 'p-value of the test:' + str(d0[1]))
    plot_acf(data_set.value, ax=axes[0, 1])

    # 1st Differencing
    d1 = adfuller(data_set.value.diff().dropna())
    axes[1, 0].plot(data_set.value.diff())
    axes[1, 0].set_title('1st Order Differencing  ' + 'p-value of the test:' + str(d1[1]))
    plot_acf(data_set.value.diff().dropna(), ax=axes[1, 1])

    # 2nd Differencing
    d2 = adfuller(data_set.value.diff().diff().dropna())
    axes[2, 0].plot(data_set.value.diff().diff())
    axes[2, 0].set_title('2nd Order Differencing  ' + 'p-value of the test:' + str(d2[1]))
    plot_acf(data_set.value.diff().diff().dropna(), ax=axes[2, 1])

    # print(data_set)
    plt.show()


class DataProcessing:
    @staticmethod
    def split_set(data_set):
        split = int(len(data_set) * .9)
        train = data_set.iloc[:split]
        test = data_set.iloc[split:data_set.shape[0]]
        return train, test

    @staticmethod
    def read_set(path):
        df = pd.read_csv(path, parse_dates=True, index_col='date')
        df.columns = ['value']

        return df

    @staticmethod
    def check_if_data_is_stationary_ADF(data_set):
        result = adfuller(data_set.dropna())
        p = result[1]

        if p > 0.05:
            print("Data is not stationary")
            return False
        else:
            print("Data is stationary")
            return True

    @staticmethod
    def make_data_stationary(df):
        df_stat = pd.DataFrame(np.log(df.value).diff().diff())
        df_stat = df_stat.dropna()

        DataProcessing.check_if_data_is_stationary_ADF(df_stat)

        plot_acf(df_stat)
        plot_pacf(df_stat)

        return pd.DataFrame(df_stat).dropna()

    @staticmethod
    def AR(df, p):
        for i in range(1, p + 1):
            df['Shifted_values_%d' % i] = df['value'].shift(i)

        df_train, df_sample = DataProcessing.split_set(df)

        df_train = df_train.dropna()
        df_sample = df_sample.dropna()

        x_train = df_train.iloc[:, 1:].values.reshape(-1, p)
        y_train = df_train.iloc[:, 0].values.reshape(-1, 1)

        lr = LinearRegression()
        lr.fit(x_train, y_train)

        theta = lr.coef_.T
        intercept = lr.intercept_

        df_train['predicted_values'] = x_train.dot(theta) + intercept
        x_sample = df_sample.iloc[:, 1:].values.reshape(-1, p)
        df_sample['predicted_values'] = x_sample.dot(theta) + intercept

        rmse = np.sqrt(mean_squared_error(df_sample['value'], df_sample['predicted_values']))

        return [df_train, df_sample, theta, intercept, rmse]

    @staticmethod
    def MA(res, q):
        for i in range(1, q + 1):
            res['Shifted_values_%d' % i] = res['residuals'].shift(i)

        res_train, res_test = DataProcessing.split_set(res)

        res_train = res_train.dropna()
        x_train = res_train.iloc[:, 1:].values.reshape(-1, q)
        y_train = res_train.iloc[:, 0].values.reshape(-1, 1)

        lr = LinearRegression()
        lr.fit(x_train, y_train)

        theta = lr.coef_.T
        intercept = lr.intercept_
        res_train['predicted_values'] = x_train.dot(theta) + intercept

        x_test = res_test.iloc[:, 1:].values.reshape(-1, q)
        res_test['predicted_values'] = x_test.dot(theta) + intercept

        RMSE = np.sqrt(mean_squared_error(res_test['residuals'], res_test['predicted_values']))

        return [res_train, res_test, theta, intercept, RMSE]

    @staticmethod
    def ARIMA(df):
        print("STAGE 1 - stationary")
        df_stat = DataProcessing.make_data_stationary(df)

        print("\nSTAGE 2 - auto-regressive model")
        best_rmse = math.inf
        best_p = -1

        for i in range(1, 21):
            [df_train, df_test, theta, intercept, rmse] = DataProcessing.AR(pd.DataFrame(df_stat.value), i)
            if rmse < best_rmse:
                best_rmse = rmse
                best_p = i

        print("The RMSE is:", best_rmse, ", value of p: ", best_p)

        [df_train, df_test, theta, intercept, rmse] = DataProcessing.AR(pd.DataFrame(df_stat.value), best_p)
        df_c = pd.concat([df_train, df_test])

        res = pd.DataFrame()
        res['residuals'] = df_c.value - df_c.predicted_values

        print("\nSTAGE 3 - moving average model")
        best_rmse2 = math.inf
        best_q = -1

        for i in range(1, 21):
            [res_train, res_test, theta, intercept, rmse] = DataProcessing.MA(pd.DataFrame(res.residuals), i)

            if rmse < best_rmse2:
                best_rmse2 = rmse
                best_q = i

        print("The RMSE is:", best_rmse2, ", value of q: ", best_q)

        [res_train, res_test, theta, intercept, rmse] = DataProcessing.MA(pd.DataFrame(res.residuals), best_q)
        res_c = pd.concat([res_train, res_test])

        df_c.predicted_values += res_c.predicted_values

        print("\nSTAGE 4 - ARIMA")
        df_c.value += np.log(df).shift(1).value
        df_c.value += np.log(df).diff().shift().value
        df_c.predicted_values += np.log(df).shift(1).value
        df_c.predicted_values += np.log(df).diff().shift().value
        df_c.value = np.exp(df_c.value)
        df_c.predicted_values = np.exp(df_c.predicted_values)

        df_c.value = df_c.value.iloc[:-1]
        df_c[['value', 'predicted_values']].plot()
        DataProcessing.calculate_accuracy(df_c, 0.05)
        print("\nPredicted value for the next day is:", df_c.predicted_values.iloc[-1])
        plt.show()

        return df_c

    @staticmethod
    def calculate_accuracy(df_c, per_cent):
        df_c['difference'] = abs(df_c.value - df_c.predicted_values)
        counter = 0

        for i in range(len(df_c)):
            diff_per_cent = per_cent * df_c.value[i]
            real_diff = df_c.difference[i]

            if real_diff < diff_per_cent:
                counter += 1

        accuracy = counter / len(df_c) * 100
        print("The difference between real and predicted values isn't bigger than", 100 * per_cent,
              "% in ", accuracy, "% of all values")


def init():
    create_csv.init()
    df = DataProcessing.read_set("crypto.csv")
    DataProcessing.ARIMA(df)


init()
