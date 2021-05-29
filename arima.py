import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_model import ARIMA

pd.options.mode.chained_assignment = None  # default='warn'

"""
ŚREDNIA KROCZĄCA - to średnia wartości (y) z jakiegoś okresu 'n', 
np. n = 100, więc średnia krocząca na dzis jest to srednia ze 100 ostatnich wartości,
dla jutra bedzie kolejna wartości, bo zbiór n się przesunie o jeden dzień itd.


p is the order of the AR(auto-regressive) term
Jest to liczba 'lagów' Y, które mają zostać użyte jako predyktory

q is the order of the MA(średnia krocząca) term
Jest to liczba opóżnionych błędów prognoz, które powinny zostąć zostąć uwzględnione w ARIMA model.

d is the number of differencing required to make the time series stationary
Wartość zmiennej d, jest to minimalna liczba różnicować, 
potrzebnych żeby dane były 'stationary' (nieruchome/stacjonarne)
Kiedy dane są stacjonarne d = 0


celem jest znalezenie p,q,d

//WZORY
Pure Auto-regressive - auto regresja
Y_t = a + B_1*Y_(t-1) + B_2*Y_(t-2) + ... + B_p*T_(t-p) + e_1

Pure model average
Yt = a + e_t + theta_1*e_(t-1) + theta_2*e_(t-2)  + ... + theta_p*e_(t-p)

Połączone
Y_t = a + B_1*Y_(t-1) + B_2*Y_(t-2) + ... + B_p*T_(t-p) + e_1 + theta_1*e_(t-1) + theta_2*e_(t-2)  + ... + theta_p*e_(t-p)



Step 1 - make date stationary:
    Bo AR jest modelem regresji liniowej, a regresja liniowa działa najlepiej,
    gdy predyktory nie są ze sobą skorelowane i są od siebie niezależne.

    Najpopularniejszym podejściem jest różnicowanie, 
    czyli odejmowanie poprzedniej wartości od obecnej wartości.
    Jeśli dane są skomplikowane czasami proces trzeba powtórzyć.


Step 2 auto regression:
    Sprawdzanie czy wymagane są jakieś warunki

Step 3 - 


"""


# pomocniczna metoda
def plot_stat(data_set):
    # # Original Series
    fig, axes = plt.subplots(3, 2)
    axes[0, 0].plot(data_set.value)
    axes[0, 0].set_title('Original Series')
    plot_acf(data_set.value, ax=axes[0, 1])

    # 1st Differencing
    axes[1, 0].plot(data_set.value.diff())
    axes[1, 0].set_title('1st Order Differencing')
    plot_acf(data_set.value.diff().dropna(), ax=axes[1, 1])

    # 2nd Differencing
    axes[2, 0].plot(data_set.value.diff().diff())
    axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(data_set.value.diff().diff().dropna(), ax=axes[2, 1])

    # print(data_set)
    plt.show()


def with_library(df):
    model = ARIMA(df.value, order=(2, 1, 1))
    model_fit = model.fit(disp=0)

    model_fit.plot_predict(dynamic=False)
    plt.show()
    # print(model_fit.summary())


def with_library_forecast(df):
    train, test = DataProcessing.split_set(df)

    model = ARIMA(train, order=(1, 1, 1))
    fitted = model.fit(disp=-1)

    # Forecast
    fc, se, conf = fitted.forecast(40, alpha=0.05)

    # Make as pandas series
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(conf[:, 0], index=test.index)
    upper_series = pd.Series(conf[:, 1], index=test.index)

    # Plot
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train, label='training')
    plt.plot(test, label='actual')
    plt.plot(fc_series, label='forecast')
    plt.fill_between(lower_series.index, lower_series, upper_series,
                     color='k', alpha=.15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
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
        # print("ADF statistics:", result[0])
        # print("p-value:", p)
        # print("number of lags:", result[2])
        # print("number of observations", result[3])

        if p > 0.05:
            print("Data is not stationary", p)
            return False
        else:
            print("Data is stationary", p)
            return True

    @staticmethod
    def make_data_stationary(df):
        df_stat = pd.DataFrame(np.log(df.value).diff().diff())
        df_stat = df_stat.dropna()

        return df_stat

    @staticmethod
    def AR(df, p):
        # adding columns to df with shifted values
        for i in range(1, p + 1):
            df['Shifted_values_%d' % i] = df['value'].shift(i)

        # split data into to training and validation set
        df_train, df_sample = DataProcessing.split_set(df)

        # .dropna() removes empty values
        df_train = df_train.dropna()
        df_sample = df_sample.dropna()

        # x, contains the lagged values, without the first column
        x_train = df_train.iloc[:, 1:].values.reshape(-1, p)

        # y - values, the first column
        y_train = df_train.iloc[:, 0].values.reshape(-1, 1)

        lr = LinearRegression()
        lr.fit(x_train, y_train)

        # w = (w_1,w_2,...,w_p)
        # Oszacowane współczynniki dla problemu regresji liniowej
        theta = lr.coef_.T
        # w_0 = intercept_ (przecięcie?)
        intercept = lr.intercept_

        # predicted values,
        # .dot - This method computes the matrix product between the DataFrame
        # and the values of any other Series, DataFrame etc.
        df_train['predicted_values'] = x_train.dot(theta) + intercept

        x_sample = df_sample.iloc[:, 1:].values.reshape(-1, p)
        df_sample['predicted_values'] = x_sample.dot(lr.coef_.T) + lr.intercept_
        # df_test[['Value','Predicted_Values']].plot()

        # print(x_sample)
        # root mean square error
        rmse = np.sqrt(mean_squared_error(df_sample['value'], df_sample['predicted_values']))

        # print("The RMSE is :", rmse, ", value of p : ", p)
        return [df_train, df_sample, theta, intercept, rmse]

    @staticmethod
    # moving average model
    def MA(res, q):
        for i in range(1, q + 1):
            res['Shifted_values_%d' % i] = res['residuals'].shift(i)

        res_train, res_test = DataProcessing.split_set(res)

        res_train_2 = res_train.dropna()
        x_train = res_train_2.iloc[:, 1:].values.reshape(-1, q)
        y_train = res_train_2.iloc[:, 0].values.reshape(-1, 1)

        # print(x_train)
        lr = LinearRegression()
        lr.fit(x_train, y_train)

        theta = lr.coef_.T
        intercept = lr.intercept_
        res_train_2['predicted_values'] = x_train.dot(lr.coef_.T) + lr.intercept_
        # res_train_2[['Residuals','Predicted_Values']].plot()

        x_test = res_test.iloc[:, 1:].values.reshape(-1, q)
        res_test['predicted_values'] = x_test.dot(lr.coef_.T) + lr.intercept_
        # res_test[['Residuals','Predicted_Values']].plot()

        RMSE = np.sqrt(mean_squared_error(res_test['residuals'], res_test['predicted_values']))

        # print("The RMSE is :", RMSE, ", Value of q : ", q)

        return [res_train_2, res_test, theta, intercept, RMSE]

    @staticmethod
    def ARIMA(df):
        print("STAGE 1 - stationary")
        df_stat = DataProcessing.make_data_stationary(df)

        best_rmse = math.inf
        best_p = -1

        for i in range(1, 21):
            [df_train, df_test, theta, intercept, rmse] = DataProcessing.AR(pd.DataFrame(df_stat.value), i)
            if rmse < best_rmse:
                best_rmse = rmse
                best_p = i

        # print(df_train)

        print("\nSTAGE 2 - auto-regressive model")
        print("The RMSE is:", best_rmse, ", value of p: ", best_p)
        # final AR
        [df_train, df_test, theta, intercept, rmse] = DataProcessing.AR(pd.DataFrame(df_stat.value), best_p)
        df_c = pd.concat([df_train, df_test])

        # difference between forecast value and the real value
        res = pd.DataFrame()
        res['residuals'] = df_c.value - df_c.predicted_values

        print("\nSTAGE 3 - moving average model")
        best_rmse2 = math.inf
        best_q = -1

        for i in range(1, 13):
            [res_train, res_test, theta, intercept, rmse] = DataProcessing.MA(pd.DataFrame(res.residuals), i)
            if rmse < best_rmse2:
                best_rmse2 = rmse
                best_q = i

        print("The RMSE is:", best_rmse2, ", value of q: ", best_q)
        # final MA
        [res_train, res_test, theta, intercept, RMSE] = DataProcessing.MA(pd.DataFrame(res.residuals), best_q)
        res_c = pd.concat([res_train, res_test])

        df_c.predicted_values += res_c.predicted_values

        # plt.show()
        print("\nSTAGE 4 - ARIMA")
        df_c.value += np.log(df).shift(1).value
        df_c.value += np.log(df).diff().shift().value
        df_c.predicted_values += np.log(df).shift(1).value
        df_c.predicted_values += np.log(df).diff().shift().value
        df_c.value = np.exp(df_c.value)
        df_c.predicted_values = np.exp(df_c.predicted_values)

        # df_c.predicted_values = df_c.iloc[-30:].predicted_values
        df_c[['value', 'predicted_values']].plot()
        plt.show()
        DataProcessing.calculate_accuracy(df_c, 0.05)
        # print(df_c)
        return df_c

    @staticmethod
    def calculate_accuracy(df_c, per_cent):
        df_c.iloc[:, :][['value', 'predicted_values']].plot()
        df_c['difference'] = abs(df_c.value - df_c.predicted_values)
        df_c[['difference']].plot()
        # plt.show()
        counter = 0

        for i in range(len(df_c)):
            diff_per_cent = per_cent * df_c.value[i]
            real_diff = df_c.difference[i]
            # print("forecast", diff_per_cent, "real: ", real_diff)

            if real_diff < diff_per_cent:
                counter += 1

        accuracy = counter / len(df_c) * 100
        print("The difference between real and predicted values isn't bigger than", per_cent,
              "% in ", accuracy, "% of all values")
        # print(df_c)


def init():
    df = DataProcessing.read_set("crypto.csv")

    # df_train, df_sample = DataProcessing.split_set(df_stat)
    # is_stationary = DataProcessing.check_if_data_is_stationary_ADF(data_set)
    # df_stationary, d = DataProcessing.make_data_stationary(data_set)
    # DataProcessing.AR(df, 5)

    df_c = DataProcessing.ARIMA(df)

    #

    # with_library_forecast(df)


init()



