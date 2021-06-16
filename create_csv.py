import requests
import json
import csv
import pandas as pd
import numpy as np
import datetime


class GatherInformation:
    @staticmethod
    # Check connection with an API
    def check_connection():
        url = 'https://api.coingecko.com/api/v3/ping'
        response = requests.get(url)
        if response.text == '{"gecko_says":"(V3) To the Moon!"}':
            return True
        return False

    @staticmethod
    # Download date
    def get_info(crypto_name, days):
        url = "https://api.coingecko.com/api/v3/coins/" + crypto_name + "/market_chart?vs_currency=usd&days=" + str(days) + "&interval=daily"
        response = requests.get(url)

        if str(response) != "<Response [200]>":
            print("ERROR -", response, crypto_name)
            return 0

        # extract only price of bitcoin
        # make dictionary from request and sort it
        json_dict = dict(response.json())
        dates_and_prices = json_dict.get('prices')
        sorted(dates_and_prices)

        price = []
        date = []
        for i in dates_and_prices:
            date.append(i[0])
            price.append(i[1])

        return date, price

    @staticmethod
    # Collect data about every crypto currency
    def prep_rows(crypto_names, days):
        dictionary = {}

        for name in crypto_names:
            # first crypto goes with dates
            # for the rest, price is the important thing
            if name == crypto_names[0]:
                dates_and_prices = GatherInformation.get_info(name, days)
                dictionary['date'] = dates_and_prices[0]
                dictionary[name] = np.round(dates_and_prices[1], 2)
                continue

            # extract only prices
            dates_and_prices = GatherInformation.get_info(name, days)
            dictionary[name] = np.round(dates_and_prices[1], 2)
        return dictionary

    @staticmethod
    # Take all rows and put them into .csv file
    def create_csv_file(path, dictionary, crypto_names):
        with open(path, "w", newline='') as file:
            writer = csv.writer(file)
            rows_quantity = len(dictionary.get('date'))

            # first row with column's names
            temp = ['date']
            temp.extend(crypto_names)
            writer.writerow(temp)

            for row in range(rows_quantity - 1):
                new_row = []
                # create row
                for crypto_name in dictionary:
                    new_row.append(dictionary.get(crypto_name)[row])

                # write row
                writer.writerow(new_row)

            # print(new_row[0])
            tom = datetime.date.today() + datetime.timedelta(days=1)
            doubled_row = [tom, new_row[1]]
            writer.writerow(doubled_row)
            file.close()

    @staticmethod
    def read_csv_file(csv_file_path):
        csv_file = pd.read_csv(csv_file_path)
        return csv_file

    @staticmethod
    def convert_timestamps_to_dates(dictionary):
        for i in range(len(dictionary.get('date'))):
            s = int(dictionary.get('date')[i]/1000)
            dictionary.get('date')[i] = datetime.datetime.fromtimestamp(s).date()


def init():
    # crypto_names = ["bitcoin", "litecoin", "ethereum", "monero"]
    crypto_names = ["bitcoin"]
    days = 400
    path = './crypto.csv'

    dictionary = GatherInformation.prep_rows(crypto_names, days)
    GatherInformation.convert_timestamps_to_dates(dictionary)

    # write to file
    GatherInformation.create_csv_file(path, dictionary, crypto_names)

    # read from file
    cryptos = GatherInformation.read_csv_file(path)


init()
