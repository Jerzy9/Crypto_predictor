import requests
import json
import csv


# Check connection with an API
def check_connection():
    url = 'https://api.coingecko.com/api/v3/ping'
    response = requests.get(url)
    if response.text == '{"gecko_says":"(V3) To the Moon!"}':
        return True
    return False


# Download date
def get_info(crypto_name):
    url = "https://api.coingecko.com/api/v3/coins/" + crypto_name + "/market_chart?vs_currency=usd&days=2&interval=daily"
    response = requests.get(url)

    if str(response) != "<Response [200]>":
        print("ERROR -", response)
        return 0

    # extract only price of bitcoin
    # make dictionary from request and sort it
    json_dict = dict(response.json())
    dates_and_prices = json_dict.get('prices')
    sorted(dates_and_prices)

    return dates_and_prices


# Collect data about every crypto currency
def prep_rows():
    dictionary = {"bitcoin": [], "litecoint": []}
    names = ["bitcoin", "litecoin"]
    for name in names:
        dates, prices = get_info(name)
        dictionary[name] = prices
    return dictionary


# Take all rows and put them into .csv file
def create_csv_file(dictionary):
    with open("crypto.csv", "w", newline='') as file:
        writer = csv.writer(file)
        for i in dictionary:
            #todo
            for j in i:
                writer.writerow()
        file.close()


print(check_connection())
# create_csv_file(prep_rows())
print(get_info("ethereum"))
