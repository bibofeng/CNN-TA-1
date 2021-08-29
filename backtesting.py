import os
import pandas as pd
from keras.models import load_model
import talib as ta  # TA-Lib for calculation of indicators
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


def add_indicators(data):
    data["RSI"] = ta.RSI(data["Close"])
    data["EMA"] = ta.EMA(data["Close"])
    data["WMA"] = ta.WMA(data["Close"])
    data["ROC"] = ta.ROC(data["Close"])
    data["TEMA"] = ta.TEMA(data["Close"])
    data["CMO"] = ta.CMO(data["Close"])
    data["SAR"] = ta.SAR(data["High"], data["Low"])
    data["WILLR"] = ta.WILLR(data["High"], data["Low"], data["Close"], timeperiod=15)
    data["CCI"] = ta.CCI(data["High"], data["Low"], data["Close"], timeperiod=15)
    data["PPO"] = ta.PPO(data["Close"], fastperiod=6, slowperiod=15)
    data["MACD"] = ta.MACD(data["Close"], fastperiod=6, slowperiod=15)[0]
    a = ta.WMA(data["Close"], timeperiod=15 // 2)
    b = data["WMA"]
    data["HMA"] = ta.WMA(2 * a - b, timeperiod=int(15 ** (0.5)))
    data["ADX"] = ta.ADX(data["High"], data["Low"], data["Close"], timeperiod=15)
    data.dropna(inplace=True)


rootpath = os.path.dirname(__file__)
##C:\dev\bbhub\CNN-TA-1\MODEL\type_1\phase_1

fp1 = "C:\\dev\\bbhub\\CNN-TA-1\\MODEL\\type_1\\phase_1\\saved_model.pb"
fp2 = "C:\\dev\\bbhub\\CNN-TA-1\\MODEL\\type_1\\phase_2\\saved_model.pb"
fp3 = "C:\\dev\\bbhub\\CNN-TA-1\\MODEL\\type_1\\phase_3\\saved_model.pb"


# fp1 = rootpath+'/MODEL/type_1/phase_1/saved_model.pb'
# fp2 = rootpath+'/MODEL/type_1/phase_2/saved_model.pb'
# fp3 = rootpath+'/MODEL/type_1/phase_3/saved_model.pb'

model_1 = load_model(fp1)
model_2 = load_model(fp2)
model_3 = load_model(fp3)

root = "C:/temp/CNNTA/Data/AAPL/"  # root data path

data = pd.read_csv(rootpath + "/Data/AAPL.csv")
print("loaded CSV\n")


data.dropna(inplace=True)  # Drop missing entries
data.set_index("Date", inplace=True)  # Set Date as the index column

# Use Adj Close instead of Close
data.drop(labels=["Close"], axis=1, inplace=True)
data.rename(columns={"Adj Close": "Close"}, inplace=True)
add_indicators(data)  # Add indicators to the dataframe


image_generator = ImageDataGenerator(rescale=1 / 255)

testing_dataset = image_generator.flow_from_directory(
    directory="c:/temp/CNNTA/AAPL/Images/",
    target_size=(15, 15),
    batch_size=32,
    color_mode="grayscale",
)

result_1 = model_1.predict(testing_dataset)
result_2 = model_2.predict(testing_dataset)
result_3 = model_3.predict(testing_dataset)

max_index_1 = np.argmax(result_1, axis=1)
max_index_2 = np.argmax(result_2, axis=1)
max_index_3 = np.argmax(result_3, axis=1)

temp = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # padding 14 days

max_index_1 = np.insert(max_index_1, 0, temp)
max_index_2 = np.insert(max_index_2, 0, temp)
max_index_3 = np.insert(max_index_3, 0, temp)


# add 'result' array as new column in DataFrame
data["model_1"] = max_index_1.tolist()
data["model_2"] = max_index_2.tolist()
data["model_3"] = max_index_3.tolist()

# { 'Buy': 0, 'Hold': 1, 'Sell': 2}
data.to_csv("Data/AAPL/test.csv")  # Save Data as CSV
