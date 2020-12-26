import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.metrics import mean_squared_error
from openpyxl import load_workbook


# data_weather = pd.read_csv('weather_features.csv')
data_energy = pd.read_csv('energy_dataset.csv')

# plt.style.use('fivethirtyeight')

hour = data_energy.time.str.slice(11, 13)


# df1 = pd.DataFrame(data_weather)
df = pd.DataFrame(data_energy)

df['hour'] = hour

# df = pd.merge(df1, df2, on='time')

df = df.dropna()


# https://github.com/pandas-dev/pandas/issues/16898, ryanjdillon (utc=True)
# https://www.youtube.com/watch?v=yCgJGsg0Xa4 (Data School)


# 'temp', 'temp_min', 'temp_max', 'pressure',
#                    'humidity', 'wind_speed', 'wind_deg', 'rain_1h', 'rain_3h', 'snow_3h',
#                    'clouds_all', 'weather_id',
numeric_columns = ['time', 'generation biomass',
                   'generation fossil brown coal/lignite',
                   'generation fossil coal-derived gas', 'generation fossil gas',
                   'generation fossil hard coal', 'generation fossil oil',
                   'generation fossil oil shale', 'generation fossil peat',
                   'generation geothermal', 'generation hydro pumped storage consumption',
                   'generation hydro run-of-river and poundage',
                   'generation hydro water reservoir', 'generation marine',
                   'generation nuclear', 'generation other', 'generation other renewable',
                   'generation solar', 'generation waste', 'generation wind offshore',
                   'generation wind onshore', 'forecast solar day ahead',
                   'forecast wind onshore day ahead', 'total load forecast',
                   'total load actual', 'price day ahead', 'price actual', 'hour']

# 'temp', 'pressure', 'humidity', 'wind_speed', 'wind_deg', 'rain_1h', 'rain_3h', 'snow_3h','clouds_all', 'weather_id',

graph_columns = ['generation biomass',
                 'generation fossil brown coal/lignite', 'generation fossil gas',
                 'generation fossil hard coal', 'generation fossil oil',
                 'generation hydro pumped storage consumption',
                 'generation hydro run-of-river and poundage',
                 'generation hydro water reservoir',
                 'generation nuclear', 'generation other', 'generation other renewable',
                 'generation solar', 'generation waste',
                 'generation wind onshore', 'forecast solar day ahead',
                 'forecast wind onshore day ahead', 'total load forecast',
                 'total load actual', 'price day ahead', 'price actual', 'hour']

X = ['forecast solar day ahead',
     'forecast wind onshore day ahead', 'total load forecast', 'hour']
y = ['price day ahead', 'price actual']

for i in df[numeric_columns]:
  # https://datatofish.com/string-to-integer-dataframe/
  df[i] = pd.to_numeric(df[i])

print(df.dtypes)

dtr_model = DecisionTreeRegressor(random_state=3)

train_X, val_X, train_y, val_y = train_test_split(df[X], df[y], random_state=3)


print(train_y['price actual'])

dtr_model.fit(train_X, train_y['price actual'])

val_predictions = dtr_model.predict(val_X)

rms_my_pred = sqrt(mean_squared_error(val_y['price actual'], val_predictions))
rms_TSO_pred = sqrt(mean_squared_error(
    val_y['price actual'], val_y['price day ahead']))

mean_abs_error_my_pred = mean_absolute_error(
    val_y['price actual'], val_predictions)
mean_abs_error_TSO_pred = mean_absolute_error(
    val_y['price actual'], val_y['price day ahead'])

print('Decision Tree Error: ' +
      str(rms_my_pred))
print('Electric Company Error: ' +
      str(rms_TSO_pred))

error1 = 0
error2 = 0

val_y_actual_price = val_y['price actual']
val_y_predict_price = val_y['price day ahead']

diff1 = val_y['price actual'] - val_predictions
diff2 = val_y['price actual'] - val_y['price day ahead']


print(diff1.describe())
print(diff2.describe())

gr = sns.distplot(diff1, bins=50, label='Actual Price - My Predictions')
gr = sns.distplot(diff2, bins=50, label='Actual Price - TSO Predictions')
gr.set(xlabel="Price Difference (€/MWh)", ylabel="Frequency")
plt.legend()
plt.show()


gr = sns.distplot(val_y['price actual'], bins=50, label='Actual Price')
gr = sns.distplot(val_y['price day ahead'], bins=50, label='TSO Prediction')
gr = sns.distplot(val_predictions, bins=50, label='My Prediction')
gr.set(xlabel="Price of Electricity(€/MWh)", ylabel="Frequency")
plt.legend()
plt.show()


# for i in val_y_actual_price:

#     error1 = error1 + abs(val_y['price actual'].loc[i] - val_predictions[i, 0])
#     error2 = error2 + abs(val_y_actual_price[i, 0] - val_y_predict_price[i, 0])

# print(error1)
# print(error2)


# filt_Val = (df['city_name'] == 'Valencia')
# df_Val = df[filt_Val]


# sns.distplot(val_y['price actual'])
# sns.distplot(val_y['price day ahead'])

# sns.distplot(df['total load forecast'])
# sns.distplot(df['total load actual'])


gr = sns.distplot(df['price day ahead'], bins=20, label='ISO Prediction')
gr = sns.distplot(df['price actual'], bins=20, label='Actual Price')
gr.set(xlabel="Price of Electricity", ylabel="Frequency")
plt.legend()
plt.show()

# print(df.info())


# plt.figure(figsize=(8, 12))
# corr = df[graph_columns].corr()
# sns.set(font_scale=0.5)
# sns.heatmap(corr, cmap="RdBu", linewidths=1.0, vmin=-
#             1, vmax=1, annot=True)

# plt.xlabel('', fontsize=2)
# plt.ylabel('', fontsize=5)
# plt.xticks(rotation=20)
# plt.show()


# path = '/Users/nathanoliver/Desktop/Python/Electricity Prediction/price_comparison.xlsx'


# print('loading workbook')
# wb = load_workbook(path)

# print('loaded workbook')

# sheet = wb["Sheet1"]


# wb2.save(path)
