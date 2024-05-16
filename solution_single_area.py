import pandas as pd

pd.set_option('display.max_columns', 100)

files = [
    ("ConsumerPrices", "Consumer prices indicators - FAOSTAT_data_en_2-22-2024.csv"),
    ("CropsProduction", "Crops production indicators - FAOSTAT_data_en_2-22-2024.csv"),
    ("Emissions", "Emissions - FAOSTAT_data_en_2-27-2024.csv"),
    ("Employment", "Employment - FAOSTAT_data_en_2-27-2024.csv"),
    ("ExchangeRate", "Exchange rate - FAOSTAT_data_en_2-22-2024.csv"),
    ("FertilizersUse", "Fertilizers use - FAOSTAT_data_en_2-27-2024.csv"),
    ("FoodBalance", "Food balances indicators - FAOSTAT_data_en_2-22-2024.csv"),
    ("FoodSecurity", "Food security indicators  - FAOSTAT_data_en_2-22-2024.csv"),
    ("FoodTrade", "Food trade indicators - FAOSTAT_data_en_2-22-2024.csv"),
    ("ForeignDirectInvestment", "Foreign direct investment - FAOSTAT_data_en_2-27-2024.csv"),
    ("LandTemperatureChange", "Land temperature change - FAOSTAT_data_en_2-27-2024.csv"),
    ("LandUse", "Land use - FAOSTAT_data_en_2-22-2024.csv"),
    ("PesticidesUse", "Pesticides use - FAOSTAT_data_en_2-27-2024.csv")
]


def load_csv(filename):
    data = pd.read_csv('resources/' + filename, low_memory=False)
    return data


def filter_by_train_and_country(trains, train_name, country_name):
    train_df = trains[train_name]
    filtered_df = train_df[train_df['Area'] == country_name]

    return filtered_df


def fill_values(series):
    fill_value = series.describe()['25%']
    mean_value = series.mean()
    next_year = series.shift(-1)

    if fill_value == 0:
        fill_value = mean_value

    series = series.mask((series == 0.0) | series.isna() & (next_year > 0), next_year)

    max_iterations = 30
    counter = 0
    while series.isna().any() and counter < max_iterations:
        series = series.mask((series == 0.0) | series.isna(), fill_value)
        counter += 1

    if series.isna().any():
        series.fillna(mean_value, inplace=True)

    return series


def rename_columns(input_df, column_mapping_information):
    input_df = input_df.rename(columns=column_mapping_information)
    return input_df


trains = {}
combined_df_result = {}
over_fitting_countries = []

for desired_name, file in files:
    trains[desired_name] = load_csv(file)

unique_areas = trains['FoodTrade']['Area'].unique()

print("=====================================")
print("Unique Areas")
# print(unique_areas)


country_name = 'Ethiopia PDR'
print("Country Name: ", country_name)
print("=====================================")
print("Data Pivoting")
print("=====================================")

food_trade_unique_item_code_list = [
    'F1888', 'F1844', 'F1890', 'F1889'
    , 'F1907', 'F1908', 'F1848', 'F1896'
]

filtered_df_food_trade = filter_by_train_and_country(trains, 'FoodTrade', country_name)
filtered_df_food_trade_exported_value = filtered_df_food_trade[filtered_df_food_trade['Element Code'] == 5922]
pivot_df_food_trade = filtered_df_food_trade_exported_value.pivot(index='Year Code', columns='Item Code (CPC)',
                                                                  values='Value')
existing_columns_food_trade = [col for col in food_trade_unique_item_code_list if
                               col in pivot_df_food_trade.columns]
pivot_df_food_trade = pivot_df_food_trade[existing_columns_food_trade]
pivot_df_food_trade.reset_index(inplace=True)
pivot_df_food_trade_columns = {
    'F1888': 'FT1',
    'F1844': 'FT2',  # partially crops
    'F1890': 'FT3',
    'F1889': 'FT4',
    'F1907': 'FT5',
    'F1908': 'FT6',
    'F1848': 'FT7',  # partially crops
    'F1896': 'FT8'
}
pivot_df_food_trade = rename_columns(pivot_df_food_trade, pivot_df_food_trade_columns)
# print(pivot_df_food_trade)

## feature engineering

# 1. ConsumerPrices
filtered_df_consumer_price = filter_by_train_and_country(trains, 'ConsumerPrices', country_name)

grouped_df = filtered_df_consumer_price.groupby(['Year Code', 'Item Code'])['Value'].sum().reset_index()
pivot_df_consumer_prices = grouped_df.pivot(index='Year Code', columns='Item Code', values='Value')
pivot_df_consumer_prices.reset_index(inplace=True)

print("=====================================")
print("Consumer Prices")
pivot_df_consumer_prices_columns = {
    23013: 'CSP1',
    23014: 'CSP2'
}
pivot_df_consumer_prices = rename_columns(pivot_df_consumer_prices, pivot_df_consumer_prices_columns)

# print(pivot_df_consumer_prices)

# 2. CropsProduction
filtered_df_crops_production = filter_by_train_and_country(trains, 'CropsProduction', country_name)
crops_production_unique_item_list = ['F1717', 'F1804', 'F17530', 'F1738'
    , 'F1841', 'F1732', 'F1726', 'F1720'
    , 'F1723', 'F1729', 'F1735']

pivot_df_crops_production = filtered_df_crops_production.pivot(index='Year Code', columns='Item Code (CPC)',
                                                               values='Value')
crops_production_existing_columns = [col for col in crops_production_unique_item_list if
                                     col in pivot_df_crops_production.columns]
pivot_df_crops_production = pivot_df_crops_production[crops_production_existing_columns]
pivot_df_crops_production.reset_index(inplace=True)
pivot_df_crops_production = pivot_df_crops_production.apply(fill_values)

print("=====================================")
print("Crops Production")
pivot_df_crops_production_columns = {
    'F1717': 'CRP1',
    'F1804': 'CRP2',
    'F17530': 'CRP3',
    'F1738': 'CRP4',
    'F1841': 'CRP5',
    'F1732': 'CRP6',
    'F1726': 'CRP7',
    'F1720': 'CRP8',
    'F1723': 'CRP9',
    'F1729': 'CRP10',
    'F1735': 'CRP11'
}
pivot_df_crops_production = rename_columns(pivot_df_crops_production, pivot_df_crops_production_columns)
# print(pivot_df_crops_production)

# 3. Emissions - Distinctive Data Relations
filtered_df_emissions = filter_by_train_and_country(trains, 'Emissions', country_name)
emissions_unique_element_code = [72430, 72440, 7230, 7273]

grouped_emissions = {}

for code in emissions_unique_element_code:
    grouped_df = filtered_df_emissions[filtered_df_emissions['Element Code'] == code]
    group_pivot_df_emissions = grouped_df.pivot(index='Year Code', columns='Item Code (CPC)', values='Value')
    grouped_emissions[code] = group_pivot_df_emissions

pivot_df_emissions = pd.concat(grouped_emissions.values(), axis=1, keys=grouped_emissions.keys())
pivot_df_emissions.reset_index(inplace=True)

print("=====================================")
print("Emissions")
columns_list = pivot_df_emissions.columns.tolist()
for i in range(len(columns_list)):
    if i == 0:
        columns_list[i] = 'Year Code'
    else:
        columns_list[i] = 'EM' + str(i)

pivot_df_emissions.columns = columns_list
pivot_df_emissions = pivot_df_emissions.loc[:, (pivot_df_emissions != 0).any(axis=0)]
# print(pivot_df_emissions)

# 4. Employment - Indicator 21150 - Not enough data
filtered_df_employment = filter_by_train_and_country(trains, 'Employment', country_name)
filtered_df_employment_indicator_code_21144 = filtered_df_employment[
    filtered_df_employment['Indicator Code'] == 21144]
pivot_df_employment = filtered_df_employment_indicator_code_21144.pivot(index='Year Code', columns='Indicator Code',
                                                                        values='Value')

print("=====================================")
print("Employment")

pivot_df_employment.reset_index(inplace=True)
pivot_df_employment = pivot_df_employment.rename(columns={21144: 'EMP1'})
# print(pivot_df_employment)

# 5. ExchangeRate
filtered_df_exchange_rate = filter_by_train_and_country(trains, 'ExchangeRate', country_name)
pivot_df_exchange_rate = filtered_df_exchange_rate.groupby('Year Code')['Value'].sum().reset_index()
pivot_df_exchange_rate.columns = ['Year Code', 'LCUValues']

print("=====================================")
print("Exchange Rate")
pivot_df_exchange_rate = pivot_df_exchange_rate.rename(columns={'LCUValues': 'EXC1'})
# print(pivot_df_exchange_rate)

# 6. [X] FertilizersUse - Too Fewer & Spread Data to train
# filtered_df_fertilizers_use = filter_by_train_and_country(trains, 'Fertilizers', 'Afghanistan')
filtered_df_fertilizers_use = filter_by_train_and_country(trains, 'FertilizersUse', country_name)
distinct_item_codes = filtered_df_fertilizers_use['Item Code'].unique().tolist()
print("=====================================")
print("Fertilizers Use")
print(distinct_item_codes)

# 7. FoodBalance - Need To Handle 0 Data
filtered_df_food_balance = filter_by_train_and_country(trains, 'FoodBalance', country_name)

filtered_df_food_balance_exported_quantity = filtered_df_food_balance[
    filtered_df_food_balance['Element Code'] == 5911]
food_balance_unique_item_code_list = ['S2905', 'S2907', 'S2908', 'S2909'
    , 'S2911', 'S2912', 'S2913', 'S2914'
    , 'S2918', 'S2919', 'S2922', 'S2923']
existing_columns_food_balance = [col for col in food_balance_unique_item_code_list if
                                 col in pivot_df_food_trade.columns]
pivot_df_food_balance = filtered_df_food_balance_exported_quantity.pivot(index='Year Code',
                                                                         columns='Item Code (FBS)', values='Value')
pivot_df_food_balance = pivot_df_food_balance[existing_columns_food_balance]

pivot_df_food_balance.reset_index(inplace=True)
pivot_df_food_balance = pivot_df_food_balance.apply(fill_values)

print("=====================================")
print("Food Balance")
pivot_df_food_balance_columns = {
    'S2905': 'FB1',
    'S2907': 'FB2',
    'S2908': 'FB3',
    'S2909': 'FB4',  # partially crops
    'S2911': 'FB5',
    'S2912': 'FB6',
    'S2913': 'FB7',
    'S2914': 'FB8',
    'S2918': 'FB9',
    'S2919': 'FB10',
    'S2922': 'FB11',
    'S2923': 'FB12'
}

pivot_df_food_balance = rename_columns(pivot_df_food_balance, pivot_df_food_balance_columns)
# print(pivot_df_food_balance)

# 8. FoodSecurity
filtered_df_food_security = filter_by_train_and_country(trains, 'FoodSecurity', country_name)
food_security_unique_item_code_list = [21030, 21031]
filtered_df_food_security_by_unique_item_code = filtered_df_food_security[
    filtered_df_food_security['Item Code'].isin(food_security_unique_item_code_list)]

pivot_df_food_security = filtered_df_food_security_by_unique_item_code.pivot(index='Year Code', columns='Item Code',
                                                                             values='Value')
pivot_df_food_security.reset_index(inplace=True)
pivot_df_food_security = pivot_df_food_security.apply(fill_values)

print("=====================================")
print("Food Security")
pivot_df_food_security_columns = {
    21030: 'FS1',
    21031: 'FS2'
}

pivot_df_food_security = rename_columns(pivot_df_food_security, pivot_df_food_security_columns)
# print(pivot_df_food_security)

# 10. ForeignDirectInvestment - Item Code = 23085:Total FDI outflows
filtered_df_foreign_direct_investment = filter_by_train_and_country(trains, 'ForeignDirectInvestment', country_name)
filtered_df_foreign_direct_investment_23080 = filtered_df_foreign_direct_investment[
    filtered_df_foreign_direct_investment['Item Code'] == 23080]
pivot_df_foreign_direct_investment = filtered_df_foreign_direct_investment_23080.pivot(index='Year Code',
                                                                                       columns='Item Code',
                                                                                       values='Value')
pivot_df_foreign_direct_investment.reset_index(inplace=True)

print("=====================================")
print("Foreign Direct Investment")
pivot_df_foreign_direct_investment = pivot_df_foreign_direct_investment.rename(columns={23081: 'FDI1'})
# print(pivot_df_foreign_direct_investment)

# 11. LandTemperatureChange
# - Element Code 7271 Temperature change # [X] 6078 Standard Deviation
# - Month Code 7020	Meteorological year
filtered_df_land_temperature_change = filter_by_train_and_country(trains, 'LandTemperatureChange', country_name)
filtered_df_land_temperature_change_by_codes = filtered_df_land_temperature_change[
    (filtered_df_land_temperature_change['Element Code'] == 7271) &
    (filtered_df_land_temperature_change['Months Code'] == 7020)
    ]

pivot_df_land_temperature_change = filtered_df_land_temperature_change_by_codes.pivot(index='Year Code',
                                                                                      columns='Element Code',
                                                                                      values='Value')
pivot_df_land_temperature_change.reset_index(inplace=True)
print("=====================================")
print("Land Temperature Change")
pivot_df_land_temperature_change = pivot_df_land_temperature_change.rename(columns={7271: 'LTC1'})
# print(pivot_df_land_temperature_change)

# 12. LandUse
filtered_df_land_use = filter_by_train_and_country(trains, 'LandUse', country_name)
distinct_item_codes = filtered_df_land_use['Item Code'].unique().tolist()

grouped_df = \
    filtered_df_land_use[filtered_df_land_use['Item Code'].isin(distinct_item_codes)].groupby(
        ['Year Code', 'Item Code'])['Value'].sum().reset_index()

pivot_df_land_use = grouped_df.pivot(index='Year Code', columns='Item Code', values='Value')
pivot_df_land_use.reset_index(inplace=True)
pivot_df_land_use.apply(fill_values)

print("=====================================")
print("Land Use")
pivot_df_land_use_columns = {
    6620: 'LU1',
    6621: 'LU2',
    6630: 'LU3',
    6640: 'LU4',
    6650: 'LU5',
}

pivot_df_land_use = rename_columns(pivot_df_land_use, pivot_df_land_use_columns)
# print(pivot_df_land_use)

# 13. PesticidesUse
filtered_df_pesticides_use = filter_by_train_and_country(trains, 'PesticidesUse', country_name)
filtered_df_pesticides_use = filtered_df_pesticides_use[
    (filtered_df_pesticides_use['Element Code'] == 5157) & (filtered_df_pesticides_use['Item Code'] == 1357)]
grouped_df = filtered_df_pesticides_use.groupby(['Year Code', 'Item Code']).sum().reset_index()
grouped_df.set_index('Year Code', inplace=True)
pivot_df_pesticides_use = grouped_df.pivot(columns='Item Code', values='Value')

print("=====================================")
print("Pesticides Use")
pivot_df_pesticides_use_columns = {
    1357: 'PU1'
}
pivot_df_pesticides_use = rename_columns(pivot_df_pesticides_use, pivot_df_pesticides_use_columns)
# print(pivot_df_pesticides_use)

## Data Preparation
print("=====================================")
print("Data Preparation")
print("=====================================")

# Concatenate all dataframes
dataframes = [pivot_df_food_trade, pivot_df_consumer_prices, pivot_df_crops_production, pivot_df_emissions,
              pivot_df_employment, pivot_df_exchange_rate, pivot_df_food_balance, pivot_df_food_security,
              pivot_df_foreign_direct_investment, pivot_df_land_temperature_change, pivot_df_land_use,
              pivot_df_pesticides_use]

for df in dataframes:
    if 'Year Code' in df.columns:
        df.set_index('Year Code', inplace=True)

print(" Step1: Concatenate all dataframes")
merged_df_data = pd.concat(dataframes, axis=1)
print(" Step2: Fill missing values")
merged_df_data = merged_df_data.apply(fill_values)
print(" Step3: Sort by index")
merged_df_data = merged_df_data.sort_index(ascending=False)
print(" Step4: Drop NA columns")
merged_df_data = merged_df_data.dropna(axis=1)

print("=====================================")
print("Merged Data")
# print(merged_df_data)
print("=====================================")

last_year = merged_df_data.index.max()
start_year = last_year - 19
# print("Start Year: ", start_year, "Last Year: ", last_year)
mask = (merged_df_data.index >= start_year) & (merged_df_data.index <= last_year)
merged_df_data = merged_df_data[mask]

print("=====================================")
print("Merged Data - filter year")
# print(merged_df_data.index.unique())
# print(merged_df_data.index.dtype)
# print(merged_df_data)
print("=====================================")

ft_columns = ['FT1', 'FT2', 'FT3', 'FT4', 'FT5', 'FT6', 'FT7', 'FT8']
csp_columns = ['CSP1', 'CSP2']
cpr_columns = ['CRP1', 'CRP2', 'CRP3', 'CRP4', 'CRP5', 'CRP6', 'CRP7', 'CRP8', 'CRP9', 'CRP10', 'CRP11']
em_columns = pivot_df_emissions.columns.tolist()[1:]
emp_columns = ['EMP1']
exc_columns = ['EXC1']
fb_columns = ['FB1', 'FB2', 'FB3', 'FB4', 'FB5', 'FB6', 'FB7', 'FB8', 'FB9', 'FB10', 'FB11', 'FB12']
fs_columns = ['FS1', 'FS2']
fdi_columns = ['FDI1']
ltc_columns = ['LTC1']
lu_columns = ['LU1', 'LU2', 'LU3', 'LU4', 'LU5']
pu_columns = ['PU1']

all_columns = ft_columns + csp_columns + cpr_columns + em_columns + emp_columns + exc_columns + fb_columns + fs_columns + fdi_columns + ltc_columns + lu_columns + pu_columns
existing_columns = [col for col in all_columns if col in merged_df_data.columns]
correlation_matrix = merged_df_data[existing_columns].corr()
# print(correlation_matrix)

for column in correlation_matrix.columns:
    if column not in ft_columns:
        correlations = correlation_matrix[column].loc[ft_columns].abs()
        if correlations.max() < 0.4:
            merged_df_data.drop(column, axis=1, inplace=True)

print("=====================================")
print("Filtered Data")
# print(merged_df_data)

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


X = merged_df_data.drop(ft_columns, axis=1).values
y = merged_df_data[ft_columns].values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Number of columns in y_train: ", y_train.shape[1])

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

total_instances = len(merged_df_data)
train_instances = len(X_train)
test_instances = len(X_test)

print(f"Total instances: {total_instances}")
print(f"Training instances: {train_instances}")
print(f"Test instances: {test_instances}")

from sklearn.metrics import mean_squared_error

input_size = X_train.shape[1]
hidden_size = 50  # This is an arbitrary number
output_size = y_train.shape[1]

model = MLP(input_size, hidden_size, output_size)

train_predictions = model(X_train)
test_predictions = model(X_test)

train_mse = mean_squared_error(y_train.detach().numpy(), train_predictions.detach().numpy())
test_mse = mean_squared_error(y_test.detach().numpy(), test_predictions.detach().numpy())

print("=====================================")
print("Mean Squared Error")
print(f"Mean Squared Error for the training set: {train_mse}")
print(f"Mean Squared Error for the test set: {test_mse}")
if test_mse > train_mse:
    print("Overfitting", country_name)
    over_fitting_countries.append(country_name)


batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

input_size = X_train.shape[1]
hidden_size = 64
output_size = y_train.shape[1]
model = MLP(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
# weight_decay for L2 regularization
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.01)

import matplotlib.pyplot as plt

print("=====================================")
print("Training the model")
loss_values = []
epochs = 50
for epoch in range(epochs):
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_values.append(loss.item())
    # print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

plt.plot(range(1, epochs + 1), loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.show()

# Make predictions in 3 years
model.eval()
future_years = 3
with torch.no_grad():
    last_year_data = X[-1].unsqueeze(0)
    predictions = []
    for _ in range(future_years):
        prediction = model(last_year_data)
        predictions.append(prediction)
        last_year_data = torch.cat((last_year_data[:, 8:], prediction), dim=1)

predictions = torch.stack(predictions).squeeze().numpy()
predictions = scaler.inverse_transform(predictions)

first_index_value = merged_df_data.index[0]
years = range(first_index_value, first_index_value + future_years)

print("Predictions in 3 years")
for year, prediction in zip(years, predictions):
    print(f"Year: {year}")
    for feature, value in zip(ft_columns, prediction):
        print(f"{feature}: {value}")
    print("\n")

import pandas as pd
import os

data = {
    "Year": [],
    "Feature": [],
    "Value": []
}

for year, prediction in zip(years, predictions):
    for feature, value in zip(ft_columns, prediction):
        data["Year"].append(year)
        data["Feature"].append(feature)
        data["Value"].append(value)

df = pd.DataFrame(data)

pivot_df_food_trade_columns_revert = {
    'FT1': 'Cereals and Preparations',
    'FT2': 'Fats and Oils (excluding Butter)',
    'FT3': 'Sugar and Honey',
    'FT4': 'Fruit and Vegetables',
    'FT5': 'Alcoholic Beverages',
    'FT6': 'Non-alcoholic Beverages',
    'FT7': 'Other food',
    'FT8': 'Tobacco'
}

reshaped_df = df.pivot(index='Year', columns='Feature', values='Value')

result_file_name = "solution.csv"
if os.path.exists("%s" % result_file_name):
    os.remove(result_file_name)

reshaped_df = rename_columns(reshaped_df, pivot_df_food_trade_columns_revert)
reshaped_df.insert(0, 'Country', country_name)
reshaped_df.insert(1, 'ValueType', "Predicted")

pivot_df_food_trade = rename_columns(pivot_df_food_trade, pivot_df_food_trade_columns_revert)
pivot_df_food_trade.insert(0, 'Country', country_name)
pivot_df_food_trade.insert(1, 'ValueType', "Real")

combined_df = pd.concat([pivot_df_food_trade, reshaped_df], axis=0)
combined_df_result[country_name] = combined_df

print("#######################")
print("Combined Data")
print(combined_df_result)
all_data = pd.concat(combined_df_result.values(), axis=0)

# Write the result to a CSV file
all_data.to_csv("combined_data.csv")
