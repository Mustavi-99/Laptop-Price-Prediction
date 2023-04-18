import pandas as pd
import numpy as np
from sklearn import metrics

from sklearn.model_selection import train_test_split

ds = pd.read_csv('laptop_prices.csv')

# for col in ds.columns:
#    print(col)

X = ds.drop(['Price'], axis=1)
Y = ds['Price']

# Convert Brand column into categorical values
brands = pd.get_dummies(X['Brand'])
X = X.drop(['Brand'], axis=1)
X = pd.concat([X, brands], axis=1)

# Convert Processor column into categorical values
processor = pd.get_dummies(X['Processor'])
X = X.drop(['Processor'], axis=1)
X = pd.concat([X, processor], axis=1)

# Convert Processor_Generation column into categorical values
processor_gen = pd.get_dummies(X['Processor_Generation'])
X = X.drop(['Processor_Generation'], axis=1)
X = pd.concat([X, processor_gen], axis=1)

# Convert Storage_Type column into categorical values
storage_type = pd.get_dummies(X['Storage_Type'])
X = X.drop(['Storage_Type'], axis=1)
X = pd.concat([X, storage_type], axis=1)

# Convert Graphics column into categorical values
graphics = pd.get_dummies(X['Graphics'])
X = X.drop(['Graphics'], axis=1)
X = pd.concat([X, graphics], axis=1)

# Convert Display_Type column into categorical values
display_type = pd.get_dummies(X['Display_Type'])
X = X.drop(['Display_Type'], axis=1)
X = pd.concat([X, display_type], axis=1)

Z = X.head(1)
# print(Z)

x_train, xtest, y_train, ytest = train_test_split(X, Y, test_size=.25, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

prediction = regressor.predict(xtest)
linear_error = metrics.mean_absolute_percentage_error(ytest, prediction)
# print(ytest)
# print(prediction)
print()
print('Performance for Multiple Linear Regression:')
print('-------------------------------------------')
print('Mean absolute percentage error: ', "{:.2f}".format(linear_error * 100), ' %')
print('Mean Absolute Error:', "{:.2f}".format(metrics.mean_absolute_error(ytest, prediction)))
print('Mean Squared Error:', "{:.2f}".format(metrics.mean_squared_error(ytest, prediction)))
print('Root Mean Squared Error:', "{:.2f}".format(np.sqrt(metrics.mean_squared_error(ytest, prediction))))
print('R2 Score:', "{:.2}".format(metrics.r2_score(ytest, prediction)))

from sklearn import tree

regressor = tree.DecisionTreeRegressor()
regressor.fit(x_train, y_train)

prediction = regressor.predict(xtest)
decision_tree_error = metrics.mean_absolute_percentage_error(ytest, prediction)
# print(ytest)
# print(prediction)
print()
print('Performance for Decision Tree Regression:')
print('-------------------------------------------')
print('Mean absolute percentage error: ', "{:.2f}".format(decision_tree_error * 100), ' %')
print('Mean Absolute Error:', "{:.2f}".format(metrics.mean_absolute_error(ytest, prediction)))
print('Mean Squared Error:', "{:.2f}".format(metrics.mean_squared_error(ytest, prediction)))
print('Root Mean Squared Error:', "{:.2f}".format(np.sqrt(metrics.mean_squared_error(ytest, prediction))))
print('R2 Score:', "{:.2}".format(metrics.r2_score(ytest, prediction)))

from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor(n_neighbors=15)
regressor.fit(x_train, y_train)

prediction = regressor.predict(xtest)
knn_error = metrics.mean_absolute_percentage_error(ytest, prediction)
# print(ytest)
# print(prediction)
print()
print('Performance for K Nearest Neighbor(KNN):')
print('-------------------------------------------')
print('Mean absolute percentage error: ', "{:.2f}".format(knn_error * 100), ' %')
print('Mean Absolute Error:', "{:.2f}".format(metrics.mean_absolute_error(ytest, prediction)))
print('Mean Squared Error:', "{:.2f}".format(metrics.mean_squared_error(ytest, prediction)))
print('Root Mean Squared Error:', "{:.2f}".format(np.sqrt(metrics.mean_squared_error(ytest, prediction))))
print('R2 Score:', "{:.2}".format(metrics.r2_score(ytest, prediction)))

from sklearn.naive_bayes import GaussianNB

regressor = GaussianNB()
regressor.fit(x_train, y_train)

prediction = regressor.predict(xtest)
gaussian_error = metrics.mean_absolute_percentage_error(ytest, prediction)
# print(ytest)
# print(prediction)
print()
print('Performance for Gaussian Naive Bayes:')
print('-------------------------------------------')
print('Mean absolute percentage error: ', "{:.2f}".format(gaussian_error * 100), ' %')
print('Mean Absolute Error:', "{:.2f}".format(metrics.mean_absolute_error(ytest, prediction)))
print('Mean Squared Error:', "{:.2f}".format(metrics.mean_squared_error(ytest, prediction)))
print('Root Mean Squared Error:', "{:.2f}".format(np.sqrt(metrics.mean_squared_error(ytest, prediction))))
print('R2 Score:', "{:.2}".format(metrics.r2_score(ytest, prediction)))

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=200)
regressor.fit(x_train, y_train)

prediction = regressor.predict(xtest)
random_forest_error = metrics.mean_absolute_percentage_error(ytest, prediction)
# print(ytest)
# print(prediction)
print()
print('Performance for Random Forest:')
print('-------------------------------------------')
print('Mean absolute percentage error: ', "{:.2f}".format(random_forest_error * 100), ' %')
print('Mean Absolute Error:', "{:.2f}".format(metrics.mean_absolute_error(ytest, prediction)))
print('Mean Squared Error:', "{:.2f}".format(metrics.mean_squared_error(ytest, prediction)))
print('Root Mean Squared Error:', "{:.2f}".format(np.sqrt(metrics.mean_squared_error(ytest, prediction))))
print('R2 Score:', "{:.2}".format(metrics.r2_score(ytest, prediction)))

# for col in x_train.columns:
#    print(col)


if random_forest_error <= decision_tree_error and random_forest_error <= gaussian_error and random_forest_error <= knn_error and random_forest_error <= linear_error:
    regressor = RandomForestRegressor()
else:
    if decision_tree_error <= gaussian_error and decision_tree_error <= knn_error and decision_tree_error <= linear_error:
        regressor = tree.DecisionTreeRegressor()
    else:
        if gaussian_error <= knn_error and gaussian_error <= linear_error:
            regressor = GaussianNB()
        else:
            if knn_error <= linear_error:
                regressor = KNeighborsRegressor()
            else:
                regressor = LinearRegression()

print(regressor)

print()
print("Brands:\n--------------\nAcer Apple Asus Dell HP Huawei Laptop Lenovo MSI Microsoft Realme")
input_brand = input("Enter Brand: ")
print()
print(
    "Processor:\n--------------\nAMD Ryzen 3, AMD Ryzen 5, AMD Ryzen 7, AMD Ryzen 9, Apple M1 Chip, Core i5 1035G1, Intel Celeron Dual Core, Intel Core M3, Intel Core i3, Intel Core i5, Intel Core i7, Intel Core i9, Intel Pentium Gold, Intel Pentium Silver")
input_processor = input("Enter Processor: ")
input_ram = int(input("Enter Ram(GB): "))
input_storage = int(input("Enter Storage(GB): "))
input_storageType = input("Enter Storage Type(SSD/HDD): ")
print()
print("Graphics:\n--------------\nAMD Radeon Graphics, AMD Radeon Vega 8 Graphics, Apple 14-Core GPU, Apple 14-core GPU, Apple 16-core GPU, Apple 24-Core GPU, Apple 32-Core GPU, Apple 32-core GPU, Apple 7-core GPU, Apple 8-core GPU, Inte UHD Graphics 600, Intel Iris Xe Graphics, Intel UHD Graphics, Intel UHD Graphics 600, Intel UHD Graphics 605, Intel UHD Graphics 615, Nvidia GTX 1650 Graphics, Nvidia GTX 1660 Graphics, Nvidia MX330 Graphics, Nvidia MX350 Graphics, Nvidia RTX 3050 Graphics, Nvidia RTX 3060 Graphics, Nvidia RTX 3070 Graphics")
input_graphics = input("Enter Graphics: ")
print()
input_displaySize = float(input("Enter Display Size: "))
print()
print("Display Type:\n--------------\n2K IPS LED Display, FHD IPS LED Display, FHD LED, FHD OLED Display, HD IPS LED Display, HD LED, Liquid Retina XDR Display, PixelSense Flow MultiTouch Display, PixelSense MultiTouch Display")
input_displayType = input("Enter Display Type: ")
input_warranty = int(input("Enter warranty: "))

Z.loc[0, input_brand] = 1
Z.loc[0, input_processor] = 1
Z.loc[0, 'RAM'] = input_ram
Z.loc[0, 'Storage (GB)'] = input_storage
Z.loc[0, input_storageType] = 1
Z.loc[0, input_graphics] = 1
Z.loc[0, 'Display_Size'] = input_displaySize
Z.loc[0, input_displayType] = 1
Z.loc[0, 'Warranty'] = input_warranty

regressor.fit(x_train, y_train)
predictedPrice = regressor.predict(Z)
print("Predicted Price: ", predictedPrice)
