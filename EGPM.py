import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures



class EGPM:
    def __init__(self):
        self.data = pd.read_csv('train.csv')
        # Trim the data to make it numeric
        self.data = self.data.select_dtypes(include=[np.number]).interpolate()

        # eta: step size
        self.eta = 0.001
        # U: max l1 norm of weight vectors
        self.U = 0.5

        #scaling all columns so that vals are in [0,1]
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(self.data.values)
        self.data = pd.DataFrame(x_scaled)

        self.NUM_SAMPLES = self.data.shape[0]
        self.NUM_FEATURES = self.data.shape[1]

        # To be set in either initialize() or poly_initialize()
        self.N = self.wm = self.wp = 0


    def initialize(self):
        self.N = self.data.shape[1]
        # plt.hist(self.data.iloc[:, self.N - 1], color='blue')
        self.wm = self.wp = 1.0 / (2 * (self.N - 1)) * np.ones(self.N - 1)


    def poly_initialize(self):
        X = self.data.iloc[:, range(self.NUM_FEATURES - 1)]
        poly = PolynomialFeatures(degree=2)
        X = poly.fit_transform(X)

        self.N = X.shape[1]
        #print self.N
        self.wm = self.wp = 1.0 / (2 * (self.N - 1)) * np.ones(self.N - 1)


    # fit linear model to batch data
    def batch_algo(self):
        lr = LinearRegression()
        X = self.data.iloc[:, range(self.N - 1)]
        y_actual = self.data.iloc[:, self.N - 1]
        model = lr.fit(X, y_actual)
        print lr.coef_

        y_pred = model.predict(X)
        batch_error = mean_squared_error(y_actual, y_pred)
        return batch_error

    def predicted_price(self, house_feature):
        house_feature = house_feature.iloc[:self.N - 1]
        return np.dot((self.wp - self.wm), house_feature)

    # wt updates: Same as in Kivinen and Warmuth
    def update_hypothesis(self, pred_price, house_feature):
        #eal_price = 1
        real_price = house_feature.iloc[self.N - 1]
        temp = self.eta * (pred_price - real_price) * self.U

        mul_wp = [math.exp(-1.0 * temp * house_feature[i]) for i in range(self.N - 1)]
        mul_wm = [1.0 / mul_wp[i] for i in range(self.N - 1)]
        normalizer = np.multiply(mul_wp, self.wp) + np.multiply(mul_wm, self.wm)

        self.wp = np.array([self.wp[i] * mul_wp[i] for i in range(self.N - 1)] * (self.U / normalizer))
        self.wm = np.array([self.wm[i] * mul_wm[i] for i in range(self.N - 1)] * (self.U / normalizer))

    # the actual algorithm
    def algo(self):
        self.initialize()

        # mean_squared_error at each round
        mse_list = list()
        # list of prices predicted
        pred_price_list = list()
        for itr in range(self.NUM_SAMPLES):
            house_feature = self.data.iloc[itr]
            pred_price = self.predicted_price(house_feature)
            pred_price_list.append(pred_price)
            # print (pred_price - house_feature.iloc[self.N - 1]) ** 2
            self.update_hypothesis(pred_price, house_feature)

        # return mean_squared_error(self.data.iloc[:, self.N - 1], pred_price_list)
        print (self.wp - self.wm)* self.U
        return pred_price_list


# How does the mse depend on eta
def eta_effect():
    temp = EGPM()

    # mean-squared-error list
    mse = list()

    for eta in np.linspace(pow(10,-2), 5*pow(10,-1), 10):
        temp.eta = eta
        pred_price_list = temp.algo()
        mse.append(mean_squared_error(temp.data.iloc[:, temp.N - 1], pred_price_list))

    batch_error = temp.batch_algo()
    be_list = batch_error * np.ones(len(mse))
    plt.plot(mse)
    plt.plot(be_list)
    plt.show()


temp = EGPM()
temp.poly_initialize()
# temp.algo()
# temp.batch_algo()

# data = pd.read_csv('train.csv')
# #
# house1 = data.iloc[:2]
# print house1.iloc[:,80]
# print house1.iloc[:, house1.shape[1]-1]
# print data.shape[0]


# df = pd.read_csv('train.csv')
# df = df.select_dtypes(include=[np.number]).interpolate()
#
# x = df.values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# df = pd.DataFrame(x_scaled)
# print df.iloc[:2]