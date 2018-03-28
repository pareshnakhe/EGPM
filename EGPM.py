import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



class EGPM:
    def __init__(self):
        self.data = pd.read_csv('train.csv')
        # Trim the data to make it numeric
        self.data = self.data.select_dtypes(include=[np.number]).interpolate()

        #scaling all columns so that vals are in [0,1]
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(self.data.values)
        self.data = pd.DataFrame(x_scaled)

        self.NUM_SAMPLES = self.data.shape[0]
        self.N = self.NUM_FEATURES = self.data.shape[1]
        # plt.hist(self.data.iloc[:, self.N - 1], color='blue')
        # plt.show()
        # exit(1)

        # eta: step size
        self.eta = 0.0009
        # U: max l1 norm of weight vectors
        self.U = 10.4
        self.wm = self.wp = 1.0 / (2 * self.N) * np.ones(self.N)


    def batch_algo(self):
        lr = LinearRegression()
        X = self.data.iloc[:, range(self.N - 1)]
        y_actual = self.data.iloc[:, self.N - 1]
        model = lr.fit(X, y_actual)

        y_pred = model.predict(X)
        print mean_squared_error(y_actual, y_pred)


    def predicted_price(self, house_feature):
        house_feature = house_feature.iloc[:self.N]
        return np.dot((self.wp - self.wm), house_feature)

    def update_hypothesis(self, pred_price, house_feature):
        #eal_price = 1
        real_price = house_feature.iloc[self.N - 1]
        temp = self.eta * (pred_price - real_price) * self.U

        mul_wp = [math.exp(-1.0 * temp * house_feature[i]) for i in range(self.N)]
        mul_wm = [1.0 / mul_wp[i] for i in range(self.N)]
        normalizer = np.multiply(mul_wp, self.wp) + np.multiply(mul_wm, self.wm)

        self.wp = np.array([self.wp[i] * mul_wp[i] for i in range(self.N)] * (self.U / normalizer))
        self.wm = np.array([self.wm[i] * mul_wm[i] for i in range(self.N)] * (self.U / normalizer))

    def algo(self):
        pred_price_list = list()
        for itr in range(self.NUM_SAMPLES):
            house_feature = self.data.iloc[itr]
            pred_price = self.predicted_price(house_feature)
            pred_price_list.append(pred_price)
            # print (pred_price - house_feature.iloc[self.N - 1]) ** 2
            self.update_hypothesis(pred_price, house_feature)

        print mean_squared_error(self.data.iloc[:, self.N - 1], pred_price_list)

temp = EGPM()
temp.algo()
temp.batch_algo()

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