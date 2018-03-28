# https://www.dataquest.io/blog/kaggle-getting-started/
# https://www.dataquest.io/blog/data-science-portfolio-project/

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

train = pd.DataFrame()
train = pd.read_csv('OnlineNewsPopularity.csv')


plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

# print "Skew:", train.SalePrice.skew()
# plt.hist(train.iloc[:, 60], color='blue')
# plt.show()
# exit(1)
#
# target = np.log(train.iloc[:, 60])
# print "Skew:", target.skew()
# plt.hist(target, color='blue')
# plt.show()
# exit(1)
#

# target = np.log(train.iloc[:, 60])
# numeric_features = train.select_dtypes(include=np.number)
# print numeric_features.shape
# print numeric_features.dtypes

#corr = train.corr()
#print corr.iloc[:, 59].sort_values(ascending=False)[:5], corr.iloc[:, 59].sort_values(ascending=False)[-5:]


y = np.log(train.iloc[:, 60])
#print train.columns
X = train.drop(columns=['url', ' shares'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.02)
print len(X_train)
lr = LinearRegression()
model = lr.fit(X_train, y_train)

print model.score(X_test, y_test)


model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                  ('linear', LinearRegression())])
# fit to an order-3 polynomial data
print model.fit(X_train, y_train).score(X_test, y_test)


# svr_poly = SVR(kernel='poly', C=1e3, degree=2)
# score = svr_poly.fit(X_train, y_train).score()
# print "scoring", score

# clf = Ridge(alpha=5.0)
# model = clf.fit(X, y)
# print model.score(X_test, y_test)

#predictions = model.predict(X_test)
#print predictions
