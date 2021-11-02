import pandas as pd

# from pylab import *
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

data_test = pd.read_csv("./ex1data1.txt")
X = data_test.iloc[:, : 1].values
Y = data_test.iloc[:, 1:].values

# 划分训练和测试用的数据比重
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/4, random_state=0)  # 随机数种子，以后每次运行结果都相同

regressor = LinearRegression()

"""
LinearRegression(fit_intercept, normalize=False, copy_X=True, n_jobs=None)
parameters:
fit_intercept: 是否计算截距
normalize: 是否规范化
copy_X: 是否复制X
Attributes:  
coef_: 系数
intercept: 截距
Methods:
fie(X, y, sample_weight=None): 拟合 // 样本权重
get_params(deep=True): 得到参数，如果deep为TRUE，则得到estimate
set_params(**params): 设置参数
predict(x): 预测
score(X, y, sample_weight=None): 预测的准确度，X,测试样本；y，X的真实结果
eight： 权重

"""

rg = regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)


# plt.scatter(X_train, Y_train, color="r", marker="x", alpha=0.48)
# plt.plot(X_test, regressor.predict(X_test), color="g")
# plt.legend(labels=["Linear line", "Test data"], loc="lower right")
# plt.xlabel("The size of restaurant")
# plt.ylabel("profit per year")
# plt.figure()
# plt.scatter(X_test, Y_test, color="g", marker="x", alpha=0.48)
# plt.plot(X_test, regressor.predict(X_test), color="g")
# plt.legend(labels=["Linear line", "Test data"], loc="lower right")
# plt.xlabel("The size of restaurant")
# plt.ylabel("profit per year")
# plt.show()

m_s_e = mean_squared_error(Y_test, Y_pred)
print("均方差为：%.2f" % m_s_e)
r_2 = r2_score(Y_test, Y_pred)
print("决定系数为：%.2f" % r_2)


print("")
print("相关系数")
print("训练后结果")
print("gradient: %.2f" % float(regressor.coef_))
print("intercept: %.2f" % float(regressor.intercept_))


def visaulizemode(x1, y1, x2, y2):
    fig = plt.figure(figsize=(12, 6), dpi=80)
    fig1 = plt.figure(figsize=(8, 8), dpi=100)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax3 = fig1.add_subplot(111)

    ax1.scatter(x1, y1, color="r", marker="x", alpha=0.5)
    ax1.plot(x1, regressor.predict(x1), color="g")
    ax1.legend(labels=["Linear line", "Test data"], loc="lower right")
    ax1.set_title("OLS")
    ax1.set_xlabel("The size of restaurant")
    ax1.set_ylabel("profit per year")

    ax2.scatter(x2, y2, color="b", marker="x", alpha=0.5)
    ax2.plot(x1, regressor.predict(x1), color="g")
    ax2.legend(labels=["Linear line", "Test data"], loc="lower right")
    ax2.set_title("OLS")
    ax2.set_xlabel("The size of restaurant")
    ax2.set_ylabel("profit per year")

    ax3.scatter(regressor.predict(x2), regressor.predict(x2) - y2, color="g", marker="x")
    ax3.set_xlabel("predicted values")
    ax3.set_ylabel("residuals")
    ax3.hlines(y=0, xmin=0, xmax=50, colors="r")
    ax3.legend(labels=["residuals", "zero"], loc="lower right")

    plt.show()


if __name__ == "__main__":
    visaulizemode(X_test, Y_test, X_train, Y_train)

