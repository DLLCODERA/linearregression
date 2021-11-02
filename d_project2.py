# pandas库相关，用于读取csv文件
import pandas as pd
# statsmodels库相关
# 用于定义线性回归中一个被称为“切比雪夫准则（也称“最小一乘法”）”的损失函数
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

# matpoltlib库相关
# 用于绘制图表
from pylab import *
import matplotlib.pyplot as plt

# sklearn库相关
# 用于定义线性回归中一个被称为“最小二乘准则（也称“最小二乘法”）”的损失函数
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import linear_model


def generate_data():
    np.random.seed(4889)
    x = np.array([10] + list(range(10, 29)))
    error = np.round(np.random.randn(20), 2)
    y = x + error
    x = np.append(x, 29)
    y = np.append(y, 20)
    return pd.DataFrame({"x": x, "y": y})


# OLS回归模型具体实现
# 定义损失函数“最小二乘准则（也称“最小二乘法”）”
def train_OLS(x, y):
    # 调用sklearn中的linear_model中的现成的最小二乘法模型，名为LinearRegression
    model = linear_model.LinearRegression()
    # 进行模型拟合操作
    model.fit(x, y)
    # 打印模型计算出的参数
    # intercept_是参数b（第一张图片中标题为OLS的图表里那个斜线的的截
    print(model.intercept_)
    # coef_是参数a（第一张图片中标题为OLS的图表里那个斜线的的斜率值）
    print(model.coef_)
    # 返回预测值，存放在变量re
    re = model.predict(x)
    # 打印均方差MSE
    print(mean_squared_error(y, re))
    # 打印R2值
    print(r2_score(y, re))
    # 将re作为本函数trainOLS的返回值
    return re

    # LAD回归模型具体实现
    # 定义损失函数“切比雪夫准则（也称“最小一乘法”）”


def train_LAD(x, y):
    # 加入全1列作为扰动项系数
    X = sm.add_constant(x)
    # 调用statsmodels的现成的“分位数回归模型”
    model = QuantReg(y, X)
    # 用分位数回归做替代，进行拟合。可行的原因如下两行
    # 分位数回归参数q为0.5时，等同于“切比雪夫准则（也称“最小一乘法”）”
    # 因为切比雪夫准则是分位数回归的一种特殊情况
    model = model.fit(q=0.5)
    # 返回预测值，存放在变量re
    re = model.predict(X)
    # 将re作为本函数trainOLS的返回值
    return re


# 定义模型可视化函数
def visualizeModel(x, y, ols, lad):
    # 定义显示框的长宽figsize和分辨率dpi
    fig = plt.figure(figsize=(12, 6), dpi=80)
    # 定义了两个图标ax2和ax3以及其放置的位置
    # 121 一行二列第一个位置
    # 122 一行二列第二个位置
    ax2 = fig.add_subplot(121)
    ax3 = fig.add_subplot(122)



    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$y$")
    ax2.set_title('OLS')
    ax3.set_xlabel("$x$")
    ax3.set_ylabel("$y$")
    ax3.set_title('LAD')

    # 定义点的颜色是（b）lue蓝色，阿尔法值为0.4，图示显示为实验数据
    ax2.scatter(x, y, color='b', marker='d', alpha=0.4, label='实验数据')
    # 定义线的图示显示是预测数据
    ax2.plot(x, ols, color='b', linestyle='-.', label='预测数据')

    # 图表ax3样式同ax2
    ax3.scatter(x, y, color='y', marker='D', alpha=0.4, label='实验数据')
    ax3.plot(x, lad, color='y', linestyle=':', label='预测数据')

    # 显示阴影
    plt.legend(shadow=True)
    # 将结果显示到屏幕
    plt.show()


def OLS_vs_LAD(data):
    features = ["x"]
    label = ["y"]
    # 返回的ols是OLS回归的预测值
    ols = train_OLS(data[features], data[label])
    # 返回的lad是ALD回归的预测值
    lad = train_LAD(data[features], data[label])
    # 调用可视化函数将图标送至屏幕
    visualizeModel(data[features], data[label], ols, lad)


def residualPlots(data):
    # 传入参数
    features = ["x"]
    label = ["y"]
    ols = train_OLS(data[features], data[label])

    # 这里到倒数第二行为止，定义残差评估的可视化图的样式
    fig1 = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig1.add_subplot(111)
    # 定义训练数据的点的样式：蓝色，形状o样式，图示为训练数据
    ax.scatter(ols, ols - data[label], c='blue', marker='o', label='训练数据')
    # 定义和纵坐标的名称
    ax.set_xlabel('Predicted values')
    ax.set_ylabel('Residuals')
    # 图示规定显示在左上
    ax.legend(loc='upper left')
    # 绘制残差评估的参考线
    # 样式为红色，显示范围是x轴0到30的范围
    ax.hlines(y=0, xmin=0, xmax=30, colors='red')

    # 显示残差估计图
    plt.show()


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    data = generate_data()
    OLS_vs_LAD(data)
    residualPlots(data)

