import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def incoming_parameters():

    data = pd.read_csv("./ex1data1.txt")
    x = data.iloc[:, :1].values
    y = data.iloc[:, 1:].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    data_dir = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}

    return data_dir


def train_model(x, y):

    regression = LinearRegression()
    regression.fit(x, y)
    y_prediction = regression.predict(x)
    m_s_e = mean_squared_error(y, y_prediction)
    r_2 = r2_score(y, y_prediction)
    coefficient = regression.coef_
    intercept = regression.intercept_

    print("gradient: %.2f" % coefficient)
    print("intercept: %.2f" % intercept)
    print("")
    print("mean squared error: %.3f" % m_s_e)
    print("correlation: %.3f" % r_2)

    return y_prediction


def configure_1(data):

    features_train = data["x_train"]
    label_train = data["y_train"]

    features_test = data["x_test"]
    label_test = data["y_test"]

    print("parameters of train")
    pred_train = train_model(features_train, label_train)
    print("=" * 50)
    print("parameters of test")
    pred_test = train_model(features_test, label_test)

    visualization(features_train,
                  label_train,
                  pred_train,
                  features_test,
                  label_test,
                  pred_test)


def visualization(x_train, y_train, pred_train, x_test, y_test, pred_test):

    fig_1 = plt.figure(figsize=(12, 6), dpi=100)
    fig_2 = plt.figure(figsize=(8, 8), dpi=100)

    ax1 = fig_1.add_subplot(121)
    ax2 = fig_1.add_subplot(122)
    ax3 = fig_2.add_subplot(111)

    ax1.scatter(x_train, y_train, color="r", marker="x", alpha=0.5)
    ax1.plot(x_train, pred_train, color="g")
    ax1.legend(labels=["Linear line", "Test data"], loc="lower right")
    ax1.set_title("train diagram")
    ax1.set_xlabel("area and number of bedroom")
    ax1.set_ylabel("price")

    ax2.scatter(x_test, y_test, color="g", marker="x", alpha=0.5)
    ax2.plot(x_test, pred_test, color="b")
    ax2.legend(labels=["Linear line", "Test data"], loc="lower right")
    ax2.set_title("test diagram")
    ax2.set_xlabel("area and number of bedroom")
    ax2.set_ylabel("price")

    ax3.scatter(pred_train, pred_train - y_train, color="r", marker="x", alpha=0.5)
    ax3.hlines(y=0, xmin=0, xmax=60, colors="g")
    ax3.legend(labels=["residuals", "zero"], loc="lower right")
    ax3.set_title("residuals diagram")
    ax3.set_xlabel("predicted values")
    ax3.set_ylabel("residuals")

    plt.show()


if __name__ == "__main__":

    plt.rcParams["axes.unicode_minus"] = False
    date = incoming_parameters()
    configure_1(date)
