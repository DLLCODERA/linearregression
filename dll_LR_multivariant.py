import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import seaborn as sns


def incoming_parameters():

    data = pd.read_csv("./text1", names=["area", "num_bedroom", "price"])
    x = data[["area", "num_bedroom"]].values
    y = data[["price"]].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    data_dir = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}
    tup = (data_dir, data)
    return tup


def train_model(x, y):

    regression = LinearRegression()
    regression.fit(x, y)
    y_prediction = regression.predict(x)
    m_s_e = mean_squared_error(y, y_prediction)
    r_2 = r2_score(y, y_prediction)
    coefficient = regression.coef_
    intercept = regression.intercept_

    print("coefficient:", end="")
    print(coefficient)
    print("intercept:", end="")
    print(intercept)
    print("")
    print("均方差：", end="")
    print(m_s_e)
    print("相关系数：", end="")
    print(r_2)
    print("")
    print(y_prediction)

    return y_prediction


def configure(data, data_all):

    features_train = data["x_train"]
    label_train = data["y_train"]

    features_test = data["x_test"]
    label_test = data["y_test"]

    print("=" * 60)
    print("parameters of train")
    pred_train = train_model(features_train, label_train)
    print("=" * 60)
    print("parameters of test")
    pred_test = train_model(features_test, label_test)
    print("=" * 60)

    visualization(features_train,
                  label_train,
                  features_test,
                  label_test,
                  data_all,
                  pred_train,
                  pred_test
                  )


def visualization(x_train, y_train, x_test, y_test, data, y_pred, y_pred_t):

    fig_1 = plt.figure(figsize=(10, 10), dpi=80)

    ax1 = fig_1.add_subplot(221, projection="3d")
    ax2 = fig_1.add_subplot(222, projection="3d")

    ax1.scatter(x_train[:, :1], x_train[:, 1:], y_train, color="r", marker="x", alpha=0.5)
    ax1.legend(labels=["Test data"], loc="lower right")
    ax1.set_title("train diagram")
    ax1.set_xlabel("area")
    ax1.set_ylabel("num_bedroom")
    ax1.set_zlabel("price")

    ax2.scatter(x_test[:, :1], x_test[:, 1:], y_test, color="g", marker="x", alpha=0.5)
    ax2.legend(labels=["Test data"], loc="lower right")
    ax2.set_title("test diagram")
    ax2.set_xlabel("area")
    ax2.set_ylabel("num_bedroom")
    ax2.set_zlabel("price")

    sns.pairplot(data, x_vars=["area", "num_bedroom"], y_vars=["price"], size=6, aspect=0.8, kind="reg")

    ax3 = fig_1.add_subplot(223)
    ax4 = fig_1.add_subplot(224)

    ax3.plot(range(len(y_pred)), y_pred, 'b', label="predict")
    ax4.plot(range(len(y_pred_t)), y_pred_t, 'r', label="test")
    ax3.set_title("predict the price,train")
    ax3.legend(loc="upper right")
    ax3.set_xlabel("number")
    ax3.set_ylabel("price")

    ax4.set_title("predict the price,test")
    ax4.legend(loc="upper right")
    ax4.set_xlabel("number")
    ax4.set_ylabel("price")

    plt.show()


if __name__ == "__main__":

    plt.rcParams["axes.unicode_minus"] = False
    date = incoming_parameters()[0]
    date_all = incoming_parameters()[1]
    configure(date, date_all)
