import numpy as np
import matplotlib.pyplot as plt


# 定义向量的内积
def multiVector(A, B):
    C = np.zeros(len(A))

    for i in range(len(A)):
        C[i] = A[i] * B[i]

        return sum(C)


def invector(A, b, a):
    D = np.zeros(b - a + 1)

    for i in range(b - a + 1):
        D[i] = A[i + a]
        return D[::-1]


# LMS算法的函数
def LMS(xn, dn, M, mu, itr):
    en = np.zeros(itr)
    W = [[0] * M for i in range(itr)]

    for k in range(itr)[M - 1:itr]:

        x = invector(xn, k, k-M+1)
        y = multiVector(W[k-1], x)
        en[k] = dn[k] - y
        W[k] = np.add(W[k-1], 2*mu*en[k]*x)

        # 求最优时的滤波器输出序列
        yn = np.inf * np.ones(len(xn))
        for k in range(len(xn))[M-1:len(xn)]:
            x = invector(xn, k, k-M+1)
            yn[k] = multiVector(W[len(W)-1], x)
            return yn, en


if __name__ == "__main__":
    # 参数初始化
    itr = 1000  # 采样点数
    M = 5  # 滤波器阶数
    mu = 3.034e-005  # 步长因子
    t = np.linspace(0, 99, itr)
    xs = 10 * np.sin(0.5 * t)
    xn1 = np.random.randn(itr)
    xn = np.add(xn1, xs)
    dn = xs  # 对于自适应滤波，用dn作为期望
    # 调用LMS
    (yn, en) = LMS(xn, dn, M, mu, itr)

    # 作图
    fig = plt.figure(figsize=(12, 10), dpi=80)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(224)

    ax1.plot(t, xn, label="$xn$")
    ax1.plot(t, dn, label="$dn$")
    ax1.set_xlabel("time")
    ax1.set_ylabel("volt")
    ax1.set_title("original signal xn and desired signal dn")

    ax2.plot(t, dn, label="$dn$")
    ax2.plot(t, yn, label="$yn$")
    ax2.set_xlabel("time")
    ax2.set_ylabel("volt")
    ax2.set_title("original signal xn and processing yn")

    ax3.plot(t, en, label="$en$")
    ax3.set_xlabel("time")
    ax3.set_ylabel("volt")
    ax3.set_title("error between processing signal yn and desired voltage dn")

    plt.show()
