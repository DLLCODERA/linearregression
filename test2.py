import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("ex1data2.txt")
x = data.iloc[:, 0:2].values
print(x)
print(type(x))
data.boxplot()
fig_1 = plt.figure(figsize=(12, 6), dpi=100)


plt.show()