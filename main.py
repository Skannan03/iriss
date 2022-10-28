import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris=pd.read_csv("C:\\Iris.csv")
print(iris)
print(iris.info())
print(iris.isna().sum())
print(iris.describe())
print(iris['SepalLengthCm'].skew())
print(iris['PetalLengthCm'].skew())
print(iris.groupby(by='SepalLengthCm')['PetalLengthCm'].mean())
print(iris['PetalLengthCm'].value_counts().plot())
plt.show()
print(iris['PetalLengthCm'].value_counts())
print(iris.corr())
print(plt.boxplot(iris['PetalLengthCm']))
plt.show()
print(iris.boxplot(column=['SepalLengthCm']))
plt.show()
print(iris.plot.scatter(x=['SepalLengthCm'],y=['PetalWidthCm']))
plt.show()
from scipy.stats import chi2_contingency
def chi(var1,var2):
    tab=pd.crosstab(var1,var2)
    _,p,_,_=chi2_contingency(tab,correction=False)
    if p<0.05:
        print("reject null hypo and accept alternate hypo",round(p,2))
    else:
        print("failed to reject null hypo",round(p,2))
    print(tab)
chi(iris['SepalLengthCm'],iris['PetalWidthCm'])
chi(iris['SepalLengthCm'],iris['Id'])
iris['Id']=np.where(iris['Id'].isna(),iris['Id'].median(skipna=True),iris['Id'])
print(iris['Id'].isna().sum())
print(iris.info())