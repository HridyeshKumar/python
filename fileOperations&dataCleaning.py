#unique values & value counts
import numpy as np 
import pandas as pd 
obj=pd.Series(["c","a","d","a","a","b","b","c","c"])
uniques=obj.unique()
print(uniques)
print(obj.value_counts())
#data loading
df=pd.read_csv("examples/ex1.csv")
df.head()
pd.read_csv("examples/ex2.csv")
pd.read_csv("examples/ex2.csv",header=None)
pd.read_csv("examples/ex2.csv",names=["a","b","c","d","message"])
names=["a","b","c","d","message"]
pd.read_csv("examples/ex2.csv",names=names,index_col="message")
result=pd.read_csv("examples/ex3.txt",sep="\s+")
print(result)
pd.read_csv("examples/ex4.csv",skiprows=[0,2,3])
result.to_csv("out.csv") #saving as a csv file
#data cleaning & preparation
#handling missing data
float_data=pd.Series([1.2,-3.5,np.nan,0])
print(float_data)
float_data.isna()
string_data=pd.Series(["aardvark",np.nan,None,"avacado"])
print(string_data&