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
print(string_data)
string_data.isna()
float_data=pd.Series([1,2,None],dtype="float64")
print(float_data)
float_data.isna()
data=pd.Series([1,np.nan,3.5,np.nan,7])
data.dropna()
data[data.notna()]
data=pd.DataFrame([1.,6.5,3.],[1.,np.nan,np.nan],[np.nan,np.nan,np.nan],[np.nan,6.5,3.]])
print(data)
data.dropna()
data.dropna(how="all")
data[4]=np.nan
print(data)
data.dropna(axis="columns",how="all")
df=pd.DataFrame(np.random.standard_normal((7,3)))
df.iloc[:4,1]=np.nan
df.iloc[:2,2]=np.nan
print(df)
df.dropna()
df.dropna(thresh=2)

#filling in missing data
print(df)
df.fillna(0)
df.fillna({1:0.5,2:0.9})
df=pd.DataFrame(np.random.standard_normal((6,3)))
df.iloc[2:,1]=np.nan
df.iloc[4:,2]=np.nan
print(df)
df.fillna(method="ffill")
df.fillna(method="ffill",limit=2)
data=pd.DataFrame({"k1":["one","two"]*3+["two"],"k2":[1,1,2,3,3,4,4]})
print(data)
data.duplicated()
data.drop_duplicates()
data['v1']=range(7)
print(data)
data.drop_duplicates(subset=["k1"])
data.drop_duplicates(["k1","k2"],keep="last")
data=pd.Series([1.,-999.,2,-999.,-1000.,3.])
print(data)
data.replace(-999,np.nan)
data.replace([-999,1000],np.nan)
data.replace([-999,1000],[np.nan,0])
data.replace({-999:np.nan,-1000:0})
data=pd.DataFrame(np.arange(12).reshape((3,4)),index=["Ohio","Colorado","New York"],columnns=["one","two","three","four"])
print(data)
def transform(x):
   return x[:4].upper()
data.index=data.index.map(transform)
print(data)
data.rename(index=str.title,columns=str.upper)
data.rename(index={"OHIO":"INDIANA"},columns={"three":"peekaboo"})
