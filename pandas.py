#PANDAS
'''Pandas contains data structures and data manipulation tools designed to make data cleaning and analysis fast and convenient in Python.
Series and Dataframe
Series is a one-dimensional array like object containing a sequence of value.'''
import pandas as pd
obj=pd.Series([4,7,-5,3])
print(obj)
obj2=pd.Series([4,7,-5,3],index=["d","b","a","c"])
print(obj2)
print(obj2["a"])
obj2["d"]=6
print(obj2)
print(obj2[["c","a","d"]])
obj2=pd.Series([4,7,-5,3,5],index=["d","b","a","a","c"])
print(obj2)
print(obj2[obj2>0])
print(obj2*2)
import numpy as np
np.exp(obj2)
sdata={"Ohio":35000,"Texas}":71000,"Oregon":16000,"Utah":5000}
obj3=pd.Series(sdata)
print(obj3)
print(obj3.to_dict())
#DATA FRAME
'''A DataFrame represents a rectangular table of data and contains an ordered, named collection of columns each of which can be a different value type.
The DataFrame has both a row index and column index.'''
data={"states":["Ohio","Ohio","Ohio","Nevada","Nevada","Nevada"],"year":[2000,2001,2002,2001,2002,2003],"pop":[1.5,1.7,3.6,2.4,2.9,3.2]}
frame=pd.DataFrame(data)
print(frame)
print(frame.head())
print(frame.tail())
print(pd.DataFrame(data,columns=["year","states","pop"]))
frame2=pd.DataFrame(data,columns=["year","states","pop","debt"])
print(frame2)
print(frame2.columns)
print(frame2["states"])
print(frame2.year)
print(frame2[["states","year"]])
print(frame2.loc[1])
print(frame2.iloc[2])
frame2["debt"]=16.5
print(frame2)
frame2["debt"]=np.arange(6.)
print(frame2)
frame2["eastern"]=frame2["states"]=="ohio"
print(frame2)
del frame2["eastern"]
print(frame2.columns)
frame2.index.name="year"
frame2.columns.name="state"
print(frame2)
data=pd.DataFrame(np.arange(16).reshape((4,4)),index=["Ohio","Colorado","Utah","New York"],columns=["one","two","three","four"])
print(data)
print(data["two"])
print(data[["three","one"]])
print(data[data["three"]>5])
print(data.loc["Colorado"])
print(data.loc[["Colorado","New York"],["two","three"]])
print(data.iloc[0:2,0:3])
print(data.loc[data.three>=2])