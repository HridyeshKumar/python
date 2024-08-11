#data aggregation &#String Manipulations
val="a,b,,guido"
val.split(",")
pieces=[x.strip() for x in val.split(",")
print(pieces)
first,second,third=pieces
first+"::"+second+"::"+third
"::".join(pieces)

#Data Wrangling
data=pd.Series(np.random.uniform(size=9),index=[["a","a","a","b","b","c","c","d","d"],[1,2,3,1,3,1,2,2,3]])
print(data)
data.index
data['b']
data['b'][3]
data["b":"c"]
data.loc[["b","d"]]
data.unstack()
frame=pd.DataFrame(np.arange(12).reshape((4,3)),index=[["a","a","b","b"],["Green","Red","Green"]])
print(frame)
frame.index.names=["key1","key2"]
frame.columns.names=["state","color"]
frame.index.nlevels

#combining and merging datasets
'''pandas.merge->
Connect rows in DataFrames based on one or more keys
pandas.concat->
Concatenate or stack objects together along an axis
combine_first->
Splice together overlapping data to fill in miing values in one object with values from another'''
df1=pd.DataFrame({"key":["b","b","a","c","a","a","b"],"data1":pd.Series(range(7),dtype="Int64")})
df2=pd.DataFrame({"key":["a","b","d"],"data2":pd.Series(range(3),dtype="Int64")})
print(df1)
print(df2)
pd.merge(df1,df2)
df3=pd.DataFrame({"lkey":["b","b","a","c","a","a","b"],"data1":pd.Series(range(7),dtype="Int64")})
df4=pd.DataFrame({"rkey":["a","b","d"],"data2":pd.Series(range(3),dtype="Int64")})
pd.merge(df3,df4,left_on="lkey",right_on="rkey")
pd.merge(df1,df2,how="outer")
pd.merge(df3,df4,left_on="lkey",right_on="rkey",how="outer")
df1=pd.DataFrame({"key":["b","b","a","c","a","b"],"data1":pd.Series(range(6),dtype="Int64")})
df2=pd.DataFrame({"key":["a","b","a","b","d"],"data2":pd.Series(range(5),dtype="Int64")})
print(df1)
print(df2)
pd.merge(df1,df2,on="key",how="left")
pd.merge(df1,df2,how="inner")
left=pd.DataFrame({"key1":["foo","foo","bar"],"key2":["one","two","three"],"lval":pd.Series([1,2,3),dtype="Int64")})
right=pd.DataFrame({"key1":["foo","foo","bar","bar"],"key2":["one","two","three"],"lval":pd.Series([1,2,3),dtype="Int64")})
print(df1)
print(df2)