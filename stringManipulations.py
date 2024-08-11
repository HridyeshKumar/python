#String Manipulations
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
Splice together overlapping data to fill in miing values in one object with values
