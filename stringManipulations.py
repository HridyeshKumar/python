#data aggregation & grouping operations, Visualisation using Matplotlib
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
left=pd.DataFrame({"key1":["foo","foo","bar"],"key2":["one","two","three"],"lval":pd.Series([1,2,3],dtype="Int64")})
right=pd.DataFrame({"key1":["foo","foo","bar","bar"],"key2":["one","one","one","two"],"rval":pd.Series([4,5,6,7],dtype="Int64")})
print(left)
print(right)
pd.merge(left,right,on=["key1","key2"],how="outer")
left1=pd.DataFrame({"key":["a","b","a","a","b","c"],"value":pd.Series(range(6),dtype="Int64")})
right1=pd.DataFrame({"group_val":[3.5,7]},index=["a","b"])
print(left1)
print(right1)
pd.merge(left1,right1,left_on="key",right_index=True)

#Concatenating along an axis
arr=np.arange(12).reshape((3,4))
print(arr)
np.concatenate([arr,arr],axis=1)
np.concatenate([arr,arr])
s1=pd.Series([0,1],index=["a","b"],dtype="Int64")
s2=pd.Series([2,3,4],index=["c","d","e"],dtype="Int64")
s3=pd.Series([5,6],index=["f","g"],dtype="Int64")
pd.concat([s1,s2,s3])
pd.concat([s1,s2,s3],axis="columns")
a=pd.Series([np.nan,2.5,0.0,3.5,4.5,np.nan],index=["f","e","d","c","b","a"])
b=pd.Series([0.,np.nan,2.,np.nan,np.nan,5.],index=["a","b","c","d","e","f"])
print(a)
print(b)
np.where(pd.isna(a),b,a)
a.combine_first(b)

#Plotting and Visualisation
import matplotlib.pyplot as plt
data=np.arange(10)
print(data)
plt.plot(data)

#Plots in Matplotlib reside within a figure object
fig=plt.figure()
ax1=fig.add_subplot(2,2,1)
ax1.hist(np.random.standard_normal(100),bins=20,color="black",alpha=0.6)
ax2=fig.add_subplot(2,2,2)
ax2.scatter(np.arange(30),np.arange(30)+3*np.random.standard_normal(30))
ax3=fig.add_subplot(2,2,3)
ax3.plot(np.random.standard_normal(50).cumsum(),color="black",linestyle="dashed")
ax4=fig.add_subplot(2,2,4)
fig,axes=plt.subplots(2,2,sharex=True,sharey=True)
for i in range(2):
   for j in range(2):
      axes[i,j].hist(np.random.standard_normal(500),bins=50,color="black",alpha=0.5)
fig.subplot_adjust(wspace=0,hspace=0)
fig=plt.figure()
ax=fig.add_subplot()
ax.plot(np.random.standard_normal(30).cumsum(),color="black",linestyle="dashed",marker="s")
fig=plt.figure()
fig,ax=plt.subplots()
ax.plot(np.random.standard_normal(1000).cumsum())
ticks=ax.set_xticks([0,250,500,750,1000])
labels=ax.set_xticklabels(["one","two","three","four","five"],rotation=30,fontsize=10)
ax.set_xlabel("Stages")
ax.set_title("Matplotlib Plot")
fig=plt.figure()
fig,ax=plt.subplots()
ax.plot(np.random.randn(1000).cumsum(),color="black",labels="one")
ax.plot(np.random.randn(1000).cumsum(),color="blue",linestyle="dashed",label="two")
ax.plot(np.random.randn(1000).cumsum(),color="red",linestyle="dotted",labels="three")
ax.legend()
fig=plt.figure()
fig,ax=plt.subplots(2,1)
data=pd.Series(np.random.uniform(size=16),index=list("abcdefghijklmnop"))
data.plot.bar(ax=axes[0],color="red",alpha=0.7)
data.plot.barh(ax=axes[1],color="purple",alpha=.5)
df=pd.DataFrame(np.random.uniform(size=(6,4)),index=["one","two","three","four","five","six"],columns=pd.Index(["A","B","C","D"],name="Genius"))
print(df)