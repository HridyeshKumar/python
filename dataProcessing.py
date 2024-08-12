import pandas as pd 
tips=pd.read_csv("examples/tips.csv")
tips.head()

import seaborn as sns 
tips["tip_pct"]=tips["tip"]/(tips["total_bill"]-tips["tip"])
tips.head()

#Histograms & Density Plots
'''A Histogram is a kind of bar plot that gives a discretized display of continuous data.'''
tips["tip_pct"].plot.hist(bins=50)
tips["tip_pct"].plot.density()

#Scatter or Point Plots
import numpy as np
macro=pd.read_csv("examples/macrodata.csv")
data=macro[["cpi","mi","tbilrate","unemp"]]
trans_data=np.log(data).diff().dropna()
trans_data.tail()
ax=sns.regplot(x="m1",y="unemp",data=trans_data)
#ax.title("Changes in log(m1) versus log(unemp)")

#Pairplot
sns.pairplot(trans_data,diag_kind="kde",plot_kws={"alpha":0.5})
sns.catplot(x="day",y="tip_pct",hue="time",col="smoker",kind="bar",data=tips[tips.tip_pct<1])
sns.catplot(x="day",y="tip_pct",row="time",col="smoker",kind="bar",data=tips[tips.tip_pct<1])

#Box Plot explains 5 number theory
sns.catplot(x="tip_pct",y="day",kind="box",data=tips[tips.tip_pct<0.5])

#Data Aggregation and Group Operations
df=pd.DataFrame({"key1":["a","a",None,"b","b","a",None],"key2":pd.Series([1,2,1,2,1,None,1],dtype="Int64"),"data1":np.random.standard_normal(7),"data2":np.random.standard_normal(7)})
print(df)
grouped=df["data1"].groupby(df["key1"])
print(grouped)
grouped.mean()
means=df["data1"].groupby([df["key1"],df["key2"]]).mean()
print(means)
sums=df["data1"].groupby([df["key1"],df["key2"]]).sum()
sum.unstack()
states=np.array(["OH","CA","CA","OH","CA","OH"])
years=[2005,2005,2006,2005,2006,2005,2006]
df["data1"].groupby([states,years]).mean()
df.groupby("key1").mean()
df.groupby("key2").count()
df.groupby(["key1","key2"]).mean()
df.groupby("key1",dropna=False).size()
df.groupby(["key1","key2"],dropna=False).size()
df.groupby("key1").count()
for name,group in df.groupby("key1"):
   print(name)
   print(group)
df.groupby("key1")["data1"].sum()
df.groupby(["key1","key2"])["data2"].mean()
tips.head()
tips.groupby(["day","smoker"])["tip_pct"].mean()
tips.groupby(["day","smoker"])["tip_pct"].agg("mean")
tips.groupby(["day","smoker"])["tip_pct"].agg(["mean","std","count"])
tips.groupby(["day","smoker"])["tip_pct"].agg(["average","mean"),("stddev",np.std)])
functions=["count","mean","max"]
result=tips.groupby(["day","smoker"])[["tip_pct","total_bill"]].agg(functions)
print(functions)
