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
#ax.title("Changes in log