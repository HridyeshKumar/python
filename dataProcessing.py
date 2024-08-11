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
