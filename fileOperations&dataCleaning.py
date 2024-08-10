#unique values & value counts
import numpy as np 
import pandas as pd 
obj=pd.Series(["c","a","d","a","a","b","b","c","c"])
uniques=obj.unique()
print(uniques)
print(obj.value_counts())