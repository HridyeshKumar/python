'''Write a program to compute summary statistics such as mean, median, mode, standard 
deviationand variance of the given different types of data.''' # type: ignore
import numpy as np
a=np.array([[1,23,78],[98,60,75],[79,25,48]])
print("Entered array:",a)
#Minimum function 
print("Minimum=",np.amin(a))
#Maximum Function
print("Maximum=",np.amax(a))
#Mean Function
print("Mean=",np.mean(a))
#Median Function
print("Median=",np.median(a))
#std Function
print("Standard Deviation=",np.std(a))
#var Function
print("Variance=",np.var(a))