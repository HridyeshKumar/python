'''Write a program to demonstrate array indexing such as slicing, integer array indexing 
and Boolean array indexing along with their basic operations in NumPy.'''
import numpy as np
a=np.arange(10,1,-2)
print("Sequential array with nagative step value:",a)
newarr=[a[3],a[1],a[2]]
print("Elements at these indices are:",newarr)
a=np.arange(20)
print("Array is:",a)
print("a[-8:17:1]=",a[-8:17:1])
print("a[10:]=",a[10:])