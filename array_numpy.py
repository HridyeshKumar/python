#Write a program to demonstrate arrays in numpy
import numpy as np
a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print("Entered array is:",a,"\nand its dimension is:",a.ndim)
print("\nEntered array is:",b,"\nand its dimension is:",b.ndim) 
print("\nEntered array is:",c,"\nand its dimension is:",c.ndim)
print("\nEntered array is:",d,"and its dimension is:",d.ndim)
