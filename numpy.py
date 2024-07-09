#NUMPY
'''NumPy, short for Numerical Python, is one of the most important fundamental packages for numerical computing in python'''
import numpy as np
arr=np.arange(1_000_000)
print(arr)
list=list(range(1_000_000))
print(list[1:10])
%timeit arr2=arr*2
%timeit list2=[x*2 for x in list]
'''One of the key features of NumPy is its N-dimensional array object or n-D array, which is fast, flexible container for large datasets in python.'''
data=np.array([[1.5,0.1,3],[0,-3,6.5]])
print(data)
print(data*10)
print(data+data)
print(data.shape)
print(data.dtype)
print(data.ndim)
data1=[6,7.5,8.0,1]
arr1=np.array(data1)
print(arr1)
print(arr1.ndim)
print(np.zeros(10))
print(np.zeros((3,6)))
print(np.ones((3,6)))
print(np.arange(15))
arr1=np.array([1,2,3],dtype=np.float64)
arr2=np.array([1,2,3],dtype=np.int32)
print(arr1.dtype)
print(arr2.dtype)
arr=np.array([1,2,3,4,5])
print(arr.dtype)
float_arr=arr.astype(np.float64)
print(float_arr.dtype)
print(float_arr)
arr=np.array([1.,2.,3.],[4.,5.,6.])
print(arr)
print(arr*arr)
print(arr-arr)
print(1/arr)
print(arr**2)
arr2=np.array([0.,4.,1.],[7.,2.,12.])
print(arr2)
print(arr2>arr)
arr=np.arange(10)
print(arr)
print(arr[5])
print(arr[5:8])
arr[5:8]=12
print(arr)
arr_slice=arr[5:8]
print(arr_slice)
arr_slice[1]=12345
print(arr_slice)
arr2d=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr2d)
print(arr2d[2])
print(arr2d[2][1])
print(arr2d[:2])
print(arr2d[:2,1:])
arr=np.arange(15).reshape((3,5))
print(arr)
arr=np.arange(15).reshape((5,3))
print(arr)
print(arr.T)#Transpose
arr=np.array([[0,1,0],[1,2,-2],[6,3,2],[-1,0,-1],[1,0,1]])
print(arr)
print(np.dot(arr.T,arr))#Matrix Multiplication
print(arr.T@arr)#Matrix Multiplication