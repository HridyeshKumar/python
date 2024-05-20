#Nested List Comprehension
data=[["John","Emily","Michael","Mary","Steven"],["Maria","Juan","Javier","Natalia","Pilar"]]
print(data)
#We want to get a single list containing all names with two or more a's in them.
interest=[]
for names in data:
	enough=[name for name in names if name.count("a")>=2]
	interest.extend(enough)
print(interest)
result=[name for names in data for name in names if name.count("a")>=2]
print(result)
#Indentation is colon mark which is 4 space or a tab
x=10
if x>5:
	print("X is greater than 5")
else:
	print("X is not greater than 5")
#FUNCTIONS
'''Functions are the primary and most important method of code organisation and reuse in python.
Functions are declared with the def keyword. A function contains  a block of code with an optimal use of the return keyword.'''
def fun(x,y):
 	return x+y
print(fun(10,20))
result=fun(20,30)
print(result)
def fun_with_return(x):
	print(x)
result=fun_with_return("hello")
print(result)
def fun_with_return():
	print("hello")
result=fun_with_return()
print(result)
#Positional Arguments
#Keyword Arguments
def fun(x,y,z=1.5):
	if z>1:
		return z*(x+y)
	else:
		return z/(x+y)
print(fun(5,6))
print(fun(5,6,z=0.7))
print(fun(x=10,y=20,z=30))
print(fun(5,6,0.7))
a=[]
def fun():
	for i in range(5):
		a.append(i)
print(fun())
print(a)
def fun():
	global a
	a=[]
	for i in range(5):
		a.append(i)
fun()
print(a)
def f():
	a=5
	b=6
	c=7
	return a,b,c
a,b,c=f()
print(a,b,c)
def f():
	a=5
	b=6
	c=7
	return {"a":a,"b":b,"c":c}
print(f())
states=["  Alabama  ","Georgia!","georgia","Georgia","Florida","south carolina##","West virginia?"]
import re   # regular expressions
def clean_strings(strings):
	result=[]
	for value in strings:
		value=value.strip()
		value=re.sub("[!#?]","",value)
		value=value.title()
		result.append(value)
	return result
print(clean_strings(states))
#Lambda Functions
'''Python has support for anonymous or lambda functions, which are a way of writing functions consisting of a single statement, result of which is the return value'''
def short_function(x):
	return x*2
print(short_function(20))
equiv=lambda x:x*2
print(equiv(20))
equiv=lambda x,y:x*y*2
print(equiv(20,40))
def apply_to_list(some_list,f):
	return [f(x) for x in some_list]
ints=[4,0,1,5,6]
print(apply_to_list(ints,lambda x:x*2))
strings=["foo(","card","bar","aaaa","abab"]
#sorting based on unique characters=>set(x)
strings.sort(key=lambda x:len(set(x)))
print(strings)
#ERRORS AND EXEPTION HANDLING
print(float("1.2345"))
#print(float("something"))#ValueError
def attempt_float(x):
	try:
		return float(x)
	except:
		return x
#The code in the except part of the block will only be executed if float(x) raises and exception
print(attempt_float("1.2345"))
print(attempt_float("something"))
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
a
