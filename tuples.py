#Write a program to demonstrate working with tuples in python
#creating an empty tuple 
empty_tup=()
print("Empty tuple=",empty_tup)
#creating single element tuple
single_tup=(10,)
print("Single element tuple=",single_tup)
#creating a tuple with multiple elements
my_tup=(10,3.7,'program','a')
print("Tuple with multiple elements is:",my_tup)
print("Length of the tuple is:",len(my_tup))
T1=(10,20,30,40,70.5,33.3) 
print("Maximum value of the tuple T1 is:",max(T1))
print("Minimum value of the tuple T1 is:",min(T1))
str1='tuple'
T=tuple(str1)  #converting string into tuple
print("After converting a string into tuple,the new tuple is:",T)
L=[2,4,6,7,8]
T2=tuple(L)    #converting list into tuple
print("After converting a list into tuple,the new tuple is:",T2)