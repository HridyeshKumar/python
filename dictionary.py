#Write a program to demonstrate working with dictionaries in python
#empty dictionary
my_dict={}
print("Empty dictionary is:",my_dict)
#dictonary with integer keys
my_dict={1:'apple',2:'ball'}
print("Dictionary with integer keys:",my_dict)
#dictionary with mixed keys
my_dict={'name':'rishi',1:[2,4,3]}
print("Dictionary with mixed keys",my_dict)
#using dict.fromkeys()
my_dict=dict.fromkeys("abcd",'alphabet')
print("Dictionary created by using dict.fromkeys method=",my_dict)
#using get method
my_dict={'name':'jack','age':25}
print(my_dict['name'])      #output jack
#changing and adding dictionary elements
my_dict['age']=18           #update value
my_dict['class']="B.Tech"   #updating value
print("After changing and adding the values,the new dictionary=",my_dict)
#using items()
print("Items in the dictionary is:",my_dict.items())
#using keys()
print("Keys in the dictionary is:",my_dict.keys())
#using values()
print("Values in the dictionary is:",my_dict.values())