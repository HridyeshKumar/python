#conditional statements
x=10
if x>=10:
	print("YES")
else:
	print("NO")
#Loops
list_a=[10,20,30,40,50]
for i in list_a:
	print(i)
x=5
while x>=0:
	print(x)
	x=x-1
#Dictionary
'''A dictionary stores a collection of key-value pairs, where key and value are python objects.
Each key is associated with a value so that a value can be conveniently retrieved, inserted modified or deleted given a particular key.
One approach for creating a dictionary is to use curly braces{}'''
dict={}
print(dict)
d1={"a":"some value","b":[1,2,3,4]}
print(d1)
print(type(d1))
d1[7]="an integer"
print(d1)
d1[5]="some value"
print(d1)
d1["dummy"]="another value"
print(d1)
del d1[5]
print(d1)
print(d1.pop("dummy"))
print(d1)
print(d1.keys())
print(d1.values())
# If we need to iterate over both the keys and values, we can use the items method over the keys and values as 2-tuples
print(list(d1.items()))
d1.update({"b":"fool","c":12})
print(d1)
#Categorize a list of words by their letter as a dictionary of lists
words=["apple","bat","bar","atom","book","cook"]
by_letter={}
for word in words:
	#print(word)
	letter=word[0]  #letter=a
	if letter not in by_letter:
		by_letter[letter]=[word] #by_letter["a"]=["apple"] by_letter["b"]=["bat"]
	else:
		by_letter[letter].append(word) #by_letter["b"].append("bar")
print(by_letter)
# Set
'''A set is an unordered collection of unique elements. A set can be created in two ways via the set function or via the set literal with curly braces{}'''
print(set([2,2,2,3,4,3,4,1,2,5]))
a={1,2,3,4,5}
b={3,4,5,6,7,8}
print(a.union(b))
print(a|b)
print(a.intersection(b))
print(a&b)
# List,Set and Dictionary comprehension
'''List Comprehension are a convenient and widely used python language feature .
It allows us to concisely form a new list by filtering the elements of a collection, transforming the elements passing the filter into one concise expression.
Filter out string with length greater then 2 and convert them to upper case.'''
strings=["a","as","bat","car","dove","python"]
result=[]
for i in strings:
	if len(i)>2:
		result.append(i.upper())
print(result)
print([x.upper() for x in strings if len(x)>2])