#"python" or 'python 'both are same
#tuple is a fixed length,immutable sequence of python objects which,once assigned ,cannot be changed.
tup_a=(4,5,6)
print(tup_a)
print(type(tup_a))
tup=tuple('string') 
print(tup)
print(tup_a)
#tup_a[0]=20 cannot change value of tuple
print(tup_a[0])
print(tup[0:3]) #last index is excluded
print(tup[:]) #start:stop
print(tup[::2]) #start:stop:step
nested_tup=(4,5,6),(7,8)
print(nested_tup)
print(nested_tup[0])
print(nested_tup[0][1])
tuple=(4,None,'fool')+('bar',)#concatenate tuple
print(tuple)
print(('fool','bar')*4)#multiplying size of tuple
tup=(4,5,6)
a,b,c=tup
print(a,b)
print(a,b,c)
print(tup[::-1])
print(tup[-1])
a=(1,2,2,2,2,3,4,2)
print(a.count(2))
# lists are variable length and their contents can be modified in place.Lists are mutable.We can define them using square brackets[] or using list type function.
list1=[2,3,7,None]
print(list1)
print(type(list1))
gen=range(20)
print(gen)
print(list(range(20)))
# Adding and removing elements
# Elements can be appended to the end of the list with the append method
list2=['fool','peeka','bar']
list2.append('war')
print(list2)
list2.insert(1,'red')
print(list2)
print(list2.pop(2))
print(list2)
list2.remove('fool')
print(list2)
# concatenate
print([4,None,'fool']+[7,8,(2,3)])
x=[4,None,'fool']
x.extend([7,8,(2,3)])
print(x)
a=[7,2,5,1,3]
a.sort()
print(a)
a.sort(reverse=True)
print(a)
b=["saw","small","He","foxes","six"]
b.sort(key=len)
print(b)
seq=[7,2,3,7,5,6,0,1]
print(seq[1:5])
seq[3:5]=[6,3]
print(seq)
print(seq[:5])
print(seq[3:])
print(seq[-1])
print(seq[::2])
print(seq[::-1])
