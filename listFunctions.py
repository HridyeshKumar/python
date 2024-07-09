l=[20,30,50,60]
print(l)
#del()=>delete element through index
del l[1]
print(l)
#pop()=>delete element through index and returns the value of deleted element
print(l.pop(2))
print(l)
#remove()=>delete element through value rather than indexing
l.remove(20)
print(l)
#clear()=>it returns a empty list
l.clear()
print(l)
l=[20,30,40,50]
l[0]=90
print(l)
l.insert(0,10)
print(l)
l.append(70)
print(l)
n=[60,80]
l.append(n)
print(l)
l.extend(n)
print(l)
l=[10,20,20,10,30,40,10,50]
a=l.count(10)
print(l)
print(a)
m=max(l)
print(m)
l1=["Hello","World"]
k=max(l1)
print(k)
m=min(l)
print(m)
k=min(l1)
print(k)
l.sort()
print(l)
l.reverse()
print(l)
l1.reverse()
print(l1)
a=l1.index("World")
print(a)
