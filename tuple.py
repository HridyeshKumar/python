t=(10,20,30,40,50)
print(type(t))
print(t)
n=t[3]
print(n)
l=len(t)
for a in range(l):
	print(t[a])
for a in t:
	print(a)
print(min(t))
print(max(t))
print(t.count(10))
print(t.index(50))
print(sum(t))
print(sum(t,10))
