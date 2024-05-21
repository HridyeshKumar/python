d={
		'name':'python',
		'fees':8000,
		'duration':'2 months'
}
print(d)      
print(type(d))
print(d['fees'])
for n in d:
	print(n)
	print(d[n])
print(d.get('name'))
for a in d.keys():
	print(a)
for a in d.values():
	print(a)
for a,b in d.items():
	print(a,b)
del d['fees']
print(d)
print(d.pop('duration'))
print(d)
d=dict(name='python',fees=8000)
print(d)
d.update({'fees':10000})
print(d)
print(d.clear())
print(d)
d['desc']="This is Python"
print(d)
course={
		'php':{'duration':'3 months','fees':15000},
		'java':{'duration':'2 months','fees':10000},
		'python':{'duration':'1 months','fees':12000},
}
print(course)
print(course['php'])
print(course['php']['fees'])
for k,v in course.items():
	print(k,v)
for k,v in course.items():
	print(k,v['duration'],v['fees'])
course['java']['fees']=20000
print(course)
