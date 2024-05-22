import json
d={
		'course_name':'Python',
		'fees':15000
}
f=json.dumps(d)
print(type(f))#string
print(f)
d='{"cname":"Python","fees":12000,"duration":"2 months"}'
x=json.loads(d)
print(type(x))#dictionary
print(x)
for a in x:
	print(a,x[a])
#How to read and write JSON file in python
import json 
file=open("posts.json","r")
x=file.read()
finaldata=json.loads(x)
for a in finaldata:
	print(a)
	print(a['title'],a['userId'])
