#simple function
def showdata():
	print("WELCOME TO CSVTU")
showdata()
#function with arguments
def sum(a,b):
	print(a+b)
sum(10,20)
sum(40,20)
def sum(a,b=1):
	print(a+b)
sum(10)
sum(40,20)
#function with return type
def square(x):
	return x*x,x*2
s=square(5)
print(s)
