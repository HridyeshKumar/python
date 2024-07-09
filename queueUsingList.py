l=[]
while True:
	c=int(input('''
			1 Enqueue
			2 Dequeue
			3 Front Elements
			4 Rear Elements
			5 Display Elements
			6 Exit
			'''))
	if c==1:
		n=input("Enter The Value:")	
		l.append(n)	
		print(l)
	elif c==2:
		if len(l)==0:
			print("Empty Queue")
		else:
			del l[0]
			print(l)
	elif c==3:
		if len(l)==0:
			print("Empty Queue")
		else:
			print("Front Queue Value=>",l[0])
	elif c==4:
		if len(l)==0:
			print("Empty Queue")
		else:
			print("Front Queue Value=>",l[-1])
	elif c==5:
		print("Display Queue=>",l)
	elif c==6:
		break
	else:
		print("Invalid Operation")
