import random
l=["rock","scissor","paper"]
'''
rock vs paper => paper wins
rock vs scissor => rock wins
paper vs scissor => scissor wins

'''
while True:
	ocount=0
	ucount=0
	uc=int(input('''
Game Start.....
1 Yes
2 No | Exit
				'''))
	if uc==1:
		for a in range(1,6):
			userInput= int(input('''
1 Rock
2 Scissor
3 Paper
'''			))
			if userInput==1:
				uchoice="rock"
			elif userInput==2:
				uchoice="scissor"
			elif userInput==3:
				uchoice="paper"
			Ochoice=random.choice(l)
			if Ochoice==uchoice:
				print("Opponent choice",Ochoice)
				print("User choice",uchoice)
				print("Game Draw")
				ucount=ucount+1
				ocount=ocount+1
			elif(uchoice=="rock" and Ochoice=="scissor") or (uchoice=="paper" and Ochoice=="rock") or (uchoice=="scissor" and Ochoice=="paper"):
				print("Opponent Choice",Ochoice)
				print("User choice",uchoice)
				print("You Win")
				ucount=ucount+1
			else:
				print("Opponent Choice",Ochoice)
				print("User choice",uchoice)
				print("Opponent Win")
				ocount=ocount+1
		if ucount==ocount:
			print("Final Game Draw....." )
			print("User Score",ucount )
			print("Opponent Score",ocount )
		elif ucount>ocount:
			print("Final You Win The Game....." )
			print("User Score",ucount )
			print("Opponent Score",ocount )
		else:
			print("Final Opponent Win The Game....." )
			print("User Score",ucount )
			print("Opponent Score",ocount )
	else:
		break
