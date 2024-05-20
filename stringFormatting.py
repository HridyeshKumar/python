#String Formating
#named indexes:
txt1="Welcome to {fname} {lname}".format(fname="AI",lname="World !!!")
#numbered indexes:
txt2="Welcome to {0} {1}".format("AI","World !!!")
#empty placeholders
txt3="Welcome to {} {}".format("AI","World !!!")
txt4="Welcome to {b:10} {a}".format(a="AI",b="World !!!")
'''   ^ ---- use it for center
      < ---- use it for left align
      > ---- use it for right align'''
txt5="Welcome to {a:^10} {b}".format(a="AI",b="World !!!")
print(txt1)
print(txt2)
print(txt3)
print(txt4)
print(txt5)
