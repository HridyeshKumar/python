infile=input("Enter 1st file name:")
outfile=input("Enter 2nd file name:")
f1=open("firstfile.txt",'r')
f2=open("secondfile.txt",'w+')
content=f1.read()
f2.write(content)