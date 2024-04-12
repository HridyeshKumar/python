list1 = [(1,2),(3,3),(1,1)]
list1.sort() 
print(list1)
list1.sort(reverse=True)
print(list1)
# Original list of strings
words = ["apple", "banana", "kiwi", "orange", "grape"]
words.sort()
print("Sorted in alphabetical order:",words)
# Sorting by length using the len() function as the key
words.sort(key=len)
# Displaying the sorted list
print("Sorted by Length:", words)
# Original list of tuples
people = [("Alice", 25), ("Bob", 30), ("Charlie", 22), ("David", 28)]
# Sorting by the second element of each tuple (age)
people.sort(key=lambda x: x[1])
# Displaying the sorted list
print("Sorted by Age in tuple:", people)
# Original list of dictionaries
students = [
	{"name": "Alice", "age": 25},
	{"name": "Bob", "age": 30},
	{"name": "Charlie", "age": 22},
	{"name": "David", "age": 28}]
# Sorting by the 'age' key in each dictionary
students.sort(key=lambda x: x["age"])
# Displaying the sorted list
print("Sorted by Age in dictionary:", students)