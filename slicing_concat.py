import pandas as pd # type: ignore

# Initializing the nested list with Data set
player_list = [['M.S.Dhoni', 36, 75, 5428000],
			['A.B.D Villers', 38, 74, 3428000],
			['V.Kohli', 31, 70, 8428000],
			['S.Smith', 34, 80, 4428000],
			['C.Gayle', 40, 100, 4528000],
			['J.Root', 33, 72, 7028000],
			['K.Peterson', 42, 85, 2528000]]


# creating a pandas dataframe
df = pd.DataFrame(player_list, columns=['Name', 'Age', 'Weight', 'Salary'])
df # data frame before slicing
