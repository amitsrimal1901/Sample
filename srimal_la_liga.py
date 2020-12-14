import pandas as pd
data= pd.read_csv("C:/Users/amit_srimal/Downloads/1-liga.csv", usecols=['Team 1', 'FT', 'Team 2'])
# splitting score of team 1/2 based on '-' separator
data[['Team 1 score','Team 2 score']]= data.FT.str.split("-", expand=True)
print(data.head(5))