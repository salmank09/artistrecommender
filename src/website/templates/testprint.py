import pandas as pd

df = pd.read_csv('items.csv')

df.to_excel('items.xlsx')
