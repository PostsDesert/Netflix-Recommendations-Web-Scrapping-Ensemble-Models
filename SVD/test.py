# pivot table without crashing notebook
import pandas as pd

df = pd.read_csv('../checkpoints/netflix_post_clean.csv')

df_p = pd.pivot_table(df,values='Rating',index='Cust_Id',columns='Movie_Id')

df_p.to_csv('../checkpoints/netflix_pivot.csv')