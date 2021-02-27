#%%
import pandas as pd
from pprint import pprint
pprint(pd.read_csv('./cv_results_mu_25.csv').to_latex(index=False))


#%%
pd.read_csv('./best_as_ts_vals.csv')['RMSE'].mean(), pd.read_csv('./best_as_ts_vals.csv')['RMSE'].std()

#%%
pd.read_csv('./best_as_scope_vals.csv')['RMSE'].mean(), pd.read_csv('./best_as_scope_vals.csv')['RMSE'].std()
