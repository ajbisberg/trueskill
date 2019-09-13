#%%
import pandas as pd

#%%
res = pd.read_csv('./elo/cv_results_regress_k_mov.csv')
best = res[res['avg_acc']==res['avg_acc'].max()]
best = best.append(res[ (res['avg_calib']-1).abs()== (res['avg_calib']-1).abs().min()])
best = best.append(res[res['avg_log_loss']==res['avg_log_loss'].min()])
best

#%% 
res[res['avg_acc']==res['avg_acc'].min()]

#%%
res = pd.read_csv('./elo/cv_results_cut_red_w90.csv')
best = res[res['avg_acc']==res['avg_acc'].max()]
# best = best.append(res[ (res['avg_calib']-1).abs()== (res['avg_calib']-1).abs().min()])
best = best.append(res[res['avg_log_loss']==res['avg_log_loss'].min()])
best

#%%
