import copy
import pandas as pd
np = pd.np
from time import perf_counter
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
import csv

from models import DynamicElo, Tournament
from utils import TournamentData, initialize_elo, merge_elo
from model_params import mov_none, mov_lin, mov_log, mov_sqrt, mov_exp

data_path = '../../cwl-data/data/'
season = TournamentData(None,data_path,wwii=True)

def regress_none(elo_i):
    return elo_i
def regress_10(elo_i):
    elo_i.elo = 0.9*elo_i.elo + 0.1*1500
    return elo_i
def regress_20(elo_i):
    elo_i.elo = 0.8*elo_i.elo + 0.2*1500
    return elo_i
def regress_30(elo_i):
    elo_i.elo = 0.7*elo_i.elo + 0.3*1500
    return elo_i

tourneys = []
t0 = perf_counter()
elo_i = initialize_elo(Tournament(TournamentData('CWL Champs 2017',data_path).data).raw_data)
for i,td in enumerate(season.tourneys):
    t = Tournament(td['data'])
    t.build_tournament()
    t.populate_next_series()
    tourneys.append(t)
t1 = perf_counter()

print('built tournaments in',round(t1-t0,1))


def validate(tourneys,elo_i,regressfunc,base_k,mov,cutoff,reduction,w90,do_test=False):
    trains = []
    tr_len = 0
    v_perf = []
    t_perf = []
    elo = DynamicElo(base_k,cutoff,reduction,mov,w90)

    for i in range(len(tourneys)-2):
    
        trains += [tourneys[i]]
        validate = tourneys[i+1]
        test = tourneys[i+2]
        for train in trains:
            train.set_first_series_elo(elo_i)
            train.play(elo)
            elo_f = train.final_elos()
            elo_i = regressfunc(merge_elo(elo_i,elo_f))
            tr_len += len(train.tournament)
        
        validate.set_first_series_elo(elo_i)
        validate.play(elo,acc=True)
        v_correct = [series['correct'] for series in validate.tournament]
        max_w = [series['max_wp'] for series in validate.tournament]
        v_perf.append(
            (i+1, 
            tr_len,
            len(validate.tournament),
            sum(v_correct)/len(v_correct), 
            sum(max_w)/sum(v_correct),
            np.mean(-np.log(np.trim_zeros(np.sort(np.multiply(v_correct,max_w)))))
        ))
        # elo_i = regressfunc(merge_elo(elo_i,validate.final_elos()))

        if do_test:
            test.set_first_series_elo(elo_i)
            test.play(elo,acc=True)
            t_correct = [series['correct'] for series in test.tournament]
            max_w = [series['max_wp'] for series in test.tournament]

            t_perf.append(
                (season.tourneys[i+2]['name'], 
                len(test.tournament),
                sum(t_correct)/len(t_correct), 
                # sum(max_w)/sum(t_correct),
                # np.mean(-np.log(np.trim_zeros(np.sort(np.multiply(t_correct,max_w)))))
            ))
    
    if do_test:
        t_out = pd.DataFrame(columns=['name','n_series','acc'],data=t_perf)
        t_out = t_out.append(pd.DataFrame(columns=['name','acc'],data=[['avg',t_out.t_acc.mean()]])).round(3)
        t3 = perf_counter()
        # print(t_out)
        return t_out.t_acc.mean()

    v_out = pd.DataFrame(columns=['n_tr_tourney','n_tr_series','n_v_series','v_acc','v_calib','v_log_loss'],data=v_perf)
    wt_avg_metrics = (
        round(np.sum(np.multiply(v_out.n_v_series,v_out.v_acc))/np.sum(v_out.n_v_series),3),
        round(np.sum(np.multiply(v_out.n_v_series,v_out.v_calib))/np.sum(v_out.n_v_series),3),
        round(np.sum(np.multiply(v_out.n_v_series,v_out.v_log_loss))/np.sum(v_out.n_v_series),3)
    )
    return  wt_avg_metrics

pool = multiprocessing.Pool(6)

# f_args = []
# ks = [1,5,10,20,30,40,50]
# rs = [regress_none,regress_10,regress_20,regress_30]
# movs = [mov_none, mov_lin, mov_log, mov_sqrt, mov_exp]
# for k in ks: 
#     for r_func in rs: 
#         for mov in movs:
#             f_args.append((tourneys,elo_i,r_func,k,mov,1600,1,200))
# # cv_metrics = validate(tourneys,elo_i,regress_none,40)
# cv_metrics = pool.starmap(validate, f_args)
# rows = []
# for i,m in enumerate(cv_metrics):
#     rows.append([
#         f_args[i][2].__name__,
#         f_args[i][3],
#         f_args[i][4].__name__,
#         m[0],
#         m[1],
#         m[2]
#     ])
# cv_metric_frame = pd.DataFrame(columns=['regress_func','base_k','mov_func','avg_acc','avg_calib','avg_log_loss'],data=rows)
# print(cv_metric_frame)

f_args = []
w90s = [100,200,300,400,500]
cutoffs = [1600,1650,1700,1750]
reduction = [0.1,0.25,0.5,0.75,0.9]

for w90 in w90s:
    for c in cutoffs:
        for r in reduction:
            f_args.append((tourneys,elo_i,regress_20,5,mov_exp,c,r,w90))
cv_metrics = pool.starmap(validate, f_args)
rows = []
for i,m in enumerate(cv_metrics):
    rows.append([
        f_args[i][5],
        f_args[i][6],
        f_args[i][7],
        m[0],
        m[1],
        m[2]
    ])
cv_metric_frame = pd.DataFrame(columns=['cutoff','reduction','w90','avg_acc','avg_calib','avg_log_loss'],data=rows)
cv_metric_frame.to_csv('./cv_results_cut_red_w90.csv',index=False)
print(cv_metric_frame)
import pdb; pdb.set_trace()