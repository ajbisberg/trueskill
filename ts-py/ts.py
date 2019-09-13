from models import Tournament # DynamicElo, 
from utils import TournamentData, initialize_rating, ts_win_probability

import multiprocessing
from time import perf_counter
import trueskill as ts
import pandas as pd
np = pd.np
data_path = '../../cwl-data/data/'
season = TournamentData(None,data_path,wwii=True)

tourneys = []
t0 = perf_counter()
for i,td in enumerate(season.tourneys):
    t = Tournament(td['data'])
    t.build_tournament()
    t.populate_next_series()
    tourneys.append(t)
t1 = perf_counter()
print('built tournaments in',round(t1-t0,1))

def validate(tourneys,mu,sigma,beta,tau,do_test=False):

    default = False
    if default:
        env = ts.setup(draw_probability=0)
    else:
        if beta is None:
            beta = sigma/2
        if tau is None:
            tau = sigma/100
    
    env = ts.setup(mu=mu,sigma=sigma,draw_probability=0,beta=beta,tau=tau)
    print('ts env is:',env)
    rating_i = initialize_rating(Tournament(TournamentData('CWL Champs 2017',data_path).data).raw_data,env)

    trains = []
    tr_len = 0
    v_perf = []
    t_perf = []

    for i in range(len(tourneys)-2):
    
        trains += [tourneys[i]]
        validate = tourneys[i+1]
        test = tourneys[i+2]
        for train in trains:
            train.set_first_series_rating(rating_i,env)
            train.play(env)
            rating_f = train.final_ratings()
            rating_i = rating_f #regressfunc(merge_elo(elo_i,elo_f))
            tr_len += len(train.tournament)
        
        validate.set_first_series_rating(rating_i,env)
        validate.play(env,acc=True)
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
            test.set_first_series_rating(rating_i,ts)
            test.play(ts,acc=True)
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
        # t3 = perf_counter()
        # print(t_out)
        return t_out.t_acc.mean()

    v_out = pd.DataFrame(columns=['n_tr_tourney','n_tr_series','n_v_series','v_acc','v_calib','v_log_loss'],data=v_perf)
    wt_avg_metrics = (
        round(np.sum(np.multiply(v_out.n_v_series,v_out.v_acc))/np.sum(v_out.n_v_series),3),
        round(np.sum(np.multiply(v_out.n_v_series,v_out.v_calib))/np.sum(v_out.n_v_series),3),
        round(np.sum(np.multiply(v_out.n_v_series,v_out.v_log_loss))/np.sum(v_out.n_v_series),3)
    )
    return (wt_avg_metrics, [mu,sigma,beta,tau])


# mus = [10,25,100,1500,5000,10000]
# sigmas = [2,8,20,100,500,1000]
# cv_args = []
# for mu in mus:
#     for sigma in sigmas: 
#         cv_args.append((tourneys,mu,sigma,None,None))

mu = 1500
sigma = 100
betas = [10,25,50,100,175,250]
taus = [0.01,0.1,1,10,50]
cv_args = []
for beta in betas:
    for tau in taus: 
        cv_args.append((tourneys,mu,sigma,beta,tau))

pool = multiprocessing.Pool(6)
cv_metrics = pool.starmap(validate, cv_args)
rows = []
for i,cv in enumerate(cv_metrics):
    m = cv[0]
    env = cv[1]
    rows.append([
        env[0],
        env[1],
        env[2],
        env[3],
        m[0],
        m[1],
        m[2]
    ])
cv_metric_frame = pd.DataFrame(columns=['mu','sigma','beta','tau','avg_acc','avg_calib','avg_log_loss'],data=rows)
cv_metric_frame.to_csv('cv_results_1500_100.csv',index=False)
print(cv_metric_frame)
