from models import Tournament  
from utils import TournamentData, initialize_rating, ts_win_probability

import multiprocessing
from time import perf_counter
import trueskill as ts
import pandas as pd
from pprint import pprint
np = pd.np
data_path = '../../cwl-data/data/'
season = TournamentData(None,data_path,wwii=True)

env = ts.setup(draw_probability=0)

tourneys = []
for i,td in enumerate(season.tourneys):
    td = Tournament(td['data'])
    td.build_for_players()
    td.populate_next_series_players()
    tourneys.append(td)

iw_final = Tournament(TournamentData('CWL Champs 2017',data_path).data)
iw_final.build_for_players()
iw_final.populate_next_series_players()

def validate(tourneys,mu=None,sigma=None,beta=None,tau=None,do_test=False,default=True):

    if default:
        env = ts.setup(draw_probability=0)
    else:
        if sigma is None:
            sigma = mu/3
        if beta is None:
            beta = sigma/2
        if tau is None:
            tau = sigma/100
        env = ts.setup(mu=mu,sigma=sigma,draw_probability=0,beta=beta,tau=tau)
    
    print('ts env is:',env)
    iw_final.set_initial_player_ratings(env)
    iw_final.play_with_players(env)
    rating_i = iw_final.players

    trains = []
    tr_len = 0
    v_perf = []
    t_perf = []

    for i in range(len(tourneys)-2):
    
        trains += [tourneys[i]]
        validate = tourneys[i+1]
        test = tourneys[i+2]
        for i,train in enumerate(trains):
            train.set_initial_player_ratings(env) if i == 0 else train.set_initial_player_ratings(env,rating_i)
            train.play_with_players(env)
            rating_i = train.players
            tr_len += len(train.tournament)
        
        validate.set_initial_player_ratings(env,rating_i)
        validate.play_with_players(env,use_best_player=False,acc=True)
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
        round(np.sum(np.multiply(v_out.n_v_series,v_out.v_log_loss))/np.sum(v_out.n_v_series),3),
        round(np.std(v_out.v_acc),3),
        round(np.std(v_out.v_calib),3),
        round(np.std(v_out.v_log_loss),3),
    )
    return (wt_avg_metrics, [env.mu,env.sigma,env.beta,env.tau])

mu = 110
sigma = mu/15
beta = sigma
tau = sigma/10

# print(validate(tourneys,mu,sigma,beta,tau,False,False))

mu = 1500
sigmas = [mu/3,mu/5,mu/10]
betas = [sigmas[0],sigmas[0]/3,sigmas[0]/5]
taus = [sigmas[0]/200,sigmas[0]/100,sigmas[0]/50]
cv_args = []
# for mu in mus:
for sigma in sigmas:
    for beta in betas:
        for tau in taus:
            cv_args.append((tourneys,mu,sigma,beta,tau,False,False))

pool = multiprocessing.Pool(6)
cv_metrics = pool.starmap(validate, cv_args)
rows = []
for i,cv in enumerate(cv_metrics):
    m = cv[0]
    env = cv[1]
    rows.append(env + list(m))
cv_metric_frame = pd.DataFrame(columns=[
    'mu','sigma','beta','tau',
    'avg_acc','avg_calib','avg_log_loss',
    'std_acc','std_calib','std_log_loss'
    ],data=rows)
cv_metric_frame.to_csv('cv_res_1500_players_Mar_19.csv',index=False)
print(cv_metric_frame)
