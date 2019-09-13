import pandas as pd
np = pd.np

import itertools
import math

def ts_win_probability(team1, team2, env):
    delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
    sum_sigma = sum(r.sigma ** 2 for r in itertools.chain(team1, team2))
    size = len(team1) + len(team2)
    denom = math.sqrt(size * (env.beta * env.beta) + sum_sigma)
    return env.cdf(delta_mu / denom)

class TournamentData:

    def __init__(self, tourney, base_path, wwii=False):
        self.iw = { 'CWL Champs 2017' : 'data-2017-08-13-champs.csv'}
        self.wwii = { 
            'CWL Dallas' : 'data-2017-12-10-dallas.csv',
            'CWL New Orleans' : 'data-2018-01-14-neworleans.csv',
            'CWL Pro League, Stage 1' : 'data-2018-04-08-proleague1.csv',
            'CWL Atlanta' : 'data-2018-03-11-atlanta.csv',
            'CWL Birmingham' : 'data-2018-04-01-birmingham.csv',
            'CWL Pro League, Relegation' : 'data-2018-04-19-relegation.csv',
            'CWL Seattle' :'data-2018-04-22-seattle.csv',
            'CWL Pro League, Stage 2' : 'data-2018-07-29-proleague2.csv', 
            'CWL Anaheim' : 'data-2018-06-17-anaheim.csv', 
            'CWL Champs 2018' : 'data-2018-08-19-champs.csv'
        }

        if wwii == True:
            self.tourneys = []
            for t in list(self.wwii.keys()):
                self.tourneys.append({'name' : t, 'data' : self._get_tourney(t,base_path)})
        else:
            self.data = self._get_tourney(tourney,base_path)

    def _get_tourney(self,tourney,base_path):
        all_t = dict(self.iw) 
        all_t.update(dict(self.wwii))
        base_url = 'https://raw.githubusercontent.com/Activision/cwl-data/master/data/'
        base = base_path if base_path else base_url
        return pd.read_csv(base+all_t[tourney])

def initialize_rating(td,env):
    np.random.seed(10)
    team_match = td[['team','end_dt','win_b','series id','match id','mode']] \
        .groupby(['team','end_dt','win_b','match id','mode'],as_index=False)['series id'].max() \
        .sort_values(['series id','end_dt'],ascending=[False,True])
    win_totals = team_match[['team','series id','win_b']].groupby(['team'],as_index=False)['win_b'].sum() \
        .sort_values('win_b',ascending=False)
    win_totals.columns = ['team','ws']
    loss_totals = team_match[['team','win_b']][team_match['win_b'] == 0] \
        .groupby('team',as_index=False).count()
    loss_totals.columns = ['team','ls']
    tourney_wl = win_totals.merge(loss_totals)
    # tourney_wl['games'] = tourney_wl['ws'] + tourney_wl['ls']
    # tourney_wl['gd'] = tourney_wl['games'] - floor(tourney_wl['games'].median())
    tourney_wl['w-l'] = (tourney_wl['ws'] - tourney_wl['ls'])
    tourney_wl['w/l'] = round(tourney_wl['ws'] / tourney_wl['ls'],2)
    wls = tourney_wl['w/l'].unique()
    sz = len(wls)
    avg_elos = -np.sort(-np.random.normal(loc=env.mu,scale=env.sigma,size=sz))
    elo_map = {}
    for i,elo in enumerate(avg_elos):
        elo_map[wls[i]] = env.Rating(elo)
    tourney_wl['rating'] = tourney_wl['w/l'].apply(lambda wl: elo_map[wl])
    tourney_wl.index = tourney_wl['team']
    return tourney_wl[['team','w-l','rating']]

def merge_elo(elo_i,elo_f):
    for row in elo_f.iterrows():
        if ( len(elo_i[elo_i.team==row[0]]) > 0 ):
            elo_i.at[row[0],'elo'] = row[1].elo
        else:
            elo_i = elo_i.append(row[1])
    return elo_i

def validate(tourneys,elo_i,regressfunc,kfunc,do_test=False):
    trains = []
    tr_len = 0
    v_perf = []
    t_perf = []

    t2 = perf_counter()
    for i in range(len(tourneys)-2):
    
        trains += [tourneys[i]]
        validate = tourneys[i+1]
        test = tourneys[i+2]
        for train in trains:
            train.set_first_series_elo(elo_i)
            train.play(kfunc)
            elo_f = train.final_elos()
            elo_i = regressfunc(merge_elo(elo_i,elo_f))
            tr_len += len(train.tournament)
        
        validate.set_first_series_elo(elo_i)
        validate.play(kfunc,acc=True)
        v_correct = [series['correct'] for series in validate.tournament]
        max_w = [series['max_wp'] for series in validate.tournament]
        v_perf.append(
            (i+1, 
            len(validate.tournament),
            sum(v_correct)/len(v_correct), 
            # sum(max_w)/sum(v_correct),
            # np.mean(-np.log(np.trim_zeros(np.sort(np.multiply(v_correct,max_w)))))
        ))
        elo_f = validate.final_elos()
        elo_i = merge_elo(elo_i,elo_f)
        elo_i = regressfunc(merge_elo(elo_i,elo_f))

        if do_test:
            test.set_first_series_elo(elo_i)
            test.play(kfunc,acc=True)
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
        # print('tested in ',round(t3-t2,1))
        return t_out.t_acc.mean()

    v_out = pd.DataFrame(columns=['n_folds','n_series','acc'],data=v_perf)
    t3 = perf_counter()
    # print(v_out)
    # print('xvalidated in ',round(t3-t2,1))
    return round(np.sum(np.multiply(v_out.n_series,v_out.acc))/np.sum(v_out.n_series),3)

# Do this later for another paper
def get_player_stats(i):
    # look at the ith position in TD list
    # df[['player','mode','k/d']].groupby(['player','mode'])['k/d'].mean().unstack().reset_index()
    pass