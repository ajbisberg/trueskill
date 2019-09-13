import copy
import pandas as pd
np = pd.np
from time import perf_counter
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from models import Tournament, Simulation
from utils import TournamentData, initialize_elo, merge_elo
from model_params import attenuate, mov_lin, mov_log, mov_sqrt, mov_exp

data_path = '../../cwl-data/data/'
season = TournamentData(None,data_path,wwii=True)
elo_i = initialize_elo(Tournament(TournamentData('CWL Champs 2017',data_path).data).raw_data)

def k1(win,elo_win,elo_loss,score_win,score_loss):
    # if win:
    #     k = mov_exp(attenuate(k,elo_win),score_win-score_loss)
    # else:
    #     k = attenuate(k,elo_loss) 
    # return k
    return 5

def regress(elo_i):
    elo_i.elo = 0.9*elo_i.elo + 0.1*1500
    return elo_i

accs = []
all_correct = []
total_series = 0
for i,td in enumerate(season.tourneys[:-1]):
    tourney = Tournament(td['data'])
    tourney.build_tournament()
    tourney.set_first_series_elo(elo_i)
    tourney.populate_next_series()
    tourney.play(k1,acc=True)
    correct = [tr['correct'] for tr in tourney.tournament]
    max_w = [tr['max_wp'] for tr in tourney.tournament]
    accs.append(
        (list(season.wwii.keys())[i], 
        len(tourney.tournament),
        sum(correct)/len(correct), 
        sum(max_w)/sum(correct),
        np.mean(-np.log(np.trim_zeros(np.sort(np.multiply(correct,max_w))))))
    )
    elo_f = tourney.final_elos()
    # if list(season.wwii.keys())[i] == 'CWL Pro League, Relegation':
    #     print(elo_i.sort_values('elo',ascending=False))
    #     import ipdb; ipdb.set_trace()
    # print(td['name'],elo_f)
    elo_i = regress(merge_elo(elo_i,elo_f))

acc_out = pd.DataFrame(columns=['tournament','n_series','acc','calib','logloss'],data=accs)
ax = acc_out.sort_values('n_series',ascending=False).plot(kind='line',x='n_series',y='acc')
ax.set_ylabel('accuracy')
ax.set_ylim(0.4,1)
plt.savefig('../img/acc.png')
acc_out = acc_out.append(
    pd.DataFrame(columns=['tournament','n_series','acc','calib','logloss'],data=[(
        '*weighted avg*',
        np.sum(acc_out.n_series),
        np.sum(np.multiply(acc_out.n_series,acc_out.acc))/np.sum(acc_out.n_series),
        np.sum(np.multiply(acc_out.n_series,acc_out.calib))/np.sum(acc_out.n_series),
        np.sum(np.multiply(acc_out.n_series,acc_out.logloss))/np.sum(acc_out.n_series)
    )])
)
acc_weighted = np.sum(np.multiply(acc_out.n_series,acc_out.acc))/np.sum(acc_out.n_series)
print(acc_out.round(3))
acc_out.round(3).to_csv('../generated_data/acc.csv')
import pdb; pdb.set_trace()
# print('weighted avg acc over all tourneys: ',round(acc_weighted,3))