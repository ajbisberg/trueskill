import copy
import pandas as pd
from time import perf_counter
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from models import Tournament, Simulation, DynamicElo
from utils import TournamentData, initialize_elo, merge_elo
from model_params import mov_exp

def regress_20(elo_i):
    elo_i.elo = 0.8*elo_i.elo + 0.2*1500
    return elo_i

regressfunc = regress_20
base_k = 5
cutoff = 1650
reduction = 0.1
mov = mov_exp
w90 = 200
elo = DynamicElo(base_k,cutoff,reduction,mov,w90)
data_path = '../../cwl-data/data/'
season = TournamentData(None,data_path,wwii=True)
elo_i = initialize_elo(Tournament(TournamentData('CWL Champs 2017',data_path).data).raw_data)

for td in season.tourneys[:-1]:
    t = Tournament(td['data'])
    t.build_tournament()
    t.populate_next_series()
    t.set_first_series_elo(elo_i)
    t.play(elo)
    elo_f = t.final_elos()
    elo_i = regressfunc(merge_elo(elo_i,elo_f))

t0 = perf_counter()
first = pd.DataFrame()
second = pd.DataFrame()
third = pd.DataFrame()
for i in range(1000):
    tc = copy.deepcopy(Tournament(season.tourneys[-1]['data']))
    tc.build_tournament()
    tc.populate_next_series()
    tc.set_first_series_elo(elo_i)
    sim = Simulation(tc)
    sim.run(elo)
    finals = sim.tournament[-1]['data']
    semis = sim.tournament[-3:-2][0]['data']
    first = first.append(finals[finals.win_b == 1])
    second = second.append(finals[finals.win_b == 0])
    third = third.append(semis[semis.win_b == 0])
t1 = perf_counter()
print('simulation time: ', round(t1-t0,1))
first.elo = first.elo.astype('float')
first.groupby('team').agg({'win_b' : 'sum','elo' : 'mean'}) \
    .sort_values('win_b',ascending=False) \
    .to_csv('../generated_data/1000_sim_champs_first.csv')
second.elo = second.elo.astype('float')
second.groupby('team').agg({'win_b' : 'count','elo' : 'mean'}) \
    .sort_values('win_b',ascending=False) \
    .to_csv('../generated_data/1000_sim_champs_second.csv')
third.elo = third.elo.astype('float')
third.groupby('team').agg({'win_b' : 'count','elo' : 'mean'}) \
    .sort_values('win_b',ascending=False) \
    .to_csv('../generated_data/1000_sim_champs_third.csv')

