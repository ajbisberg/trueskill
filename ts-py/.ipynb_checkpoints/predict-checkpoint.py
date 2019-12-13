import copy
import pandas as pd
np = pd.np
from time import perf_counter
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
import csv

from models import Tournament, Simulation
from utils import TournamentData, initialize_elo, merge_elo
from model_params import attenuate, mov_lin, mov_log, mov_sqrt, mov_exp
from xvalidate import validate

data_path = '../../cwl-data/data/'
season = TournamentData(None,data_path,wwii=True)

elo_i = initialize_elo(Tournament(TournamentData('CWL Champs 2017',data_path).data).raw_data)
# import pdb; pdb.set_trace()
tourneys = []
t0 = perf_counter()
for i,td in enumerate(season.tourneys):
    t = Tournament(td['data'])
    t.build_tournament()
    t.populate_next_series()
    tourneys.append(t)
t1 = perf_counter()
print('built tournaments in',round(t1-t0,1))

def k_simple_40(win,elo_win,elo_loss,score_win,score_loss):
    return 40

def regress_none(elo_i):
    return elo_i

m_acc = validate(tourneys,elo_i,regress_none,k_simple_40)

print(m_acc)