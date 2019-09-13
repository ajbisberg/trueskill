import pandas as pd
np = pd.np
from time import perf_counter

from models import Tournament, Simulation
from utils import TournamentData, initialize_elo, merge_elo
from model_params import attenuate, mov_lin, mov_log, mov_sqrt, mov_exp
data_path = '../../cwl-data/data/'

def k1(win,elo_win,elo_loss,score_win,score_loss):
    k = 30
    if win:
        k = mov_exp(attenuate(k,elo_win),score_win-score_loss)
    else:
        k = attenuate(k,elo_loss) 
    return k


# TODO: run perf counter on each of these steps 
t0 = perf_counter()
season = TournamentData(None,data_path,wwii=True)
t1= perf_counter()
elo_i = initialize_elo(Tournament(TournamentData('CWL Champs 2017',data_path).data).raw_data)
t2 = perf_counter()
tr = Tournament(season.tourneys[9]['data'])
t25 = perf_counter()
tr.build_tournament()
t3 = perf_counter()
tr.set_first_series_elo(elo_i)
t4 = perf_counter()
tr.populate_next_series()
t5 = perf_counter()
tr.play(k1)
t6 = perf_counter()
tr.final_elos()
t7 = perf_counter()

print(
    round(t1-t0,3),'load wwii data\n',
    round(t2-t1,3),'init elo from 2017\n',
    round(t25-t2,3),'instantiate tourney\n',
    round(t3-t25,3),'build tourney\n',
    round(t4-t3,3),'set first series elo\n',
    round(t5-t4,3),'pop next series\n',
    round(t6-t5,3),'play tourney\n',
    round(t7-t6,3),'final elos\n'
)
