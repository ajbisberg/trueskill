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

t1 = Tournament(season.tourneys[0]['data'])
t1.build_for_players()
t1.populate_next_series_players()
t1.set_initial_player_ratings(env)
pprint(t1.players[:5])