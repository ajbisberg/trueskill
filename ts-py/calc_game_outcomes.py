from trueskill import Rating, rate_1vs1, global_env, setup
import numpy as np
from model_params import mov_exp, mov_lin
from models import DynamicElo
from as_ts import sim_games as sim_games_ts
from as_scope import sim_games_scope

env = setup(mu=125.000, sigma=2.500, beta=1.667, tau=0.013, draw_probability=0)
elod = {'base_k': 0.5, 'cutoff': 1550, 'reduction': 0.75, 'mov': mov_lin, 'w90': 50}
elo = DynamicElo(elod['base_k'], elod['cutoff'],elod[ 'reduction'], elod['mov'], elod['w90'])

#sim_games_ts(env)
sim_games_scope(elo)