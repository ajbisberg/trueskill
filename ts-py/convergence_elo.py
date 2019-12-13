import json
import matplotlib.pyplot as plt
from model_params import mov_exp, mov_lin
from models import DynamicElo
import numpy as np
from scipy.stats import bernoulli
from sklearn.metrics import mean_squared_error

# Experiment Parameters
varying_base = 1650
varying_increment = 100
static_base = 1600

# ELO Model
base_k = 30
cutoff = 1650
reduction = 0.75
mov = mov_lin
w90 = 200
elo = DynamicElo(base_k, cutoff, reduction, mov, w90)

# Calculate the game outcomes by varying one players skill score while holding
# the others static
game_outcomes = []
win_probs = []
varying_skill, static_skill = varying_base, static_base
win_prob = elo.calc_win_prob(varying_skill, static_skill)
win_probs.append(win_prob)
game_outcomes.extend(bernoulli.rvs(win_prob, size=200))

varying_skill, static_skill = varying_base + varying_increment, static_base
win_prob = elo.calc_win_prob(varying_skill, static_skill)
win_probs.append(win_prob)
game_outcomes.extend(bernoulli.rvs(win_prob, size=100))

varying_skill, static_skill = varying_base + 2 * varying_increment, static_base
win_prob = elo.calc_win_prob(varying_skill, static_skill)
win_probs.append(win_prob)
game_outcomes.extend(bernoulli.rvs(win_prob, size=100))

varying_skill, static_skill = varying_base + 3 * varying_increment, static_base
win_prob = elo.calc_win_prob(varying_skill, static_skill)
win_probs.append(win_prob)
game_outcomes.extend(bernoulli.rvs(win_prob, size=100))

# Plot ground truth skill for each game
game_number = np.arange(0, 501)
ground_truth_skill = np.zeros(game_number.shape)
ground_truth_skill[0:200] = varying_base
ground_truth_skill[200:300] = varying_base + varying_increment
ground_truth_skill[300:400] = varying_base + 2 * varying_increment
ground_truth_skill[400:501] = varying_base + 3 * varying_increment

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(game_number, ground_truth_skill, 'r-', label='Ground Truth Skill')
ax.set_xlabel('Number of games')
ax.set_ylabel('Skill')
ax.legend()
fig.savefig('../figures/GroundTruthOnlyELO.png')
plt.show()

# Using game outcomes calculate skill for each game using true-skill model
elo_skill = [varying_base]
varying_player, static_player = varying_base, static_base
for game in game_outcomes:
    if game == 1:
        varying_player, static_player = elo.update(varying_player, static_player, 3, 2)
    else:
        static_player, varying_player = elo.update(static_player, varying_player, 3, 2)
    elo_skill.append(varying_player)
    static_player = static_base


# calculate mse between true and predicted skill
# def relative_mse(true, pred):
#     val = (true - pred)/true
#     val = val ** 2
#     return np.average(val)
def relative_mse(true, pred):
    mse = mean_squared_error(true, pred)
    average = np.average(true)
    mse_den = mean_squared_error(np.full(true.shape, average), true)
    return mse/mse_den

# calculate mse between true and predicted skill
mse_elo_skill = mean_squared_error(ground_truth_skill, elo_skill)
rmse_elo_skill = relative_mse(ground_truth_skill, elo_skill)

# plot results
fig, ax = plt.subplots(figsize=(8, 5))
# ground truth
ax.plot(game_number, ground_truth_skill, 'r-', label='Ground Truth Skill')
# elo
ax.plot(game_number, elo_skill, 'b-', label='Elo Skill (MSE = {:.2f})'.format(rmse_elo_skill))
# figure formatting
ax.set_xlabel('Number of games')
ax.set_ylabel('Skill')
ax.set_xlim(-5, 505)
ax.legend()
fig.savefig('../figures/GroundAndElo.png')
plt.show()

# Export data
data = {'base_k': base_k, 'cutoff':cutoff, 'reduction':reduction, 'mov': str(mov_lin), 'w90':w90,
        'varying_base': varying_base, 'varying_increment': varying_increment,
        'static_base': static_base, 'win_probabilities': win_probs, 'mse': mse_elo_skill,
        'rmse': rmse_elo_skill, 'game_outcomes': [int(game) for game in game_outcomes]}
# Export Data
with open('../model_params/convergEloParams.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)

