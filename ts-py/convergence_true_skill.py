import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bernoulli
from sklearn.metrics import mean_squared_error
from trueskill import Rating, rate_1vs1, global_env, setup
from utils import ts_win_probability

# sigma = 8.33
# varying_base = 25
# varying_increment = 10
# static_base = 20
# setup(mu=varying_base, sigma=sigma, beta=sigma/2, tau=sigma/100)
sigma = 8.333
varying_base = 110
varying_increment = 10
static_base = 100

# Calculate the game outcomes by varying one players skill score while holding
# the others static
game_outcomes = []
win_probs = []
varying_skill, static_skill = Rating(varying_base, sigma), Rating(static_base, sigma)
win_prob = ts_win_probability([varying_skill], [static_skill], global_env())
win_probs.append(win_prob)
game_outcomes.extend(bernoulli.rvs(win_prob, size=200))

varying_skill, static_skill = Rating(varying_base + varying_increment, sigma), Rating(static_base, sigma)
win_prob = ts_win_probability([varying_skill], [static_skill], global_env())
win_probs.append(win_prob)
game_outcomes.extend(bernoulli.rvs(win_prob, size=100))

varying_skill, static_skill = Rating(varying_base + 2 * varying_increment, sigma), Rating(static_base, sigma)
win_prob = ts_win_probability([varying_skill], [static_skill], global_env())
win_probs.append(win_prob)
game_outcomes.extend(bernoulli.rvs(win_prob, size=100))

varying_skill, static_skill = Rating(varying_base + 3 * varying_increment, sigma), Rating(static_base, sigma)
win_prob = ts_win_probability([varying_skill], [static_skill], global_env())
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
fig.savefig('../figures/GroundTruthOnlyTrueSkill.png')
plt.show()

# Using game outcomes calculate skill for each game using true-skill model
true_skill_mean = [varying_base]
true_skill_var = [sigma]
varying_player, static_player = Rating(varying_base, sigma), Rating(static_base, sigma)
for game in game_outcomes:
    if game == 1:
        varying_player, static_player = rate_1vs1(varying_player, static_player)
    else:
        static_player, varying_player = rate_1vs1(static_player, varying_player)
    true_skill_mean.append(varying_player.mu)
    true_skill_var.append(varying_player.sigma)
    static_player = Rating(static_base, sigma)


def relative_mse(true, pred):
    mse = mean_squared_error(true, pred)
    average = np.average(true)
    mse_den = mean_squared_error(np.full(true.shape, average), true)
    return mse/mse_den


mse_true_skill = mean_squared_error(ground_truth_skill, true_skill_mean)
rmse_true_skill = relative_mse(ground_truth_skill, true_skill_mean)

# plot results
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(game_number, ground_truth_skill, 'r-', label='Ground Truth Skill')
ax.plot(game_number, true_skill_mean, 'b-', label='True Skill (MSE = {:.2f})'.format(rmse_true_skill))
error_below = np.array(true_skill_mean)-np.array(true_skill_var)
error_above = np.array(true_skill_mean)+np.array(true_skill_var)
ax.fill_between(game_number, error_below, error_above, facecolor=(173/256, 216/256, 230/256))
ax.set_xlabel('Number of games')
ax.set_ylabel('Skill')
ax.set_xlim(-5, 505)
ax.set_ylim(100, 145)
ax.legend()
fig.savefig('../figures/GroundAndTrueSkill.png')
plt.show()

# Export Data
data = {'sigma': sigma, 'varying_base': varying_base, 'varying_increment': varying_increment,
        'static_base': static_base, 'win_probabilities': win_probs, 'mse': mse_true_skill, 'rmse': rmse_true_skill,
        'game_outcomes': [int(game) for game in game_outcomes]}
# Export Data
with open('../model_params/convergTrueSkillParams.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)
