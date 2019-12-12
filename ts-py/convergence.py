import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bernoulli
from sklearn.metrics import mean_squared_error
from trueskill import Rating, rate_1vs1, global_env
from utils import ts_win_probability

# Calculate the game outcomes by varying one players skill score while holding
# the others static
game_outcomes = []

varying_skill, static_skill = Rating(110), Rating(100)
win_prob = ts_win_probability([varying_skill], [static_skill], global_env())
game_outcomes.extend(bernoulli.rvs(win_prob, size=200))

varying_skill, static_skill = Rating(120), Rating(100)
win_prob = ts_win_probability([varying_skill], [static_skill], global_env())
game_outcomes.extend(bernoulli.rvs(win_prob, size=100))

varying_skill, static_skill = Rating(130), Rating(100)
win_prob = ts_win_probability([varying_skill], [static_skill], global_env())
game_outcomes.extend(bernoulli.rvs(win_prob, size=100))

varying_skill, static_skill = Rating(140), Rating(100)
win_prob = ts_win_probability([varying_skill], [static_skill], global_env())
game_outcomes.extend(bernoulli.rvs(win_prob, size=100))

# Plot ground truth skill for each game
game_number = np.arange(1, 501)
ground_truth_skill = np.zeros(game_number.shape)
ground_truth_skill[0:200] = 110
ground_truth_skill[200:300] = 120
ground_truth_skill[300:400] = 130
ground_truth_skill[400:500] = 140

# fig, ax = plt.subplots(figsize=(10, 5))
# ax.plot(game_number, ground_truth_skill, 'r-', label='Ground Truth Skill')
# ax.set_xlabel('Number of games')
# ax.set_ylabel('Skill')
# ax.legend()
# fig.savefig('../figures/GroundTruthOnly.png')
# plt.show()

# Using game outcomes calculate skill for each game using true-skill model
true_skill_mean = []
true_skill_var = []
varying_player, static_player = Rating(110), Rating(100)
for game in game_outcomes:
    if game == 1:
        varying_player, static_player = rate_1vs1(varying_player, static_player)
    else:
        varying_player, static_player = rate_1vs1(static_player, varying_player)
    true_skill_mean.append(varying_player.mu)
    true_skill_var.append(varying_player.sigma)
    static_player = Rating(100)

# calculate mse between true and predicted skill
mse_true_skill = mean_squared_error(ground_truth_skill, true_skill_mean)

# plot results
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(game_number, ground_truth_skill, 'r-', label='Ground Truth Skill')
ax.plot(game_number, true_skill_mean, 'b-', label='True Skill (MSE = {:.2f})'.format(mse_true_skill))
error_below = np.array(true_skill_mean)-np.array(true_skill_var)
error_above = np.array(true_skill_mean)+np.array(true_skill_var)
ax.fill_between(game_number, error_below, error_above, facecolor=(173/256, 216/256, 230/256))
ax.set_xlabel('Number of games')
ax.set_ylabel('Skill')
ax.set_xlim(-5, 505)
ax.legend()
fig.savefig('../figures/GroundAndTrueSkill.png')
plt.show()
