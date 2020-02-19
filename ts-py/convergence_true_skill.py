import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bernoulli
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from trueskill import Rating, rate_1vs1, global_env
from utils import ts_win_probability

sigma = 110 / 3
elliot_base = 110
 # this is elliot in MBML
elliot_increment = 10  # elliot skill increases after so many games - first 200 and every 100 after up to 500

# create player pool
possible_opponents = np.random.normal(loc=125.0, scale=10.0, size=99)
opponents_for_games = np.random.choice(possible_opponents, size=500)

game_outcomes = []
win_probabilities = []
elliot_gt_skill = elliot_base
for index, opponent in enumerate(opponents_for_games):
    if index in [200, 300, 400]:
        elliot_gt_skill += elliot_increment

    elliot_wins = 0
    for i in range(1000):
        elliot_performance = np.random.normal(loc=elliot_gt_skill, scale=5.0)
        opponent_performance = np.random.normal(loc=opponent, scale=5.0)
        if elliot_performance > opponent_performance: elliot_wins += 1
    win_probability = float(elliot_wins)/1000
    win_probabilities.append(win_probability)
    game_outcomes.append(bernoulli.rvs(win_probability))

# Plot ground truth skill for each game
game_number = np.arange(0, 501)
ground_truth_skill = np.zeros(game_number.shape)
ground_truth_skill[0:200] = elliot_base
ground_truth_skill[200:300] = elliot_base + elliot_increment
ground_truth_skill[300:400] = elliot_base + 2 * elliot_increment
ground_truth_skill[400:501] = elliot_base + 3 * elliot_increment

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(game_number, ground_truth_skill, 'r-', label='Ground Truth Skill')
ax.set_xlabel('Number of games')
ax.set_ylabel('Skill')
ax.legend()
fig.savefig('../figures/GroundTruthOnlyTrueSkill.png')
plt.show()

# Using game outcomes calculate skill for each game using true-skill model
true_skill_mean = [125]
true_skill_var = [5]
elliot_rating = Rating(125, 5)  # intialize
for index, game in enumerate(game_outcomes):
    opponent_rating = Rating(opponents_for_games[index], 5.0)
    if game == 1:
        elliot_rating = rate_1vs1(elliot_rating, opponent_rating)[0]
    else:
        elliot_rating = rate_1vs1(opponent_rating, opponent_rating)[1]
    true_skill_mean.append(elliot_rating.mu)
    true_skill_var.append(elliot_rating.sigma)


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

# # Export Data
# data = {'sigma': sigma, 'varying_base': elliot_base, 'varying_increment': elliot_increment,
#         'static_base': static_base, 'win_probabilities': win_probs, 'mse': mse_true_skill, 'rmse': rmse_true_skill,
#         'game_outcomes': [int(game) for game in game_outcomes]}
# # Export Data
# with open('../model_params/convergTrueSkillParams.json', 'w') as outfile:
#     json.dump(data, outfile, indent=4)
