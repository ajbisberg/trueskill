from scipy.stats import bernoulli
import pandas as pd
np = pd.np
from model_params import mov_exp, mov_lin
from models import DynamicElo
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
def relative_mse(true, pred):
    mse = mean_squared_error(true, pred)
    average = np.average(true)
    mse_den = mean_squared_error(np.full(true.shape, average), true)
    return mse/mse_den

def sim_games(elo):
    possible_opponents = np.random.normal(loc=1500.0, scale=(1500.0/12.5), size=99)
    opponents_for_games = np.random.choice(possible_opponents, size=500)
    game_outcomes = []
    win_probabilities = []
    skill_start = 1320.
    elliot_increment = (1500.0/12.5)
    elliot_gt_skill = skill_start
    true_elliot_skill = [skill_start]
    for index, opponent in enumerate(opponents_for_games):
        if index in [200,300,400]:
            elliot_gt_skill += elliot_increment
        true_elliot_skill.append(elliot_gt_skill)
        win_probability = elo.calc_win_prob(elliot_gt_skill,opponent)
        win_probabilities.append(win_probability)
        game_outcomes.append(bernoulli.rvs(win_probability))
    return opponents_for_games, game_outcomes, true_elliot_skill

def update_skill(elo, opponents_for_games, game_outcomes):
    elliot = 1500.
    elos = [elliot]
    for index, opponent in enumerate(opponents_for_games):
        if game_outcomes[index] == 1:
            elliot, opponent = elo.update(elliot, opponent, 3, 2) 
        else:
            opponent, elliot = elo.update(opponent, elliot, 3, 2)
        elos.append(elliot)
    return elos

cv_data = []

for base_k in [0.5,1,2,3,20]:
    for cutoff in [1550,1600,1650]:
        for w90 in [50,100,200]:
            mov = mov_lin
            reduction = 0.75
            elo = DynamicElo(base_k, cutoff, reduction, mov, w90)
            opponents_for_games, game_outcomes, true_elliot_skill = sim_games(elo)
            skill_mean = update_skill(elo,opponents_for_games,game_outcomes)
            cv_data.append([base_k,cutoff,'mov_lin',w90, relative_mse(np.array(true_elliot_skill),skill_mean), skill_mean])

cv_res = pd.DataFrame(columns=['k','cutoff','mov','w90','RMSE','skill_mean'],data=cv_data)
best_vals = cv_res.sort_values('RMSE').reset_index()
print(best_vals[['k','cutoff','mov','w90','RMSE']][:5])
best_vals[['k','cutoff','mov','w90','RMSE']].to_csv('cv_as_scope_Mar_02.csv',index=False)

if True:
    skill_mean = best_vals['skill_mean'][0]
    fig, ax = plt.subplots()
    ax.plot(list(range(501)),skill_mean, label='SCOPE Estimate (RMSE={})'.format(round(best_vals['RMSE'][0],3)))
    ax.plot(list(range(501)),true_elliot_skill, label='Ground Truth Skill')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # print(best_vals['mu'][0], best_vals['sigma'][0], best_vals['beta'][0], best_vals['tau'][0])
    ax.set_title('Ancestral Sampling - SCOPE \n k = {}, cutoff = {}, mov = {}, w90 = {}'.format(
        best_vals['k'][0], best_vals['cutoff'][0], best_vals['mov'][0], best_vals['w90'][0])
    )
    ax.legend()
    ax.set_xlabel('game number')
    ax.set_ylabel('score')
    plt.savefig('scope_synth.png')