from scipy.stats import bernoulli
import pandas as pd
np = pd.np
from trueskill import Rating, rate_1vs1, global_env, setup
from utils import ts_win_probability
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
def relative_mse(true, pred):
    mse = mean_squared_error(true, pred)
    average = np.average(true)
    mse_den = mean_squared_error(np.full(true.shape, average), true)
    return mse/mse_den

def sim_games(env,fix_variance=True):
    possible_opponents = np.random.normal(loc=125.0, scale=10.0, size=99)
    opponents_for_games = np.random.choice(possible_opponents, size=500)
    game_outcomes = []
    win_probabilities = []
    elliot_gt_skill = 110
    elliot_increment = 10
    true_elliot_skill = [elliot_gt_skill]
    for index, opponent in enumerate(opponents_for_games):
        if index in [200,300,400]:
            elliot_gt_skill += elliot_increment
        true_elliot_skill.append(elliot_gt_skill)
        
        if fix_variance:
            ewins = 0
            for i in range(1000):
                opponent_perf = np.random.normal(loc=opponent, scale=5.0)
                elliot_perf = np.random.normal(loc=elliot_gt_skill, scale=5.0)
                if elliot_perf > opponent_perf: ewins += 1
            win_probability = float(ewins)/1000
        else:
            elliot_skill, opponent_skill = Rating(elliot_gt_skill, sigma), Rating(opponent, sigma)
            win_probability = ts_win_probability([elliot_skill], [opponent_skill], env)

        win_probabilities.append(win_probability)
        game_outcomes.append(bernoulli.rvs(win_probability))
    return opponents_for_games, game_outcomes, true_elliot_skill

def update_skill(env, opponents_for_games, game_outcomes):
    e_ts_perf = env.create_rating()
    skill_mean = [mu]
    skill_var = [sigma]
    for index, opponent in enumerate(opponents_for_games):
        o_ts_perf = env.create_rating(opponent)
        rating_groups = [
            {'elliot'   :  e_ts_perf}, 
            {'opponent' :  o_ts_perf}
        ]
        ranking = [int(not game_outcomes[index]),game_outcomes[index]]
        rated = env.rate(rating_groups, ranks=ranking)
        skill_mean.append(rated[0]['elliot'].mu)
        skill_var.append(rated[0]['elliot'].sigma)
        e_ts_perf = rated[0]['elliot']
    return skill_mean, skill_var

mu = 125
# sigma = 5
# beta = sigma/2
# tau = sigma/100
# env = setup(mu=mu,sigma=sigma,beta=beta,tau=tau,draw_probability=0)

cv_data = []
fix_variance = True
if fix_variance:
    opponents_for_games, game_outcomes, true_elliot_skill = sim_games(None,True)

for sigma in [mu/50,mu/25,mu/10,mu/3]:
    for beta in [sigma/1.5,sigma/2,sigma/5]:
        for tau in [sigma/200,sigma/100,sigma/50]:
            env = setup(mu=mu,sigma=sigma,beta=beta,tau=tau,draw_probability=0)
            if not fix_variance:
                opponents_for_games, game_outcomes, true_elliot_skill = sim_games(env,False)
            skill_mean, skill_var = update_skill(env, opponents_for_games, game_outcomes)
            cv_data.append([mu, sigma, beta, tau, relative_mse(np.array(true_elliot_skill),skill_mean), skill_mean, skill_var])

cv_res = pd.DataFrame(columns=['mu','sigma','beta','tau','RMSE','skill_mean','skill_var'],data=cv_data)
best_vals = cv_res.sort_values('RMSE').reset_index()
print(best_vals[['mu','sigma','beta','tau','RMSE']][:5])
best_vals[['mu','sigma','beta','tau','RMSE']].to_csv('cv_as_ts_Mar_02.csv',index=False)

if True:
    skill_mean = best_vals['skill_mean'][0]
    skill_var = best_vals['skill_var'][0]
    fig, ax = plt.subplots()
    error_below = np.array(skill_mean)-np.array(skill_var)
    error_above = np.array(skill_mean)+np.array(skill_var)
    ax.plot(list(range(501)),skill_mean, label='TrueSkill Estimate (RMSE={})'.format(round(best_vals['RMSE'][0],3)))
    ax.plot(list(range(501)),true_elliot_skill, label='Ground Truth Skill')
    ax.fill_between(list(range(501)), error_below, error_above, facecolor=(173/256, 216/256, 230/256))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # print(best_vals['mu'][0], best_vals['sigma'][0], best_vals['beta'][0], best_vals['tau'][0])
    ax.set_title('Ancestral Sampling - TrueSkill \n mu = {}, sigma = {}, beta = {}, tau = {}'.format(
        round(best_vals['mu'][0],3), 
        round(best_vals['sigma'][0],3), 
        round(best_vals['beta'][0],3), 
        round(best_vals['tau'][0],3))
    )
    ax.legend()
    ax.set_xlabel('game number')
    ax.set_ylabel('score')
    plt.savefig('ts_synth.png')