import sys
import pandas as pd
np = pd.np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from math import log
from scipy.special import erf

##
# Functions for setting model hyperparameters
##

# K attenuation
k = 30
def attenuate(k,elo_win):
    if elo_win >= 1600:
        k = k - 10
    return k

# margin of victory

def mov_none(k,delta_w):
    return k

def mov_lin(k,delta_w):
    return k + 3*(delta_w-1)

def mov_log(k,delta_w):
    return k + log((delta_w-1)*150+1)

def mov_sqrt(k,delta_w):
    mov = delta_w if delta_w > 1 else 0
    return k + (mov*100)**(1/2)

def mov_exp(k,delta_w):
    mov = delta_w if delta_w > 1 else 0
    return k + (mov)**(3)

##
# plot hyperparams
##
if len(sys.argv) > 1 and sys.argv[1] == '--plot':
    # MOV
    deltas = np.arange(1,4)
    scales = [mov_lin(k,d) for d in deltas]
    mf = pd.concat([pd.Series(deltas,name='mov'),pd.Series(scales,name='linear')],axis=1)
    scales = [mov_log(k,d) for d in deltas]
    mf = pd.concat([mf,pd.Series(scales,name='log')],axis=1)
    scales = [mov_sqrt(k,d) for d in deltas]
    mf = pd.concat([mf,pd.Series(scales,name='sqrt')],axis=1)
    scales = [mov_exp(k,d) for d in deltas]
    mf = pd.concat([mf,pd.Series(scales,name='exp')],axis=1)
    print(deltas)

    fig = plt.figure(figsize=(4,3))
    fig.tight_layout()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    mf.plot(kind='line',x='mov',y='linear',ax=ax)
    mf.plot(kind='line',x='mov',y='log',ax=ax)
    mf.plot(kind='line',x='mov',y='sqrt',ax=ax)
    mf.plot(kind='line',x='mov',y='exp',ax=ax)

    plt.xticks([0,1,2])
    plt.ylabel('K value')
    plt.xlabel('Margin of Victory')
    plt.title('K Scale for Margin of Victory')
    plt.savefig('../img/mov.png',dpi=300)

    # attenuation
    elos = np.arange(1200,1850,50)
    att1 = [attenuate(k,elo) for elo in elos]
    att2 = [attenuate(20,elo) for elo in elos]
    af = pd.concat([
        pd.Series(elos,name='elo'),
        pd.Series(att1,name='k(30)'),
        pd.Series(att2,name='k(20)')],axis=1)
    fig = plt.figure(figsize=(4,3))
    fig.tight_layout()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    af.plot(kind='line',x='elo',y='k(30)',ax=ax,label='K=30')
    af.plot(kind='line',x='elo',y='k(20)',ax=ax,label='K=20')
    plt.ylabel('K value')
    plt.xlabel('Elo')
    plt.title('K Attenuation for High Elo Teams')
    plt.savefig('../img/att.png',dpi=300)

    # sigmoid vs erf
    def win_prob(d,s):
        return 1 / (10**(-d/s) + 1)

    def sig(d,s):
        return 2 / (10**(-d/s) + 1) - 1

    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    rng = np.arange(-5,5,0.1)
    ax.plot(rng,[sig(x,1) for x in rng], label = '2 / (10^-x) + 1) - 1')
    ax.plot(rng,erf(rng), label = 'erf(x)',alpha=0.7)
    ax.legend(loc=6)
    plt.title('Sigmoid is a close approximation of erf')
    fig.tight_layout()
    plt.savefig('../img/sig-erf.png',dpi=300)

    # setting w90 param
    deltas = np.arange(-500,500,10)
    wp1 = [win_prob(d,200) for d in deltas]
    wp2 = [win_prob(d,300) for d in deltas]
    wp3 = [win_prob(d,400) for d in deltas]
    delf = pd.concat(
        [pd.Series(deltas,name='delta'),
        pd.Series(wp1,name='p_200'),
        pd.Series(wp2,name='p_300'),
        pd.Series(wp3,name='p_400'),
        pd.Series(erf(deltas),name='erf')],axis=1)
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    delf.plot(kind='line',x='delta',y='p_200',label='w90(200)',ax=ax)
    delf.plot(kind='line',x='delta',y='p_300',ax=ax, label='w90(300)')
    delf.plot(kind='line',x='delta',y='p_400',ax=ax, label='w90(400)')
    plt.plot([-500,400],[0.9,0.9],'--k')
    plt.plot([400,400],[0,0.9],'--k')
    plt.ylabel('Team A Win Probability')
    plt.xlabel('Difference in Elo')
    plt.title('Setting w90 in Elo Win Probability')
    ax.legend(loc=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    plt.savefig('../img/win_prob.png',dpi=300)
