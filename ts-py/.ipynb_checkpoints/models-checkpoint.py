import pandas as pd
from copy import copy, deepcopy
from utils import ts_win_probability
np = pd.np

class Elo:

    def __init__(self,k_func,tourney=None):
        self.k_func = k_func
        self.tourney = tourney

    def update(self,elo_win,elo_loss,score_win,score_loss):
        wp = self.calc_win_prob(elo_win,elo_loss)
        # Expectation for win =1, loss = 0
        new_elo_win = elo_win + self.k_func(True,elo_win,elo_loss,score_win,score_loss)*(1-wp)
        new_elo_loss = elo_loss + self.k_func(False,elo_win,elo_loss,score_win,score_loss)*(0-wp)
        return (new_elo_win, new_elo_loss)

    def calc_win_prob(self,elo0,elo1):
        return 1 / (10**(-(elo0-elo1)/200) + 1)

class DynamicElo:

    def __init__(self,base_k,cutoff,reduction,mov,w90,tourney=None):
        self.base_k = base_k
        self.cutoff = cutoff
        self.reduction = reduction
        self.mov = mov
        self.w90 = w90
        self.tourney = tourney

    def attn(self,k,elo,cutoff,reduction):
        if elo >= cutoff:
            k = reduction*k
        return k

    def k_func(self,win,elo_win,elo_loss,score_win,score_loss):
        if win:
            k = self.mov(self.attn(self.base_k,elo_win,self.cutoff,self.reduction),score_win-score_loss)
        else:
            k = self.attn(self.base_k,elo_loss,self.cutoff,self.reduction)
        return k

    def update(self,elo_win,elo_loss,score_win,score_loss):
        wp = self.calc_win_prob(elo_win,elo_loss)
        # Expectation for win =1, loss = 0
        new_elo_win = elo_win + self.k_func(True,elo_win,elo_loss,score_win,score_loss)*(1-wp)
        new_elo_loss = elo_loss + self.k_func(False,elo_win,elo_loss,score_win,score_loss)*(0-wp)
        return (new_elo_win, new_elo_loss)

    def calc_win_prob(self,elo0,elo1):
        return 1 / (10**(-(elo0-elo1)/self.w90) + 1)

class Tournament:

    def __init__(self, raw_data):
        self.raw_data = raw_data
        # format columns
        self.raw_data['end_dt'] = pd.to_datetime(self.raw_data['end time'])
        self.raw_data['win_b'] = (self.raw_data['win?'] == 'W').astype('int')
        self.rating_f = pd.DataFrame(columns=['team','rating','series_out'])

    def build_tournament(self):
        '''
            tournament - list (asc. order by time) of the series played in that tournament
                    series
                        id - series id
                        ended_at - time the last match in the series ended
                        data - df of series info
                            index = team name
                            win_b = # matches won
                            elo 
            teams - list of team names (str)
        '''
        self.tournament = []
        self.teams = self.raw_data.team.unique()
        # self.players = self.raw_data[['team','player']].groupby('team')['player'].apply(set)
        # get series in order
        ordered_series = self.raw_data[['series id','end_dt']] \
            .groupby('series id',as_index=False)['end_dt'].max() \
            .sort_values('end_dt').reset_index()

        for i in range(len(ordered_series)):
            series = self.raw_data[self.raw_data['series id'] == ordered_series.loc[i,'series id']]
            match_score = series.groupby(['team','match id','win_b'],as_index=False)['team'].max().reset_index() \
                .groupby('team',as_index=False)['win_b'].sum()
            match_score['rating'] = None
            self.tournament.append(
                { 
                    # add field for players 
                    'id' : ordered_series.loc[i,'series id'],
                    'ended_at' : ordered_series.loc[i,'end_dt'],
                    'data' : match_score  
                }
            )
        return self.tournament

    def set_first_series_rating(self,rating_i,env):
        _teams = list(self.teams)
        while len(_teams) > 0:
            # get first game for each team in tournament
            teams = list(filter(lambda s: _teams[0] in s['data'].team.unique(),self.tournament))
            # get the index of the first time we see that team
            t_idx = next((i for (i, d) in enumerate(self.tournament) if d['id'] == teams[0]['id']), None)
            # try finding rating, if not set it to average (maybe try bottom 1/8)
            try:
                rating = rating_i.loc[_teams[0],'rating']
            except KeyError:
                # use rating of bottom 10%
                # use z score from norm dist to pick this
                rating = env.Rating(mu=(env.mu - 0.84*env.sigma),sigma=env.sigma)
            tdata = self.tournament[t_idx]['data']
            tdata.at[tdata[tdata.team == _teams[0]].index.item(),'rating'] = rating
            # remove that team from the list
            _teams.pop(0)
        return self.tournament

    def populate_next_series(self):
        i = 0
        _tournament = list(self.tournament)
        while len(_tournament) > 1:
            i += 1
            series = self.tournament[self.tournament.index(_tournament[0])]['data']
            winner = series[series.win_b==series.win_b.max()]
            loser = series[series.win_b==series.win_b.min()]
            _tournament.pop(0)
            w_next_idx = next((i for i,s in enumerate(_tournament) if winner.team.item() in list(s['data'].team)),None)
            l_next_idx = next((i for i,s in enumerate(_tournament) if loser.team.item() in list(s['data'].team)),None)
            self.tournament[i-1]['w_ns'] = None
            self.tournament[i-1]['l_ns'] = None
            if w_next_idx is not None:
                self.tournament[i-1]['w_ns'] = self.tournament[w_next_idx+i]['id']
            if l_next_idx is not None:
                self.tournament[i-1]['l_ns'] = self.tournament[l_next_idx+i]['id']
        return self.tournament    

    def play(self,env,acc=False):
        # could parallelize group play
        i = 0
        _tournament = list(self.tournament)
        while len(_tournament) > 0:
            i += 1
            sdata = self.tournament[self.tournament.index(_tournament[0])]
            series = sdata['data']      
            if acc:
                wp0 = ts_win_probability([series.at[0,'rating']],[series.at[1,'rating']],env)
                t0_win = series.at[0,'win_b'] > series.at[1,'win_b']
                sdata['max_wp'] = max([wp0,1-wp0])
                if t0_win and wp0 > 0.5 or not t0_win and wp0 < 0.5:
                    sdata['correct'] = 1
                else:
                    sdata['correct'] = 0
            winner = series[series.win_b==series.win_b.max()]
            loser = series[series.win_b==series.win_b.min()]
            new_rating = env.rate_1vs1(winner.rating.item(),loser.rating.item())
            _tournament.pop(0)
            w_next_idx = next((i for i,s in enumerate(_tournament) if s['id'] == sdata['w_ns']),None)
            l_next_idx = next((i for i,s in enumerate(_tournament) if s['id'] == sdata['l_ns']),None)
            if w_next_idx is not None:
                next_data = self.tournament[w_next_idx+i]['data']
                next_data.at[next_data[next_data.team==winner.team.item()].index[0],'rating'] = new_rating[0]
            else: 
                self.rating_f = self.rating_f.append(pd.DataFrame(columns=['team','rating','series_out'],data=[[winner.team.item(),new_rating[0],sdata['id']]]))     
            if l_next_idx is not None:
                next_data = self.tournament[l_next_idx+i]['data']
                next_data.at[next_data[next_data.team==loser.team.item()].index[0],'rating'] = new_rating[1]
            else: 
                self.rating_f = self.rating_f.append(pd.DataFrame(columns=['team','rating','series_out'],data=[[loser.team.item(),new_rating[1],sdata['id']]]))
        return

    def elos_for_team(self,team):
        out = []
        all_series = list(filter(lambda s: team in s['data'].team.unique(),self.tournament))
        all_series.sort(key=lambda x: x['ended_at'])
        if len(all_series) == 0:
            return pd.DataFrame(columns=['team','elo','series','win'],data=out)
        for series in all_series:
            team_row = series['data'][series['data'].team==team]
            out.append([
                team_row.team.item(),
                team_row.elo.item(),
                series['id'],
                team_row.win_b.item() > series['data'][series['data'].team!=team].win_b.item()])
        eloft = self.elo_f[self.elo_f.team==team]
        out.append([eloft.team.item(),eloft.elo.item(),eloft.series_out.item(),None])
        return pd.DataFrame(columns=['team','elo','series','win'],data=out)
    
    def final_ratings(self):
        out = []
        for team in self.raw_data.team.unique():
            all_series = list(filter(lambda s: team in s['data'].team.unique(),self.tournament))
            all_series.sort(key=lambda x: x['ended_at'])
            last = all_series[-1]['data']
            out.append((team,last[last.team==team].rating.item()))
        out_f = pd.DataFrame(data=out,columns=['team','rating'])
        out_f.index = out_f.team
        return out_f

    # player based stuff
    def build_for_players(self):
        self.tournament = []
        ordered_series = self.raw_data[['series id','end_dt']] \
            .groupby('series id',as_index=False)['end_dt'].max() \
            .sort_values('end_dt').reset_index()
        team_players = self.raw_data.sort_values('end_dt')[['team','player','win_b','series id','end_dt']] \
            .groupby(['end_dt','series id','win_b','team'])['player'].apply(set).reset_index()
        for s in ordered_series['series id'].unique(): 
            sdata = team_players[team_players['series id']==s]
            series = {'id' : s}
            series['ended_at'] = sdata['end_dt'].max()
            score = sdata.groupby('team')['win_b'].sum().reset_index()
            winner = score[score['win_b']==score.win_b.max()]
            loser = score[score['win_b']==score.win_b.min()]
            series['data'] = {
                'winner' : {
                    'team' : winner.team.item(),
                    'score' : winner.win_b.item(),
                    'players': set.union(*sdata[sdata['team']==winner.team.item()].player)
                },
                'loser' : {
                    'team' : loser.team.item(),
                    'score' : loser.win_b.item(),
                    'players' : set.union(*sdata[sdata['team']==loser.team.item()].player)
                }
                
            }
            self.tournament.append(series)
        return self.tournament

    def populate_next_series_players(self):
        # pretty much the same, just switch to DICT syntax for queries
        i = 0
        _tournament = list(self.tournament)
        while len(_tournament) > 1:
            i += 1
            series = self.tournament[self.tournament.index(_tournament[0])]['data']
            winner = series['winner']
            loser = series['loser']
            _tournament.pop(0)
            w_next_idx = next((i for i,s in enumerate(_tournament) if winner['team'] in [v['team'] for k,v in s['data'].items()]),None)
            l_next_idx = next((i for i,s in enumerate(_tournament) if loser['team'] in [v['team'] for k,v in s['data'].items()]),None)
            self.tournament[i-1]['w_ns'] = None
            self.tournament[i-1]['l_ns'] = None
            if w_next_idx is not None:
                self.tournament[i-1]['w_ns'] = self.tournament[w_next_idx+i]['id']
            if l_next_idx is not None:
                self.tournament[i-1]['l_ns'] = self.tournament[l_next_idx+i]['id']
        return self.tournament  

    def set_initial_player_ratings(self,env):
        # build frame of players
        # set to mean
        names = self.raw_data.player.unique()
        ratings = np.full((1,len(names)),env.Rating())[0]
        self.players = pd.DataFrame(columns=['player','rating'],data=np.array([names,ratings]).T)

    def play_with_players(self):
        # instead of storing ratings with players, query knowledge source of player rratings
        pass

class Simulation: 

    def __init__(self,tourney):
        '''
            takes a single tourney in initialized state and plays it out
        '''
        self.tournament = self._clear_tournament(tourney.tournament)

    def _clear_tournament(self,tournament):
        '''
            removes team names from all teams don't have elo
        '''
        for i in range(len(tournament)):
            tdata = tournament[i]['data']
            # if any of the teams have no elo
            if False in list(tdata.elo.notnull()):
                # set both teams win count to null
                tdata.win_b = None
                # set just the null teams name to null
                if tdata.at[0,'elo'] is None:
                    tdata.at[0,'team'] = None
                if tdata.at[1,'elo'] is None:
                    tdata.at[1,'team'] = None
                tdata[tdata.elo.isnull()].team = None
        return tournament
    
    def run(self,elo):
        '''
            call tournament.play() using sample from distribution using calc_win_prob
        '''
        i = 0
        _tournament = list(self.tournament)
        while len(_tournament) > 0:
            i += 1
            sdata = self.tournament[self.tournament.index(_tournament[0])]
            series = sdata['data']
            # calc win probability 
            t0 = series.iloc[0]
            t1 = series.iloc[1]
            p_t0 = elo.calc_win_prob(t0.elo,t1.elo)
            # create distribution, sample it
            t0_win = np.random.binomial(1,p_t0)
            if t0_win:
                winner = t0; loser = t1
            else:  
                winner = t1; loser = t0
            series.at[series[series.team==winner.team].index.item(),'win_b'] = 1
            series.at[series[series.team==loser.team].index.item(),'win_b'] = 0
            _tournament.pop(0)
            if len(_tournament) == 0: return 
            w_next_idx = next((i for i,s in enumerate(_tournament) if s['id'] == sdata['w_ns']),None)
            l_next_idx = next((i for i,s in enumerate(_tournament) if s['id'] == sdata['l_ns']),None)
            new_elo = elo.update(winner.elo,loser.elo,winner.win_b,loser.win_b)
            # print(sdata,'\n')
            # print('\n\n',w_next_idx+i,l_next_idx+i,'\n\n')
            if w_next_idx is not None:
                w_next_data = self.tournament[w_next_idx+i]['data']
                # print('\n\n',w_next_data[w_next_data.team.isnull()],'\n\n')
                next_idx = w_next_data[w_next_data.team.isnull()].index[0]
                w_next_data.at[next_idx,'elo'] = new_elo[0]
                w_next_data.at[next_idx,'team'] = winner.team
            if l_next_idx is not None:
                l_next_data = self.tournament[l_next_idx+i]['data']
                # print('\n\n',self.tournament[l_next_idx+i],'\n\n')
                next_idx = l_next_data[l_next_data.team.isnull()].index[0]
                l_next_data.at[next_idx,'elo'] = new_elo[1]
                l_next_data.at[next_idx,'team'] = loser.team
        return self.tournament    