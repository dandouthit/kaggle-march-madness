# # Elo ratings based on regular-season games

# This notebook implements Elo ratings for NCAA regular-season games using the same formula as FiveThirtyEight's NBA Elo ratings. My resources for this were:
# 
# - https://en.wikipedia.org/wiki/Elo_rating_system
# - https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/
# - https://github.com/fivethirtyeight/nfl-elo-game/blob/master/forecast.py
# 
# (The last link above is for 538's NFL Elos (not NBA), but it was useful for a code example of the approach. )
# 
# The idea here is to get another feature to be plugged in (alongside seeds, etc.) when predicting tournament games.

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

K = 20.
HOME_ADVANTAGE = 100.

def elo_pred(elo1, elo2):
    return(1. / (10. ** (-(elo1 - elo2) / 400.) + 1.))

def expected_margin(elo_diff):
    return((7.5 + 0.006 * elo_diff))

def elo_update(w_elo, l_elo, margin):
    elo_diff = w_elo - l_elo
    pred = elo_pred(w_elo, l_elo)
    mult = ((margin + 3.) ** 0.8) / expected_margin(elo_diff)
    update = K * mult * (1 - pred)
    return(pred, update)
	
def final_elo_per_season(df, team_id):
    d = df.copy()
    d = d.loc[(d.WTeamID == team_id) | (d.LTeamID == team_id), :]
    d.sort_values(['Season', 'DayNum'], inplace=True)
    d.drop_duplicates(['Season'], keep='last', inplace=True)
    w_mask = d.WTeamID == team_id
    l_mask = d.LTeamID == team_id
    d['season_elo'] = None
    d.loc[w_mask, 'season_elo'] = d.loc[w_mask, 'w_elo']
    d.loc[l_mask, 'season_elo'] = d.loc[l_mask, 'l_elo']
    out = pd.DataFrame({
        'team_id': team_id,
        'season': d.Season,
        'season_elo': d.season_elo
    })
    return(out)
	
	
rs = pd.read_csv("../datasets/womens-machine-learning-competition-2018/WRegularSeasonCompactResults.csv")

team_ids = set(rs.WTeamID).union(set(rs.LTeamID))
len(team_ids)

# This dictionary will be used as a lookup for current
# scores while the algorithm is iterating through each game
elo_dict = dict(zip(list(team_ids), [1500] * len(team_ids)))

# New columns to help us iteratively update elos
rs['margin'] = rs.WScore - rs.LScore
rs['w_elo'] = None
rs['l_elo'] = None



assert np.all(rs.index.values == np.array(range(rs.shape[0]))), "Index is out of order."

preds = []

# Loop over all rows of the games dataframe
for i in range(rs.shape[0]):
    
    # Get key data from current row
    w = rs.at[i, 'WTeamID']
    l = rs.at[i, 'LTeamID']
    margin = rs.at[i, 'margin']
    wloc = rs.at[i, 'WLoc']
    
    # Does either team get a home-court advantage?
    w_ad, l_ad, = 0., 0.
    if wloc == "H":
        w_ad += HOME_ADVANTAGE
    elif wloc == "A":
        l_ad += HOME_ADVANTAGE
    
    # Get elo updates as a result of the game
    pred, update = elo_update(elo_dict[w] + w_ad,
                              elo_dict[l] + l_ad, 
                              margin)
    elo_dict[w] += update
    elo_dict[l] -= update
    preds.append(pred)

    # Stores new elos in the games dataframe
    rs.loc[i, 'w_elo'] = elo_dict[w]
    rs.loc[i, 'l_elo'] = elo_dict[l]

np.mean(-np.log(preds))

df_list = [final_elo_per_season(rs, i) for i in team_ids]
season_elos = pd.concat(df_list)

season_elos.to_csv("updated_season_elos_women.csv", index=None)

