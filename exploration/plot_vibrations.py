# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 09:32:36 2023

Content of log file:
mu=self.mu
mu_dot=self.mu_dot
actions=self.actions
actions_dot=self.actions_dot,
angles=self.angles,
est_angles=self.est_angles,
target_pos=self.target_pos,
est_target_pos=self.est_target_pos,
hand_pos=self.hand_pos,
est_hand_pos=self.est_hand_pos,
mode=self.mode, 
int_errors=self.int_errors,
config=log_start

@author: cqw485
"""

import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import config as c

# make figure of errors across trials
fig, ax = plt.subplots(3, 1, figsize=(15, 15))

fn = 'simulation/data/log_blind-vibrations10_incl-noise.npz'

# Load log
log = np.load(fn)

# get final positions for target and hand
target = log['target_pos'][:,-1,:]
hand = log['hand_pos'][:,-1,:]
belief_target = log['est_target_pos'][:,-1,:]
belief_hand = log['est_hand_pos'][:,-1,:]

# calculate Euclidean distance
error = np.linalg.norm(target - hand, axis=1)
error_target_belief = np.linalg.norm(target - belief_target, axis=1)
error_hand_belief = np.linalg.norm(hand - belief_hand, axis=1)

data = pd.DataFrame(error, columns=['Error'])
data['Target belief error'] = error_target_belief
data['Hand belief error'] = error_hand_belief

#sns.lineplot(data=data, y='Error', x=data.index, ax=ax[0])
sns.lineplot(data=data.iloc[:20, :], ax=ax[0])
sns.lineplot(data=data.iloc[20:40, :], ax=ax[1])
sns.lineplot(data=data.iloc[40:, :], ax=ax[2])
#plt.axhline(y=10, c='r')
#ax.title.set_text('Alpha = {}'.format(int(val)/10))

#plt.title = 'Errors in reaching, target- and hand belief across various alpha values'
#plt.savefig('simulation/data/Errors_across-alpha_beta_01.png', bbox_inches='tight')


#%%

def frame_adjust(df):
    df.columns = ['S'+ str(i) for i in df.columns]
    df['Trial'] = df.index.astype(int)
    df = pd.melt(df, id_vars='Trial')
    return df

# get final positions for target and hand
target = log['target_pos']
hand = log['hand_pos']
belief_target = log['est_target_pos']
belief_hand = log['est_hand_pos']

# calculate Euclidean distance
errors = frame_adjust(pd.DataFrame(np.linalg.norm(target - hand, axis=2)))
errors_tb = frame_adjust(pd.DataFrame(np.linalg.norm(target - belief_target, axis=2)))
errors_hb = frame_adjust(pd.DataFrame(np.linalg.norm(hand - belief_hand, axis=2)))

errors['error'] = errors['value']
errors['tb'] = errors_tb['value']
errors['hb'] = errors_hb['value']
errors['condition'] = 'BL'

errors.loc[(errors['Trial'] > 20) & (errors['Trial'] < 40), 'condition'] = 'Vibration noise (-)'
errors.loc[errors['Trial'] > 40, 'condition'] = 'Vibration noise (+)'

# make figure of errors throughout individual trials
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

sns.lineplot(data=errors[errors['Trial'] < 20], x='variable', y='error', ax=ax[0], estimator=np.mean)
ax[0].set_ylim(0, 60)
sns.lineplot(data=errors[(errors['Trial'] > 20) & (errors['Trial'] < 40)], x='variable', y='error', ax=ax[1], estimator=np.mean)
ax[1].set_ylim(0, 60)
sns.lineplot(data=errors[errors['Trial'] > 40], x='variable', y='error', ax=ax[2], estimator=np.mean)
ax[2].set_ylim(0, 60)

#plt.title = 'Errors in reaching, target- and hand belief across various alpha values'
#plt.savefig('simulation/data/Errors-across-steps_across-alpha_beta_01.png', bbox_inches='tight')

# make figure of errors throughout individual trials
fig = plt.figure(figsize=(15, 15))

sns.lineplot(data=errors, x='variable', y='error', hue='condition', estimator=np.mean)


#%%

final_errors = errors[errors['variable'] == 'S499']

# make figure of errors throughout individual trials
fig = plt.figure(figsize=(10, 8))

sns.barplot(x='alpha', y='value', data=final_errors)
plt.savefig('simulation/data/Final-errors_across-alpha_beta_01.png', bbox_inches='tight')

#%%
import pickle

with open('simulation/data/config_alpha_10_beta_01.pkl', 'rb') as f:
    log_start = pickle.load(f)
     
#print(log_start)
log_start['w_vel']
