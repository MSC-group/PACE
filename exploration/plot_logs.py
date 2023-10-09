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
fig = plt.figure(figsize=(15, 15))

all_files = sorted(glob.glob('simulation/data/log_alpha_[0-9][0-9]_beta_01.npz'))
ncols = 2
nrows = len(all_files) // ncols + (len(all_files) % ncols > 0)

# load data per logfile
for n, log_name in enumerate(all_files):
    # Load log
    log = np.load(log_name)
    print(log_name)
    val = re.findall(r'log_alpha_(.*)_beta_01.npz', log_name)[0]
    
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
    
    ax = plt.subplot(nrows, ncols, n + 1, ylim=[-10, 105])
    sns.lineplot(data=data, ax=ax)
    plt.axhline(y=10, c='r')
    ax.title.set_text('Alpha = {}'.format(int(val)/10))
    
plt.title = 'Errors in reaching, target- and hand belief across various alpha values'
plt.savefig('simulation/data/Errors_across-alpha_beta_01.png', bbox_inches='tight')


#%%

def frame_adjust(df, val):
    df.columns = ['S'+ str(i) for i in df.columns]
    df['Trial'] = df.index.astype(str)
    df = pd.melt(df, id_vars='Trial')
    df['alpha'] = val
    return df


all_files = sorted(glob.glob('simulation/data/log_alpha_[0-9][0-9]_beta_01.npz'))
ncols = 2
nrows = len(all_files) // ncols + (len(all_files) % ncols > 0)

errors = pd.DataFrame([])
errors_tb = pd.DataFrame([])
errors_hb = pd.DataFrame([])

for n, log_name in enumerate(all_files):
    # Load log
    log = np.load(log_name)
    val = re.findall(r'log_alpha_(.*)_beta_01.npz', log_name)[0]
    
    # get final positions for target and hand
    target = log['target_pos']
    hand = log['hand_pos']
    belief_target = log['est_target_pos']
    belief_hand = log['est_hand_pos']
    
    # calculate Euclidean distance
    error = frame_adjust(pd.DataFrame(np.linalg.norm(target - hand, axis=2)), val)
    error_target_belief = frame_adjust(pd.DataFrame(np.linalg.norm(target - belief_target, axis=2)), val)
    error_hand_belief = frame_adjust(pd.DataFrame(np.linalg.norm(hand - belief_hand, axis=2)), val)
    
    errors = pd.concat([errors, error])
    errors_tb = pd.concat([errors_tb, error_target_belief])
    errors_hb = pd.concat([errors_hb, error_hand_belief])

# make figure of errors throughout individual trials
fig = plt.figure(figsize=(15, 15))

sns.relplot(
    data=errors_hb, x="variable", y="value",
    col="alpha", kind="line")

#sns.lineplot(data=error, x='variable', y='value', ax=ax, estimator='mean')
ax.title.set_text('Alpha = {}'.format(int(val)/10))

plt.title = 'Errors in reaching, target- and hand belief across various alpha values'

plt.savefig('simulation/data/Errors-hb-across-steps_across-alpha_beta_01.png', bbox_inches='tight')

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
