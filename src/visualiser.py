import numpy as np
import h5py
from visdom import Visdom
import time

viz = Visdom()

#assert viz.check_connection()

path = '../data.h5'
data = h5py.File(path, 'r', libver='latest', swmr=True)
group = data['epochs']  # I know there are 10 epochs

# include visdom window creation
reward_dset = group['rewards']
speed_dset = group['speeds']
quota_dset = group['quota_met']
losses_dset = group['losses']
overall_rewards_dset = group['aggr_reward']
quota_err_dset = group['quota_error']


# Visdom populate
reward_read_from = 0
speed_read_from = 0
quota_read_from = 0
losses_read_from = 0
overall_rewards_read_from = 0
quota_err_read_from = 0
wins = []
while True:
    try:
        reward_dset.id.refresh()
        if reward_read_from == 0:
            r_win = viz.line(
                    X=np.arange(reward_read_from, reward_dset.shape[0]),
                    Y=reward_dset[:],
                    opts=dict(title='epoch rewards')
                    )
            wins.append(r_win)
        else:
            if reward_read_from <= reward_dset.shape[0]:
                update = 'append'
            else:
                reward_read_from = 0
                update = None
            viz.line(
                    X=np.arange(reward_read_from, reward_dset.shape[0]),
                    Y=reward_dset[reward_read_from:],
                    win=wins[0],
                    update=update
                    )
        reward_read_from = reward_dset.shape[0]

        speed_dset.id.refresh()
        if speed_read_from == 0:
            s_win = viz.line(
                    X=np.arange(speed_read_from, speed_dset.shape[0]),
                    Y=speed_dset[:],
                    opts=dict(title='epoch speeds')
                    )
            wins.append(s_win)
        else:
            if speed_read_from <= speed_dset.shape[0]:
                update = 'append'
            else:
                speed_read_from = 0
                update = None
            viz.line(
                    X=np.arange(speed_read_from, speed_dset.shape[0]),
                    Y=speed_dset[speed_read_from:],
                    win=wins[1],
                    update=update
                    )
        speed_read_from = speed_dset.shape[0]

        quota_dset.id.refresh()
        if quota_read_from == 0:
            q_win = viz.line(
                    X=np.arange(quota_read_from, quota_dset.shape[0]),
                    Y=quota_dset[:],
                    opts=dict(title='epoch quotas')
                    )
            wins.append(q_win)
        else:
            if quota_read_from <= quota_dset.shape[0]:
                update = 'append'
            else:
                quota_read_from = 0
                update = None
            viz.line(
                    X=np.arange(quota_read_from, quota_dset.shape[0]),
                    Y=quota_dset[quota_read_from:],
                    win=wins[2],
                    update=update
                    )
        quota_read_from = quota_dset.shape[0]

        losses_dset.id.refresh()
        if losses_read_from == 0:
            l_win = viz.line(
                    X=np.arange(losses_read_from, losses_dset.shape[0]),
                    Y=losses_dset[:],
                    opts=dict(title='epoch losses')
                    )
            wins.append(l_win)
        else:
            if losses_read_from < losses_dset.shape[0]:
                update = 'append'
            else:
                losses_read_from = 0
                update = None
            viz.line(
                    X=np.arange(losses_read_from, losses_dset.shape[0]),
                    Y=losses_dset[losses_read_from:],
                    win=wins[3],
                    update=update
                    )
        losses_read_from = losses_dset.shape[0]

        overall_rewards_dset.id.refresh()
        if overall_rewards_read_from == 0:
            or_win = viz.line(
                    X=np.arange(overall_rewards_read_from, overall_rewards_dset.shape[0]),
                    Y=overall_rewards_dset[:],
                    opts=dict(title='overall rewards')
                    )
            wins.append(or_win)
        else:
            if overall_rewards_read_from <= overall_rewards_dset.shape[0]:
                update = 'append'
            else:
                overall_rewards_read_from = 0
                update = None
            viz.line(
                    X=np.arange(overall_rewards_read_from, overall_rewards_dset.shape[0]),
                    Y=overall_rewards_dset[overall_rewards_read_from:],
                    win=wins[4],
                    update=update
                    )
        overall_rewards_read_from = overall_rewards_dset.shape[0]

        quota_err_dset.id.refresh()
        if quota_err_read_from == 0:
            qe_win = viz.line(
                    X=np.arange(quota_err_read_from, quota_err_dset.shape[0]),
                    Y=quota_err_dset[:],
                    opts=dict(title='quota error')
                    )
            wins.append(qe_win)
        else:
            if quota_err_read_from <= quota_err_dset.shape[0]:
                update = 'append'
            else:
                quota_err_read_from = 0
                update = None
            viz.line(
                    X=np.arange(quota_err_read_from, quota_err_dset.shape[0]),
                    Y=quota_err_dset[quota_err_read_from:],
                    win=wins[5],
                    update=update
                    )
        quota_err_read_from = quota_err_dset.shape[0]


        time.sleep(0.5)
    except KeyboardInterrupt:
        break

data.close()
