"""Allow visualisation of test h5"""

import numpy as np
import h5py
from visdom import Visdom
import time

viz = Visdom()


def visualise_test(test_path, for_epoch):
    data = h5py.File(test_path, 'r', libver='latest', swmr=True)
    group = data['epochs']

    reward_dset = group['rewards']
    speed_dset = group['speeds']
    quota_dset = group['quota_met']
    quota_err_dset = group['quota_error']

    # Visdom populate
    reward_read_from = 0
    speed_read_from = 0
    quota_read_from = 0
    quota_err_read_from = 0
    wins = []
    try:
        reward_dset.id.refresh()
        if reward_read_from == 0:
            r_win = viz.line(
                    X=np.arange(reward_read_from, reward_dset.shape[0]),
                    Y=reward_dset[:],
                    opts=dict(title='Rewards - Epoch {}'.format(for_epoch))
                    )
            wins.append(r_win)

        speed_dset.id.refresh()
        if speed_read_from == 0:
            s_win = viz.line(
                    X=np.arange(speed_read_from, speed_dset.shape[0]),
                    Y=speed_dset[:],
                    opts=dict(title='Speed Schedule - Epoch {}'.format(for_epoch))
                    )
            wins.append(s_win)

        quota_dset.id.refresh()
        if quota_read_from == 0:
            q_win = viz.line(
                    X=np.arange(quota_read_from, quota_dset.shape[0]),
                    Y=quota_dset[:],
                    opts=dict(title='Quota Progression - Epoch {}'.format(for_epoch))
                    )
            wins.append(q_win)

        quota_err_dset.id.refresh()
        if quota_err_read_from == 0:
            qe_win = viz.line(
                    X=np.arange(quota_err_read_from, quota_err_dset.shape[0]),
                    Y=quota_err_dset[:],
                    opts=dict(title='Quota Offset - Epoch {}'.format(for_epoch))
                    )
            wins.append(qe_win)

    except Exception as ex:
        print('error in visualisation: \n{}'.format(ex))


if __name__ == '__main__':
    visualise_test('../tests/')
