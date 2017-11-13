import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from .replay import ReplayMemory

last_sync = 0


def optimize_model(optimizer, memory, model, BATCH_SIZE, GAMMA, target=None, train_target=False,
        test=False, freeze_micro=False,
        freeze_macro=False):
    global last_sync
    if len(memory) < BATCH_SIZE:
        return

    # random idea
    for i, param in enumerate(model.parameters()):
        if test and i == 2:
            if freeze_micro:
                param.requires_grad = False
            elif freeze_macro:
                param.requires_grad = True
            else:
                param.requires_grad = True
        if test and i == 3:
            if freeze_micro:
                param.requires_grad = True
            elif freeze_macro:
                param.requires_grad = False
            else:
                param.requires_grad = True


    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = ReplayMemory.Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.cuda.ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(torch.cuda.FloatTensor))
    if target is None:
        next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    else:
        next_state_values[non_final_mask] = target(non_final_next_states).max(1)[0]

    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()


    for i, param in enumerate(model.parameters()):
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1)
    optimizer.step()

    if train_target and target is not None:
        target.load_state_dict(model.state_dict())
    return loss
