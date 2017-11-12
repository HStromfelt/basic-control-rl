import numpy as np
import random

import torch
from torch.autograd import Variable

def select_action(model, state, num_actions, eps_threshold, debug=False):
    sample = random.random()
    if sample > eps_threshold:
        return model(
            Variable(
                state,
                volatile=True
                ).type(torch.cuda.FloatTensor),
            debug=debug).data.max(1)[1].view(1, 1)
    else:
        return torch.cuda.LongTensor([[random.randrange(num_actions)]])
