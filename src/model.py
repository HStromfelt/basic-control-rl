import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class vanilla_Linear_Net(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(vanilla_Linear_Net, self).__init__()
        #self.fc1 = nn.Linear(num_inputs - 1, 64)
        self.fc1 = nn.Linear(num_inputs - 2, 64)
        #self.fc1_q = nn.Linear(1, 32)
        self.fc1_q = nn.Linear(2, 32)

        self.fc2_micro = nn.Linear(96, 64)
        self.fc2_macro = nn.Linear(96, 64)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_outputs)

        #self.fc3 = nn.linear(64, num_outputs)
        #self.fc1 = nn.Linear(num_inputs, num_outputs)


    def forward(self, x, debug=False):
        # forward model
        x_a = F.hardtanh(self.fc1(x[:,:-2]))
        x_b = F.hardtanh(self.fc1_q(x[:,-2:]))
        x = torch.cat((x_a, x_b), 1)

        x_micro = F.hardtanh(self.fc2_micro(x))
        x_macro = F.hardtanh(self.fc2_macro(x))

        x = torch.cat((x_micro, x_macro), 1)

        x = F.hardtanh(self.fc3(x))

        Qout = self.fc4(x)
        #Qout = self.fc3(x)



        #Qout = self.fc1(x)

        if debug:
            import pdb
            pdb.set_trace()
        return Qout

class Linear_Net(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Linear_Net, self).__init__()
        #self.fc1_time = nn.Linear(32, 128)
        self.fc1_time = nn.Linear(10, 128)
        self.fc1_system = nn.Linear(20, 128)
        self.fc1_q = nn.Linear(2, 32)

        self.fc2_groups = nn.Linear(num_inputs, num_inputs)

        self.fc2_system = nn.Linear(128, 128)
        self.fc2_time = nn.Linear(128, 128)

        self.fc2_adv = nn.Linear(256 + num_inputs + 32, 38)
        self.fc2_value = nn.Linear(256 + num_inputs + 32, 38)

        self.fc3_adv = nn.Linear(38, num_outputs)
        self.fc3_value = nn.Linear(38, num_outputs)

        #self.fc1 = nn.Linear(148, 256)
        #self.fc2= nn.Linear(256, 21)

    def forward(self, x, debug=False):
        # forward model
        #x_time = F.relu(self.fc1_time(x[:, :32]))
        x_time = F.relu(self.fc1_time(x[:, :10]))
        x_time = F.relu(self.fc2_time(x_time))

        #x_system = F.relu(self.fc1_system(x[:, 32:-2]))
        x_system = F.relu(self.fc1_system(x[:, 10:-2]))
        x_system = F.relu(self.fc2_system(x_system))

        x_q = F.relu(self.fc1_q(x[:, -2:]))

        x_grouped = F.relu(self.fc2_groups(x))

        x = torch.cat((x_time, x_system, x_grouped, x_q), 1)
        # Adv stream
        adv_x = F.relu(self.fc2_adv(x))
        adv_x = self.fc3_adv(adv_x)
        norm_adv_x = adv_x - torch.mean(adv_x)
        # V stream
        value_x = F.relu(self.fc2_value(x))
        value_x = self.fc3_value(value_x)
        # Q values
        Qout = value_x + norm_adv_x

        #x = F.relu(self.fc1(x))
        #Qout = self.fc2(x)

        if debug:
            import pdb
            pdb.set_trace()
        return Qout



