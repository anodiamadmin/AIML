"""
   frac.py
   ZZEN9444, CSE, UNSW
"""

import torch
import torch.nn as nn

class Full3Net(nn.Module):
    def __init__(self, hid):
        super(Full3Net, self).__init__()
        self.hid_dim = hid

        self.fc1 = nn.Linear(2, hid)      # input (x, y) → hid
        self.fc2 = nn.Linear(hid, hid)              # hid → hid
        self.fc3 = nn.Linear(hid, 1)     # hid → output (1)

    def forward(self, input):
        self.hid1 = torch.tanh(self.fc1(input))     # first hidden layer
        self.hid2 = torch.tanh(self.fc2(self.hid1)) # second hidden layer
        output = torch.sigmoid(self.fc3(self.hid2)) # output layer
        return output


class Full4Net(nn.Module):
    def __init__(self, hid):
        super(Full4Net, self).__init__()
        self.fc1 = nn.Linear(2, hid)      # Input to Hidden Layer 1
        self.fc2 = nn.Linear(hid, hid)    # Hidden Layer 1 to Hidden Layer 2
        self.fc3 = nn.Linear(hid, hid)    # Hidden Layer 2 to Hidden Layer 3
        self.fc4 = nn.Linear(hid, 1)      # Hidden Layer 3 to Output

    def forward(self, x):
        # Hidden layers with tanh activation
        self.hid1 = torch.tanh(self.fc1(x))
        self.hid2 = torch.tanh(self.fc2(self.hid1))
        self.hid3 = torch.tanh(self.fc3(self.hid2))
        # Output layer with sigmoid activation
        out = torch.sigmoid(self.fc4(self.hid3))
        return out

class DenseNet(nn.Module):
    def __init__(self, hid: int):
        super().__init__()
        self.hid = hid
        # x ∈ R^2
        self.fc1 = nn.Linear(2, hid)            # W10, b1
        self.fc2 = nn.Linear(2 + hid, hid)      # [W20 | W21], b2
        self.fc3 = nn.Linear(2 + hid + hid, 1)  # [W30 | W31 | W32], b_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h1
        self.hid1 = torch.tanh(self.fc1(x))
        # h2 uses both x and h1
        h2_in = torch.cat([x, self.hid1], dim=1)
        self.hid2 = torch.tanh(self.fc2(h2_in))
        # out uses x, h1, h2
        out_in = torch.cat([x, self.hid1, self.hid2], dim=1)
        out = torch.sigmoid(self.fc3(out_in))
        return out
