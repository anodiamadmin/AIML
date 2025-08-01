"""
   encoder.py
   ZZEN9444, CSE, UNSW
"""

import torch

# REPLACE aus26 WITH YOUR OWN DATA
# TO REPRODUCE IMAGE SHOWN IN SPEC

# aus26 = torch.Tensor([[0]])

star16 = torch.Tensor(
    [[1,1,0,0,0,0,0,0],
     [0,1,1,0,0,0,0,0],
     [0,0,1,1,0,0,0,0],
     [0,0,0,1,1,0,0,0],
     [0,0,0,0,1,1,0,0],
     [0,0,0,0,0,1,1,0],
     [0,0,0,0,0,0,1,1],
     [1,0,0,0,0,0,0,1],
     [1,1,0,0,0,0,0,1],
     [1,1,1,0,0,0,0,0],
     [0,1,1,1,0,0,0,0],
     [0,0,1,1,1,0,0,0],
     [0,0,0,1,1,1,0,0],
     [0,0,0,0,1,1,1,0],
     [0,0,0,0,0,1,1,1],
     [1,0,0,0,0,0,1,1]])

aus26 = torch.Tensor([
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # bottom-left anchor
  [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1],  # top-left anchor
  [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],  # bottom-right anchor
  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],  # top-right anchor
  [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],  # mid-left edge anchor (0, 6)
  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],  # mid-right edge anchor (10, 5)
  [1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0],  # WA  (4, 9) Pt#1
  [1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0],  # NT  (5, 9) Pt#2
  [1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0],  # NT  (5, 8) Pt#3
  [1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,0,0],  # NT  (6, 8) Pt#4
  [1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,0],  # QLD (7, 9) Pt#5
  [1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0],  # QLD (8, 8) Pt#6
  [1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,0,0],  # QLD (9, 7) Pt#7
  [1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,0,0,0],  # QLD (9, 6) Pt#8
  [1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0],  # NSW (9, 5) Pt#9
  [1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,0,0,0,0,0],  # NSW (8, 4) Pt#10
  [1,1,1,1,1,1,1,0,0,0,1,1,1,1,0,0,0,0,0,0],  # VIC (7, 4) Pt#11
  [1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0],  # SA  (6, 5) Pt#12
  [1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],  # SA  (5, 5) Pt#13
  [1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0],  # WA  (4, 4) Pt#14
  [1,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0],  # WA  (3, 4) Pt#15
  [1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],  # WA  (2, 5) Pt#16
  [1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],  # WA  (2, 6) Pt#17
  [1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0],  # WA  (2, 7) Pt#18
  [1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0],  # WA  (3, 8) Pt#19
  [1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0]   # TAS (8, 2) Pt#20
])