import random
import numpy as np

# Location equals that of Figure 1: Grid World
policy = [['L', 'D', 'R',  1],
          ['U', ' ', 'L', -1], 
          ['R', 'U', 'L', 'L']]

# Probabilities of the target directions
direction = {
    'target_up': 0.6,
    'target_down': 0.1,
    'target_left': 0.2,
    'target_right': 0.1
    }

# Keys and probabilites to find next move
keys = list(direction.keys())
probabilities = list(direction.values())

# Init row and col sizes
row_size = len(policy)
col_size = len(policy[0])

#TODO Look at task 2