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


# Init number of experiments, reward array, and gamma
experiments = 10
rewards = []
gamma = 0.8


for col in range(col_size):
    for row in range(row_size-1, -1, -1):
        if row == 1 and col == 1: continue #  On Rock
        if row == 0 and col == 3: continue #  On +1 terminal state
        if row == 1 and col == 3: continue #  On -1 terminal state

        r = row # init row to traverse
        c = col # init col to traverse
        state = policy[r][c] # starting state
        for _ in range(experiments):
            reward = 0 # Init reward
            time_step = 1 # Init time step
            while True:
                # Winning terminal state is obtained
                if state == 1:
                    reward += 1 * gamma ** time_step
                    break
                # Losing terminal state is obtained
                if state == -1:
                    reward -= 1 * gamma ** time_step
                    break

                # Non Terminal state
                reward -= 0.04 * gamma ** time_step

                # Choose a direction
                chosen_direction = random.choices(keys, weights=probabilities, k=1)[0]

                # Find next state given the next chosen direction
                if state == 'U':
                    if chosen_direction == 'target_up' and r > 0: r-=1
                    if chosen_direction == 'target_down' and r < row_size-1: r+=1
                    if chosen_direction == 'target_left' and c > 0: c-=1
                    if chosen_direction == 'target_right' and c < col_size-1: c+=1
                if state == 'D':
                    if chosen_direction == 'target_up' and r < row_size-1: r+=1
                    if chosen_direction == 'target_down' and r > 0: r-=1
                    if chosen_direction == 'target_left' and c < col_size-1: c+=1
                    if chosen_direction == 'target_right' and c > 0: c-=1
                if state == 'L':
                    if chosen_direction == 'target_up' and c > 0: c-=1
                    if chosen_direction == 'target_down' and c < col_size-1: c+=1
                    if chosen_direction == 'target_left' and r < row_size-1: r+=1
                    if chosen_direction == 'target_right' and r > 0: r-=1
                if state == 'R':
                    if chosen_direction == 'target_up' and c < col_size: c+=1
                    if chosen_direction == 'target_down' and c > 0: c-=1
                    if chosen_direction == 'target_left' and r > 0: r-=1
                    if chosen_direction == 'target_right' and r < row_size-1: r+=1

                # Update the time step
                time_step += 1

                # If we run into the rock, we stay in the same state
                if policy[r][c] == ' ':
                    continue
                else:
                    state = policy[r][c]

            # Collect all rewards into the rewards array
            rewards.append(reward)

        # Print Results
        print("(", col+1,",", row_size-row, ") ", np.average(rewards), sep="")