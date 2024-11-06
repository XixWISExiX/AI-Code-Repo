import numpy as np
import matplotlib.pyplot as plt


# Location equals that of Figure 1: Grid World
policy = [['L', 'D', 'R',  1],
          ['U', ' ', 'L', -1], 
          ['R', 'U', 'L', 'L']]


# Init Values for each non terminal state to 0 (random enough)
values = [[0,   0,  0,  1],
          [0, ' ',  0, -1], 
          [0,   0,  0,  0]]


# Init row and col sizes
row_size = len(policy)
col_size = len(policy[0])

# Init gamma
gamma = 0.8

# Number of iterations
iterations = 10000

for _ in range(iterations):
    for col in range(col_size):
        for row in range(row_size-1, -1, -1):
            if row == 1 and col == 1: continue #  On Rock
            if row == 0 and col == 3: continue #  On +1 terminal state
            if row == 1 and col == 3: continue #  On -1 terminal state

            neighboring_states = []
            if row < row_size-1 and values[row+1][col] != ' ': neighboring_states.append(values[row+1][col]) # Add potential above neighbors
            if row > 0 and values[row-1][col] != ' ': neighboring_states.append(values[row-1][col]) # Add potential below neighbors
            if col < col_size-1 and values[row][col+1] != ' ': neighboring_states.append(values[row][col+1]) # Add potential above neighbors
            if col > 0 and values[row][col-1] != ' ': neighboring_states.append(values[row][col-1]) # Add potential below neighbors

            values[row][col] = -0.04 + gamma * np.max(neighboring_states) # Bellman's Equation

print('(1, 1)', round(values[2][0], 3))
print('(1, 2)', round(values[1][0], 3))
print('(1, 3)', round(values[0][0], 3))

print('(2, 1)', round(values[2][1], 3))
print('(2, 3)', round(values[0][1], 3))

print('(3, 1)', round(values[2][2], 3))
print('(3, 2)', round(values[1][2], 3))
print('(3, 3)', round(values[0][2], 3))

print('(4, 1)', round(values[2][3], 3))


# Convert the matrix to a NumPy array, with ' ' replaced by NaN for easier handling
values_array = np.array([[0 if x == ' ' else x for x in row] for row in values], dtype=float)
values_array[[0, -1]] = values_array[[-1, 0]]

# Create the plot
plt.figure(figsize=(6, 6))
plt.imshow(values_array, cmap="coolwarm", interpolation='none', alpha=0.8)
plt.colorbar(label="Values")

# Add the values to each cell
for i in range(values_array.shape[0]):
    for j in range(values_array.shape[1]):
        value = values[row_size-1-i][j]
        # value = values[i][j]
        if not isinstance(value, str):
            value = round(value, 3)            
        plt.text(j, i, str(value), ha='center', va='center', color='black', fontsize=12)

# Customize the plot appearance
plt.xticks(ticks=np.arange(col_size), labels=np.arange(1, col_size + 1))
plt.yticks(ticks=np.arange(row_size), labels=np.arange(1, row_size + 1))
plt.gca().invert_yaxis()
plt.title("Bellman's Equation Values for Grid World")
plt.show()