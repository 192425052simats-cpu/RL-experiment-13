import random

# -------------------------------
# Tic-Tac-Toe Environment
# -------------------------------

def empty_board():
    return [0] * 9   # 0 = empty, 1 = agent (X), -1 = opponent (O)

def available_actions(board):
    return [i for i in range(9) if board[i] == 0]

def check_winner(board):
    win_positions = [
        (0,1,2),(3,4,5),(6,7,8),
        (0,3,6),(1,4,7),(2,5,8),
        (0,4,8),(2,4,6)
    ]
    for a, b, c in win_positions:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    if 0 not in board:
        return 0   # draw
    return None    # game continues

# -------------------------------
# SARSA Agent
# -------------------------------

Q = {}   # Q-table dictionary

alpha = 0.1     # learning rate
gamma = 0.9     # discount factor
epsilon = 1.0   # exploration rate
epsilon_decay = 0.999
min_epsilon = 0.01

def get_Q(state, action):
    return Q.get((tuple(state), action

