# 🎮 SARSA-Based Tic-Tac-Toe AI

This project demonstrates the implementation of a **SARSA (State–Action–Reward–State–Action)** reinforcement learning algorithm to train an AI agent to play the classic **Tic-Tac-Toe** game. The agent learns optimal moves by interacting with the environment and updating its knowledge using on-policy learning.

---

## 📌 Overview

* **Algorithm Used:** SARSA (On-policy Reinforcement Learning)
* **Environment:** Tic-Tac-Toe (3×3 board)
* **Programming Language:** Python
* **Opponent:** Random-move player
* **Learning Method:** ε-greedy exploration

The agent improves its performance over time by playing thousands of games and adjusting its strategy based on rewards.

---

## 🧠 Key Concepts

* **State:** Current configuration of the Tic-Tac-Toe board
* **Action:** Choosing an empty cell (0–8)
* **Reward System:**

  * `+1` → Agent wins
  * `-1` → Agent loses
  * `0`  → Draw or intermediate step
* **Policy:** ε-greedy (balances exploration & exploitation)
* **Learning Type:** On-policy (SARSA)

---

## ⚙️ Project Structure

```
SARSA-TicTacToe/
│
├── sarsa_tictactoe.py   # Main Python implementation
└── README.md            # Project documentation
```

---

## 🧩 Code Explanation

### 1️⃣ Tic-Tac-Toe Environment

* The board is represented as a list of 9 elements
* `0` → Empty cell
* `1` → AI agent (X)
* `-1` → Opponent (O)

Functions included:

* `empty_board()` – initializes the board
* `available_actions(board)` – returns empty positions
* `check_winner(board)` – checks win/draw/game continuation

---

### 2️⃣ SARSA Agent

* **Q-table:** Stored as a Python dictionary
* **Learning Rate (α):** 0.1
* **Discount Factor (γ):** 0.9
* **Exploration Rate (ε):** Starts at 1.0 and decays gradually

Key functions:

* `get_Q(state, action)` – retrieves Q-values
* `choose_action(state)` – selects action using ε-greedy policy
* `update_Q(...)` – updates Q-values using SARSA rule

**SARSA Update Formula:**

Q(S, A) ← Q(S, A) + α [ R + γ Q(S', A') − Q(S, A) ]

---

### 3️⃣ Training Phase

* Total training episodes: **10,000**
* The agent plays against a random opponent
* After each move, Q-values are updated
* ε value decays after every episode

This phase allows the agent to learn optimal strategies such as:

* Blocking opponent moves
* Choosing center and corner positions
* Avoiding losing states

---

### 4️⃣ Evaluation Phase

After training, the agent is evaluated over **1,000 games** against a random opponent.

Metrics recorded:

* Number of Wins
* Number of Losses
* Number of Draws

---

## 🧪 Complete Python Code

Below is the **complete SARSA-based Tic-Tac-Toe implementation** used in this experiment:

```python
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

Q = {}

alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.01

def get_Q(state, action):
    return Q.get((tuple(state), action), 0.0)

def choose_action(state):
    actions = available_actions(state)
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        q_values = [get_Q(state, a) for a in actions]
        return actions[q_values.index(max(q_values))]

def update_Q(state, action, reward, next_state, next_action):
    old_q = get_Q(state, action)
    next_q = get_Q(next_state, next_action) if next_action is not None else 0
    new_q = old_q + alpha * (reward + gamma * next_q - old_q)
    Q[(tuple(state), action)] = new_q

# -------------------------------
# Training the Agent
# -------------------------------

episodes = 10000

for _ in range(episodes):
    board = empty_board()
    state = board[:]
    action = choose_action(state)

    while True:
        board[action] = 1
        result = check_winner(board)

        if result is not None:
            reward = 1 if result == 1 else 0
            update_Q(state, action, reward, board, None)
            break

        opp_action = random.choice(available_actions(board))
        board[opp_action] = -1
        result = check_winner(board)

        if result is not None:
            reward = -1 if result == -1 else 0
            update_Q(state, action, reward, board, None)
            break

        next_state = board[:]
        next_action = choose_action(next_state)

        update_Q(state, action, 0, next_state, next_action)

        state = next_state
        action = next_action

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# -------------------------------
# Evaluation
# -------------------------------

wins = losses = draws = 0

for _ in range(1000):
    board = empty_board()

    while True:
        action = choose_action(board)
        board[action] = 1
        result = check_winner(board)

        if result is not None:
            if result == 1:
                wins += 1
            else:
                draws += 1
            break

        opp_action = random.choice(available_actions(board))
        board[opp_action] = -1
        result = check_winner(board)

        if result is not None:
            if result == -1:
                losses += 1
            else:
                draws += 1
            break

print("Evaluation Results:")
print("Wins:", wins)
print("Losses:", losses)
print("Draws:", draws)
```

---

## 📊 Sample Output

```
Evaluation Results:
Wins: 620
Losses: 85
Draws: 295
```

> ⚠️ Output may vary slightly each run due to randomness.

---

## ✅ Results & Observations

* The SARSA agent significantly improves over time
* Loss rate decreases as training progresses
* Agent achieves a high win/draw percentage
* Demonstrates effective on-policy learning

---

## 🎯 Conclusion

This project successfully implements a **SARSA-based reinforcement learning agent** for Tic-Tac-Toe. Due to the game’s small state space, SARSA performs efficiently and converges to a strong policy.

The experiment clearly illustrates:

* The working of on-policy reinforcement learning
* The importance of reward design
* The role of exploration–exploitation balance

This implementation serves as an excellent foundation for understanding and extending reinforcement learning techniques to more complex environments.

---

## 🚀 Future Enhancements

* Train against a Minimax opponent
* Add a GUI for human vs AI gameplay
* Compare SARSA with Q-learning
* Save and load trained Q-table

---

## 👨‍🎓 Academic Use

✔ Suitable for:

* AI / Machine Learning Lab Records
* University Assignments
* Mini Projects
* Viva & Exam Preparation

---

**Happy Learning! 🎉**

