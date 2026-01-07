import random
import math
import matplotlib.pyplot as plt

# -----------------------------
# Environment Setup
# -----------------------------

# Five price points
prices = [10, 15, 20, 25, 30]

# True (unknown) conversion probabilities for each price
true_probs = [0.30, 0.25, 0.20, 0.15, 0.10]

T = 2000  # Number of time steps
epsilon = 0.1
c = 2

# -----------------------------
# Helper Function
# -----------------------------
def get_reward(price_index):
    """Simulates customer purchase"""
    if random.random() < true_probs[price_index]:
        return prices[price_index]  # revenue earned
    return 0

# -----------------------------
# ε-Greedy Algorithm
# -----------------------------
def epsilon_greedy():
    Q = [0] * len(prices)
    N = [0] * len(prices)
    revenue = []

    total = 0
    for _ in range(T):
        if random.random() < epsilon:
            action = random.randint(0, len(prices) - 1)
        else:
            action = Q.index(max(Q))

        reward = get_reward(action)
        total += reward

        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        revenue.append(total)

    return revenue

# -----------------------------
# UCB Algorithm
# -----------------------------
def ucb():
    Q = [0] * len(prices)
    N = [0] * len(prices)
    revenue = []

    total = 0
    for t in range(1, T + 1):
        ucb_values = []
        for i in range(len(prices)):
            if N[i] == 0:
                ucb_values.append(float('inf'))
            else:
                ucb_values.append(Q[i] + c * math.sqrt(math.log(t) / N[i]))

        action = ucb_values.index(max(ucb_values))
        reward = get_reward(action)
        total += reward

        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        revenue.append(total)

    return revenue

# -----------------------------
# Thompson Sampling Algorithm
# -----------------------------
def thompson_sampling():
    alpha = [1] * len(prices)
    beta = [1] * len(prices)
    revenue = []

    total = 0
    for _ in range(T):
        samples = [random.betavariate(alpha[i], beta[i]) for i in range(len(prices))]
        action = samples.index(max(samples))

        if random.random() < true_probs[action]:
            reward = prices[action]
            alpha[action] += 1
        else:
            reward = 0
            beta[action] += 1

        total += reward
        revenue.append(total)

    return revenue

# -----------------------------
# Run Experiment
# -----------------------------
eps_revenue = epsilon_greedy()
ucb_revenue = ucb()
ts_revenue = thompson_sampling()

# -----------------------------
# Plot Results
# -----------------------------
plt.figure()
plt.plot(eps_revenue, label="Epsilon-Greedy")
plt.plot(ucb_revenue, label="UCB")
plt.plot(ts_revenue, label="Thompson Sampling")
plt.xlabel("Time Steps")
plt.ylabel("Cumulative Revenue")
plt.title("Comparison of Bandit Algorithms for Dynamic Pricing")
plt.legend()
plt.show()

# -----------------------------
# Final Revenue Output
# -----------------------------
print("Final Revenue after 2000 steps:")
print("Epsilon-Greedy:", eps_revenue[-1])
print("UCB:", ucb_revenue[-1])
print("Thompson Sampling:", ts_revenue[-1])

