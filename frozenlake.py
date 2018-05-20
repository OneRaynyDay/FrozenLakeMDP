"""
Let's use Value Iteration to solve FrozenLake!

Setup
-----
We start off by defining our actions:
A = {move left, move right...} = {(0,1),(0,-1),...}
S = {(i,j) for 0 <= i,j < 4}
Reward for (3,3) = 1, and otherwise 0.
Probability distribution is a 4x(4x4) matrix of exactly the policy.
We have pi(a|s), where a in A, and s in S.

Problem formulation : https://gym.openai.com/envs/FrozenLake-v0/

Algorithm
---------
Because our situation is deterministic for now, we have the value iteration eq:

v <- 0 for all states.
v_{k+1}(s) = max_a (\sum_{s',r} p(s',r|s,a) (r + \gamma * v_k(s'))

... which decays to:

v_{k+1}(s = max_a (\sum_{s'} 1_(end(s')) + \gamma * v_k(s'))

Because of our deterministic state and the deterministic reward.
"""
import numpy as np
import random
from numpy.linalg import norm
import matplotlib.pyplot as plt

N = 4
v = np.zeros((N, N), dtype=np.float32) # Is our value vector.
ITER = 1000
A = [(0,1),(0,-1),(1,0),(-1,0)]

# If you're feeling adventurous, make your own MAP
MAP = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

def proj(n, minn, maxn):
    """
    projects n into the range [minn, maxn). 
    """
    return max(min(maxn-1, n), minn)

def move(s, tpl, stochasticity=0):
    """
    Set stochasticity to any number in [0,1].
    This is equivalent to "slipping on the ground"
    in FrozenLake.
	"""
    if MAP[s[0]][s[1]] == 'H': # Go back to the start
        return (0,0)
    if np.random.random() < stochasticity:
        return random.choice(A)
    return (proj(s[0] + tpl[0], 0, N), proj(s[1] + tpl[1], 0, N))

def reward(s):
    return MAP[s[0]][s[1]] == 'G'
    
def run_with_value(v, gamma=0.9):
    old_v = v.copy()
    for i in range(N):
        for j in range(N):
            best_val = 0
            for a in A:
                new_s = move((i,j), a)
                best_val = max(best_val, gamma * old_v[new_s])
            v[i,j] = best_val + reward((i,j))
    return old_v

# Extracting policy from v:
def pi(s, v):
    cur_best = float("-inf")
    cur_a = None
    for a in A:
        val = v[move(s, a, stochasticity=0)]
        if val > cur_best:
            cur_a = a
            cur_best = val
    return cur_a

if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    # Performing Value Iteration
    old_v = run_with_value(v)
    for i in range(ITER):
        old_v = run_with_value(v)

    # Plotting a nice arrow map.
    action_map = np.array([
        [pi((i,j), v) for j in range(N)] for i in range(N)])
    Fx = np.array([ [col[1] for col in row] for row in action_map ])
    Fy = np.array([ [-col[0] for col in row] for row in action_map ])
    plt.matshow(v) 
    plt.quiver(Fx,Fy)
    plt.show()
