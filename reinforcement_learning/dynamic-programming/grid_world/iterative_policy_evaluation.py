import numpy as np
from grid_world import standard_grid, negative_grid

def print_values(V, g):
  for i in range(g.width):
    print("---------------------------")
    for j in range(g.height):
      v = V.get((i,j), 0)
      if v >= 0:
        print(" %.2f|" % v, end="")
      else:
        print("%.2f|" % v, end="") # -ve sign takes up an extra space
    print("")


def print_policy(P, g):
  for i in range(g.width):
    print("---------------------------")
    for j in range(g.height):
      a = P.get((i,j), ' ')
      print("  %s  |" % a, end="")
    print("")
    
grid = standard_grid()

states = grid.all_states()


V = {}
for i in states:
    V[i] = 0
gamma = 1
while True:
    delta= 0
    for i in states:
        old_v = V[i]
        if i in grid.actions:
            v = 0
            p = 1.0 / len(grid.actions[i])
            for j in grid.actions[i]:
                grid.set_state(i)
                reward =grid.move(j)
                v += p * (reward + gamma * V[grid.current_state()])
            V[i] = v
            delta = max(delta, abs(old_v - V[i]))
    if delta <1e-3:
        break
print('The result value function')
print_values(V, grid)

policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'R',
    (2, 1): 'R',
    (2, 2): 'R',
    (2, 3): 'U',
  }


V = {}
for i in states:
    V[i] = 0
gamma = 0.9
while True:
    delta= 0
    for i in states:
        old_v = V[i]
        if i in grid.actions:
            v = 0
            
            for j in grid.actions[i]:
                if j == policy.get(i):
                    grid.set_state(i)
                    reward =grid.move(j)
                    v +=  (reward + gamma * V[grid.current_state()])
            V[i] = v
            delta = max(delta, abs(old_v - V[i]))
    if delta <1e-3:
        break
print('The result value function for Fixed Policy')
print_values(V, grid)

################## policy iteration
grid = negative_grid()
def q_from_v(s, V):
    q = np.zeros(len(grid.actions[s]))
    for j in range(len(q)):
        q[j] = 0
        for i in grid.actions[s]:
            grid.set_state(s)
            reward = grid.move(i)
            q[j] += reward + gamma * V[grid.current_state()]
    return q
        
q_from_v((2,0) , V)


    
         