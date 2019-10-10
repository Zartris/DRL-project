

# Temporal Difference Methods

Monte Carlo (MC) control methods require us to complete an entire episode of interaction before updating the Q-table.  Temporal Difference (TD) methods will instead update the Q-table after every time step.  



Instead of using an whole episode to update the Q-table, just use a small frame instead.

**Frame**: *Sarsa* 

Take an example of a frame: $(s_0, a_0, r_0) \mapsto (s_1, a_1)$

* From state $s_0$ given action $a_0$, it will yield an reward of $r_0$ and take us to state $s_1$ and we select action $a_1$ 

Then we can update the Q-table:

* $G_t = r_0 + Q(s_1, a_1)$
* $Q(s_0, a_0) = Q(s_0, a_0) + \alpha (\gamma G_t - Q(s_0, a_0))$
  * This is just the constant alpha update from monte carlo.

**The pseudo code for this implementation:**

<img src="images\TD_sarsa.png" style="zoom: 25%;" />

In the algorithm, the number of episodes the agent collects is equal to *num_episodes*.  For every time step $t\geq 0$, the agent:

- **takes** the action $A_t$ (from the current state $S_t$) that is $\epsilon$-greedy with respect to the Q-table,
- receives the reward $R_{t+1}$ and next state $S_{t+1}$,
- **chooses** the next action $A_{t+1}$ (from the next state $S_{t+1}$) that is $\epsilon$-greedy with respect to the Q-table,
- uses the information in the tuple ($S_t$, $A_t$, $R_{t+1}$, $S_{t+1}$, $A_{t+1}$) to update the entry $Q(S_t, A_t)$ in the Q-table corresponding to the current state $S_t$ and the action $A_t$.

## TD control Q-learning

Q-learning or sarsamax is another method, that builds upon the sarsa method.

When updating Q-table, instead of selecting an action $a_{t+1}$ with an $\epsilon$-greedy policy, we are using the current uptimal action or the action that maximizes $Q(s_t, a_t)$. 
$$
G_t = r_{t+1} + \gamma \max_{a\in A} Q(s_{t+1}, a) \\
Q(s_t, a_t) = Q(s_t, a_t) +\alpha (G_t - Q(s_t, a_t))
$$
**The pseudo code for this implementation:**

<img src="images\sarsamax_qlearning.png" style="zoom: 25%;" />



### Expected Sarsa

Expected sarsa works like the other sarsa examples but instead of using max or just the next actions, we estimate based on the expected future reward.
$$
G_t = r_{t+1} + \gamma \sum_{a\in A} \Big( \pi (a|S_{t+1}) Q(s_{t+1}, a)\Big) \\
Q(s_t, a_t) = Q(s_t, a_t) +\alpha (G_t - Q(s_t, a_t))
$$
<img src="images\expected_sarsa.png" style="zoom:25%;" />

