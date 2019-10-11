

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

### Optimism

You have learned that for any TD control method, you must begin by initializing the values in the Q-table. It has been shown that [initializing the estimates to large values](http://papers.nips.cc/paper/1944-convergence-of-optimistic-and-incremental-q-learning.pdf) can improve performance. For instance, if all of the possible rewards that can be received by the agent are negative, then initializing every estimate in the Q-table to zeros is a good technique. In this case, we refer to the initialized Q-table as **optimistic**, since the action-value estimates are guaranteed to be larger than the true action values.

### Similarities

All of the TD control methods we have examined (Sarsa, Sarsamax, Expected Sarsa) converge to the optimal action-value function $q_*$ (and so yield the optimal policy $\pi_*$) if:

1. the value of $\epsilon$ decays in accordance with the GLIE conditions, and
2. the step-size parameter $\alpha$ is sufficiently small.

### Differences

The differences between these algorithms are summarized below:

- Sarsa and Expected Sarsa are both **on-policy** TD control algorithms.  In this case, the same ($\epsilon$-greedy) policy that is evaluated and improved is also used to select actions.
- Sarsamax is an **off-policy** method, where the (greedy) policy that is evaluated and improved is different from the ($\epsilon$-greedy) policy that is used to select actions.
- On-policy TD control methods (like Expected Sarsa and Sarsa) have  better online performance than off-policy TD control methods (like  Sarsamax). 
- Expected Sarsa generally achieves better performance than Sarsa.

If you would like to learn more, you are encouraged to read Chapter 6 of the [textbook](http://go.udacity.com/rl-textbook) (especially sections 6.4-6.6).