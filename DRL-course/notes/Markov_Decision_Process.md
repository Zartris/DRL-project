# Markov Decision Process

Terms and descriptions:

____

Policies:





**Finding Policies: - Value iteration.**

[Missing the algorithm]
$$
U(s) = R(s) + \gamma \times \text{max}_a \sum_{s´} T(s,a,s') U_t(s')
$$
The utility at a given state is computed by :

1. The reward of the current state - $R(s)$
2. The long term discounted reward gamma - $\gamma$
3. The best possible action a - $\text{max}_a$  
4. Sum of possible transition given action a and state to the possible states
5. times the possible states utility value.



**Finding Policies: - Policy iteration.**

- Start with $\pi_0$ as just random guesses of actions.

- *Evaluate*: Given $\pi_t$ compute  $U_t = U^{\pi_t}$ which is the utility for the given policy.

- *Improve:*
  $$
  \pi_{t+1} = \arg\max_a \sum_{s´} T(s,a,s´) U(s´)
  $$
  To do this we need to know how to evaluate $U^{\pi_t}$:

$$
U_t(s) = U^{\pi_t}(s) = R(s) + \gamma \times \sum_{s´}{T(s,\pi_t(s), s´)\times U_t(s´)}
$$

​		So we are evaluating the current policy by following the action given from the policy.

This will be done until the policy is no changing anymore.