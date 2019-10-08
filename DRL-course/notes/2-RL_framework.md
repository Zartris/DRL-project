## Monte Carlo method

We can use the monte Carlo method to estimate the action value function.
The Monte Carlo methods is using randomness to solve the **prediction problem**.

**Prediction Problem**: Given a policy, how might the agent estimate the value function for that policy?

#### MC Prediction

* An algorithm that solves the prediction problem and determine the value function $v_{\pi} (\text{or } q_{\pi})$ corresponding to a policy $\pi$.
* 

This is done by using a Q-table:

| States/actions | action 1. | action 2. | action 3. |
| -------------- | --------- | --------- | --------- |
| State 1.       |           |           |           |
| State 2.       |           |           |           |
| State 3.       |           |           |           |

2-types of MC Prediction:

* **Every-visit MC Prediction**: Average the returns following all visits to each state-action pair, in all episodes.
* **First-visit MC Prediction**: For each episode, we only consider the first visit to the state-action pair. The pseudocode for this option can be found below.

<img src="images\first_visit_pseudo.png" style="zoom: 33%;" />

- *Q*-table, with a row for each state and a column for each action. The entry corresponding to state *s* and action *a* is denoted Q*(*s*,*a*).
- *N* - table that keeps track of the number of first visits we have made to each state-action pair.
- *returns\_sum* - table that keeps track of the sum of the rewards obtained after first visits to each state-action pair.

**First-visit or Every-visit?**

Both the first-visit and every-visit method are **guaranteed to converge** to the true action-value function, as the number of visits to each state-action pair approaches infinity. (*So, in other words, as long as the agent gets enough experience with each state-action pair, the value function estimate will be pretty close to the true value.*)

- Every-visit MC is [biased](https://en.wikipedia.org/wiki/Bias_of_an_estimator), whereas first-visit MC is unbiased (see Theorems 6 and 7).
- Initially, every-visit MC has lower [mean squared error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error), but as more episodes are collected, first-visit MC attains better MSE (see Corollary 9a and 10a, and Figure 4).

### Epsilon greedy policies

It helps the agent explore and don't get stuck on a policy.
This is done by introducing probability to take another action.

- If the coin lands tails (so, with probability $1-\epsilon$), the agent selects the greedy action.
- If the coin lands heads (so, with probability $\epsilon$), the agent selects an action *uniformly* at random from the set of available (non-greedy **AND** greedy) actions.

In order to construct a policy $\pi$ that is $\epsilon$-greedy with respect to the current action-value function estimate $Q$, we will set:
$$
\pi(a|s) \leftarrow 
\begin{cases}
1 - \epsilon + \dfrac{\epsilon}{|A(s)|}, & \text{if } a \text{ maximizes } Q(s,a)\\
    \dfrac{\epsilon}{|A(s)|},              & \text{Otherwise}
\end{cases}
$$
For each $s \in S$ and $a\in A(s)$.

We add the $\frac{\epsilon}{|A(s)|}$ to make the probability to 1.
So for each possible action in given state $s$: we have $\frac{\epsilon}{|A(s)|}$ to choose this option unless you are the action that maximizes the $Q$-table at row $s$, then we have $1 - \epsilon + \frac{\epsilon}{|A(s)|}$ probability to chose the greedy action :)



Notes:

* Equiprobable random policy is setting $\epsilon = 1$, all action are favoured equally.
  * Setting $\epsilon = 0$, favoures only the greedy descision.

### Greedy in the Limit with Infinite Exploration (GLIE)

In order to guarantee that MC control converges to the optimal policy $\pi_*$, we need to ensure that two conditions are met.  We refer to these conditions as **Greedy in the Limit with Infinite Exploration (GLIE)**.  In particular, if:

- every state-action pair $s,a$ (for all $s\in\mathcal{S}$ and $a\in\mathcal{A}(s)$) is visited infinitely many times, and 
- the policy converges to a policy that is greedy with respect to the action-value function estimate $Q$,

then MC control is guaranteed to converge to the optimal policy (in  the limit as the algorithm is run for infinitely many episodes).  These  conditions ensure that:

- the agent continues to explore for all time steps, and
- the agent gradually exploits more (and explores less).

One way to satisfy these conditions  is to modify the value of $\epsilon$ when specifying an $\epsilon$-greedy policy.  In particular, let $\epsilon_i$ correspond to the $i$-th time step.  Then, both of these conditions are met if:

- $\epsilon_i > 0$ for all time steps $i$, and 
- $\epsilon_i$ decays to zero in the limit as the time step $i$ approaches infinity (that is, $\lim_{i\to\infty} \epsilon_i = 0$).

For example, to ensure convergence to the optimal policy, we could set $\epsilon_i = \frac{1}{i}$.