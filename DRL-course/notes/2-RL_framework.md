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

* Equiprobable random policy is setting $\epsilon = 1$, all action are favored equally.
  * Setting $\epsilon = 0$, favors only the greedy decision.

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

***READ LATER*** = https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

#### Setting the Value of $\epsilon$, in Practice





### Incremental Mean - Update the policy each episode.

<img src="\images\incremental_mean.png" style="zoom: 33%;" />

There are two relevant tables:

- $Q$ -table, with a row for each state and a column for each action. The entry corresponding to state $s$ and action $a$ is denoted $Q(s,a)$.
- $N$ - table that keeps track of the number of first visits we have made to each state-action pair.

The number of episodes the agent collects is equal to $num\_episodes$.

The algorithm proceeds by looping over the following steps:

- **Step 1**: The policy $\pi$ is improved to be $\epsilon$-greedy with respect to $Q$, and the agent uses $\pi$ to collect an episode.
- **Step 2**: $N$ is updated to count the total number of first visits to each state action pair.
- **Step 3**: The estimates in $Q$ are updated to take into account the most recent information.

In this way, the agent is able to improve the policy after every episode!

### Constant-alpha

<img src="\images\constant_alpha.png" style="zoom: 25%;" />

We can denote the error between what we predicted $Q(s,a)$ and the actual result given as delta:
$$
\delta_t := G_t - Q(S_t, A_t)
$$


* If $\delta_t > 0$, then increase $Q(S_t, A_t)$ since our estimate is to low.
* if $\delta_t <0 $, then decrease $Q(S_t, A_t)$ since the estimate is to high.

​	But how much should we increase or decrease this?

Well we have seen $\alpha = \dfrac{1}{N(S_t, A_t)}$ which is the average over everything we have seen.

But the more we train the less we can change this average. So using a constant **step size**: $\alpha$ we emphasizes the changes that comes later more than we do of what we have seen long ago.

#### Setting the value of $\alpha$

Recall the update equation that we use to amend the values in the Q-table:
$$
Q(S_t, A_t) \leftarrow   Q(S_t, A_t) + \alpha (G_t - Q(S_t, A_t))
$$
To examine how to set the the value of $\alpha$ in more detail, we will slightly rewrite the equation as follows:
$$
Q(S_t,A_t) \leftarrow (1-\alpha)Q(S_t,A_t) + \alpha G_t
$$
We have to set $\alpha$ between ZERO and ONE:  $0< \alpha \leq 1$

* Setting $\alpha$ to **ZERO** makes the function keeps the value of the Q-table and forget anything new. We never want this to happen since we will never learn anything new.
  * $Q(S_t,A_t) \leftarrow Q(S_t,A_t)$
* Setting $\alpha$ to **ONE** makes the function forget everything we learn in the past and update the value in the Q-table with the newest result.
  * $Q(S_t,A_t) \leftarrow G_t$

Smaller values for $\alpha$ encourage the agent to consider a longer history of returns when calculating the action-value function estimate. Increasing the value of $\alpha$ ensures that the agent focuses more on the most recently sampled returns.