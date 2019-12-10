# Actor Critic Methods

- **Value based methods** (Q-learning, Deep Q-learning): where we learn a value function **that will map each state action pair to a value.** Thanks to these methods, we find the best action to take for each state — the action with the biggest value. This works well when you have a finite set of actions.
- **Policy based methods** (REINFORCE with Policy Gradients): where we directly optimize the policy without using a value function. This is useful when the action space is continuous or stochastic. The main problem is finding a good score function to compute how good a policy is. We **use total rewards of the episode.**

https://www.freecodecamp.org/news/an-intro-to-advantage-actor-critic-methods-lets-play-sonic-the-hedgehog-86d6240171d/

____

The actor-critic method is a combination of value-based and policy-based.
We are training a network **Actor** as a policy-based method to decide in a given state what the corresponding distribution over the action space. Another network is made, the **Critic** which is a value-based network, that will learn to evaluate the state value function $V_\pi$ using TD estimation. Using this **Critic** we will calculate the advantage function and train the **Actor** using this value.

![](images\actor_critic_basic.png)

**One training step in the Actor-critic cycle:**

1. Given a state:

   * $\pi(a|s;\theta_\pi) =$ **Actor**(state)
   * best_action = max($\pi(a|s;\theta_\pi)$)
   * reward, state_next = env(state, best_action)
   * (s,a,r,s') = (state, best_action, reward, state_next)

2. Given reward, state_next and discount value $\gamma$:

   * $\text{value}_{state}$ = $V(s,\theta_v)$ = **Critic** (s)

   * $\text{value}_{Nstate}$ = $V(s',\theta_v)$ = **Critic** (s')
   * train **Critic** (reward+$\gamma V(s'; \theta_v)$)

3. Given state, action, reward, $\text{value}_{state}$, $\text{value}_{Nstate}$ and discount value $\gamma$

   * Advantage(state,action) = $\text{reward} + \gamma V(s';\theta_v) - V(s;\theta_v)$
   * train **Actor** (Advantage(state,action))



### 1. A3C: Asynchronous Advantage Actor-critic



#### 1.1 N-step bootstrapping

#### 1.2 Parallel Training instead of replay buffer

The main reason that we needed the replay buffer was so we could decorrelate experienced tuples and sequentailly stroing them for later processing. We would then randomly (or weighted) select small mini batches of experiences without each experience being correlated.

A3C replaces the replay buffer with parallel training, by creating multiple instances of the environment and agent, and run them all at the same time. Our agent will receive mini batches of the correlated experiences juist as we need. The samples will be decorrelated because agents will likely be experiencing different states at any given time.

![](images\A3C_parallel.png)

On top of that, this way of training allows us to use on-policy learning in our learning algorithm, which is often associated with more stable learning.

#### 1.3 Off-policy vs On-policy

