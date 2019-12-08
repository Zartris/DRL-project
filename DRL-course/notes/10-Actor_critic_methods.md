# Actor Critic Methods

- **Value based methods** (Q-learning, Deep Q-learning): where we learn a value function **that will map each state action pair to a value.** Thanks to these methods, we find the best action to take for each state — the action with the biggest value. This works well when you have a finite set of actions.
- **Policy based methods** (REINFORCE with Policy Gradients): where we directly optimize the policy without using a value function. This is useful when the action space is continuous or stochastic. The main problem is finding a good score function to compute how good a policy is. We **use total rewards of the episode.**

https://www.freecodecamp.org/news/an-intro-to-advantage-actor-critic-methods-lets-play-sonic-the-hedgehog-86d6240171d/

