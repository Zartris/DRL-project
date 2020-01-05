# Notes on Alpha Star

### Combined learning

The alpha star is using a combined learning strategy; Imitation learning (supervised learning), Reinforcement learning and Multi-agent learning.

Since the complexity of the game is so high, they are initiating each model using imitation learning, training the model to first predict players movement and then uses reinforcement learning to improve the agent after.

That said, even throughout the reinforcement learning, they uses a **z** statistic.

### What is Z Statistic

z statistic is a summarized strategy sampled from human data, example:

* First 20 constructed buildings and units 
* Units, buildings, effects, upgrades present during a game
* During initation learning this is set to zero 10% of the time.

So for each player the statistic ‘z’ characterizes a summary of actions done througout the game.

During reinforcement learning, they either **condition the agent** on a statistic z, in which case agents receive a reward for following the strategy corresponding to z, or **train the agent unconditionally**, in which case the agent is free to choose its own strategy. 

Agents also receive a penalty whenever their action probabilities differ from the supervised policy. 

This human exploration ensures that a wide variety of relevant modes of play continue to be explored throughout training.

### Imitation learning (supervised)











### Reward

Reward is the outcome of the game without discount:

* win = 1
* Tie=0
* Loss=-1

Using multi-agent actor-critic paradigm:

* Value function predicts reward at a given time and updates policy.

Introducing a pseudo reward for following "z" statistics

