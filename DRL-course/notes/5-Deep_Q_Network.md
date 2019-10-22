# Deep Q-Network

#### 1. Introduction

Q-learning is a form of Temporal Difference (TD) learning



____

#### 2. Experience Replay

Experience replays helps us address one tupe of correlation. That is between consecutive experience tupels

When the agent interacts with the environment, the sequence of  experience tuples can be highly correlated (connected to eachother).  The naive Q-learning  algorithm that learns from each of these experience tuples in sequential  order runs the risk of getting swayed by the effects of this  correlation, or in other words priotizing the actions that we have seen the most.

By instead keeping track of a **replay buffer** and using **experience replay** to sample from the buffer at random, we can prevent action values from oscillating or diverging catastrophically.

The **replay buffer** contains a collection of experience tuples (SSS, AAA, RRR, S′S'S′).  The tuples are gradually added to the buffer as we are interacting with the environment.

The act of sampling a small batch of tuples from the replay buffer in order to learn is known as **experience replay**.   In addition to breaking harmful correlations, experience replay allows  us to learn more from individual tuples multiple times, recall rare  occurrences, and in general make better use of our experience.

TLDR; So instead of learning sequentail, we are keeping a buffer of <S,A,R,S'> pairs and learn from these in a random order. We can learn multiple times from each pair, and by doing so we can priotize seeing the rare cases more often than the normal pairs.

____

#### 3. Fixed Q-Targets

There is another kind of correlation that Q-learning is susceptible to.

In Q-Learning, we **update a guess with a guess**, and this can potentially lead to harmful correlations.  To avoid this, we can update the parameters $w$ in the network $\hat{q}$ to better approximate the action value corresponding to state $S$ and action $A$ with the following update rule:

<img src="images\Q-learning_update_rule.jpg" style="zoom: 80%;" />

where $w^-$ are the weights of a separate target network that are not changed during the learning step, and $(S, A, R, S')$ is an experience tuple.

The main idea is to use two separarete networks with identical architectures. 
Lets call it target Q-Network and primary Q-Network, where the target Q-Network is updated less often to have a stable (fixed) target $\hat{q} (S',a ,w^-)$.

