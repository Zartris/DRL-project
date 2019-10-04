# 1. General Reinforcement learning.

These note's is taken from **grokking-deep-reinforcement-learning** 

Which intro you can find here: https://livebook.manning.com/book/grokking-deep-reinforcement-learning/welcome/v-7/

The book is still under development, but it is describing the content of Deep reinforcement learning very good and is filled with examples and coding exercises. 
I have bought the book and I would highly recommend you doing the same.

## Agents

An **agent** is the decision maker only and nothing else. That means if you  are training a robot to pick up objects, the robot arm is not part of  the agent. Only the code that makes decisions is referred to as the 
agent.

#### State

A state could be many things but in general it is a description of how the environment looks at a given time.

* This could be where all the pieces of a chess game is located on the board.
* or a set of images and a score.

Interestingly, often agents don't have access to the actual full state 
of the environment. The part of the state that the agent can observe is 
called an **observation**. Observations depend on states but are what the agent can in fact see.

#### Actions

Actions is a set of possible commands the agent can chose from:

* This we define as ´a´ or $a \in A$ which is an action a in the set of actions.

#### Environments

The environment is represented by a set of variables related to the problem. This could be the position and the velocity of a robot.  If we are training a robot to pick up objects, the objects to be picked up, the tray where the objects lay, the wind, if any, and so on, are all part of the environment. But the robot arm is also part of the environment because it is not part of the agent. And even though the agent can decide to move the arm, the actual arm movement is noisy, and thus the arm is part of the environment.

The environment is taking in an actions and responds with a *reward* (which are described better later) and a new state.

* You can see the environment as the game rules and observer of the "Game". 

## Tasks

* A task is an instance (A job to be done) of the reinforcement learning (RL) problem.
  * A task can be categorised into two classes
* **Episodic tasks**: are task with a well-defined starting and ending point.
  * This is task that would at some point end.
  * A good example is chess, each task could be a round of chess.
* **Continuing task**: are tasks that continue forever, without ending.
  * A good example of this is the stock market.
    * Buying and selling stocks will never end, hence no real goal is provided.
    * If a goal is described (earn 100$), it would then be a Episodic task.



## Rewards

