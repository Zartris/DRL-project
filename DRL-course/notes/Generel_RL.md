





## Agents

#### State

A state could be many things but in general it is a description of how the environment looks at a given time.

* This could be where all the pieces of a chess game is located on the board.
* or a set of images and a score.

#### Actions

Actions is a set of possible commands the agent can chose from:

* This we define as ´a´ or $a \in A$ which is an action a in the set of actions.

#### environments

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

