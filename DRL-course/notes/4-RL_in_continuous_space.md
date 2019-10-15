# RL in Continuous Space

#### 1. Intro

So far we have worked with reinforcement learning environments where the number of states and actions is limited. But what about MDPs with much larger spaces?  Consider that the Q-table must have a **row for each state**. So, for instance, if there are 10 million possible states, the Q-table
must have 10 million rows.  Furthermore, if the state space is the set of continuous [real-valued numbers](https://en.wikipedia.org/wiki/Real_number) (an **infinite** set!), it becomes impossible to represent the action values in a **finite** structure! 

____

#### 2. Discrete vs. Continuous Space



____

