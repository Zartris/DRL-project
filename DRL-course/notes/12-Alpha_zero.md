# Alpha zero case-study!

The best part of the alphazero algorithm is simplicity: it consists of a Monte Carlo tree search, guided by a deep neural network. This is  analogous to the way humans think about board games -- where  professional players employ hard calculations guides with intuitions. 

Alpha zero specializes in zero-sum games

### 1. Zero-sum games (Turn based)

* Tic-tac-toe
* Chess
* go
* shogu

We start with a board game environment, a grid for example, then two competing agents take turns to perform actions to try to win the game. Where in the end, one agent's win is another agent's loss.
Usually, we also assumed that the game contains no hidden information, so there's no element of luck,
and winning or losing is entirely determined by skill.

Let's look at a tic-tac-toe example.
The goal is to get three in a row,  where we can represent board mathematically.
The board can be represented by a matrix where zero indicates empty space and plus or minus one indicates the pieces of player one and player two.

Given that there are only circles and crosses in tic-tac-toe, each entry can then only be zero,
plus one, or minus one. We can also encode the final outcome by a score, where plus one indicates a win by the first player, and negative one indicates a win by the second player, and zero indicates a draw.

![](images\zero_sum_games_matrix.png)


This way of representing the board is convenient because the board can easily be fed into a neural network. Also, if you want to switch other player's pieces, we can just multiply the matrix by negative one. We can also flip the score by multiplying it by negative one.

This property will come in handy when we build an agent to play the game, since each move can be represented as each player.
Now that we've encoded the game mathematically, we can rephrase everything in the language of reinforcement learning. We have a sequence of states for the board game denoted by: $s_t$,
and we have two players denoted by plus or minus one: Player:$(-1)^t$.
Here, I've simplified the negative one to the power of $t$, assuming we start with $t=0$.
Player plus one:

* Performs actions at all the even timesteps
* Goal: tries to maximize the final score plus z,

while player negative one 

* performs actions at all the odd timesteps 
* and tries to maximize negative one times the final score.

Now, imagine we've constructed an intelligent agent who is able to play a perfect game as player plus one,
by then it should be able to play a perfect game as player negative one as well as long as we flipped the state $s_t$ at all the odd timesteps.

Then we can have the agent play against itself with a common policy.
$$
\pi_0(a_t| (-1)^t\cdot s_t)
$$
Now, besides having a common policy, we can also have a common critic that can
evaluate the expected outcome from the perspective of the current player.
$$
v_\theta ((-1)^t \cdot s_t)\\
= \text{Estimates: } (-1)^t\cdot z
$$
This essentially estimates the expected value of negative one to the t power times a score z.
We will see later that this is the basic idea behind AlphaZero, where we have one agent playing against itself along with one critic that self-improves as more and more games are played.

____

