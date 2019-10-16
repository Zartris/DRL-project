# RL in Continuous Space

#### 1. Intro

So far we have worked with reinforcement learning environments where the number of states and actions is limited. But what about MDPs with much larger spaces?  Consider that the Q-table must have a **row for each state**. So, for instance, if there are 10 million possible states, the Q-table
must have 10 million rows.  Furthermore, if the state space is the set of continuous [real-valued numbers](https://en.wikipedia.org/wiki/Real_number) (an **infinite** set!), it becomes impossible to represent the action values in a **finite** structure! 

____

#### 2. Discrete vs. Continuous Space

**Discrete Spaces** is defined to have a finite amount of states and action space.

* States : $s\in \{s_0,s_1,\dots,s_n\}$
* Actions : $a\in \{a_0,a_1,\dots,a_m\}$

where $n$ and $m$ does not have to be equal.

Many algorithms for computing the Value Functions, needs a discrete space, since they loop over all states to compute the value-function. One of these algorithm is Q-learning.

**Continuous Spaces** is when the space goes to infinity, as real-numbers

- States : $s\in \mathbb{R}^n$
- Actions : $a\in \mathbb{R}^m$

So if we before could think of a Q-table as spaces x actions we now have to think of it as a density plot over a desired range.

<img src="D:\dev\learning\DRL-project\DRL-course\notes\images\Continuous_space.jpg" style="zoom: 50%;" />

Actions can be Continuous as well, in fact most real-world applications are.
For example the smallest change in velocity $\in R^n$ can change the output.

we define continuous space as a Box() type:

example Box(2)

| num  | Observation | min   | max  |
| ---- | ----------- | ----- | ---- |
| 0    | position    | -1.2  | 0.6  |
| 1    | velocity    | -0.07 | 0.07 |

So we have 2 states, with min max values.



So how do we deal with Continuous spaces?

* Discretization
* Function approximation

____

#### 3. Discretization

Discretization is basically converting a continuous space into a discrete one. All we're saying is let's bring back a grid structure with discrete positions identified. Note that we're not really forcing our agent to be in exactly the center of these positions. Since the underlying world is continuous, we don't have control over that. But in our representation of the state space, we only identify certain positions as relevant.

* Round up or down within the grid.

Ways to alter the grid structure (none-Uniform Discretization):

* Alter the size of the grid to fit with obstacles
*  An alternate approach would be to divide up the grid into smaller cells where required.

<img src="D:\dev\learning\DRL-project\DRL-course\notes\images\discretization.jpg" alt="1: Uniform discretization, 2: Non-Uniform, 3: Non-Uniform smaller cells" style="zoom:150%;" />

*Exercise "Discretization"*

____

#### 3.1 Tile-coding

Tile coding is still a Discretization method, but functions a little bit different from what we have seen before. 

As we seen in the previous Discretization method, some kind of prior knowledge is need for choosing the right grid size. 

A more generic method is Tile-coding.

![](D:\dev\learning\DRL-project\DRL-course\notes\images\tile_coding.png)

The underlying space is continuous and 2 dimensional. Then we overlay multiple grids or tilings on top of this space, each slightly offset from each other. 

Now any position S in the state space can be coarsely identified by the tiles that it activates (see right image). If we assign a bit to each tile, then we can represent our new discretised state as a bit vector.

* ones for the tiles that get activated
* and zero elsewhere

$V(s) = \sum_{i=1}^n b_i(s)w_i \quad | b_i == 1 \text{  if active}$

~~~~python
def TILE_CODING (S, A, P, R, gamma, m, n)
	for i in range(1, m):
		# initialize tiling i with n/m tiles
		initialize_tiling(i, n/m)
		for j in range(1, n/m):
			#set weight of tile j in i to w=0.
			initilize_weight(j, w=0)
	while True:
        s = random.random(S)
        shift_in_V(s) = max_(a|s)[R(s,a) + gamma*V(P(s,a))] - V(s)
        for i in range (1, m):
            w = weight_of_active_tile(s,i)
            w = w + (a/m)*shift_in_V(s)
        if times_up:
			break
~~~~



Tile-coding has some drawbacks, one is we still have to manually select the tile sizes, there offsets and number of tiling's.

A more adaptive approach is Adaptive Tile-coding.

Each tile start very large, but we divide each tile into two whenever appropriated. 

**So when do we split?**

* We can use a heuristic for that, so we basically want to split the state space when we no longer learn anything from the current representation.
  * That is when our value function isn't changing.
* We can stop when we have reached some upper limit on the number of splits.

**So which tile do we split?**

* We have to look at which one is likely to have the greatest effect on the value function.
* We need to keep track of subtiles and their projected weights, then we just pick the tile with the greatest difference between subtile weights



There are many other heuristics we can use but the main advantage of adaptive tile coding is that it does not rely on a human to specify a discretisation ahead of time.
The resulting space is appropriately partitioned based on its complexity.

*Exercise Tile_coding*

____



