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

<img src="images\Continuous_space.jpg" style="zoom: 50%;" />

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

<img src="images\discretization.jpg" alt="1: Uniform discretization, 2: Non-Uniform, 3: Non-Uniform smaller cells" style="zoom:150%;" />

*Exercise "Discretization"*

____

#### 3.1 Tile-coding

Tile coding is still a Discretization method, but functions a little bit different from what we have seen before. 

As we seen in the previous Discretization method, some kind of prior knowledge is need for choosing the right grid size. 

A more generic method is Tile-coding.

![](images\tile_coding.png)

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

#### 4. Function approximation

Given a problem domain with continuous states $s \in \mathcal{S} = {\mathbb{R}^{n}}$, we wish to find a way to represent the value function $v_{\pi}(s)$ (for prediction) or $q_{\pi}(s, a)$ (for control).

We can do this by choosing a parameterized function that *approximates* the true value function:

$\hat{v}(s, \mathbf{w}) \approx v_{\pi}(s)$
 $\hat{q}(s, a, \mathbf{w}) \approx q_{\pi}(s, a)$

Our goal then reduces to finding a set of parameters $\mathbf{w}$  that yield an optimal value function. We can use the general  reinforcement learning framework, with a Monte-Carlo or  Temporal-Difference approach, and modify the update mechanism according  to the chosen function.

**Feature vector**

A common intermediate step is to compute a feature vector that is representative of the state: $\mathbf{x}(s)= \begin{bmatrix} x_1(s)\\x_2(s) \\ \vdots \\ x_n(s) \end{bmatrix} $

**Dot product**

What do we do when we have two vectors and want to produce a scalar $\mathbf{w}$?

- Dot product!

$$
\hat{v}(s, \mathbf{w}) = \mathbf{x}(s)^\intercal \cdot \mathbf{w}\\
= \Big(x_1(s)\cdots x_n(s) \Big) \cdot \begin{bmatrix} w_1 \\ \vdots \\ w_n \end{bmatrix} \\
= x_1(s) \cdot w_1 + \cdots + x_n(s) \cdot w_n\\
= \sum_{j=1}^n x_j(s)*w_j
$$

This is known as linear function approximation.

____

#### 4.1 Linear function approximation

**Value approximation:**

Taking the derivate of $\triangledown_w\hat{v}(s, \mathbf{w}) = \mathbf{x}(s)$ so by this follows:

* Value function : $\hat{v}(s,w) = \mathbf{x}(s)^\intercal \mathbf{w}$
* minimize Error : $\mathbb{J}(\mathbf{w}) = \mathbb{E}\Big[ \big(v_\pi(s) - \mathbf{x}(s)^\intercal \mathbf{w}\big)^2\Big] $
* Error Gradient : $\triangledown_w \mathbb{J}(\mathbf{w}) = -2\big(v_\pi(s) - \mathbf{x}(s)^\intercal \mathbf{w} \big) \mathbf{x}(s) $
* Update rule : $\Delta \mathbf{w} = - \alpha \frac{1}{2}\triangledown_w \mathbb{J}(\mathbf{w}) = \alpha \Big(v_\pi(s) - \mathbf{x}(s)^\intercal \mathbf{w} \Big) \mathbf{x}(s)$

<img src="images\linear_function_approximation.jpg" style="zoom: 67%;" />



**Action value approximation:**

Now we want to approximate the action value function $\hat{q}(s,a,\mathbf{w})$.

First we create the feature vector as before in the following way:

$\mathbf{x}(s,a)= \begin{bmatrix} x_1(s,a)\\x_2(s,a) \\ \vdots \\ x_n(s,a) \end{bmatrix} $

For each state we would need to compute the action value function
$$
\hat{q}(s, a_1, \mathbf{w}) = ?\\
\cdots \\
\hat{q}(s, a_m, \mathbf{w}) = ?
$$
we are trying to find $n$ different action-value functions, one for each action dimension, but intuitively we know that these function are related. So it makes sense we can compute them together. 

We can do this by extending our weight vector and turning it into a matrix:
$$
\hat{q}(s,a, \mathbf{w}) = \big(x_1(s,a)\cdots x_n(s,a) \big) \cdot \begin{bmatrix} w_{11}\cdots w_{1m} \\ \vdots \quad\ddots\quad \vdots  \\  w_{n1}\cdots w_{nm} \end{bmatrix} \\
= \Big( \hat{q}(s,a_1,\mathbf{w}) \cdots \hat{q}(s,a_m,\mathbf{w})\Big)
$$
Here each column of the matrix emulates a separate linear function.

**Limitations:**

The primary limitation of linear function approximation is that we can only represent linear relationships between inputs and outputs.

With one dimensional input this is a line and in two dimensions it becomes a plane and so on.

So if our underlying function has a non-linear shape, our linear approximation may give very bad results.

____

#### 4.2 Kernel functions

We can use kernels to emulate a non-linear function in linear space.

We defined our feature vector as something generic. Something that takes a state or a state action pair and produces a feature vector.

$\mathbf{x}(s,a)= \begin{bmatrix} x_1(s,a)\\x_2(s,a) \\ \vdots \\ x_n(s,a) \end{bmatrix} $

Each element $(\cdots x_i(s,a) \cdots)$of this vector can be produced by a separate function, which can be non-linear, for example, lets say that s is defined as a single real number, then we could have that:
$$
x_1(s) = s\\
x_2(s) = s^2\\
x_3(s) = s^3\\
\vdots
$$
These are called Kernel Functions or Basis Functions. 

They transform the input state into a different space, but note that since our value function is still defined as a linear combination of these features, we can still use linear function approximation.

**Radial Basis Functions**

RBF is a very common form of Kernels used for this purpose.

The kernel is described:
$$
\phi_i(s) = \exp{\Big( -\dfrac{||s-c_i||^2}{2\sigma^2_i}\Big)}
$$
Essentially think of the current state S as a location in the continuous state space.
Each Basis Function is shown as a blob, the closer the state is to the center of this blob the higher the response returned by the function and the further you go the response falls off gradually with the radius. 

____

#### 4.3 Non-Linear Function Approximation 

Let pass our linear responds $\hat{q}(s,a, \mathbf{w}) = \mathbf{x}(s,a)^\intercal \cdot \mathbf{w}$ obtained using the dot product, through some non-linear function $f$:
$$
\hat{q}(s,a, \mathbf{w}) = f\big(\mathbf{x}(s,a)^\intercal \cdot \mathbf{w} \big)
$$
This is the basis of artificial neural networks. Such a non-linear function $f$ is generally called an activation function and immensely increase the representational capacity of our approximator.

We can iteratively update the parameters of any such function using gradient descent:
$$
\Delta\mathbf{w} = \alpha \Big( v_\pi(s) - \hat{v}(s, \mathbf{w}) \Big) \triangledown_w \hat{v}(s,\mathbf{w})
$$
