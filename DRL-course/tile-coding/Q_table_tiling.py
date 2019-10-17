import numpy as np

from Tile_Coding import create_tilings, tile_encode


class QTable:
    """Simple Q-table."""

    def __init__(self, state_size, action_size):
        """Initialize Q-table.

        Parameters
        ----------
        state_size : tuple
            Number of discrete values along each dimension of state space.
        action_size : int
            Number of discrete actions in action space.
        """
        self.state_size = state_size
        self.action_size = action_size
        # Create Q-table, initialize all Q-values to zero
        self.q_table = np.zeros((state_size[0], state_size[1], action_size))

        # Note: If state_size = (9, 9), action_size = 2, q_table.shape should be (9, 9, 2)
        print("QTable(): size =", self.q_table.shape)


class TiledQTable:
    """Composite Q-table with an internal tile coding scheme."""

    def __init__(self, low, high, tiling_specs, action_size):
        """Create tilings and initialize internal Q-table(s).

        Parameters
        ----------
        low : array_like
            Lower bounds for each dimension of state space.
        high : array_like
            Upper bounds for each dimension of state space.
        tiling_specs : list of tuples
            A sequence of (bins, offsets) to be passed to create_tilings() along with low, high.
        action_size : int
            Number of discrete actions in action space.
        """
        self.tilings = create_tilings(low, high, tiling_specs)
        self.state_sizes = [tuple(len(splits) + 1 for splits in tiling_grid) for tiling_grid in self.tilings]
        self.action_size = action_size
        self.q_tables = [QTable(state_size, self.action_size) for state_size in self.state_sizes]
        print("TiledQTable(): no. of internal tables = ", len(self.q_tables))

    def get(self, state, action):
        """Get Q-value for given <state, action> pair.

        Parameters
        ----------
        state : array_like
            Vector representing the state in the original continuous space.
        action : int
            Index of desired action.

        Returns
        -------
        value : float
            Q-value of given <state, action> pair, averaged from all internal Q-tables.
        """
        # Encode state to get tile indices
        encoded_state = tile_encode(state, self.tilings)
        # print("encoded_state", encoded_state)
        q_value = 0.0
        # Retrieve q-value for each tiling, and return their average
        for tiling_nr in range(len(encoded_state)):
            q_value += self.q_tables[tiling_nr].q_table[encoded_state[tiling_nr][0]][encoded_state[tiling_nr][1]][
                action]
        q_value = q_value / len(encoded_state)
        # print("Q-value", str(q_value))
        return q_value

    def update(self, state, action, value, alpha=0.1):
        """Soft-update Q-value for given <state, action> pair to value.

        Instead of overwriting Q(state, action) with value, perform soft-update:
            Q(state, action) = alpha * value + (1.0 - alpha) * Q(state, action)

        Parameters
        ----------
        state : array_like
            Vector representing the state in the original continuous space.
        action : int
            Index of desired action.
        value : float
            Desired Q-value for <state, action> pair.
        alpha : float
            Update factor to perform soft-update, in [0.0, 1.0] range.
        """
        # Encode state to get tile indices
        encoded_state = tile_encode(state, self.tilings)
        # Update q-value for each tiling by update factor alpha
        for grid_state, Q in zip(encoded_state, self.q_tables):
            _value = Q.q_table[grid_state[0]][grid_state[1]][action]
            Q.q_table[grid_state[0]][grid_state[1]][action] = alpha * value + (1.0 - alpha) * _value


