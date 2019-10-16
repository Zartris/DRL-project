# Import common libraries
import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from Q_table_tiling import TiledQTable


def create_tiling_grid(low, high, bins=(10, 10), offsets=(0.0, 0.0)):
    """Define a uniformly-spaced grid that can be used for tile-coding a space.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins or tiles along each corresponding dimension.
    offsets : tuple
        Split points for each dimension should be offset by these values.

    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    """
    result = []
    dim = len(bins)
    for i in range(dim):
        step = (high[i] - low[i]) / bins[i]
        value = low[i] + step
        grid = []
        while value < high[i]:
            grid.append(value + offsets[i])
            value = value + step
        result.append(np.asarray(grid))
    print(result)
    return result


def create_tilings(low, high, tiling_specs):
    """Define multiple tilings using the provided specifications.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    tiling_specs : list of tuples
        A sequence of (bins, offsets) to be passed to create_tiling_grid().

    Returns
    -------
    tilings : list
        A list of tilings (grids), each produced by create_tiling_grid().
    """
    return [create_tiling_grid(low, high, bin, offsets) for bin, offsets in tiling_specs]


def visualize_tilings(tilings):
    """Plot each tiling as a grid."""
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    linestyles = ['-', '--', ':']
    legend_lines = []

    fig, ax = plt.subplots(figsize=(10, 10))
    for i, grid in enumerate(tilings):
        for x in grid[0]:
            l = ax.axvline(x=x, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], label=i)
        for y in grid[1]:
            l = ax.axhline(y=y, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)])
        legend_lines.append(l)
    ax.grid('off')
    ax.legend(legend_lines, ["Tiling #{}".format(t) for t in range(len(legend_lines))], facecolor='white',
              framealpha=0.9)
    ax.set_title("Tilings")
    return ax  # return Axis object to draw on later, if needed


def visualize_encoded_samples(samples, encoded_samples, tilings, low=None, high=None):
    """Visualize samples by activating the respective tiles."""
    samples = np.array(samples)  # for ease of indexing

    # Show tiling grids
    ax = visualize_tilings(tilings)

    # If bounds (low, high) are specified, use them to set axis limits
    if low is not None and high is not None:
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
    else:
        # Pre-render (invisible) samples to automatically set reasonable axis limits, and use them as (low, high)
        ax.plot(samples[:, 0], samples[:, 1], 'o', alpha=0.0)
        low = [ax.get_xlim()[0], ax.get_ylim()[0]]
        high = [ax.get_xlim()[1], ax.get_ylim()[1]]

    # Map each encoded sample (which is really a list of indices) to the corresponding tiles it belongs to
    tilings_extended = [np.hstack((np.array([low]).T, grid, np.array([high]).T)) for grid in
                        tilings]  # add low and high ends
    tile_centers = [(grid_extended[:, 1:] + grid_extended[:, :-1]) / 2 for grid_extended in
                    tilings_extended]  # compute center of each tile
    tile_toplefts = [grid_extended[:, :-1] for grid_extended in tilings_extended]  # compute topleft of each tile
    tile_bottomrights = [grid_extended[:, 1:] for grid_extended in tilings_extended]  # compute bottomright of each tile

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for sample, encoded_sample in zip(samples, encoded_samples):
        for i, tile in enumerate(encoded_sample):
            # Shade the entire tile with a rectangle
            topleft = tile_toplefts[i][0][tile[0]], tile_toplefts[i][1][tile[1]]
            bottomright = tile_bottomrights[i][0][tile[0]], tile_bottomrights[i][1][tile[1]]
            ax.add_patch(Rectangle(topleft, bottomright[0] - topleft[0], bottomright[1] - topleft[1],
                                   color=colors[i], alpha=0.33))

            # In case sample is outside tile bounds, it may not have been highlighted properly
            if any(sample < topleft) or any(sample > bottomright):
                # So plot a point in the center of the tile and draw a connecting line
                cx, cy = tile_centers[i][0][tile[0]], tile_centers[i][1][tile[1]]
                ax.add_line(Line2D([sample[0], cx], [sample[1], cy], color=colors[i]))
                ax.plot(cx, cy, 's', color=colors[i])

    # Finally, plot original samples
    ax.plot(samples[:, 0], samples[:, 1], 'o', color='r')

    ax.margins(x=0, y=0)  # remove unnecessary margins
    ax.set_title("Tile-encoded samples")
    return ax


def discretize(sample, grid):
    """Discretize a sample as per given grid.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    grid : list of array_like
        A list of arrays containing split points for each dimension.

    Returns
    -------
    discretized_sample : array_like
        A sequence of integers with the same number of dimensions as sample.
    """
    return tuple(int(np.digitize(s, g)) for s, g in zip(sample, grid))


def tile_encode(sample, tilings, flatten=False):
    """Encode given sample using tile-coding.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    tilings : list
        A list of tilings (grids), each produced by create_tiling_grid().
    flatten : bool
        If true, flatten the resulting binary arrays into a single long vector.

    Returns
    -------
    encoded_sample : list or array_like
        A list of binary vectors, one for each tiling, or flattened into one.
    """
    encoded_sample = [discretize(sample, grid) for grid in tilings]
    return np.concatenate(encoded_sample) if flatten else encoded_sample


if __name__ == '__main__':
    # Set plotting options
    ex1 = False
    ex2 = True
    ex3 = True

    plt.style.use('ggplot')
    np.set_printoptions(precision=3, linewidth=120)
    env = gym.make('Acrobot-v1')
    env.seed(505)
    if ex1:
        # Explore state (observation) space
        print("State space:", env.observation_space)
        print("- low:", env.observation_space.low)
        print("- high:", env.observation_space.high)

        # Explore action space
        print("Action space:", env.action_space)
    elif ex2:
        low = [-1.0, -5.0]
        high = [1.0, 5.0]
        # create_tiling_grid(low, high, bins=(10, 10), offsets=(-0.1, 0.5))  # [test]

        tiling_specs = [((10, 10), (-0.066, -0.33)),
                        ((10, 10), (0.0, 0.0)),
                        ((10, 10), (0.066, 0.33))]
        tilings = create_tilings(low, high, tiling_specs)
        visualize_tilings(tilings)
        plt.show()
        # Test with some sample values
        samples = [(-1.2, -5.1),
                   (-0.75, 3.25),
                   (-0.5, 0.0),
                   (0.25, -1.9),
                   (0.15, -1.75),
                   (0.75, 2.5),
                   (0.7, -3.7),
                   (1.0, 5.0)]
        encoded_samples = [tile_encode(sample, tilings) for sample in samples]
        print("\nSamples:", repr(samples), sep="\n")
        print("\nEncoded samples:", repr(encoded_samples), sep="\n")
        visualize_encoded_samples(samples, encoded_samples, tilings)
        plt.show()
    elif ex3:
        samples = [(-1.2, -5.1),
                   (-0.75, 3.25),
                   (-0.5, 0.0),
                   (0.25, -1.9),
                   (0.15, -1.75),
                   (0.75, 2.5),
                   (0.7, -3.7),
                   (1.0, 5.0)]
        low = [-1.0, -5.0]
        high = [1.0, 5.0]
        # create_tiling_grid(low, high, bins=(10, 10), offsets=(-0.1, 0.5))  # [test]

        tiling_specs = [((10, 10), (-0.066, -0.33)),
                        ((10, 10), (0.0, 0.0)),
                        ((10, 10), (0.066, 0.33))]
        tq = TiledQTable(low, high, tiling_specs, 2)
        # Test with a sample Q-table
        s1 = 3
        s2 = 4
        a = 0
        q = 1.0
        print(
            "[GET]    Q({}, {}) = {}".format(samples[s1], a,
                                             tq.get(samples[s1], a)))  # check value at sample = s1, action = a
        print("[UPDATE] Q({}, {}) = {}".format(samples[s2], a, q));
        tq.update(samples[s2], a, q)  # update value for sample with some common tile(s)
        print("[GET]    Q({}, {}) = {}".format(samples[s1], a,
                                               tq.get(samples[s1], a)))  # check value again, should be slightly updated
