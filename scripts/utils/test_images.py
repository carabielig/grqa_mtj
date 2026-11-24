import numpy as np
from perlin_noise import PerlinNoise

from scripts.utils.turing import TwoDimensionalRDEquations

### Some code snippets for test images were provided by Norbert Marwan


def get_longitudinal_wave_test_image(size):
    x = np.linspace(0, 8 * np.pi, size)
    y = np.linspace(0, 0 * np.pi, size)  # for longitudinal wave pattern
    X, Y = np.meshgrid(x, y)

    # Create a periodic pattern using sine or cosine functions
    return np.sin(X) + np.cos(Y)


def get_periodic_test_image(size, amplitude=1, n=8):
    x = np.linspace(0, n * np.pi, size)
    y = np.linspace(0, n * np.pi, size)  # for checkerboard like pattern
    X, Y = np.meshgrid(x, y)

    # Create a periodic pattern using sine or cosine functions
    return amplitude * np.sin(X) + amplitude * np.cos(Y)


def get_fractal_test_image(levels=1):

    assert levels.is_integer(), "levels must be an integer."

    size = 3**levels

    fractal = np.ones((size, size))

    for lev in range(1, levels + 1):
        l = int(size / 3**lev)
        square_pattern = np.ones((3 * l, 3 * l))
        square_pattern[l : 2 * l, l : 2 * l] = 0

        for row in range(3 ** (lev - 1)):
            for col in range(3 ** (lev - 1)):
                if (
                    not np.sum(
                        fractal[
                            row * 3 * l : (row + 1) * 3 * l,
                            col * 3 * l : (col + 1) * 3 * l,
                        ]
                    )
                    == 0
                ):
                    fractal[
                        row * 3 * l : (row + 1) * 3 * l, col * 3 * l : (col + 1) * 3 * l
                    ] = square_pattern

    return fractal


def get_logistic_map_test_image(size):

    def logistic_map(r, x):
        return r * x * (1 - x)

    # Initialize the grid with random values
    r = 3.9  # Choose a parameter in the chaotic regime
    Z_chaotic = np.random.rand(size, size)

    # Iterate the logistic map over the grid to generate chaos
    for i in range(1, size):
        for j in range(1, size):
            Z_chaotic[i, j] = logistic_map(r, Z_chaotic[i - 1, j - 1])

    return Z_chaotic


def get_correlated_noise_test_image(size):

    noise = PerlinNoise(octaves=10, seed=1)
    Z_correlated_noise = np.array(
        [[noise([i / size, j / size]) for j in range(size)] for i in range(size)]
    )

    return Z_correlated_noise


def get_uniformly_distributed_white_noise(size):
    return np.random.uniform(0, 1, (size, size))


def get_gaussian_distributed_white_noise(size):
    return np.random.normal(0, 1, (size, size))  # mean=0, stddev=1


def get_turing_test_image(size):
    Da, Db, alpha, beta = 1, 100, -0.005, 10

    def Ra(a, b):
        return a - a**3 - b + alpha

    def Rb(a, b):
        return (a - b) * beta

    dx = 1
    dt = 0.001

    Z_turing, b = TwoDimensionalRDEquations(
        Da, Db, Ra, Rb, width=size, height=size, dx=dx, dt=dt, steps=150
    ).evolution_outcome_as_array(n_steps=150)

    return Z_turing


def get_checkerboard(box_size, num_boxes):
    """
    Generate a checkerboard pattern.
    from: https://stackoverflow.com/questions/2169478/how-to-make-a-checkerboard-in-numpy
    """

    assert isinstance(box_size, int), "Box size has to be an Integer."
    assert isinstance(num_boxes, int), "Number of Boxes has to be an Integer."

    cb = np.kron(
        [[1, 0] * num_boxes, [0, 1] * num_boxes] * num_boxes,
        np.ones((box_size, box_size)),
    )

    return cb


def get_special_checkerboard(size):
    """
    Generate a special checkerboard pattern that has a LAM = 0.
    """

    im = np.zeros((size, size))
    im[::2, :] = 1
    im[::2, ::2] = 2
    im[1::2, 1::2] = 3

    return im
