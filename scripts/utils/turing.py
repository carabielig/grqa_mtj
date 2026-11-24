# code from: https://github.com/ijmbarr/turing-patterns/blob/master/turing-patterns.ipynb
# and https://github.com/ijmbarr/turing-patterns/blob/master/tutils.py

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def laplacian1D(a, dx):
    return (-2 * a + np.roll(a, 1, axis=0) + np.roll(a, -1, axis=0)) / (dx**2)


def laplacian2D(a, dx):
    return (
        -4 * a
        + np.roll(a, 1, axis=0)
        + np.roll(a, -1, axis=0)
        + np.roll(a, +1, axis=1)
        + np.roll(a, -1, axis=1)
    ) / (dx**2)


def random_initialiser(shape):
    return (
        np.random.normal(loc=0, scale=0.05, size=shape),
        np.random.normal(loc=0, scale=0.05, size=shape),
    )


class BaseStateSystem:
    """
    Base object for "State System".

    We are going to repeatedly visualise systems which are Markovian:
    the have a "state", the state evolves in discrete steps, and the next
    state only depends on the previous state.

    To make things simple, I'm going to use this class as an interface.
    """

    def __init__(self):
        raise NotImplementedError()

    def initialise(self):
        raise NotImplementedError()

    def initialise_figure(self):
        fig, ax = plt.subplots()
        return fig, ax

    def update(self):
        raise NotImplementedError()

    def draw(self, ax):
        raise NotImplementedError()

    def plot_time_evolution(self, filename, n_steps=30):
        """
        Creates a gif from the time evolution of a basic state syste.
        """
        self.initialise()
        fig, ax = self.initialise_figure()

        def step(t):
            self.update()
            self.draw(ax)

        anim = animation.FuncAnimation(
            fig, step, frames=np.arange(n_steps), interval=20
        )
        anim.save(filename=filename, dpi=60, fps=10, writer="imagemagick")
        plt.close()

    def plot_evolution_outcome(self, filename, n_steps):
        """
        Evolves and save the outcome of evolving the system for n_steps
        """
        self.initialise()
        fig, ax = self.initialise_figure()

        for _ in range(n_steps):
            self.update()

        self.draw(ax)
        # plt.show()
        fig.savefig(filename)
        plt.close()

    def evolution_outcome_as_array(self, n_steps):
        """
        Returns the outcome of evolving the system for n_steps as arrays
        """

        self.initialise()

        for _ in range(n_steps):
            self.update()

        return self.a, self.b


class TwoDimensionalRDEquations(BaseStateSystem):
    def __init__(
        self,
        Da,
        Db,
        Ra,
        Rb,
        initialiser=random_initialiser,
        width=1000,
        height=1000,
        dx=1,
        dt=0.1,
        steps=1,
    ):
        self.Da = Da
        self.Db = Db
        self.Ra = Ra
        self.Rb = Rb

        self.initialiser = initialiser
        self.width = width
        self.height = height
        self.shape = (width, height)
        self.dx = dx
        self.dt = dt
        self.steps = steps

    def initialise(self):
        self.t = 0
        self.a, self.b = self.initialiser(self.shape)

    def update(self):
        for _ in range(self.steps):
            self.t += self.dt
            self._update()

    def _update(self):
        # unpack so we don't have to keep writing "self"
        a, b, Da, Db, Ra, Rb, dt, dx = (
            self.a,
            self.b,
            self.Da,
            self.Db,
            self.Ra,
            self.Rb,
            self.dt,
            self.dx,
        )

        La = laplacian2D(a, dx)
        Lb = laplacian2D(b, dx)

        delta_a = dt * (Da * La + Ra(a, b))
        delta_b = dt * (Db * Lb + Rb(a, b))

        self.a += delta_a
        self.b += delta_b

    def draw(self, ax):
        ax[0].clear()
        ax[1].clear()

        ax[0].imshow(self.a, cmap="jet")
        ax[1].imshow(self.b, cmap="brg")

        # not working for newer matplotlib versions
        # ax[0].grid(b=False)
        # ax[1].grid(b=False)
        ax[0].grid(visible=False)
        ax[1].grid(visible=False)

        ax[0].set_title("A, t = {:.2f}".format(self.t))
        ax[1].set_title("B, t = {:.2f}".format(self.t))

    def initialise_figure(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        return fig, ax
