from random import uniform


def uniform_ephemeral(mn: float = -0.5, mx: float = 0.5):
    def uniform_ephemeral_():
        return uniform(mn, mx)

    return uniform_ephemeral_

