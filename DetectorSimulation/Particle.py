from DetectorSimulation.lorentz_transformation import *


class Particle:
    """A particle with position and momentum and PID.
    """

    position: np.ndarray
    momentum: np.ndarray
    mass: float
    pid: int

    def __init__(self, position: tuple, momentum: tuple, pid: int, mass=None):
        """Set instance variables:
                position: list [x, y, z]
                momentum: list [E, px, py, pz]
                pid: particle by PDG code
                mass: mass of particle, optional to input as can be set using 4-vector also
        """
        # Check input validity
        assert len(position) == 3, "Wrong position input"
        assert len(momentum) == 4, "Wrong momentum input"
        self.position, self.momentum = np.array(position), np.array(momentum)
        if mass is None:
            self.mass = np.sqrt(max(0, self.momentum[0] ** 2 - self.momentum[1:].dot(self.momentum[1:])))
        elif mass < 0:
            self.mass = 0
            self.momentum[0] = np.linalg.norm(self.momentum[1:])
        else:
            self.mass = float(mass)
        self.pid = pid

    def boost_this_particle(self, direction: tuple, beta: float):
        """Boost this particle in direction and beta"""
        assert len(direction) == 3
        # Convert to array
        direction_arr = np.array(direction)
        # Normalize direction
        unit_direction = direction_arr / np.linalg.norm(direction_arr)
        # Apply boost
        self.boost_by(boost_matrix(beta * unit_direction))
        return None

    def boost_by(self, boost: np.ndarray):
        """Boost by given boost matrix"""
        self.momentum = boost.dot(self.momentum)
        return None

    def boost_new_particle(self, direction: tuple, beta: float):
        new_par = self.clone()
        new_par.boost_this_particle(direction, beta)
        return new_par

    def shift_this_particle(self, shift_vector: tuple):
        """Shift this particle's position by shift_vector (tuple (x, y, z))"""
        assert len(shift_vector) == 3
        self.position += np.array(shift_vector)
        return None

    def shift_new_particle(self, shift_vector: tuple):
        new_par = self.clone()
        new_par.shift_this_particle(shift_vector)
        return new_par

    def place_this_particle(self, new_position: tuple):
        """Place this particle at new_position (x, y, z)."""
        assert len(new_position) == 3
        self.position = np.array(new_position)
        return None

    def place_new_particle(self, new_position: tuple):
        new_par = self.clone()
        new_par.place_this_particle(new_position)
        return new_par

    def rotate_this_particle(self, rotation_vector: tuple, rotation_angle: float):
        """Rotate momentum by rotation_angle by rotation_vector.
            rotation_vector: list [px, py, pz]
            rotation_angle: in radians
        """
        assert len(rotation_vector) == 3
        p = self.momentum[1:]
        # Convert rotation_vector to array
        rotation_vector_arr = np.array(rotation_vector)
        # Normalize rotation axis
        axis = rotation_vector_arr / np.linalg.norm(rotation_vector_arr)
        # Rodrigues' rotation formula from wikipedia
        self.momentum[1:] = rotation(p, axis, rotation_angle)
        return None

    def rotate_new_particle(self, rotation_vector: tuple, rotation_angle: float):
        new_par = self.clone()
        new_par.rotate_this_particle(rotation_vector, rotation_angle)
        return new_par

    def clone(self):
        new_particle = Particle(tuple(self.position), tuple(self.momentum), self.pid)
        return new_particle


if __name__ == "__main__":
    pass
