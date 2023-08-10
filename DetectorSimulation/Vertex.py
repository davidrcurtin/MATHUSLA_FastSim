from typing import List
from DetectorSimulation.Particle import Particle
from DetectorSimulation.lorentz_transformation import *


class Vertex:

    position: np.ndarray
    momentum: np.ndarray
    mass: float
    pid: int
    decay_product: List[Particle]

    def __init__(self, position: tuple, momentum: tuple, pid: int, decay_product: List[Particle], mass=None):
        """Set instance variables:
                position: 1d array (x, y, z)
                four_momentum: 1d array (E, px, py, pz)
                decay_product: list of particles
        """
        # Validate input
        assert len(position) == 3, "Wrong position input"
        assert len(momentum) == 4, "Wrong momentum input"
        for par in decay_product:
            assert tuple(par.position) == position
        self.position = np.array(position)
        self.momentum = np.array(momentum)
        if mass is None:
            self.mass = np.sqrt(max(0, self.momentum[0] ** 2 - self.momentum[1:].dot(self.momentum[1:])))
        elif mass < 0:
            self.mass = 0
            self.momentum[0] = np.linalg.norm(self.momentum[1:])
        else:
            self.mass = float(mass)
        self.pid = pid
        self.decay_product = decay_product

    """
    # ======Getters and setters======
    def get_position(self):
        return self._position

    def get_momentum(self):
        return self._momentum

    def get_decay_product(self):
        return self._decay_product

    def set_position(self, position: tuple):
        assert len(position) == 3
        self._position = np.array(position)

    def set_momentum(self, momentum: tuple):
        assert len(momentum) == 4
        self._momentum = np.array(momentum)

    def set_decay_product(self, decay_product: List[Particle]):
        for par in decay_product:
            assert par.position == self._position
        self._decay_product = decay_product

    def add_decay_product(self, decay_product: List[Particle]):
        for par in decay_product:
            assert par.position == self._position
        self._decay_product.extend(decay_product)
    """

    # ======Physics======
    def boost_this_vertex(self, direction: tuple, beta: float):
        """Boost this llp and decay products in direction and beta"""
        assert len(direction) == 3
        # Convert direction to array
        direction_arr = np.array(direction)
        # Normalize direction
        unit_direction = direction_arr/np.linalg.norm(direction_arr)
        # Apply boost to vertex
        boost = boost_matrix(beta*unit_direction)
        self.momentum = boost.dot(self.momentum)
        # Boost decay product
        for par in self.decay_product:
            par.boost_by(boost)
        return None

    def boost_new_vertex(self, direction: tuple, beta: float):
        new_vertex = self.clone()
        new_vertex.boost_this_vertex(direction, beta)
        return new_vertex

    def shift_this_vertex(self, shift_vector: tuple):
        """Shift this vertex's position by shift_vector (1d array [x, y, z])"""
        assert len(shift_vector) == 3
        self.position += np.array(shift_vector)
        for par in self.decay_product:
            par.shift_this_particle(shift_vector)
        return None

    def shift_new_vertex(self, shift_vector: tuple):
        new_vertex = self.clone()
        new_vertex.shift_this_vertex(shift_vector)
        return new_vertex

    def place_this_vertex(self, new_position: tuple):
        """Place this vertex at new_position."""
        assert len(new_position) == 3
        self.position = np.array(new_position)
        for par in self.decay_product:
            par.place_this_particle(new_position)
        return None

    def place_new_vertex(self, new_position: tuple):
        new_vertex = self.clone()
        new_vertex.place_this_vertex(new_position)
        return new_vertex

    def rotate_this_vertex(self, rotation_vector: tuple, rotation_angle: float):
        """Rotate momentum by rotation_angle by rotation_vector.
            rotation_vector: 1d array [px, py, pz]
            rotation_angle: in radians
        """
        assert len(rotation_vector) == 3
        p = self.momentum[1:]
        # Convert rotation_vector to array
        rotation_vector_arr = np.array(rotation_vector)
        # Normalize rotation axis
        axis = rotation_vector_arr / np.linalg.norm(rotation_vector_arr)
        self.momentum[1:] = rotation(p, axis, rotation_angle)
        # Rotate each particle in decay product
        for par in self.decay_product:
            par.momentum[1:] = rotation(par.momentum[1:], axis, rotation_angle)
        return None

    def rotate_new_vertex(self, rotation_vector: tuple, rotation_angle: float):
        new_par = self.clone()
        new_par.rotate_this_vertex(rotation_vector, rotation_angle)
        return new_par

    def clone(self):
        decay_product = []
        for par in self.decay_product:
            decay_product.append(par.clone())
        new_vertex = Vertex(tuple(self.position), tuple(self.momentum), self.pid, decay_product, self.mass)
        return new_vertex
