## Authors: Wentao Cui, Jaipratap Singh Grewal, Yiyan (Lillian) Luo

## Note: certain functions in the sections labelled by "Functions for Trigger Studies" are not relevant to geometric efficiency purposes and can be safely ignored BUT NOT DELETED as the rest of the code might have some dependencies on them.

import os
from typing import List, Union, Dict, Optional
import numpy as np
import copy
import random

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from DetectorSimulation.Particle import Particle
from DetectorSimulation.Vertex import Vertex


class Module:
    label: int # module label for hit_tracking thru comparing with criteria
    typ: str # 'x', 'y', 'z' corresponding to alignment of area vector normal to plane of detector module
    x_index: float  # "index of module along x axis", lowest x-coord of module
    y_index: float  # "index of module along y axis", lowest y-coord of module
    z_index: float  # "index of module along z axis", lowest z-coord of module
    x_dim: int # x-dimension (0 if no dimension along x_direction)
    y_dim: int # y-dimension (0 if no dimension along y_direction)
    z_dim: int # z_dimension, (0 if no dimension along z_direction)
    long_direction: int # longitudinal direction of this module
    tracker_hits: list 
        
    def __init__(self, typ, label, x_index: float, y_index: float, z_index: float, x_dim: int, y_dim: int, z_dim: int, long_direction: int, detector):
        self.label = label
        self.typ = typ
        self.x_index = x_index
        self.y_index = y_index
        self.z_index = z_index
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.long_direction = long_direction
        self.detector = detector
        self.tracker_hits = []
        self.next = None # keep track of distance to module formed using the next values of x_mesh, y_mesh (helps with trigger)

    def add_new_hit(self, hit):
        self.tracker_hits.append(hit)

    def clear_module(self):
        self.tracker_hits = []

    def get_index(self):
        return self.x_index, self.y_index, self.z_index
    
    def get_label(self):
        return self.label
    
    def get_typ(self):
        return self.typ
    
    def set_next(self, dist):
        self.next = dist

    def __str__(self):
        return f"module {(self.x_index, self.y_index, self.z_index)}, hit layer {self.typ}, long direction {self.long_direction}"


class TrackerHit:

    hit_coordinate: np.ndarray  # position of tracker hit
    hit_module: Module

    def __init__(self, hit_coordinate: np.ndarray, hit_module: Module, track, detector, time):
        self.hit_coordinate = hit_coordinate
        self.hit_module = hit_module
        self.track = track
        self.detector = detector
        self.time = time # time of the hit, needed for scint_resolve
        # append this tracker hit to list of hits within the module
        self.hit_module.add_new_hit(self)

    def __str__(self):
        return f"{self.hit_coordinate}, {str(self.hit_module)}\n"


class DetectorParticle:
    particle: Particle
    tracker_hits: List[TrackerHit]
    _visibility: int
    recon_criteria: List[str]  # list of PASSED recon criteria
    trigger_criteria: List[str]  # list of PASSED trigger criteria
    decay_volume_hits: dict  # hits on 6 walls of decay volume
    detector_volume_hits: dict  # hits on 6 walls of detector volume
    wall_hit: dict
    trigger_grids: List[int] # list of unique labels of 3x3 grids in which particle triggered

    INVISIBLE = -1
    LOW_ENERGY = 0
    VISIBLE = 1

    def __init__(self, particle: Particle, detector):

        self.particle = particle
        self.detector = detector
        self.tracker_hits = []
        self._visibility = DetectorParticle.VISIBLE
        self.decay_volume_hits = dict()
        self._decay_volume_hit()
        self.detector_volume_hits = dict()
        self._detector_volume_hit()
        self.wall_hit = dict()
        self.wall_hit["DECAY"] = self._is_decay_hit()  # get wall hits on decay volume
        self.wall_hit["DETECTOR"] = self._is_detector_hit()  # get wall hits on detector volume
        self.recon_criteria = []
        self.trigger_criteria = []
        self.trigger_grids = []
        if self.wall_hit["DETECTOR"]:
            self._hit_detect()  # run hit detection
            # test reconstruction if particle is visible and as single particle event no scintillator_resolve needed
            if self.detector.current_event_mode == Detector.PARTICLE and self._visibility == DetectorParticle.VISIBLE:
                self._test_recon_criteria()
                self._test_trigger_criteria()
                
    def test_criteria(self): # this is only called if current_event is Detector.Vertex
        if self._visibility == DetectorParticle.VISIBLE:  # only test for reconstruction if particle is visible
            self._test_recon_criteria()
            self._test_trigger_criteria()  

    def _hit_detect(self): # remember momentum is a 4-vector
        # Check if particle momentum passes minimum
        # if particle ID is not among list of visible (charged) particles, set to invisible and do not hit detect
        
        if abs(self.particle.pid) not in self.detector.config.min_particle_momenta:
            self._visibility = DetectorParticle.INVISIBLE  # no hit detection for invisible particles
        else:  # for particles with visible PID
            if np.linalg.norm(self.particle.momentum[1:]) < self.detector.config.min_particle_momenta[abs(self.particle.pid)]:
                self._visibility = DetectorParticle.LOW_ENERGY  # low energy does not count toward reconstruction, but hit detect
            # get particle position and momentum, calculate (x, y) for each tracker layer z coordinate
            for key in self.detector.module_groups.keys():
                if self.detector.module_groups[key]["type"] == 'x' and 'x' in self.detector.types:
                    if self.particle.momentum[1] == 0:  # particle not moving in x direction
                        pass
                    else:

                        for i in range(len(self.detector.module_groups[key]["x_layer"])):

                                x_coord = self.detector.module_groups[key]["x_layer"][i] 
    
                                try:
                                    hit_yz, time = self._trajectory(x_coord, 'x')  # throw assertion error if time is -ve
                                    hit_coord = np.append([x_coord], hit_yz)  # hit coordinate (x, y, z)
                                    j = 0

                                    while j < len(self.detector.module_groups[key]["y_mesh"]):
                                        # check if hit coord y is in range of any module
                                        if self.detector.module_groups[key]["module_y_range"][j][0] <= hit_yz[0] <= self.detector.module_groups[key]["module_y_range"][j][1]:
                                            k = 0
                                            # check if hit coord z is in range of any module
                                            while k < len(self.detector.module_groups[key]["z_range"]):
                                                if self.detector.module_groups[key]["z_range"][k][0] <= hit_yz[1] <= self.detector.module_groups[key]["z_range"][k][1]:
                                                    
                                                    y = self.detector.module_groups[key]["module_y_range"][k][0]
                                                    z = self.detector.module_groups[key]["z_range"][k][0]
                                                    self.tracker_hits.append(TrackerHit(hit_coord, self.detector.modules[(x_coord,y,z)], self, self.detector, time))
                                                
                                                    break
                                                   
                                                k += 1
    
                                            break
        
                                        j += 1
                                except AssertionError:
                                    pass
                                   
                
                elif self.detector.module_groups[key]["type"] == 'y' and 'y' in self.detector.types:   
                    if self.particle.momentum[2] == 0:  # particle not moving in y direction
                        pass
             
                    else:
                        for i in range(len(self.detector.module_groups[key]["y_layer"])):

                                y_coord = self.detector.module_groups[key]["y_layer"][i] 
                                try:
                                    hit_xz, time = self._trajectory(y_coord, 'y')  # throw assertion error if time is -ve
                                    hit_coord = np.array([hit_xz[0], y_coord, hit_xz[1]])  # hit coordinate (x, y, z)
                                    j = 0

                                    while j < len(self.detector.module_groups[key]["x_mesh"]):
                                        # check if hit coord x is in range of any module
                                        if self.detector.module_groups[key]["module_x_range"][j][0] <= hit_xz[0] <= self.detector.module_groups[key]["module_x_range"][j][1]:
                                            k = 0
                                            
                                            # check if hit coord z is in range of any module
                                            while k < len(self.detector.module_groups[key]["z_range"]):
                                                if self.detector.module_groups[key]["z_range"][k][0] <= hit_xz[1] <= self.detector.module_groups[key]["z_range"][k][1]:

                                                    x = self.detector.module_groups[key]["module_x_range"][j][0]
                                                    z = self.detector.module_groups[key]["z_range"][k][0]

                                                    self.tracker_hits.append(TrackerHit(hit_coord, self.detector.modules[(x, y_coord, z)], self, self.detector, time))
                                                    
                                                    break
                                                    
                                                k += 1
                                            break

                                        j += 1
                                except AssertionError:
                                    pass
                                    
                
                elif self.detector.module_groups[key]["type"] == 'z' and 'z' in self.detector.types:
                    if self.particle.momentum[3] == 0:  # particle not moving in z direction
                        pass
                    
                    else:

                        for i in range(len(self.detector.module_groups[key]["z_layer"])):
                            z_coord = self.detector.module_groups[key]["z_layer"][i]

                            try:
                                hit_xy, time = self._trajectory(z_coord, 'z')  # throw assertion error if time is -ve
                                hit_coord = np.append(hit_xy, [z_coord])  # hit coordinate (x, y, z)
                                j = 0

                                while j < len(self.detector.module_groups[key]["x_mesh"]):
                                    # check if hit coord y is in range of any module
                                    if self.detector.module_groups[key]["module_x_range"][j][0] <= hit_xy[0] <= self.detector.module_groups[key]["module_x_range"][j][1]:
                                        k = 0
                                        # check if hit coord z is in range of any module
                                        while k < len(self.detector.module_groups[key]["y_mesh"]):
                                            if self.detector.module_groups[key]["module_y_range"][k][0] <= hit_xy[1] <= self.detector.module_groups[key]["module_y_range"][k][1]:
                                                x = self.detector.module_groups[key]["module_x_range"][j][0]
                                                y = self.detector.module_groups[key]["module_y_range"][k][0]

                                                self.tracker_hits.append(TrackerHit(hit_coord, self.detector.modules[(x,y,z_coord)], self, self.detector, time))
                                                break

                                            k += 1
                                        break

                                    j += 1
       
                            except AssertionError:
                                pass
                 
                    
           
    def _trajectory(self, coord, typ): # used within the hit_detect function
        # typ is a string: 'x' or 'y' or 'z' i.e. the unit vector of the coordinate
        if typ == 'x':
            # find yz coordinate corresponding to given x
            start_yz = self.particle.position[1:]
            start_x = self.particle.position[0]
            t = (coord - start_x) / self.particle.momentum[1] # 4-momentum is used
            assert t >= 0
            return (start_yz + t * self.particle.momentum[2:], t)
        elif typ == 'y':
            # find xz coordinate corresponding to given y
            start_xz = [self.particle.position[0], self.particle.position[-1]]
            start_y = self.particle.position[1]
            t = (coord - start_y) / self.particle.momentum[2] # 4-momentum is used
            assert t >= 0
            return (start_xz + t * np.array([self.particle.momentum[1], self.particle.momentum[3]]), t)
        
        elif typ == 'z':
            # find xy coordinate of particle trajectory at given z
            start_xy = self.particle.position[:2]
            start_z = self.particle.position[2]
            t = (coord - start_z) / self.particle.momentum[3]
            assert t >= 0
            return (start_xy + t * self.particle.momentum[1:3], t)

    def _decay_volume_hit(self):
        # back
        self.decay_volume_hits["x-"] = self._hit_surface(self.detector.config.decay_x_min, 0, 1, (self.detector.config.decay_y_min, self.detector.config.decay_y_max), 2, (self.detector.config.decay_z_min, self.detector.config.decay_z_max))
        # front
        self.decay_volume_hits["x+"] = self._hit_surface(self.detector.config.decay_x_max, 0, 1, (self.detector.config.decay_y_min, self.detector.config.decay_y_max), 2, (self.detector.config.decay_z_min, self.detector.config.decay_z_max))
        # left
        self.decay_volume_hits["y-"] = self._hit_surface(self.detector.config.decay_y_min, 1, 0, (self.detector.config.decay_x_min, self.detector.config.decay_x_max), 2, (self.detector.config.decay_z_min, self.detector.config.decay_z_max))
        # right
        self.decay_volume_hits["y+"] = self._hit_surface(self.detector.config.decay_y_max, 1, 0, (self.detector.config.decay_x_min, self.detector.config.decay_x_max), 2, (self.detector.config.decay_z_min, self.detector.config.decay_z_max))
        # bottom
        self.decay_volume_hits["z-"] = self._hit_surface(self.detector.config.decay_z_min, 2, 0, (self.detector.config.decay_x_min, self.detector.config.decay_x_max), 1, (self.detector.config.decay_y_min, self.detector.config.decay_y_max))
        # top
        self.decay_volume_hits["z+"] = self._hit_surface(self.detector.config.decay_z_max, 2, 0, (self.detector.config.decay_x_min, self.detector.config.decay_x_max), 1, (self.detector.config.decay_y_min, self.detector.config.decay_y_max))

    def _detector_volume_hit(self):
        # back
        if self.decay_volume_hits["x-"][0]:
            self.detector_volume_hits["x-"] = self.decay_volume_hits["x-"]
        else:
            self.detector_volume_hits["x-"] = self._hit_surface(self.detector.config.decay_x_min, 0, 1, (self.detector.config.decay_y_min, self.detector.config.decay_y_max), 2, (self.detector.config.detector_z_min, self.detector.config.detector_z_max))
        # front
        if self.decay_volume_hits["x+"][0]:
            self.detector_volume_hits["x+"] = self.decay_volume_hits["x+"]
        else:
            self.detector_volume_hits["x+"] = self._hit_surface(self.detector.config.decay_x_max, 0, 1, (self.detector.config.decay_y_min, self.detector.config.decay_y_max), 2, (self.detector.config.detector_z_min, self.detector.config.detector_z_max))
        # left
        if self.decay_volume_hits["y-"][0]:
            self.detector_volume_hits["y-"] = self.decay_volume_hits["y-"]
        else:
            self.detector_volume_hits["y-"] = self._hit_surface(self.detector.config.decay_y_min, 1, 0, (self.detector.config.decay_x_min, self.detector.config.decay_x_max), 2, (self.detector.config.detector_z_min, self.detector.config.detector_z_max))
        # right
        if self.decay_volume_hits["y+"][0]:
            self.detector_volume_hits["y+"] = self.decay_volume_hits["y+"]
        else:
            self.detector_volume_hits["y+"] = self._hit_surface(self.detector.config.decay_y_max, 1, 0, (self.detector.config.decay_x_min, self.detector.config.decay_x_max), 2, (self.detector.config.detector_z_min, self.detector.config.detector_z_max))
        # bottom
        self.detector_volume_hits["z-"] = self._hit_surface(self.detector.config.detector_z_min, 2, 0, (self.detector.config.decay_x_min, self.detector.config.decay_x_max), 1, (self.detector.config.decay_y_min, self.detector.config.decay_y_max))
        # top
        self.detector_volume_hits["z+"] = self._hit_surface(self.detector.config.detector_z_max, 2, 0, (self.detector.config.decay_x_min, self.detector.config.decay_x_max), 1, (self.detector.config.decay_y_min, self.detector.config.decay_y_max))

    def _is_decay_hit(self):
        for key in self.decay_volume_hits:
            if self.decay_volume_hits[key][0]:
                return True
        return False

    def _is_detector_hit(self):
        for key in self.decay_volume_hits:
            if self.detector_volume_hits[key][0]:
                return True
        return False

    def _hit_surface(self, surface: float, k, i, i_range, j, j_range):
        """
        (helper) Check if particle hits a given surface along its trajectory, and if True, where the hit is.
        :param surface: coordinate that defines the surface (e.g. x=0 defines yz plane)
        :param k: index of the coordinate defining the surface
        :param i: other position index 1
        :param i_range: range of the wall of position index 1
        :param j: other position index 2
        :param j_range: range of the wall of position index 2
        :return: [is_hit, hit_position]
        """
        r_i = self.particle.position[i]
        r_j = self.particle.position[j]
        r_k = self.particle.position[k]
        if self.particle.momentum[k+1] != 0:
            t = (surface - r_k) / self.particle.momentum[k+1]
            if t >= 0:
                hit_i = r_i + t * self.particle.momentum[i+1]
                hit_j = r_j + t * self.particle.momentum[j+1]
                if i_range[0] <= hit_i <= i_range[1] and j_range[0] <= hit_j <= j_range[1]:
                    hit = [0, 0, 0]
                    hit[i], hit[j], hit[k] = hit_i, hit_j, surface
                    return [True, hit]
                else:
                    return [False, None]
            else:
                return [False, None]
        else:
            return [False, None]

    def _test_recon_criteria(self):
        """Test if particle event passes reconstruction criteria."""
        criteria = self.detector.config.track_recon_criteria.keys()
        for criterion in criteria:
            min_hit = self.detector.config.track_recon_criteria[criterion][0]  # minimum number of hits
            layers = self.detector.config.track_recon_criteria[criterion][1]  # tracker layers the hits must come from
            if self._reconstruction(min_hit, layers):  # check if particle passes recon criteria
                self.recon_criteria.append(criterion)

    def _reconstruction(self, min_hits: int, planes: List[int]) -> bool: 
        """(helper) Return true iff track can be reconstructed."""
        hits = 0
        if len(planes) >= min_hits:  # there must be more tracker layers than hits required
            i = 0
            while i < len(self.tracker_hits) and hits < min_hits:
                if self.tracker_hits[i].hit_module.label in planes:
                    hits += 1
                i += 1
           
        return hits == min_hits

    def _test_trigger_criteria(self): # this also works for any "trigger" without any nbhd criteria i.e. just normal recon
        """Test if particle event passes triggering criteria"""
        criteria = self.detector.config.track_trigger_criteria.keys()
        for criterion in criteria:
            if len(self.detector.config.track_trigger_criteria[criterion]) == 3: # to accomodate geometry 0 for Charlie
                min_hit = self.detector.config.track_trigger_criteria[criterion][0]
                layers = self.detector.config.track_trigger_criteria[criterion][1]
                neighbourhood = self.detector.config.track_trigger_criteria[criterion][2]
                if self._trigger(min_hit, layers, neighbourhood):
                    self.trigger_criteria.append(criterion)
            else: # if length = 2, so basically track recon case, no nearest nbhd
                min_hit = self.detector.config.track_trigger_criteria[criterion][0]
                layers = self.detector.config.track_trigger_criteria[criterion][1]
                if self._reconstruction(min_hit, layers):  # check if particle passes trigger w/o nbhd (basically recon)
                    self.trigger_criteria.append(criterion)

    def _hits_among_planes(self, planes: List[int]):
        """(helper) Return all hits among given set of planes."""
        hits = []
        for hit in self.tracker_hits:
            if hit.hit_module.label in planes:
                hits.append(hit)
        return hits

    
    def _trigger(self, min_hits: int, planes: List[int], neighbourhood: int) -> bool: 
        """
        (helper) Algorithm for testing if particle event passes triggering criteria.
        :param min_hits: minimum number of hits required
        :param planes: list of tracker planes the hits must come from
        :param neighbourhood: nearest neighbour (n -> (2n+1)x(2n+1) neighbourhood)
        :return:
        
        Also saves the labels of all 3x3 grid which trigger the particle.
        """
        value = False
        hits = self._hits_among_planes(planes)
        if len(hits) >= min_hits:
            for i in range(len(hits)-min_hits+1):  # index by bottom layer of the set of min_hits layers
                if hits[i].hit_module.typ == 'z': # should always be true as only z-layer used
                    if abs(hits[i].hit_module.x_index-hits[i+min_hits-1].hit_module.x_index) <= 2*neighbourhood * np.round(hits[i].hit_module.next,3) and \
                            abs(hits[i].hit_module.y_index-hits[i+min_hits-1].hit_module.y_index) <= 2*neighbourhood * np.round(hits[i].hit_module.next,3):
                        value = True
                        for key in self.detector.grids_coord.keys():
                            if set([hits[i].hit_module.x_index, hits[i+min_hits-1].hit_module.x_index]).issubset(self.detector.grids_coord[key][0]) and set([hits[i].hit_module.y_index, hits[i+min_hits-1].hit_module.y_index]).issubset(self.detector.grids_coord[key][1]):
                                self.trigger_grids.append(key)
                                          
        return value
    
    
    def is_resolved(self, criterion: tuple) -> bool:
        """
        Check if the track can be resolved according to the given resolution criteria.

        :param criterion: the resolution criteria for a track to be resolved
        :return: if track can be resolved
        """
        # extract the number of resolved hits and resolution for each direction from the criteria
        num_hits = criterion[0]
        res_1 = criterion[1]  # transverse resolution
        res_2 = criterion[2]  # longitudinal resolution
        resolved_hits = 0
        # for each tracker hit object, go to the module it hits and iterate over each tracker hit within that module
        for hit in self.tracker_hits:
            if hit.hit_module.typ == 'z':
                resolved = True
                for other_hit in hit.hit_module.tracker_hits:
                    # check that this is a distinct hit from the tracker hit we care about
                    if other_hit != hit:
                        # check if the x, y positions of them are within the minimum resolution
                        # x <-> long_direction=0; y <-> long_direction=1; z <-> long_direction=2;
                        long_dir = hit.hit_module.long_direction  # corresponds to res_2 (0 or 1)
                        trans_dir = abs(long_dir - 1)  # corresponds to res_1 (1 or 0)
                        if abs(other_hit.hit_coordinate[trans_dir] - hit.hit_coordinate[trans_dir]) < res_1 \
                                and abs(other_hit.hit_coordinate[long_dir] - hit.hit_coordinate[long_dir]) < res_2:
                            # if too close to other_hit
                            resolved = False
                            break  # exit the for loop because this hit is not resolved
                if resolved:
                    resolved_hits += 1
                    
                    
            if hit.hit_module.typ == 'y':
                resolved = True
                for other_hit in hit.hit_module.tracker_hits:
                    # check that this is a distinct hit from the tracker hit we care about
                    if other_hit != hit:
                        # check if the x, y positions of them are within the minimum resolution
                        # x <-> long_direction=0; y <-> long_direction=1; z <-> long_direction=2;
                        long_dir = hit.hit_module.long_direction  # corresponds to res_2 (0 or 2)
                        trans_dir = abs(long_dir - 2)  # corresponds to res_1 (2 or 0)
                        if abs(other_hit.hit_coordinate[trans_dir] - hit.hit_coordinate[trans_dir]) < res_1 \
                                and abs(other_hit.hit_coordinate[long_dir] - hit.hit_coordinate[long_dir]) < res_2:
                            # if too close to other_hit
                            resolved = False
                            break  # exit the for loop because this hit is not resolved
                if resolved:
                    resolved_hits += 1 
            
            if hit.hit_module.typ == 'x':
                layer = hit.hit_module.x_index
                resolved = True
                for other_hit in hit.hit_module.tracker_hits:
                    # check that this is a distinct hit from the tracker hit we care about
                    if other_hit != hit:
                        # check if the x, y positions of them are within the minimum resolution
                        # x <-> long_direction=0; y <-> long_direction=1; z <-> long_direction=2;
                        long_dir = hit.hit_module.long_direction  # corresponds to res_2 (1 or 2)
                        trans_dir = abs(long_dir - 2) if long_dir==3 else 3 # corresponds to res_1 (2 or 1)
                        if abs(other_hit.hit_coordinate[trans_dir] - hit.hit_coordinate[trans_dir]) < res_1 \
                                and abs(other_hit.hit_coordinate[long_dir] - hit.hit_coordinate[long_dir]) < res_2:
                            # if too close to other_hit
                            resolved = False
                            break  # exit the for loop because this hit is not resolved
                if resolved:
                    resolved_hits += 1 
            
        return resolved_hits >= num_hits

    def hit_display(self, ax, hit, module_label): # should work as just plots a point in 3D
        if hit == "energy":
            show = self._visibility == DetectorParticle.VISIBLE
        elif hit == "all":
            show = self._visibility != DetectorParticle.INVISIBLE
        else:
            show = False

        if show:
            x_hit, y_hit, z_hit = [], [], []
            if module_label:
                for hit in self.tracker_hits:
                    x_hit.append(hit.hit_coordinate[0])
                    y_hit.append(hit.hit_coordinate[1])
                    z_hit.append(hit.hit_coordinate[2])
                    ax.text(hit.hit_coordinate[0], hit.hit_coordinate[1], hit.hit_coordinate[2],
                            str(hit.hit_module.get_index()), (1, 1, 0))
            else:
                for hit in self.tracker_hits:
                    x_hit.append(hit.hit_coordinate[0])
                    y_hit.append(hit.hit_coordinate[1])
                    z_hit.append(hit.hit_coordinate[2])
            ax.scatter3D(x_hit, y_hit, z_hit, marker="x")

    def track_display(self, ax, recon, zorder): 
        """Display all tracks: invisible: light
                               visible but below energy: dotted orange
                               visible energy: solid orange
        """
        if self.particle.momentum[-1] < 0:
            # travel down
            z_end = ax.get_zlim()[0]
        else:
            # travel up
            z_end = ax.get_zlim()[1]
        try:
            x_end, y_end = self._trajectory(z_end, 'z')[0]
            if self._visibility == -1:
                # Invisible
                color = "blue"
                linestyle = "-"
                alpha = 0.3
            elif self._visibility == 0:
                # Low energy
                color = "orange"
                linestyle = "--"
                alpha = 1
            else:
                color = "orange"
                linestyle = "-"
                alpha = 1
            if recon in self.recon_criteria:
                color = "r"
            ax.plot([self.particle.position[0], x_end], [self.particle.position[1], y_end],
                    [self.particle.position[2], z_end], color=color, ls=linestyle, alpha=alpha,zorder=zorder) 
        except AssertionError:
            print("Particle trajectory out of bound.")

    def raw_str(self, query): 
        report = ""
        detector = self.detector
        if query:
            report += "{\n"
            report += f"{detector.track_reconstructed()}\n" \
                f"{detector.event_pass_trigger()}\n" \
                f"{detector.track_is_in_decay_volume()}\n" \
                f"{detector.track_is_in_detector_volume()}\n"
            report += "}\n"
        report += f"{self.particle.pid};{list(self.particle.position)};{list(self.particle.momentum)};{self._visibility}\n"
        report += "{\n"
        if self.tracker_hits:
            for hit in self.tracker_hits:
                report += f"{list(hit.hit_coordinate)};{hit.hit_module.get_index()}\n"
        else:
            report += "None\n"
        report += "}\n"
        report += f"{self.recon_criteria};{self.trigger_criteria}\n"
        report += str(self.decay_volume_hits) + "\n"
        report += str(self.detector_volume_hits) + "\n"
        report += str(self.wall_hit)+"\n;\n"
        return report

    def __str__(self): 
        if self._visibility == 1:
            is_visible = "visible"
        elif self._visibility == 0:
            is_visible = "low momentum"
        else:
            is_visible = "invisible"
        hit_report = ""
        for hit in self.tracker_hits:
            hit_report += "\t" + str(hit)
        if hit_report == "":
            hit_report = "\tNone\n"
        report = f"Particle PID: {self.particle.pid}, position: {self.particle.position}, momentum: {self.particle.momentum}, {is_visible}\n" \
            f"  Hits at: \n" \
            f"  {hit_report}" \
            f"  Passed reconstruction criteria: {self.recon_criteria}\n" \
            f"         triggering criteria: {self.trigger_criteria}\n" \
            f"  Wall hits: {self.wall_hit}\n" \
            f"  \tDecay: {self.decay_volume_hits}\n" \
            f"  \tDetector: {self.detector_volume_hits}\n;\n"
        return report


class DetectorVertex:

    vertex: Vertex
    particles: List[DetectorParticle]
    recon_criteria: List[str]
    wall_hit: dict

    def __init__(self, vertex: Vertex, detector):
        self.vertex = vertex
        self.particles = [DetectorParticle(particle, detector) for particle in vertex.decay_product]
        self.detector = detector
        # First, remove dead hits from particle/module hit_lists if scintillator bars are used
        if self.detector.config.scint_resolve:
            all_hits = []
            for par in self.particles:
                all_hits.extend(par.tracker_hits) # need all hits so we can sort by time and remove dead hits
            tbr = self.detector.scint_resolved(all_hits, self.detector.config.scint_dims, self.detector.config.scint_resolve_layers) # get all the hits that are dead 
            # remove the dead hits from list of hits
            for hit in tbr:
                for par in self.particles:
                    if hit in par.tracker_hits:
                        par.tracker_hits.remove(hit)
                        break
        # Then, test recon and trigger criteria for particles in this vertex               
        for par in self.particles:
            par.test_criteria()
        self.recon_criteria = []
        for criterion in self.detector.config.vertex_recon_criteria.keys(): # this is where we test_criteria for particles too
            if self._test_recon_criteria(criterion):
                self.recon_criteria.append(criterion)
        self.wall_hit = {"LLP_DECAY": False, "LLP_DETECTOR": False, "DECAY": self._par_hit("DECAY"),
                         "DETECTOR": self._par_hit("DETECTOR")}
        self._llp_flags()
     

    def _par_hit(self, criteria) -> bool:
        for par in self.particles:
            if par.wall_hit[criteria]:
                return True
        return False

    def _llp_flags(self):
        x, y, z = self.vertex.position
        if self.detector.config.decay_x_min <= x <= self.detector.config.decay_x_max \
                and self.detector.config.decay_y_min <= y <= self.detector.config.decay_y_max:
            if self.detector.config.decay_z_min <= z <= self.detector.config.decay_z_max:
                self.wall_hit["LLP_DECAY"], self.wall_hit["LLP_DETECTOR"] = True, True
            elif self.detector.config.detector_z_min <= z <= self.detector.config.detector_z_max:
                self.wall_hit["LLP_DETECTOR"] = True

    def hit_display(self, ax, hit, module_label): # plots a group of points in 3D
        for par in self.particles:
            par.hit_display(ax, hit, module_label)

    def track_display(self, ax, recon, zorder): # a combined particle track_display
        vertex_position = self.vertex.position
        vertex_momentum = self.vertex.momentum
        ax.scatter(vertex_position[0], vertex_position[1], vertex_position[2], marker="o", color="r")
        ax.plot([self.detector.IP[0], vertex_position[0]], [self.detector.IP[1], vertex_position[1]],
                [self.detector.IP[2], vertex_position[2]], linestyle=":", color="black")
        ax.quiver(vertex_position[0], vertex_position[1], vertex_position[2], vertex_momentum[1], vertex_momentum[2], vertex_momentum[3], color="gray", label="LLP")
        for par in self.particles:
            par.track_display(ax, recon, zorder)
            
       
    def _test_recon_criteria(self, criterion: str):
        # Get a list of particles to consider
        particles = []
        if len(self.detector.config.vertex_recon_criteria[criterion]) == 2:  # resolution criteria present
            res_criteria = self.detector.config.vertex_recon_criteria[criterion][1]
            for par in self.particles:
                if par.is_resolved(res_criteria):  # if particle is resolved by this criterion
                    particles.append(par)            
        else:
            particles = self.particles  # no resolution criteria so all particles are considered
        # get a list of passed reconstruction criteria of particles that passes at least one of the track recon criteria
        particles_recon = []
        for par in particles:
            if par.recon_criteria:  # if the particle track satisfies any track criterion
                particles_recon.append(par.recon_criteria.copy())  # list will be modified, so make a copy
        # Expand list of criteria from crit1 n1 crit2 n2 to crit1 ... crit1 (n1 times) crit2 ... crit2 (n2 times)
        vertex_criteria = self.detector.config.vertex_recon_criteria[criterion][0]
        crit_list = []
        for rule in vertex_criteria:
            for _ in range(rule[0]):
                crit_list.append(rule[1])
        # Recursion
        return self._recon_helper(particles_recon, [], crit_list, 0)

    def _recon_helper(self, par_list: list, par_stack: list, crit_list: list, curr_index: int) -> bool:
        if curr_index == len(crit_list):
            return True
        else:
            # iterate over items in par_list. Since par_list will be modified along the way, we make a copy to keep for
            for par_recon in par_list.copy():  # par_recon: passed track criteria of particle object
                if crit_list[curr_index] in par_recon:
                    par_list.remove(par_recon)  # remove the particle that will be used to fulfill this criteria
                    par_stack.append(par_recon)  # but do save it somewhere in case we need to try other iterations
                    if self._recon_helper(par_list, par_stack, crit_list, curr_index+1):
                        return True
                    else:
                        par_list.append(par_stack.pop())
            return False
        
 
    def __str__(self): 
        particle_report = ""
        for par in self.particles:
            particle_report += str(par)
        report = f"Vertex position: {self.vertex.position}, momentum: {self.vertex.momentum}\n" \
            f"Final state particles: \n" \
            f"{particle_report}" \
            f"Passed vertex reconstruction criteria: {self.recon_criteria}\n" \
            f"Wall hits: {self.wall_hit}\n;\n\n"
        return report


class ParamCard:
    """
    loads the txt param card which parametrizes the detector and its geometry
    """

    filename: str
    title: str
    scint_resolve: int
    scint_resolve_layers: Union[None, List]
    decay_x_min: float
    decay_y_min: float
    decay_x_dim: float
    decay_y_dim: float
    decay_x_max: float
    decay_y_max: float
    decay_z_min: float
    decay_z_max: float
    detector_z_min: float
    detector_z_max: float
    module_groups: dict
### the commented out variables are keys of each module WITHIN module_groups 
#     x_mesh: list
#     y_mesh: list
#     x_layer: list
#     y_layer: list 
#     z_layer: list
#     z_range: list(tuples) or single tuple
#     long_direction: list
#     module_x_dim: float
#     module_y_dim: float
    track_recon_criteria: Dict[str, list]
    track_recon_default: str
    track_trigger_criteria: Dict[str, list]
    track_trigger_default: str
    vertex_recon_criteria: Dict[str, list]
    vertex_recon_default: str
    min_particle_momenta: Dict[int, float]
        
    def __init__(self, filename): 
        self.filename = filename
        # read file
        file = open(filename, "r")
        lines = file.readlines()
        file.close()
        # filter commented out lines and empty lines
        param_card = {"title": None,
                      "scint_resolve": None,
                      "scint_resolve_layers":None,
                      "scint_dims": None,
                      "decay_volume_x_min": None,
                      "decay_volume_y_min": None,
                      "decay_volume_x_dim": None,
                      "decay_volume_y_dim": None,
                      "decay_volume_z_min": None,
                      "decay_volume_z_max": None,
                      "groups": list,
                      "x_mesh": dict(),
                      "y_mesh": dict(),
                      "x_layer": dict(),
                      "y_layer": dict(),
                      "z_layer": dict(),
                      "labels": dict(),
                      "type": dict(),
                      "z_range": dict(),
                      "long_direction": dict(),
                      "module_x_dim": dict(),
                      "module_y_dim": dict(),
                      "track_recon_criteria": dict(),
                      "track_recon_default": None,
                      "track_trigger_criteria": dict(),
                      "track_trigger_default": None,
                      "vertex_recon_criteria": dict(),
                      "vertex_recon_default": None,
                      "min_particle_momenta": None}
        temp_module = -1
        for line in lines:
            if line[0] != "#" and line.strip():
                raw = line.split()
                if raw[0] == "track_recon_criteria":
                    assert len(raw[2:]) == 3, "Wrong number of arguments for track reconstruction criterion"
                    if param_card["track_recon_default"] is None:
                        param_card["track_recon_default"] = raw[2]
                    param_card["track_recon_criteria"][raw[2]] = [int(raw[3]), list(map(int, raw[4].split(",")))]
                elif raw[0] == "track_trigger_criteria":
                    assert len(raw[2:]) in [3,4], "Wrong number of arguments for track triggering criterion"
                    if param_card["track_trigger_default"] is None:
                        param_card["track_trigger_default"] = raw[2]
                    if len(raw[2:]) == 4:
                        param_card["track_trigger_criteria"][raw[2]] = [int(raw[3]), list(map(int, raw[4].split(","))), int(raw[5])]
                    else:
                        param_card["track_trigger_criteria"][raw[2]] = [int(raw[3]), list(map(int, raw[4].split(",")))]
                elif raw[0] == "vertex_recon_criteria":
                    if param_card["vertex_recon_default"] is None:
                        param_card["vertex_recon_default"] = raw[2]
                    # if no resolution, length of the list is 1. otherwise length of list is 2, where first entry is
                    # track criteria, second entry is resolution criteria
                    param_card["vertex_recon_criteria"][raw[2]] = [[]]
                    if len(raw[2:]) % 2 == 1:  # no resolution
                        for i in range(3, len(raw), 2):
                            param_card["vertex_recon_criteria"][raw[2]][0].append((int(raw[i]), raw[i+1]))
                    else:  # resolution
                        for i in range(3, len(raw)-3, 2):
                            # first entry specify track criteria
                            param_card["vertex_recon_criteria"][raw[2]][0].append((int(raw[i]), raw[i+1]))
                        # second entry specify the spacial resolution (num_hits, Z1 (m), Z2 (m))
                        param_card["vertex_recon_criteria"][raw[2]].append((int(raw[-3]), float(raw[-2])/100, float(raw[-1])/100))
                #######
                elif raw[0] == "module_group":
                    if temp_module == -1:
                        param_card["groups"] = []
                        param_card["x_mesh"] = {}
                        param_card["y_mesh"] = {}
                        param_card["x_layer"] = {}
                        param_card["y_layer"] = {}
                        param_card["z_layer"] = {}
                        param_card["z_range"] = {}
                        param_card["labels"] = {}
                        param_card["type"] = {}
                        param_card["long_direction"] = {}
                        param_card["module_x_dim"] = {}
                        param_card["module_y_dim"] = {}
                        
                    temp_module = raw[2]
                    param_card["groups"].append(raw[2])
                    param_card["x_mesh"][raw[2]] = None
                    param_card["y_mesh"][raw[2]] = None
                    param_card["x_layer"][raw[2]] = None
                    param_card["y_layer"][raw[2]] = None
                    param_card["z_layer"][raw[2]] = None
                    param_card["z_range"][raw[2]] = None
                    param_card["labels"][raw[2]] = None
                    param_card["type"][raw[2]] = None
                    param_card["long_direction"][raw[2]] = None
                    param_card["module_x_dim"][raw[2]] = None
                    param_card["module_y_dim"][raw[2]] = None
                #######
                else:
                    if len(raw[2:]) == 1:
                        if temp_module != -1:
                            param_card[raw[0]][temp_module] = raw[2]                   
                        else:
                            param_card[raw[0]] = raw[2]
                                               
                    else:
                        param_card[raw[0]] = raw[2:]

        if param_card["scint_resolve"] is not None and len(param_card["scint_resolve"]) == 5: 
            self.scint_resolve = int(param_card["scint_resolve"][0])
            self.scint_dims = [float(param_card["scint_resolve"][1]), float(param_card["scint_resolve"][2]), int(param_card["scint_resolve"][3])]
            self.scint_resolve_layers = list(map(int, param_card["scint_resolve"][4].split(",")))
        elif param_card["scint_resolve"] is not None: # no scint_resolve
            self.scint_resolve = int(param_card["scint_resolve"])
        self.title = param_card["title"]
        self.decay_x_min = float(param_card["decay_volume_x_min"])
        self.decay_y_min = float(param_card["decay_volume_y_min"])
        self.decay_x_dim = float(param_card["decay_volume_x_dim"])
        self.decay_y_dim = float(param_card["decay_volume_y_dim"])
        self.decay_x_max = self.decay_x_min+self.decay_x_dim
        self.decay_y_max = self.decay_y_min+self.decay_y_dim
        self.decay_z_min = float(param_card["decay_volume_z_min"])
        self.decay_z_max = float(param_card["decay_volume_z_max"])
        
        # using param_card dictionary to set-up and populate module_groups
        self.module_groups = {}
        for i in param_card["groups"]:
            self.module_groups[i] = {}
        
        for i in self.module_groups.keys():
            self.module_groups[i]["x_mesh"] = []
            self.module_groups[i]["y_mesh"] = []
            self.module_groups[i]["x_layer"] = []
            self.module_groups[i]["y_layer"] = []
            self.module_groups[i]["z_layer"] = []
            self.module_groups[i]["z_range"] = []
            self.module_groups[i]["labels"] = []
            self.module_groups[i]["type"] = None # type of module - 'x', 'y', 'z'
            self.module_groups[i]["long_direction"] = []
        
        for i in self.module_groups.keys():
            if param_card["module_x_dim"][i] is not None:
                self.module_groups[i]["module_x_dim"] = float(param_card["module_x_dim"][i])
            if param_card["module_y_dim"][i] is not None:
                self.module_groups[i]["module_y_dim"] = float(param_card["module_y_dim"][i])
            if param_card["x_mesh"][i] is not None:
                self.module_groups[i]["x_mesh"] = list(map(float, param_card["x_mesh"][i].split(",")))
            if param_card["y_mesh"][i] is not None:
                self.module_groups[i]["y_mesh"] = list(map(float, param_card["y_mesh"][i].split(",")))
            self.module_groups[i]["labels"] = list(map(float, param_card["labels"][i].split(",")))
            self.module_groups[i]["type"] = str(param_card["type"][i])
            if param_card["long_direction"][i] is not None:
                self.module_groups[i]["long_direction"] = list(map(int, param_card["long_direction"][i].split(",")))
            # these parameters may or may not be None depending on module type
            if param_card["x_layer"][i] is not None:
                self.module_groups[i]["x_layer"] = list(map(float, param_card["x_layer"][i].split(",")))
            if param_card["y_layer"][i] is not None:
                self.module_groups[i]["y_layer"] = list(map(float, param_card["y_layer"][i].split(",")))
            if param_card["z_layer"][i] is not None:
                self.module_groups[i]["z_layer"] = list(map(float, param_card["z_layer"][i].split(",")))
            
      
        temp_z = []
        for i in self.module_groups.keys():
            temp_z.extend(self.module_groups[i]["z_layer"])
        self.detector_z_min = min(self.decay_z_min, min(temp_z))
        self.detector_z_max = max(self.decay_z_max, max(temp_z))
        self.track_recon_criteria = param_card["track_recon_criteria"]
        self.track_recon_default = param_card["track_recon_default"]
        self.track_trigger_criteria = param_card["track_trigger_criteria"]
        self.track_trigger_default = param_card["track_trigger_default"]
        self.vertex_recon_criteria = param_card["vertex_recon_criteria"]
        self.vertex_recon_default = param_card["vertex_recon_default"]
        self.min_particle_momenta = {}
        for pid_p_pair in param_card["min_particle_momenta"]:
            pair = pid_p_pair.split(",")
            self.min_particle_momenta[int(pair[0])] = float(pair[1])/1000  # convert to GeV
         
        # after detector_min is set, change z_coords from relative to global
        for i in self.module_groups.keys():
            if param_card["z_range"][i] is not None:
                    # convert z_range to list of tuples
                    temp = list(map(float, param_card["z_range"][i].split(",")))
                    j = 0
                    while j < len(temp) - 1:
                        self.module_groups[i]["z_range"].append((temp[j] + self.detector_z_min, temp[j+1] + self.detector_z_min))
                        j += 1
                       

    def __str__(self): 
        return f"from file {self.filename}:\n\n" \
            f"title: {self.title}\n\n" \
            f"decay_volume_x_min: {self.decay_x_min}\n" \
            f"decay_volume_y_min: {self.decay_y_min}\n" \
            f"decay_volume_x_dim: {self.decay_x_dim}\n" \
            f"decay_volume_y_dim: {self.decay_y_dim}\n" \
            f"decay_volume_z_min: {self.decay_z_min}\n" \
            f"decay_volume_z_max: {self.decay_z_max}\n\n" \
            f"track_recon_criteria: {self.track_recon_criteria}\n\n" \
            f"track_trigger_criteria: {self.track_trigger_criteria}\n\n" \
            f"vertex_recon_criteria: {self.vertex_recon_criteria}\n\n" \
            f"min_particle_momenta: {self.min_particle_momenta}\n\n" \
            f"module_groups: {self.module_groups}"


class Detector:
    """
    This is the Detector object class. Implements all the sensor layers etc.
    """
    config: ParamCard

        
    modules: Dict[tuple, Module]
    module_groups: dict() # same as the one in class ParamCard but contains more keys

    current_event_mode: int
    current_particle: Particle
    current_vertex: Vertex
    current_event: Optional[Union[DetectorParticle, DetectorVertex]]

    IP = (0, 0, 0) # default interaction point (of the particle collider)

    TRUE = 1
    FALSE = 0
    NONE = -1

    NO_EVENT = 0
    PARTICLE = 1
    VERTEX = 2

    def __init__(self, param_card: str):
        self.config = ParamCard(param_card)
        self.modules = {} # all in one dictionary for all modules - gives easy access during tracking
        self.types = set([])
        self.module_groups = self.config.module_groups.copy()
        for i in self.module_groups.keys():
            self.module_groups[i]["module_x_range"] = []
            self.module_groups[i]["module_y_range"] = []
            self.module_groups[i]["modules"] = {}

        self.current_event_mode = Detector.NO_EVENT
        self.current_particle = None
        self.current_vertex = None
        self.current_event = None
       
        self._allocate_modules()
        self.grids, self.grids_coord = self.define_trigger_grids() # unique grid label and corresponding modules included 

    def _allocate_modules(self):
        """From config extract all start and end position for each module""" 
        for i in self.module_groups.keys():
            for x in self.module_groups[i]["x_mesh"]:
                self.module_groups[i]["module_x_range"].append((x,x+self.module_groups[i]["module_x_dim"]))
            for y in self.module_groups[i]["y_mesh"]:
                self.module_groups[i]["module_y_range"].append((y,y+self.module_groups[i]["module_y_dim"]))
            
        # label modules by smallest (x, y, z) coordinate it passes through
        for key in self.module_groups.keys():
            if self.module_groups[key]["type"] == 'z': # modules in xy plane (normal vector parallel to z)
                self.types.add('z')
                typ = 'z'
                for i in range(len(self.module_groups[key]["x_mesh"])):
                    x = min(self.module_groups[key]["module_x_range"][i])
                    for j in range(len(self.module_groups[key]["y_mesh"])):
                        y = min(self.module_groups[key]["module_y_range"][j])
                        for k in range(len(self.module_groups[key]["z_layer"])):
                            z = self.module_groups[key]["z_layer"][k]
                            label = int(self.module_groups[key]["labels"][k])
                            x_dim = float(self.module_groups[key]["module_x_dim"])
                            y_dim = float(self.module_groups[key]["module_y_dim"])
                            z_dim = 0
                            long_direction = int(self.module_groups[key]["long_direction"][k])
                            self.module_groups[key]["modules"][(x, y, z)] = Module(typ, label, x, y, z, x_dim, y_dim, z_dim, long_direction, self)
        
                            # get inter-module distance - to be used during trigger
                            # ASSUMES SAME INTER-MODULE DISTANCE IN x AND y DIRECTIONS
                            if i != len(self.module_groups[key]["x_mesh"]) - 1:
                                dist = abs(self.module_groups[key]["x_mesh"][i+1] - self.module_groups[key]["x_mesh"][i])
                            else:
                                dist = abs(self.module_groups[key]["x_mesh"][i] - self.module_groups[key]["x_mesh"][i-1])
                            
                            self.module_groups[key]["modules"][(x, y, z)].set_next(dist)
                            self.modules.update(self.module_groups[key]["modules"])
                            
            elif self.module_groups[key]["type"] == 'y': # modules in xz plane (normal vector parallel to y)
                self.types.add('y')
                typ = 'y'
                for i in range(len(self.module_groups[key]["x_mesh"])):
                    x = min(self.module_groups[key]["module_x_range"][i])
                    for j in range(len(self.module_groups[key]["z_range"])):
                        z = min(self.module_groups[key]["z_range"][j])
                        for k in range(len(self.module_groups[key]["y_layer"])):
                                y = self.module_groups[key]["y_layer"][k]
                                label = int(self.module_groups[key]["labels"][k])
                                z_dim = float(self.module_groups[key]["z_range"][j][1] - self.module_groups[key]["z_range"][j][0])
                                y_dim = 0
                                x_dim = float(self.module_groups[key]["module_x_dim"])
                                long_direction = int(self.module_groups[key]["long_direction"][k])
                                self.module_groups[key]["modules"][(x, y, z)] = Module(typ, label, x, y, z, x_dim, y_dim, z_dim, long_direction, self)
                                self.modules.update(self.module_groups[key]["modules"])
                
            elif self.module_groups[key]["type"] == 'x': # modules in yz plane (normal vector parallel to x)
                self.types.add('x')
                typ = 'x'
                for i in range(len(self.module_groups[key]["y_mesh"])):
                    y = min(self.module_groups[key]["module_y_range"][i])
                    for j in range(len(self.module_groups[key]["z_range"])):
                        z = min(self.module_groups[key]["z_range"][j])
                        for k in range(len(self.module_groups[key]["x_layer"])):
                                x = self.module_groups[key]["x_layer"][k] 
                                label = int(self.module_groups[key]["labels"][k])
                                x_dim = 0
                                z_dim = float(self.module_groups[key]["z_range"][j][1] - self.module_groups[key]["z_range"][j][0])
                                y_dim = float(self.module_groups[key]["module_y_dim"])
                                long_direction = int(self.module_groups[key]["long_direction"][k])
                                self.module_groups[key]["modules"][(x, y, z)] = Module(typ, label, x, y, z, x_dim, y_dim, z_dim, long_direction, self)
                                self.modules.update(self.module_groups[key]["modules"])
                                
                                
######################## FUNCTIONS SPECIFIC TO TRIGGER STUDIES #####################
    def define_trigger_grids(self) -> dict:
        
        J = {}
    
        n = 0
        for i in range(8):
            for j in range(8):
                J[n] = ((i,i+1,i+2),(j,j+1,j+2))
                n += 1

        J[64] = ((0,1),(0,1))
        J[65] = ((0,1),(8,9))
        J[66] = ((8,9),(0,1))
        J[67] = ((8,9),(8,9))
        
       
        J_coord = {}
        n = 0
        for i in range(len(self.module_groups['0']["x_mesh"])-2):
            for j in range(len(self.module_groups['0']["y_mesh"])-2):
                J_coord[n] = (sorted(self.module_groups['0']["x_mesh"])[i:i+3], sorted(self.module_groups['0']["y_mesh"])[j:j+3])
                n += 1
                
            
        # next add corner labels and modules to grid list
#         J[n] = (sorted(self.module_groups[0]["x_mesh"])[0:2], sorted(self.module_groups[0]["y_mesh"])[0:2])
#         n += 1
#         J[n] = (sorted(self.module_groups[0]["x_mesh"])[0:2], sorted(self.module_groups[0]["y_mesh"])[-2:])
#         n+=1 
#         J[n] = (sorted(self.module_groups[0]["x_mesh"])[-2:], sorted(self.module_groups[0]["y_mesh"])[0:2])
#         n+=1
#         J[n] = (sorted(self.module_groups[0]["x_mesh"])[-2:], sorted(self.module_groups[0]["y_mesh"])[-2:])
        
        return J, J_coord
        

    def scint_resolved(self, hits: List[TrackerHit], dims: List[float], labels: List[int]) -> List[TrackerHit]: # hits is the list of all tracker_hits
        """
        If particle lands on a dead scintillator in one of the layers with label in labels, remove that hit from all hit_lists.
        NOTE: This is only used for trigger layers in the z-direction (as of yet).
        """
        tbr = [] # collect the to be removed hits here
        temp = []
        for hit in hits:
             if hit.hit_module.label in labels:
                    temp.append(hit)
        for hit in temp:
            tbr_from_module = [] # collect the dead hits to be removed from this hit.hit_module's hit list
            long_dir = hit.hit_module.long_direction
            hit.hit_module.tracker_hits.sort(key=lambda x: x.time)
            for h in hit.hit_module.tracker_hits:
                if h not in tbr_from_module:
                    if long_dir: # long_dir = 1 i.e. y
                        fibre_1 = (hit.hit_module.y_index - h.hit_coordinate[1])%dims[0]
                        fibre_2 = (hit.hit_module.x_index - h.hit_coordinate[0])%dims[1]
                    else: # long_dir = 0 i.e. x
                        fibre_1 = (hit.hit_module.x_index - h.hit_coordinate[0])%dims[0]
                        fibre_2 = (hit.hit_module.y_index - h.hit_coordinate[1])%dims[1]
                    for oh in hit.hit_module.tracker_hits:
                        if oh != h and oh not in tbr_from_module:
                            if long_dir:
                                other_fibre_1 = (hit.hit_module.y_index - oh.hit_coordinate[1])%dims[0]
                                other_fibre_2 = (hit.hit_module.x_index - oh.hit_coordinate[0])%dims[1]
                            else:
                                other_fibre_1 = (hit.hit_module.x_index - oh.hit_coordinate[0])%dims[0]
                                other_fibre_2 = (hit.hit_module.y_index - oh.hit_coordinate[1])%dims[1]
                            if (other_fibre_1 == fibre_1 and other_fibre_2 == fibre_2) or (other_fibre_1 == fibre_1 + dims[2] and other_fibre_2 == fibre_2):
                                tbr_from_module.append(oh)
                                tbr.append(oh)
                            
            for tbr_hit in set(tbr_from_module):
                hit.hit_module.tracker_hits.remove(tbr_hit) 
             
        return set(tbr) # hits to be removed from particle hit_list
    
    
####################################################################################################
        
        
    # ======Detector Information====== 
    def decay_volume(self):
        """Return decay volume"""
        return self.config.decay_x_dim * self.config.decay_y_dim * \
            (self.config.decay_z_max - self.config.decay_z_min)

    def number_modules(self): 
        """Return number of modules"""
        return len(self.modules.keys())

    def detector_area(self): 
        """Return total Detector area"""
        return self.config.decay_x_dim * self.config.decay_y_dim

    def tracker_area(self): 
        """Return total area of Detector planes"""
        area = 0
        for key in self.modules.keys():
            if self.modules[key].typ == 'x':
                area += self.modules[key].y_dim * self.modules[key].z_dim
            if self.modules[key].typ == 'y':
                area += self.modules[key].x_dim * self.modules[key].z_dim
            if self.modules[key].typ == 'z':
                area += self.modules[key].x_dim * self.modules[key].y_dim 
        return area

    def param_card(self):
        """Return param_card text used to initialize this Detector object"""
        return str(self.module_groups) # return the updated version instead of self.config

    def raw_output(self, query=False):
        """Return the raw output for event loaded. This is the event summary for data analysis."""
        if self.current_event is None:
            return "N/A"
        else:
            return self.current_event.raw_str(query)

    def formatted_output(self):
        """Return the formatted output for event loaded. This is a readable version of event summary."""
        if self.current_event is None:
            return "No event loaded."
        else:
            return str(self.current_event)

    # can change quality/format of file saved by changing the arguments of the savefig function used at the end of this function
    # can change ordering of overlay of tracks vs detector planes by changing zorder
    # No input for zorder i.e. default: planes over tracks
    # zorder = 1000: tracks over planes
    def detector_display(self, save_path: str, note="", show=False, ip=False, module_label=False, hit="energy", recon=("default", "default"), trigger="default",zorder=10):
        """Show 3d graph of Detector"""
        fig = plt.figure(figsize=(14, 8))
        ax = []
        # 5 subplots each showing one perspective of the detector
        for i in range(1, 6):
            ax.append(fig.add_subplot(2, 3, i, projection="3d"))
        # 1: default, 2: top down, 3:
        perspectives = [{"elev": None, "azim": None},
                        {"elev": 90, "azim": 90},
                        {"elev": 0, "azim": 0},
                        {"elev": 20, "azim": 20},
                        {"elev": 20, "azim": 200}]
        for a, p in zip(ax, perspectives):
            self._display_initialize(a, ip, p) # 1st subfunction

            self._decay_volume_display(a) # 2nd 
            
            self._module_display(a) # 3rd
            
            if self.current_event_mode != Detector.NO_EVENT:
                if recon[0] == "default":  # recon[0] is track criteria
                    recon = (self.config.track_recon_default, recon[1]) 
                if recon[1] == "default":  # recon[1] is vertex criteria
                    recon = (recon[0], self.config.vertex_recon_default) 
                if trigger == "default":
                    trigger = self.config.track_trigger_default 
                self._hit_display(a, hit, module_label) # 5th
                self._track_display(a, recon[0], zorder) # 4th
            Detector._set_axes_equal(a) # 6th 
            

        if self.current_event_mode != Detector.NO_EVENT:
            if self.current_event_mode == Detector.PARTICLE:
                event_object = self.current_event.particle
                recon_by_crit = self.track_reconstructed(recon[0])
                recon = recon[0]
            else:
                event_object = self.current_event.vertex
                recon_by_crit = self.vertex_reconstructed(recon[1])
                recon = recon[1]
            # make a note of event_info in plot
            event_info = f"x = {tuple([round(coord, 2) for coord in event_object.position])}\n" \
                f"p = {tuple([round(p, 2) for p in event_object.momentum])}\n" \
                f"b = {round(np.linalg.norm(event_object.momentum[1:])/event_object.mass, 2)}\n" \
                f"Reconstruction by {recon} criteria: {recon_by_crit}\n" \
                f"Trigger by {trigger} criteria: {self.event_pass_trigger(trigger)}\n" \
                f"{note}"
            fig.text(0.70, 0.3, event_info, fontsize=12)
            
            
            

        # legend for tracks
        labels = ["invisible", "low energy", "visible", f"passed {recon}", "vertex momentum"]
        custom_lines = [Line2D([0], [0], color="blue", linestyle="-", alpha=0.3, lw=1),
                        Line2D([0], [0], color="orange", linestyle="--", alpha=1, lw=1),
                        Line2D([0], [0], color="orange", linestyle="-", alpha=1, lw=1),
                        Line2D([0], [0], color="r", linestyle="-", alpha=1, lw=1),
                        Line2D([0], [0], color="gray", linestyle="-", lw=1.5)]
        fig.legend(custom_lines, labels)
        fig.suptitle("Detector " + self.config.title)
        fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs", save_path+".pdf"),format="pdf",dpi=1200)
        if show:
            plt.show()
        else:
            plt.close(fig)

    def _display_initialize(self, ax, ip, pers: dict):
        """Set xyz limits."""
        x_min = min(1.25 * self.config.decay_x_min, 0.75 * self.config.decay_x_min)
        y_min = min(1.25 * self.config.decay_y_min, 0.75 * self.config.decay_y_min)
        z_min = min(1.25 * self.config.detector_z_min, 0.75 * self.config.detector_z_min)
        x_max = max(1.25 * self.config.decay_x_max, 0.75 * self.config.decay_x_max)
        y_max = max(1.25 * self.config.decay_y_max, 0.75 * self.config.decay_y_max)
        z_max = max(1.25 * self.config.detector_z_max, 0.75 * self.config.detector_z_max)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        # set perspective
        ax.view_init(**pers)

        ax.set_xlabel('$X (m)$')
        ax.set_ylabel('$Y (m)$')
        ax.set_zlabel('$Z (m)$')

        ax.plot([Detector.IP[0]], [Detector.IP[1]], [Detector.IP[2]], marker="o")
        ax.text(Detector.IP[0], Detector.IP[1], Detector.IP[2], "IP")

        if ip:
            x_min, x_max = min(x_min, Detector.IP[0]), max(x_max, Detector.IP[0])
            y_min, y_max = min(y_min, Detector.IP[1]), max(y_max, Detector.IP[1])
            z_min, z_max = min(z_min, Detector.IP[2]), max(z_max, Detector.IP[2])
            max_range = max(x_max-x_min, y_max-y_min, z_max-z_min)
            # This ensures the proportionality of the xyz axis remains 1:1:1
            ax.set(xlim=(x_min, x_min+max_range), ylim=(y_min, y_min+max_range), zlim=(z_min, z_min+max_range))

    @staticmethod
    def _set_axes_equal(ax: plt.Axes):
        """Set 3D plot axes to equal scale.

        Make axes of 3D plot have equal scale so that spheres appear as
        spheres and cubes as cubes.  Required since `ax.axis('equal')`
        and `ax.set_aspect('equal')` don't work on 3D.
        """
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        x, y, z = origin
        ax.set_xlim3d([x - radius, x + radius])
        ax.set_ylim3d([y - radius, y + radius])
        ax.set_zlim3d([z - radius, z + radius])

    def _decay_volume_display(self, ax):
        """Add decay volume onto given ax. Decay volume is shown as a lightgrey box."""
        vertices = []
        for x in [self.config.decay_x_min, self.config.decay_x_max]:
            for y in [self.config.decay_y_min, self.config.decay_y_max]:
                for z in [self.config.decay_z_min, self.config.decay_z_max]:
                    vertices.append([x, y, z])
        v = np.array(vertices)
        sides = [[v[0], v[2], v[3], v[1]],
                 [v[4], v[6], v[7], v[5]],
                 [v[0], v[4], v[6], v[2]],
                 [v[1], v[5], v[7], v[3]],
                 [v[0], v[4], v[5], v[1]],
                 [v[2], v[6], v[7], v[3]]]
        ax.add_collection3d(art3d.Poly3DCollection(sides, facecolors='lightgray', linewidths=0.5, alpha=.3))

    def _module_display(self, ax): 
        """Add tracker modules onto given ax. Each module is a rectangular region."""
        
        for key in self.module_groups.keys():
            if self.module_groups[key]['type'] == 'y':
                for z in self.module_groups[key]['z_range']:
                    for x in self.module_groups[key]['x_mesh']:
                        for y in self.module_groups[key]['y_layer']: 
                            rect = Rectangle((x, z[0]), self.module_groups[key]['module_x_dim'], z[1]-z[0], color="#013220", alpha=0.2)
                            ax.add_patch(rect)
                            art3d.pathpatch_2d_to_3d(rect, z=y, zdir='y')
                            
            elif self.module_groups[key]['type'] == 'x':
                for y in self.module_groups[key]['y_mesh']:
                    for z in self.module_groups[key]['z_range']:
                        for x in self.module_groups[key]['x_layer']:
                            rect = Rectangle((y, z[0]), self.module_groups[key]['module_y_dim'], z[1]-z[0], color="#00008b", alpha=0.2)
                            ax.add_patch(rect)
                            art3d.pathpatch_2d_to_3d(rect, z=x, zdir='x')
                
            elif self.module_groups[key]['type'] == 'z':
                for x in self.module_groups[key]['x_mesh']:
                    for y in self.module_groups[key]['y_mesh']:
                        for z in self.module_groups[key]['z_layer']:
                            rect = Rectangle((x, y), self.module_groups[key]['module_x_dim'], self.module_groups[key]['module_y_dim'], color="#8ec4e6", alpha=0.2)
                            ax.add_patch(rect)
                            art3d.pathpatch_2d_to_3d(rect, z=z, zdir='z')

    def _track_display(self, ax, recon, zorder):
        if self.current_event is not None:
            self.current_event.track_display(ax, recon, zorder)

    def _hit_display(self, ax, hit, label):
        """When an event is loaded, this method adds each tracker hit location onto given ax. A tracker hit is
        represented by an 'x'"""
        if self.current_event is not None:
            self.current_event.hit_display(ax, hit, label)

    # ======Event====== 
    def clear_detector(self):
        """Flush all event variables.
        """
        self.current_particle = None
        self.current_vertex = None
        self.current_event = None
        for key in self.modules.keys():
            # clear previous tracker hits from modules
            self.modules[key].clear_module()
        self.current_event_mode = Detector.NO_EVENT

    def new_particle_event(self, particle: Particle):
        """Create new event with single particle track
        """
        self.current_particle = particle
        self.current_vertex = None
        self.current_event = DetectorParticle(particle, self)
        self.current_event_mode = Detector.PARTICLE

    def new_vertex_event(self, vertex: Vertex):
        """Create new event with a displaced vertex arising from LLP decay
        """
        self.current_particle = None
        self.current_vertex = vertex
        self.current_event = DetectorVertex(vertex, self)
        self.current_event_mode = Detector.VERTEX

    def return_current_particle(self) -> Optional[Particle]:
        """Return the current particle event if a DetectorParticle is loaded"""
        if self.current_event_mode == Detector.PARTICLE:
            return self.current_event
        else:
            return None

    def return_current_vertex(self) -> Optional[Vertex]:
        """Return the current vertex event if a DetectorVertex is loaded"""
        if self.current_event_mode == Detector.VERTEX:
            return self.current_event
        else:
            return None
        
    def return_num_particles(self) -> Optional[int]:
        """Return the number of decay particles of the current vertex."""
        if self.current_event_mode == Detector.PARTICLE:
            return 1
        elif self.current_event_mode == Detector.VERTEX:
            return len(self.current_event.particles)
        else:
            return None
        
    def return_num_vis_particles(self) -> Optional[int]:
        """Return the number of visible decay particles of the current vertex."""
        if self.current_event_mode == Detector.PARTICLE:
            if self.current_event._visibility == DetectorParticle.VISIBLE:
                return 1
            else:
                return 0
        elif self.current_event_mode == Detector.VERTEX:
            num = 0
            for par in self.current_event.particles:
                    if par._visibility == DetectorParticle.VISIBLE:
                        num += 1
            return num
        else:
            return None
        
    ########### FUNCTIONS SPECIFIC TO TRIGGER STUDIES #####################
       
    def per_trigger_module_hits(self, trigger_label_list:list): # returns dictionary
        """Return a dictionary of dictionaries i.e. {z_layer: {module_index: # of visible hits}}. There are a 100 total
        modules per a total of 6 z-trigger layers."""
        hit_arr = np.zeros((6,10,10))
        lst = sorted(trigger_label_list)
        x,y = [],[]
        for module in self.modules.keys():
                if self.modules[module].label in lst:
                    x.append(np.round(self.modules[module].x_index,4))
                    y.append(np.round(self.modules[module].y_index,4))
        x,y = sorted(list(set(x))),sorted(list(set(y)))
        dx = {k: v for v, k in enumerate(x)}
        dy = {k: v for v, k in enumerate(y)}
        for i in range(len(lst)):
            for module in self.modules.keys():
                if self.modules[module].label == lst[i]:
                    hit_arr[i,dx[np.round(self.modules[module].x_index,4)],dy[np.round(self.modules[module].y_index,4)]] = len([hit for hit in self.modules[module].tracker_hits if hit.track._visibility == DetectorParticle.VISIBLE])
                    
        return hit_arr
    
    def grid_wise_hits(self, trigger_label_list:list):
        """Return a 68 x 15 x 50 array giving hit distribution 3x3 grid (layer- and module-wise), grids uniquely labelled from 0 to 67 (64-67 are the 2x2 corner grids).
        """
        hits = self.per_trigger_module_hits(trigger_label_list)
        grid = np.zeros((68,15,300))
        for i in self.grids.keys(): # runs from 0 to 67
            t = self.grids[i]
            l = np.sum(hits[:,t[0][0]:t[0][-1]+1,t[1][0]:t[1][-1]+1],axis=(1,2)).astype(int) # returns 6 numbers
            m = np.sum(hits[:,t[0][0]:t[0][-1]+1,t[1][0]:t[1][-1]+1],axis=0).flatten().astype(int) # returns 9/4 numbers
            for j in range(len(l)):
                if l[j] != 0:
                    grid[i][j][l[j]] += 1
            for j in range(len(m)):
                if m[j] != 0:
                    grid[i][6+j][m[j]] += 1
                    
        return grid
    
    def dc_hits(self, rank, trigs_reqd): 
        """
        Rank specifies whether we want the least or second-least overwhelming track, trigs_reqd is the minimum number of triggered tracks needed to count the event (otherwise ignored)
        """
        
        dc_l = np.zeros((15,300))
        dc_m = np.zeros((15,300))
        
        num_vis_tracks_trigger = self.event_particle_pass_trigger() # list of vis particles triggering 
        
        if num_vis_tracks_trigger is not None and len(num_vis_tracks_trigger) >= trigs_reqd:
            hits = self.per_trigger_module_hits([0,1,2,3,4,5]) # should return 6*10*10 array
            grid_labels = []
            for par in num_vis_tracks_trigger:
                grid_labels.append(list(set(par.trigger_grids))) # list of list of triggering grids for each track
            grid_dict = self.grids
            
            trackwise_L1 = [] # to store trackwise min L1 to take a overall minimum over later
            trackwise_M = [] # to store trackwise min (max hits M1-9) to take a overall minimum over later
            
            for grid_list in grid_labels: # for each triggering list
                grid_index = None
                L1_min = 1000
                equal = {}
                for i in range(len(grid_list)):
                    temp = np.sum(hits[0,
                                       grid_dict[grid_list[i]][0][0]:grid_dict[grid_list[i]][0][-1]+1,
                                       grid_dict[grid_list[i]][1][0]:grid_dict[grid_list[i]][1][-1]+1]).astype(int)
                    if temp <= L1_min:
                        L1_min = temp
                        grid_index = grid_list[i]
                        equal[grid_index] = L1_min 
                        
                val_min = min(list(equal.values()))
                for key in equal:
                    if equal[key] == val_min:
                        trackwise_L1.append((equal[key], key))
    
                # now do for J^tilde_M_alpha
            
                temp = {i: np.max(np.sum(hits[:, grid_dict[i][0][0]:grid_dict[i][0][-1]+1, grid_dict[i][1][0]:grid_dict[i][1][-1]+1],axis=0).flatten().astype(int)) for i in grid_list}
            
                min_val = min(list(temp.values()))
                
                for key in temp:
                    if temp[key] == min_val:
                        trackwise_M.append((temp[key], key))

            # deal with the collected the layer-optimized data
            hits_sorted = list(set(sorted([x[0] for x in trackwise_L1])))
            if len(hits_sorted) >= 2:
                reqd_hits = hits_sorted[rank]
            else:
                reqd_hits = hits_sorted[0]
            
            degenerate = []
            for item in trackwise_L1:
                if item[0] == reqd_hits:
                    degenerate.append(item[1])
                    
            degenerate = list(set(degenerate))
            
            L1_min = random.choice(degenerate)
   
            x = grid_dict[L1_min][0]
            y = grid_dict[L1_min][1]

            l = np.sum(hits[:,x[0]:x[-1]+1,y[0]:y[-1]+1],axis=(1,2)).astype(int) # returns 6 numbers

            m = np.sum(hits[:,x[0]:x[-1]+1,y[0]:y[-1]+1],axis=0).flatten().astype(int) # returns 9 numbers

            for j in range(len(l)):
                if l[j] != 0:
                    dc_l[j][l[j]] += 1
            for j in range(len(m)):
                if m[j] != 0:
                    dc_l[6+j][m[j]] += 1
                        
            # now deal with the collected module optimized data
            
            hits_sorted = sorted([x[0] for x in trackwise_M])
            reqd_hits = hits_sorted[rank]
            
            degenerate = []
            for item in trackwise_M:
                if item[0] == reqd_hits:
                    degenerate.append(item[1])
                    
            degenerate = list(set(degenerate))
            
            L1_min = random.choice(degenerate)

            l = np.sum(hits[:,grid_dict[L1_min][0][0]:grid_dict[L1_min][0][-1]+1,grid_dict[L1_min][1][0]:grid_dict[L1_min][1][-1]+1],axis=(1,2)).astype(int) # returns 6 numbers
            m = np.sum(hits[:,grid_dict[L1_min][0][0]:grid_dict[L1_min][0][-1]+1,grid_dict[L1_min][1][0]:grid_dict[L1_min][1][-1]+1],axis=0).flatten().astype(int) # returns 9 numbers
            for j in range(len(l)):
                if l[j] != 0:
                    dc_m[j][l[j]] += 1
            for j in range(len(m)):
                if m[j] != 0:
                    dc_m[6+j][m[j]] += 1

                        
        return dc_l, dc_m
    
    ###############################################################################################
    
    
    def num_vis_events_through_layer(self, label) -> Optional[int]: 
        """Return the number of visible events/tracks that pass through the x,y, or z-layer. Input is only one coordinate at a time, with the other two being None by default."""
        if self.current_event_mode != Detector.NO_EVENT:
            num_events = 0
            if self.current_event_mode == Detector.PARTICLE and self.current_event._visibility == DetectorParticle.VISIBLE:
                for hit in self.current_event.tracker_hits:
                    if hit.hit_module.label == label:
                        num_events += 1 
                        break
                          
            else:
                for par in self.current_event.particles:
                    if par._visibility == DetectorParticle.VISIBLE:
                        for hit in par.tracker_hits:
                            if hit.hit_module.label == label:
                                num_events += 1 
                                break
                                  
            return num_events # number of visible particles that pass through that layer 
  
        return None

    def vertex_reconstructed(self, recon_criteria="default") -> int:
        """Return 1 vertex can be reconstructed according to specified vertex reconstruction criterion, 0 if not.
           Return -1 if no vertex object is loaded."""
        if self.current_event_mode == Detector.VERTEX:
            if recon_criteria == "default":
                recon_criteria = self.config.vertex_recon_default
            if recon_criteria in self.current_event.recon_criteria:
                return Detector.TRUE
            else:
                return Detector.FALSE
        else:
            return Detector.NONE

    def vertex_num_recon_tracks(self, recon_criteria="default") -> int:
        """Return number of tracks reconstructed with this vertex by the track recon_criteria"""
        if self.current_event_mode == Detector.VERTEX:
            if recon_criteria == "default":
                recon_criteria = self.config.track_recon_default
            hits = 0
            for par in self.current_event.particles:
                if recon_criteria in par.recon_criteria:
                    hits += 1
            return hits
        else:
            return Detector.NONE

    def vertex_xy_within_modules(self) -> int:
        """Return 1 if the vertex position is within at least one of the modules, 0 if not.
           Return -1 if no vertex object is loaded."""
        if self.current_event_mode == Detector.VERTEX:
            for x_module in self.module_x_range:
                if x_module[0] <= self.current_event.vertex.position[0] <= x_module[1]:
                    for y_module in self.module_y_range:
                        if y_module[0] <= self.current_event.vertex.position[1] <= y_module[1]:
                            return Detector.TRUE
            return Detector.FALSE
        else:
            return Detector.NONE

    def track_reconstructed(self, recon_criteria="default") -> int:
        """Return 1 if particle track can be reconstructed according to specified criterion, 0 if not.
           Return -1 if no particle event is loaded."""
        if self.current_event_mode == Detector.PARTICLE:
            if recon_criteria == "default":
                recon_criteria = self.config.track_recon_default
            if recon_criteria in self.current_event.recon_criteria:
                return Detector.TRUE
            else:
                return Detector.FALSE
        else:
            return Detector.NONE

    def event_pass_trigger(self, trigger_criteria="default") -> int:
        """Return 1 if any event passes the trigger criteria, 0 if not.
           Return -1 if no event is loaded."""
        if trigger_criteria == "default":
            trigger_criteria = self.config.track_trigger_default
        if self.current_event is None:
            return Detector.NONE
        elif self.current_event_mode == Detector.PARTICLE:
            if trigger_criteria in self.current_event.trigger_criteria:
                return Detector.TRUE
            else:
                return Detector.FALSE
        else:
            for par in self.current_event.particles:
                if trigger_criteria in par.trigger_criteria:
                    return Detector.TRUE
            return Detector.FALSE

    def event_particle_pass_trigger(self, trigger_criteria="default") -> Optional[List[Particle]]:
        """Return list of particles which pass the specific trigger criterion.
           Return None if no vertex event is loaded."""
        if self.current_event_mode == Detector.VERTEX:
            if trigger_criteria == "default":
                trigger_criteria = self.config.track_trigger_default
            par_pass = []
            for par in self.current_event.particles:
                if trigger_criteria in par.trigger_criteria:
                    par_pass.append(par) 
            return par_pass
        return None

    def track_is_in_decay_volume(self) -> int:
        """
        :return: Whether any particle track passes through the decay volume.
            1 for true
            0 for false
            -1 for not applicable
        """
        if self.current_event is None:
            return Detector.NONE
        else:
            if self.current_event.wall_hit["DECAY"]:
                return Detector.TRUE
            else:
                return Detector.FALSE

    def track_is_in_detector_volume(self) -> int:
        """
        :return: Whether any particle track passes through the Detector volume
            1 for true
            0 for false
            -1 for not applicable
        """
        if self.current_event is None:
            return Detector.NONE
        else:
            if self.current_event.wall_hit["DETECTOR"]:
                return Detector.TRUE
            else:
                return Detector.FALSE

    def vertex_is_in_decay_volume(self) -> int:
        """
        :return: Whether vertex is in the decay volume.
            1 for true
            0 for false
            -1 for not applicable
        """
        if self.current_event_mode == Detector.VERTEX:
            if self.current_event.wall_hit["LLP_DECAY"]:
                return Detector.TRUE
            else:
                return Detector.FALSE
        else:
            return Detector.NONE

    def vertex_is_in_detector_volume(self) -> int:
        """
        :return: Whether vertex is in the Detector volume
            1 for true
            0 for false
            -1 for not applicable
        """
        if self.current_event_mode == Detector.VERTEX:
            if self.current_event.wall_hit["LLP_DETECTOR"]:
                return Detector.TRUE
            else:
                return Detector.FALSE
        else:
            return Detector.NONE
