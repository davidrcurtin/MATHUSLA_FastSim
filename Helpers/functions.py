#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
import random
import itertools
from typing import List,Optional
import DetectorSimulation.Detector as Detector
import DetectorSimulation.Particle as Particle
import DetectorSimulation.Vertex as Vertex
import DetectorSimulation.lorentz_transformation as lt


#############################################################################################

## read_vectors to be used with 4-vector in the format (E, px, py, pz)
## read_vectors_SMS to be used with 4-vectors in the format (weight, E, px, py, pz), where weight 
## is the number of real 4-vectors represented by given 4-vector (NOTE: SMS in the title, does
## not mean it should be only used with the SM+S case)

## argument "n" is the number of 4-vectors one needs, n = -1 reads in the entire file

def read_vectors(path: str, n=-1) -> List:
    # create llps using hadronic decay 4-vectors.
    vectors = []
    file = open(path, "r")
    for line in file.readlines():
        # split string and convert to four-vector of ints
        # skip over first index of elements of clean as they are serial number
        m = list(map(float, line.strip().split(",")))
        # data is for beam axis along z, our beam axis is y so update coordinates
        m[1], m[2], m[3] = -m[1], m[3], m[2]
        vectors.append(np.array(m))
    if n == -1:
        return vectors
    else:
        return random.sample(vectors, n)

def read_vectors_SMS(path: str, n=-1) -> List:
    # create llps using hadronic decay 4-vectors.
    vectors = []
    file = open(path, "r")
    for line in file.readlines():
        # split string and convert to four-vector of ints
        m = list(map(float, line.strip().split(",")))
        w = m[0] # get the weight
        m = m[1:] # get the fv
        # data is for beam axis along z, our beam axis is y so update coordinates
        m[1], m[2], m[3] = -m[1], m[3], m[2]
        vectors.append(np.array([w] + m)) # fvs are of form (w, fv) - same convention throughout code
    if n == -1:
        return vectors
    else:
        return random.sample(vectors, n) # randomly sample
    
#############################################################################################

## These functions take in 3-vectors and return there phi and theta angles

## get_theta is used to check if an LLP (4-vector) passes through the detector or not

## deal_with_phi takes advantage of rotational symmetry about the beam axis to randomly 
## choose a phi

# beam axis is y-axis
# theta is angle to y-axis i.e. arccos(y/r)
# phi is angle to x in x-z plane i.e arctan(x/z) 

def get_phi(u) -> float:# u is a three-vector 
    phi = np.arctan2(u[0], u[2]) # (x, z)
    return phi 

def get_theta(u) -> float:
    return np.arccos(u[1]/np.linalg.norm(u)) # (y/r)

def deal_with_phi(four_p: List[float], phi_min: float, phi_max: float) -> List[float]:
    
    curr_phi = get_phi(four_p[1:])
    new_phi = random.uniform(phi_min, phi_max)
    p_rot = lt.rotation(four_p[1:], np.array([0,1,0]), new_phi - curr_phi)
    new_4p = np.array([four_p[0], p_rot[0], p_rot[1], p_rot[2]])

    return new_4p


#############################################################################################

# function that finds max min angles OF THE DECAY VOLUME (note its not the detector volume!) 
def get_detector_angles(detector_benchmark: Detector.Detector) -> tuple:
    
    x = np.array([detector_benchmark.config.decay_x_min, detector_benchmark.config.decay_x_max])
    y = np.array([detector_benchmark.config.decay_y_min, detector_benchmark.config.decay_y_max])
    z = np.array([detector_benchmark.config.decay_z_min, detector_benchmark.config.decay_z_max])

    corners = np.array(np.meshgrid(x, y, z)).T.reshape(-1,3)
    points = corners.copy()

    detector_theta = []
    detector_phi = []

    # for efficiency, randomly generate 100 points lying in/on the detector and take min,max angles from these
    # -- ends up giving the (almost) actual min, max (using Central Limit Theorem)
    for j in itertools.product(corners, corners):
        for k in range(100):
            u = random.uniform(0,1)
            new = u * j[0] + (1-u) * j[1]
            detector_theta.append(get_theta(new))
            detector_phi.append(get_phi(new))

    theta_min, theta_max = min(detector_theta), max(detector_theta)
    phi_min, phi_max = min(detector_phi), max(detector_phi)
    
    return (phi_min, phi_max, theta_min, theta_max)


#############################################################################################

## MAIN FUNCTION

## this function takes in an LLP 4-vector and if it passes through the detector, returns
## a weight i.e. probability of decay in the detector (according to formula mentioned in paper),
## decay position (w.r.t. the exponential CDF of the probability) and its boost. Returns None is LLP 4-vector
## does not pass through detector.

## NOTE: a lot of the components of this function can be used by themselves as separate functions
## if need be depending on the context

# ctau is lifetime, might be input by hand or given by reading in a file
# detector_benchmark is a Detector object, which we show how to initialize later
def get_weight(four_p: List[float], mass: float, ctau: float, detector_benchmark: Detector.Detector) -> Optional[tuple]:
    
    ## first we check if 4-vector passes through the detector and if it does, obtain its boundary hit coordinates i.e.
    ## entry (L1) and exit points (L2)
    
    # clear the detector of past events, if any
    detector_benchmark.clear_detector()
    # choose a position for the LLP to start at - we just choose the interaction point (IP) and the sim
    # automatically uses kinematics to figure out trajectory
    position = (0,0,0)
    # create an LLP particle, Particle.Particle(position, 4-vector, PID) - choose a random PID
    llp = Particle.Particle(position, four_p, 13)
    # feed this particle to the detector as a new (particle) event - other options include new_vertex_event which 
    # feeds a vertex (i.e. an LLP with its decay products) to the detector
    detector_benchmark.new_particle_event(llp)

    # set up a list to collect boundary hit point coordinates
    L1L2 = []
    
    # this list is a short-hand way of coordinates of the detector's boundary planes
    decay_vol_boundaries=['x-','x+','y-','y+','z-','z+']
    
    # if LLP passes through the detector
    if (detector_benchmark.track_is_in_decay_volume()):
        
        # get all the sensor layer hits in within the decay volume
        current_decay_volume_intersections = detector_benchmark.return_current_particle().decay_volume_hits

        # collect all the boundary hit objects
        decay_vol_boundaries_hits=[]
        
        # going through each of the boundary planes, check if there is a hit and collect it
        for i in decay_vol_boundaries:
            if current_decay_volume_intersections[i][0]:
                decay_vol_boundaries_hits.append(current_decay_volume_intersections[i][1])

        # another sanity check, there should be 2 hits, entry and exit
        if (len(decay_vol_boundaries_hits) == 2):
            # now convert list of hit coordinates into distance from the origin/IP
            L1L2 = [math.sqrt(item[0]**2 + item[1]**2 + item[2]**2) for item in decay_vol_boundaries_hits]
            L1L2.sort()
            # L1 is distance of entry point to IP
            # L2 is distance of exit point to IP
            L1,L2 = L1L2
            # sort the decay_vol_boundaries_hits so that the first (second) entry corresponds to coordinates of L1 (L2)
            decay_vol_boundaries_hits.sort(key=lambda x:math.sqrt(x[0]**2 + x[1]**2 + x[2]**2))
            
            # get the boost
            b = np.linalg.norm(four_p[1:])/mass 
            
            # now using random variable PDF inversion (using uniform distribution) get an exponential distribution for 
            # decay probability between L1 and L2
            unif = random.uniform(1-np.exp(-L1/(b*ctau)), 1-np.exp(-L2/(b*ctau)))
            exp = -b*ctau*np.log(1-unif) # use pdf inversion to get exponential distribution
            # explicitly get the decay position by multiply unit vector along L1 with the chosen norm from the exp distribution
            decay_position = np.array(decay_vol_boundaries_hits[0])/L1 * exp
            
            # get the weight i.e. probability of decay within the detector volume
            weight = np.exp(-L1/(b*ctau)) - np.exp(-L2/(b*ctau))

            return weight, decay_position, b
        
        # or return None if LLP does not pass through the detector
        else: 
            return None