import os

from DetectorSimulation.llp_gun import *
from DetectorSimulation.Detector import Detector
from DetectorSimulation.lorentz_transformation import *
from typing import Union

import numpy as np


# particle ID -> mass
# feel free to add in more particles if need be
PARTICLES = {11: 0.511 * 1e-3, 12: 0, 13: 0.1056583755, 311: 0.5, 421: 1.86484, 423: 2.00685, 111: 0.13497, 113: 0.77526,
            411: 1.86966, 413: 2.01026, 211: 0.13957, 213: 0.77511, 431: 1.96835, 321: 0.493677, 323: 1.414, 221:         0.547862, 313: 1.414, 433: 2.112} # GeV

# hadronic decay llp files
hadronic_decay = {}


def get_llp(mode: str, mass: float, decay_position: tuple, boost: float, decay_to: Union[list,str,None]=None) -> Vertex:
    """

    :param mass: mass of llp in GeV
    :param mode: leptonic2body or hadronic
    :param decay_position: decay position of the llp
    :param boost: b = p/m, where beta = b/np.sqrt(b**2 + 1)
    :param decay_to: (optional) list of PIDs of decay products in leptonic 2-body/3-body decays or specific decay tag (in case of 
                     SMS, take a look below and at function hadronic_SMS) or None (for all other empty decays)
    :return: a llp vertex
    """
    # create an llp at IP
    if mode == "leptonic3body":
        llp = leptonic_3_body(mass, PARTICLES[decay_to[0]], PARTICLES[decay_to[1]], PARTICLES[decay_to[2]], decay_to)
    elif mode == "leptonic2body":
        llp = leptonic_2_body(mass, PARTICLES[decay_to[0]], PARTICLES[decay_to[1]], decay_to)
    elif mode == "hadronic_RHN_Ue":
        llp = hadronic_RHN_Ue(mass)
    elif mode == "hadronic_RHN_Umu":
        llp = hadronic_RHN_Umu(mass)
    elif mode == "hadronic_RHN_Utau":
        llp = hadronic_RHN_Utau(mass)
    elif mode == "hadronic_SMS":
        llp = hadronic_SMS(mass, decay_to)
    else: # mode == "hadronic_HXX"
        llp = hadronic_HXX(mass)
    if llp is None:
        return None
    else:
        new_p = mass * boost
        # aim and boost at decay position. This way the llp coming from IP will pass through the decay position in its trajectory
        new_llp = align_trajectory(llp, decay_position, new_p)
  
        return new_llp


def leptonic_2_body(mass: float, m1: float, m2: float, decay_product: list): # for generic daughters
    """
    :param mass: mass of the parent llp
    :param m1: mass of decay product 1
    :param m2: mass of decay product 2
    :param decay_product: PID of the decay products
    :return:
    """
    E1 = abs((mass**2 + m1**2 - m2**2)/(2*mass))
    E2 = abs((mass**2 + m2**2 - m1**2)/(2*mass))
    p_norm = np.sqrt(abs(E1**2 - m1**2))
    phi = np.random.uniform(0, 2*np.pi)
    theta = np.arccos(2*np.random.uniform(0, 1) - 1)
    p_decay_1 = (E1, p_norm*np.sin(theta)*np.cos(phi), p_norm*np.sin(theta)*np.sin(phi), p_norm*np.cos(theta))
    p_decay_2 = (E2, -p_norm*np.sin(theta)*np.cos(phi), -p_norm*np.sin(theta)*np.sin(phi), -p_norm*np.cos(theta))
    decay_1 = Particle(Detector.IP, p_decay_1, decay_product[0])
    decay_2 = Particle(Detector.IP, p_decay_2, decay_product[1])
    # this makes a vertex at the origin
    return Vertex(Detector.IP, (mass, 0, 0, 0), 11111, [decay_1, decay_2])

## same function as above, just with decay product masses equal
# def leptonic_2_body(mass: float, decay_product) -> Vertex:
#     """

#     :param mass: mass of the parent llp
#     :param decay_product: PID of the decay products
#     :return:
#     """
#     E_decay = mass / 2
#     p_norm = np.sqrt(E_decay**2 - PARTICLES[decay_product]**2)
#     phi = np.random.uniform(0, 2*np.pi)
#     theta = np.arccos(2*np.random.uniform(0, 1) - 1)
#     p_decay_1 = (E_decay, p_norm*np.sin(theta)*np.cos(phi), p_norm*np.sin(theta)*np.sin(phi), p_norm*np.cos(theta))
#     p_decay_2 = (E_decay, -p_norm*np.sin(theta)*np.cos(phi), -p_norm*np.sin(theta)*np.sin(phi), -p_norm*np.cos(theta))
#     decay_1 = Particle(Detector.IP, p_decay_1, decay_product)
#     decay_2 = Particle(Detector.IP, p_decay_2, decay_product)
#     # this makes a vertex at the origin
#     return Vertex(Detector.IP, (mass, 0, 0, 0), 11111, [decay_1, decay_2])


#### Leptonic 3-Body ###
    
# 2-body kinematic function
def l(x,y,z):
    return (x-y-z)**2 - 4*y*z

# in CM frame of parent
# https://arxiv.org/pdf/hep-ph/0508097.pdf pg 46
def leptonic_3_body(m, m1, m2, m3, pid, n=1):
    # pid is an array of particle id's
    assert 0 <= min(m1, m2, m3) and max(m1, m2, m3) <= m
    s =  m**2 # Mandelststam variable

    # generate angular variables
    phi1, phi2= 2*np.pi*np.random.rand(2,n)
    theta1, theta2 = np.arccos(2*np.random.rand(2,n)-1)

    # get endpoint    
    E1_max = (s + m1**2 - (m2 + m3)**2)/(2*np.sqrt(s))
    E1 = np.random.uniform(m1, E1_max, n)

    # get p1
    p1vecmag = np.sqrt(E1**2 - m1**2)
    p1x, p1y, p1z = p1vecmag*np.sin(theta1)*np.cos(phi1), \
                    p1vecmag*np.sin(theta1)*np.sin(phi1), \
                    p1vecmag*np.cos(theta1)
    p1 = np.transpose([E1] + [p1x] + [p1y] + [p1z])

    # now work in CM of p2, p3 (denoted 23)
    m_23 = np.sqrt(s - 2*np.sqrt(s)*E1 + m1**2) # invariant mass of m1 and m2 treated together
    p2vecmag_23 = np.sqrt(l(m_23**2, m2**2, m3**2))/(2*m_23)

    # get p2, p3 in 23 frame
    p2x_23, p2y_23, p2z_23 = p2vecmag_23*np.sin(theta2)*np.cos(phi2), \
                                p2vecmag_23*np.sin(theta2)*np.sin(phi2), \
                                p2vecmag_23*np.cos(theta2)
    p3x_23, p3y_23, p3z_23 = -p2x_23, -p2y_23, -p2z_23
    E2_23, E3_23 = np.sqrt(m2**2 + p2x_23**2 + p2y_23**2 + p2z_23**2), \
                    np.sqrt(m3**2 + p3x_23**2 + p3y_23**2 + p3z_23**2)
    p2_23, p3_23 = np.transpose([E2_23] + [p2x_23] + [p2y_23] + [p2z_23]), \
                    np.transpose([E3_23] + [p3x_23] + [p3y_23] + [p3z_23])

    # boost back to CM frame
    directions = p1[:,1:]
    betas = -p1vecmag/np.sqrt(p1vecmag**2 + m_23**2)
    p2 = np.asarray([boost_matrix(e[2] * (e[1]/np.linalg.norm(e[1]))).dot(e[0]) for e in zip(p2_23, directions, betas)])
    p3 = np.asarray([boost_matrix(e[2] * (e[1]/np.linalg.norm(e[1]))).dot(e[0]) for e in zip(p3_23, directions, betas)])
    

    # after getting momenta, create vertex
   
    decay_1 = Particle(Detector.IP, p1[0], pid[0])
    decay_2 = Particle(Detector.IP, p2[0], pid[1])
    decay_3 = Particle(Detector.IP, p3[0], pid[2]) # because neutrino

    return Vertex(Detector.IP, (m, 0, 0, 0), 11111, [decay_1, decay_2, decay_3])

#### hadronic functions #####
## These hadronic functions are almost the same but were separated into different functions for ease of use with the 
## different LLP analyses - but should be very easy to update them for any new analysis that need be done

########################## RHN_Ue #####################################################################

def hadronic_RHN_Ue(mass: float) -> Vertex:
    llp_file_mass = mass
    if llp_file_mass not in hadronic_decay: # each mass has a unique file so all good!
        if 0.42 <= llp_file_mass < 0.5:
            path = "/Users/jai/Desktop/MATHUSLA_TOTAL/RHN_Ue_hadronic_decays/vN_Ntoall_lightfonly_" + str(mass)+ ".txt"
        elif 0.5 <= llp_file_mass < 0.99:
            path = "/Users/jai/Desktop/MATHUSLA_TOTAL/RHN_Ue_hadronic_decays/vN_Ntoall_nocharmnoss_" + str(mass)+ ".txt"
        elif 0.99 <= llp_file_mass < 1.871:
            path = "/Users/jai/Desktop/MATHUSLA_TOTAL/RHN_Ue_hadronic_decays/vN_Ntoall_nocharm_" + str(mass)+ ".txt"
        elif 1.871 <= llp_file_mass < 1.97:
            path = "/Users/jai/Desktop/MATHUSLA_TOTAL/RHN_Ue_hadronic_decays/vN_Ntoall_inclD_" + str(mass) + ".txt"
        elif 1.97 <= llp_file_mass < 3.74:
            path = "/Users/jai/Desktop/MATHUSLA_TOTAL/RHN_Ue_hadronic_decays/vN_Ntoall_inclDs_" + str(mass) + ".txt"
        else: # m > 3.74
            path = "/Users/jai/Desktop/MATHUSLA_TOTAL/RHN_Ue_hadronic_decays/vN_Ntoall_inclDD_" + str(mass) + ".txt"
        
        hadronic_decay[llp_file_mass] = [load_llp_file(path, -1), 0]
        
    i = hadronic_decay[llp_file_mass][1] % len(hadronic_decay[llp_file_mass][0])
    hadronic_decay[llp_file_mass][1] += 1
    
    # this returns a vertex at the origin
    return allocate_llp_data(hadronic_decay[llp_file_mass][0][i], Detector.IP)


########################## RHN_Umu #####################################################################

def hadronic_RHN_Umu(mass: float) -> Vertex:
    llp_file_mass = mass
    if llp_file_mass not in hadronic_decay: # each mass has a unique file so all good!
        if 0.53 <= llp_file_mass <= 0.6:
            path = "/Users/jai/Desktop/MATHUSLA_TOTAL/RHN_Umu_hadronic_decays/vN_Ntoall_lightfonly_" + str(mass)+ ".txt"
        elif 0.6 < llp_file_mass <= 0.99:
            path = "/Users/jai/Desktop/MATHUSLA_TOTAL/RHN_Umu_hadronic_decays/vN_Ntoall_nocharmnoss_" + str(mass)+ ".txt"
        elif 0.99 < llp_file_mass <= 1.98:
            path = "/Users/jai/Desktop/MATHUSLA_TOTAL/RHN_Umu_hadronic_decays/vN_Ntoall_nocharm_" + str(mass)+ ".txt"
        elif 1.98 < llp_file_mass < 2.08:
            path = "/Users/jai/Desktop/MATHUSLA_TOTAL/RHN_Umu_hadronic_decays/vN_Ntoall_inclD_" + str(mass) + ".txt"
        elif 2.08 <= llp_file_mass < 3.75:
            path = "/Users/jai/Desktop/MATHUSLA_TOTAL/RHN_Umu_hadronic_decays/vN_Ntoall_inclDs_" + str(mass) + ".txt"
        else: # m > 3.74
            path = "/Users/jai/Desktop/MATHUSLA_TOTAL/RHN_Umu_hadronic_decays/vN_Ntoall_inclDD_" + str(mass) + ".txt"
        
        hadronic_decay[llp_file_mass] = [load_llp_file(path, -1), 0]
        
    i = hadronic_decay[llp_file_mass][1] % len(hadronic_decay[llp_file_mass][0])
    hadronic_decay[llp_file_mass][1] += 1
    
    # this returns a vertex at the origin
    return allocate_llp_data(hadronic_decay[llp_file_mass][0][i], Detector.IP)

    
########################## RHN_Utau #####################################################################

def hadronic_RHN_Utau(mass: float) -> Vertex:
    llp_file_mass = mass
    if llp_file_mass not in hadronic_decay: # each mass has a unique file so all good!
        if 0.42 <= llp_file_mass < 0.99:
            path = "/Users/jai/Desktop/MATHUSLA_TOTAL/RHN_Utau_hadronic_decays/vN_Ntoall_lightfonly_" + str(mass)+ ".txt"
        elif 0.99 <= llp_file_mass < 1.92:
            path = "/Users/jai/Desktop/MATHUSLA_TOTAL/RHN_Utau_hadronic_decays/vN_Ntoall_lightfsonly_" + str(mass)+ ".txt"
        elif 1.92 <= llp_file_mass < 2.28:
            path = "/Users/jai/Desktop/MATHUSLA_TOTAL/RHN_Utau_hadronic_decays/vN_Ntoall_lightfstau_" + str(mass)+ ".txt"
        elif 2.28 <= llp_file_mass < 3.65:
            path = "/Users/jai/Desktop/MATHUSLA_TOTAL/RHN_Utau_hadronic_decays/vN_Ntoall_lightfstauK_" + str(mass) + ".txt"
        elif 3.65 <= llp_file_mass < 3.75:
            path = "/Users/jai/Desktop/MATHUSLA_TOTAL/RHN_Utau_hadronic_decays/vN_Ntoall_lightfstauD_" + str(mass) + ".txt"
        else: # m > 3.75
            path = "/Users/jai/Desktop/MATHUSLA_TOTAL/RHN_Utau_hadronic_decays/vN_Ntoall_lightfstauDD_" + str(mass) + ".txt"
        
        hadronic_decay[llp_file_mass] = [load_llp_file(path, -1), 0]
        
    i = hadronic_decay[llp_file_mass][1] % len(hadronic_decay[llp_file_mass][0])
    hadronic_decay[llp_file_mass][1] += 1
    
    # this returns a vertex at the origin
    return allocate_llp_data(hadronic_decay[llp_file_mass][0][i], Detector.IP)


########################## SMS #####################################################################

def hadronic_SMS(mass: float, particle: str) -> Vertex: # particle is the particle that "hadronizes"
    llp_file_mass = mass
    if (llp_file_mass, particle) not in hadronic_decay:
        path = "/Users/jai/Desktop/MATHUSLA_TOTAL/SMS_LLP_decays/" + particle + "_" + str(mass) + ".txt"
        hadronic_decay[(llp_file_mass, particle)] = [load_llp_file(path, -1), 0]
    i = hadronic_decay[(llp_file_mass, particle)][1] % len(hadronic_decay[(llp_file_mass, particle)][0])
    hadronic_decay[(llp_file_mass, particle)][1] += 1
    
    # this returns a vertex at the origin
    return allocate_llp_data(hadronic_decay[(llp_file_mass, particle)][0][i], Detector.IP)

########################### HXX ######################################################################

def hadronic_HXX(mass: float) -> Vertex:
    if mass > 55:
        llp_file_mass = "55"
    else:
        llp_file_mass = str(int(5 * (mass//5)))
    if llp_file_mass not in hadronic_decay:
        path = os.path.join("/Users/jai/Desktop/H_hadronic_decays/bb_" + str(int(mass)) + ".txt")
        hadronic_decay[llp_file_mass] = [load_llp_file(path, -1), 0]
    i = hadronic_decay[llp_file_mass][1] % len(hadronic_decay[llp_file_mass][0])
    hadronic_decay[llp_file_mass][1] += 1
    # this returns a vertex at the origin
    return allocate_llp_data(hadronic_decay[llp_file_mass][0][i], Detector.IP)


