from DetectorSimulation.Particle import Particle
from DetectorSimulation.Vertex import Vertex

from typing import List
import numpy as np


def create_llp_from_file(llp_file: str, position=(0, 0, 0), num=-1) -> List[Vertex]:
    """

    :param llp_file: file name
    :param position: initial position of the llp
    :param num: number of llps to load. Default is to load all.
    :return:
    """
    llp_str = load_llp_file(llp_file, num)
    vertex = []
    for llp in llp_str:
        vertex.append(allocate_llp_data(llp, position))
    return vertex


def load_llp_file(llp_file: str, num=-1) -> List[List[str]]:
    """Import particle from llp_file.

        :param llp_file: file name
               format of stonysim output
               for each event
               { parent LLP particle info,
               {immediate descendent particle info table}
               { final states particle info table} }

               and each particle info entry has this format:
               {"px","py","pz","E", "m", "PID"}
        :param num: number of llps to read from file. Default -1 means read all
        :return: A particle or vertex
        """
    file = open(llp_file, "r")
    raw_lines = file.readlines()
    lines = [line.strip() for line in raw_lines]
    file.close()
    # strip -> split at , -> strip -> split at , ->...
    # Remove brackets at start and finish of file
    if lines[0] == lines[1] == "{":
        lines = lines[1:-1]
    llp_data_set = []
    i = 0
    last_split = -1
    while i < len(lines) and (num == -1 or len(llp_data_set) < num):
        if lines[i] == "," or not lines[i].strip:
            llp_data_set.append([lines[last_split + 2], lines[i - 1]])
            last_split = i
        i += 1
    return llp_data_set


# IMPORTANT: change the line of code before the function returns depending on whether LLPs used are in rest frame of parent or not
def allocate_llp_data(data: list, position=(0, 0, 0)) -> Vertex:
    """

    :param data:
    :param position:
    :return: a vertex
    
    """
    if data[-1] == "{{}}}": # read in invisible decay
        return None
    else:
        llp_data = [line.replace('*^', 'e') for line in data]
    
        particles = []
        par_data = llp_data[1].split("},{")
        for par_str in par_data:
            par_param = tuple(map(float, par_str.strip("{").strip("}").split(",")))
            par = Particle(position, (par_param[3],)+par_param[:3], int(par_param[5]), par_param[4])
            particles.append(par)
            
        vertex_param = tuple(map(float, llp_data[0].strip("{").strip("},").split(",")))
        
        ## NOTE: comment out one of the next two lines of code depending on the simulation details
        
        # for llp is in rest frame, fv is (mllp, 0, 0, 0)
        vertex = Vertex(position, (vertex_param[4],0,0,0), int(vertex_param[5]), particles, vertex_param[4])
        
        ## next line is only if llp NOT in rest frame (when being read in from file)
#         vertex = Vertex(position, (vertex_param[3],)+vertex_param[:3], int(vertex_param[5]), particles, vertex_param[4])
        
        return vertex


def align_trajectory(llp: Vertex, target: tuple, p_norm=None) -> Vertex:
    """Aim particle at target and boost LLP momentum to new momentum if specified.

    :param llp: a long lived particle
    :param target: position to aim llp at.
    :param p_norm: momentum llp should be boosted to.
    """
    # Find rotation axis and rotation angle
    # Angle is measured away from current momentum, toward location vector.
    p = llp.momentum[1:]
    # axis is perpendicular to llp momentum and vector from current position to target
    direction = np.array(target) - llp.position
    axis = np.cross(p, direction)
    if np.linalg.norm(p) == 0 or np.linalg.norm(direction) == 0:
        # if no rotation is necessary
        new_llp = llp.clone()
    else:
        angle = np.arccos(np.dot(p, direction) / (np.linalg.norm(p) * np.linalg.norm(direction)))
        new_llp = llp.rotate_new_vertex(tuple(axis), angle)
    new_llp.place_this_vertex(target)

    # Boost to new momentum if specified
    # momentum should be a 4 vector
    if p_norm is not None and np.linalg.norm(direction) != 0:
        # Can only boost particle with mass
        if llp.mass != 0:
            # project onto momentum axis
            energy = new_llp.momentum[0]
            momentum_1d = np.linalg.norm(new_llp.momentum[1:])
            new_energy = np.sqrt(new_llp.mass**2 + p_norm**2)
            # boost 1: exceptional case where momentum of vertex is already 0
            if momentum_1d != 0:
                beta1 = momentum_1d / energy
                new_llp.boost_this_vertex(tuple(-direction), beta1)
            # boost 2: exceptional case where target momentum is 0
            if p_norm != 0:
                beta2 = p_norm / new_energy
                new_llp.boost_this_vertex(tuple(direction), beta2)
        else:
            print("Massless particle cannot be boosted.")

    return new_llp
