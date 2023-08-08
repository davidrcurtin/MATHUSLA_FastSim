#!/usr/bin/env python
# coding: utf-8

# In[1]:


## write a function that converts geant decay files into FastSim decay files
def write_hadrons_for_fastsim(save: str, file: str):
    """
    save: the path to file where output will be saved
    file: the path to file to be read in 
    
    returns a file with fvs in format of stonysim output
               for each event
               { parent LLP particle info -> always in lab/MadGraph frame (because easy to convert to rest frame!),
               {immediate descendent particle info table}
               { final states particle info table} }

               and each particle info entry has this format:
               {"px","py","pz","E", "m", "PID"} -> always in rest frame of parent LLP
    """
    indices = [0]
    with open(file, "r") as f1:
        lines = f1.readlines()
        # get rid of first line as that is just instructions about the format
        lines = lines[1:]
        for i in range(len(lines)):
            if i+2 <= len(lines)-1:
                if lines[i] == "\n" and lines[i+1] == "\n" and i+2 <= len(lines)-2:
                    indices.append(i+2)
        indices.append(len(lines)-3)
                    
        with open(save, "w") as f:
            
            f.write("{\n")
            
            for i in range(len(indices)-1): # this is the start of the block
                index = indices[i]
                # LLP 4-vector
                E,px,py,pz,m,pid = lines[index].split(",")
                
                f.write("{\n")
                f.write("{"+f"{px},{py},{pz},{E},{m},{pid}"+"},\n")
                f.write("{")
                
                # daughter 4-vectors from above LLP
                for j in range(index+1,indices[i+1]-2):
                    E,px,py,pz,m,pid,name = lines[j].split(",")
                    if j == indices[i+1]-3:
                        f.write("{"+f"{px},{py},{pz},{E},{m},{pid}"+"}}}\n")
                        f.write(",\n")
                    else:
                        f.write("{"+f"{px},{py},{pz},{E},{m},{pid}"+"},")
                f.write("}")
        f.close() 

