# My Utility : Algorithm PSO 

import numpy    as np
import util_bp  as np
import random 

class Particle:
   
    def __init__(self):
        self.position = np.array([(-1) ** (bool(random.getrandbits(1))) * random.random()*50, (-1)**(bool(random.getrandbits(1))) * random.random()*50])
        self.pbest_position = self.position
        self.pbest_value = float('inf')
        self.velocity = np.array([0,0])
    def move(self):
        self.position_i = self.position_i + self.velocity_i

class Space():

    def __init__(self, n_particles,particles):
        self.n_particles = n_particles
        self.particles = particles
        self.gbest_value = float('inf')
        self.gbest_position = np.array([random.random()*50, random.random()*50])
    
    def set_pbest(self):
        for particle in self.particles:
            fitness_cadidate = self.fitness(particle)
            if(particle.pbest_value > fitness_cadidate):
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.position
            

    def set_gbest(self):
        for particle in self.particles:
            best_fitness_cadidate = self.fitness(particle)
            if(self.gbest_value > best_fitness_cadidate):
                self.gbest_value = best_fitness_cadidate
                self.gbest_position = particle.position

    
#Swarm: Initializing
def iniSwarm(particle_number):
    particles_vector = [Particle() for _ in range(particle_number)]
    space = Space.__init__(space,particle_number, particles_vector)
    return(space)
# Fitness by use MSE
def Fitness_mse():
    ...
    return()
    
# Update:Particles based on Fitness-MSE
def upd_pFitness():
    ...
    return()
# Update: Swarm's velocity
def upd_veloc(particle:Particle,pos_best_g):
    ...
    return()  
  
    
#-----------------------------------------------------------------------