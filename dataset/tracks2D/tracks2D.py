## tracks2D.py
##
## Simple script to generate dataset for the synthetic task in InfoGAIL paper
##

import numpy as np
import matplotlib.pyplot as plt

class TrackAgent:
   """
   A track agent is designed to simply walk around in a circle at a constant velocity
   """

   def __init__(self, radius, velocity, noise):
      """
      Create a track agent to generate a noisy circular path

      Parameters:
      	center - center of the 
        radius - the radius of the circle the agent follows
        velocity - the distance the agent covers each time step
        noise - the standard deviation of Gaussian noise added to the orientation at
                each time step
      """

      self.radius = radius
      self.velocity = velocity
      self.noise = noise

      # Start at the origin
      self.position = (0., 0.)
      self.



   def step(self):
      """
      """

