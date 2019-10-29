## abstract_model.py
##
## An abstract neural network model defining common interface for all networks used


import tensorflow as tf 
import numpy as np 

from dataset import *

class AbstractModel:
	"""
	An abstract representation of policy and value networks that defines common methods for all
	networks
	"""

	def __init__(self):
		"""
		"""

		pass


	def __call__(self):
		"""
		Reteurn the
		"""

		raise NotImplementedError