## factory.py
##
## 

import tensorflow as tf 
import numpy as np 


# Helpers for creating nn model
def create_weight(shape, name=None, std_dev=0.25):
   """
   Create a weight variable
   """

   return tf.Variable(tf.random.normal(shape, stddev=std_dev), name=name)


def create_bias(shape, name=None):
	"""
	Create a bias variable
	"""

	return tf.Variable(tf.zeros(shape), name=name)

