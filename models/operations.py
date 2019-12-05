## operations.py
##
## A collection of useful operations that are performed during reinforcement learning
## that are best encapsulted as a tensorflow operation.
##
## CloneOperation - 

import tensorflow as tf

class CloneOperation(object):
  """
  Wrapper for Tensorflow operation to perform cloning operation between a source and target network
  """

  def __init__(self, source_network, target_network, scope='clone_operation'):
    """
    """

    self.source_network = source_network
    self.target_network = target_network
    self.scope = scope

    # Get the default session to perform operations in; raise an error if no session is available 
    self.sess = tf.compat.v1.get_default_session()
    assert self.sess != None, "No Tensorflow Session found, a Session is needed for the Clone operation"

    # Create an operator to update the target weights from the current DQN
    self.update_operations = []

    with tf.compat.v1.variable_scope(self.scope):
    	for source_param, target_param in zip(self.source_network.getParams(), self.target_network.getParams()):
    		self.update_operations.append(tf.compat.v1.assign(target_param, source_param))


  def __call__(self):
    """
    Perform the update operations
    """

    self.sess.run(self.update_operations)

