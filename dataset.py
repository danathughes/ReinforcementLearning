import numpy as np


class Dataset:
	"""
	A dataset maintains a collection of related data, defined by a provided schema.  In addition,
	a dataset object allows for generating random batches for minibatch training.
	"""

	def __init__(self, schema):
		"""
		Create a new dataset with the provided schema.  The schema consists of 
		a dictionary mapping the name of each data element to the shape of a single
		entry of the element
		"""

		# Maintain the dictionary of element shapes, and create a dictionary of the
		# actual elements with no current entries
		self.schema = schema
		self.data = { name: np.zeros((0,) + shape) for name, shape in self.schema.items() }

		# Set the number of elements to 0, and create an empty index
		self.size = 0
		self.indices = np.array([])
		self.batch_index = 0


	def __getitem__(self, key):
		"""
		Return the data stored by the provided key
		"""

		# Make sure the key is available in the data
		assert key in self.data.keys(), "Key %s is not present in dataset" % repr(key)

		# Return the data indexed by the key
		return self.data[key]


	def add(self, data):
		"""
		Add the provided data to the current dataset.  Data needs to be formatted as
		a dictionary mapping element name to entries.
		"""

		# TODO: Check to make sure that the provided data 1) has all entries of the
		#       schema, 2) the shape of the data matches the schema, and 3) the number
		#       of datapoints in each of the elements are the same

		# Check the number of elements in the data


		# Append the provided data to the dataset
		for element in self.data.keys():
			#if element in data.keys():
			self.data[element] = np.concatenate([self.data[element], data[element]])		

		# Recalculate the number of elements in the dataset
		self.size = self.data.values()[0].shape[0]
		self.indices = np.array(range(self.size))


	def augment(self, data):
		"""
		Augment the existing dataset with a new keys and associated data.  Data structure
		will be inferred from the provided data and added to the schema.
		
		data - a dictionary mapping keys to numpy arrays
		"""

		for key, value in data.items():
			# Check that the provided data is the correct length
			assert value.shape[0] == self.size, "Incorrect number of elements in provided data for key %s" % repr(key)

			# Add the shape of the data to the schema, and the value of the data to the dataset
			self.schema[key] = value.shape[1:]
			self.data[key] = value


	def replace(self, data):
		"""
		Replace the data in the dataset with the provided values, leaving unprovided values
		unchanged in the dataset

		data - a dictionary maping keys to numpy arrays
		"""

		for key, value in data.items():
			# Check that the provided data exists in the schema, and the shape is correct
			assert key in self.schema.keys(), "Provided data key not in schema: %s" % repr(key)
			assert self.schema[key] == value.shape[1:], "Provided data is incorrect shape, expecting %s, provided %s" % (repr(schema[key]), repr(value.shape[1:]))

			# Replace the data
			self.data[key] = value



	def shuffle(self):
		"""
		Randomize the indices
		"""

		# Reset the batch_index to the start of the indices
		self.batch_index = 0

		# Shuffle the indices
		self.indices = np.array(range(self.size))
		np.random.shuffle(self.indices)


	def reset(self):
		"""
		De-randomize the indicies and set the batch index to 0
		"""

		# Un-shuffle the indicies, so that order of the data reflects the order
		# the data appeared
		self.batch_index = 0
		self.indicies = nu.array(range(self.size))


	def clear(self):
		"""
		Empty out the data
		"""

		self.data = { name: np.zeros((0,) + shape) for name, shape in self.schema.items() }

		self.size = 0
		self.indices = np.array([])
		self.batch_index = 0


	def sample(self, batch_size):
		"""
		Get the next available batch as a dataset
		"""

		# Create a new dataset with the same schema
		batch = Dataset(self.schema)

		# Get the indices into the dataset for this batch
		end_idx = min(self.batch_index + batch_size, self.size)
		idx = self.indices[self.batch_index:end_idx]

		# Create the batch
		for name in self.data.keys():
			batch.data[name] = self.data[name][idx]

		batch.size = end_idx - self.batch_index

		# Increment the batch index
		self.batch_index = end_idx

		return batch