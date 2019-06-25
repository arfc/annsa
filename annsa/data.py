import pandas as pd
from annsa.template_sampling import make_random_spectrum

class spectrum_data(object):
	"""docstring for spectrum_data"""
	def __init__(self, filename, folder=False, ae=False, csv=False, online=False):
		"""
		Generates a data object. 

		Parameters: 
		-----------
		filename : string
			This is the path including the name of the file
			that contains your raw data.
		folder : boolean
			This says whether filename leads to a folder of
			data, or a single dataset. 
			Default is false
		ae : boolean
			This says whether the data should be processed 
			for autoencoders or not. 
			Default is false.
		csv : boolean
			If you are inputting data from a csv file this 
			should be true. 
			Default is false.
		online : boolean, optional
			Whether or not the data set should be sampled by
			using online_data_augmentation or not.
		"""
		super(spectrum_data, self).__init__()

		self.filename = filename
		self.dataframe = pd.DataFrame()

	def add_column(self):
		"""
		Adds a column of data to the spectrum dataframe. 
		"""
		pass

	def add_row(self):
		"""
		Adds a row of data to the spectrum dataframe.
		This is useful if you want to add more data to 
		an existing dataframe.
		"""
		#name = 
		#frames = [self.dataframe, row]
		#self.dataframe = pd.concat(frames)
		pass

	def parse_fname(self, filename):
		"""
		Separates the name of a single file into key
		identifiers based on the standard annsa naming
		convention.

		Parameters: 
		-----------
		filename : string
			Name of file containing raw spe data.

		Returns: 
		--------
		parameters : list of strings
			This list of strings will become the 
			column headers for the dataframe. 
		"""
		pass

	def make_dataframe(self):
		"""
		This function is only used when folder==True. 
		Moves through a folder and gets the files, 
		"""

		pass

	def sample_dataframe(self, **kwargs):
		"""
		Returns a single row of data from the dataframe
		by creating a subset of the dataframe based on 
		keyword arguments and then randomly selecting 
		a row from that subset. 
		"""

		#pd.dataframe.sample picks a random row
		pass

	def sample_spectrum(self):

		"""
		1. rebinning --> This allows you to simulate detector
		calibrations like the gain of a detector which may 
		change based on pmt_voltage or ambient temperature.
		This function already exists from Mark. 
		2. lower level discriminator (LLD) --> typically the 
		first few channels are simply noise and thus should
		be set to zero. ~10 channels. 
		3. normalize the function by area --> divides the 
		value of each channel by the sum of the counts.
		4. scale by the number of counts. This allows you to 
		simulate a dataset with a different number of counts
		than the dataset you are sampling from. 
		"""

		pass