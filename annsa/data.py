import pandas as pd
from annsa.template_sampling import make_random_spectrum
import os

class spectrum_data(object):
	"""docstring for spectrum_data"""
	def __init__(self, filename, folder=False, ae=False, csv=False, online=False):
		"""
		Generates a data object of type SpectrumData. 

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

		self.keys = ['Isotope', 'Distance', 'Height', 'Shielding', 'Areal Density', 'FWHM', 'Spectrum']
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
		#gets the keys from the file name
		values = parse_fname(self.filename)

		#read in the data from the file
		#...
		#...

		value.append(spectrum) #spectrum is a list

		dictionary = {}

		for key in self.keys:
			i = 0
			dictionary[key] = values[i]
			i += 1

		row = pd.DataFrame(dictionary)
		
		#adds the row to the bottom of the dataframe
		frames = [self.dataframe, row]
		self.dataframe = pd.concat(frames)
		
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
			data for the preset column headers 
			based on the standardized file naming
			convention. 
		"""
		#gets rid of the file extension
		parameters = os.path.splitext(self.filename)[0]
		#splits into values by '_'
		parameters = parameters.split('_')
		#converts strings to floats if appropriate.
		for index in range(len(parameters)):
			if '.' in parameters[index]:
				parameters[index] = float(parameters[index])

		return parameters

	def make_dataframe(self):
		"""
		This function is only used when folder==True. 
		Moves through a folder and gets the files, 
		"""
		# name = os.path.splitext(self.filename)[0]
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


		Can simply use import make_random_spectrum.
		"""

		pass
