import pandas as pd
# from annsa.template_sampling import make_random_spectrum
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

class SpectrumData(object):
	"""
	SpectrumData is a data object that contains all the data
	that needs to be analyzed by using annsa
	"""
	def __init__(self,
				folder=None,
				ae=False,
				csv=False,
				output_name=None,
				online=False):
		"""
		Generates a data object of type SpectrumData.

		Parameters:
		-----------
		folder : string
			This working directory with all of your data.
			If no directory is given, data must be added one
			file at a time.
			Default is None.
		ae : boolean
			This says whether the data should be processed
			for autoencoders or not.
			Default is False.
		csv : boolean
			If you want to save the data as a single csv file.
			Default is False.
		output_name : string
			The name of the file and path you want for your
			data output.
		online : boolean, optional
			Whether or not the data set should be sampled by
			using online_data_augmentation or not.
		"""
		super(SpectrumData, self).__init__()

		self.data = {'Isotope':[],'Distance':[],'Height':[],'Shielding':[],
		'Area Density':[],'FWHM':[],'Spectra':[]}
		# self.labels = self.dataset[0]
		# self.spectra = self.dataset[1]
		self.folder = folder
		self.df=None
		self.dataset = [[],[]] # training labels and training data
		self.dataframe = pd.DataFrame()

#==========================This needs to be a function==========================
		if folder != None:
			# Opens the folder
			directory = os.fsencode(folder)
			# Iterates through the files in the directory
			for file in os.listdir(directory):
				# Gets the file name itself, the path is appended in each func
				filename = os.fsdecode(file)
				# If the file is a .spe file with spectrum data, then ...
				if filename.endswith(".spe"):
					# Grabs the isotope parameter
					isotope, _  = self.parse_fname(filename)
					# Pulls the spectrum out of the file and into a numpy array.
					spectrum = self.get_spectrum(filename)
					# Dataset implementation...
					self.dataset[0].append(isotope)
					self.dataset[1].append(spectrum)

			labels = self.dataset[0]
			spectra = self.dataset[1]
			data_dict = {'Isotopes' : labels}
			for spectrum in spectra:
				for channel, count in enumerate(spectrum):
					if str(channel) not in data_dict.keys():
						data_dict[str(channel)] = []
					data_dict[str(channel)].append(count)

			# Pandas DataFrame implementation...
			self.dataframe = pd.DataFrame(data_dict)

			if csv:
				self.dataframe.to_csv(output_name)
				print(self.df)
#===============================================================================

	def add_column(self, key, values):
		"""
		Adds a column of data to the spectrum dataframe.

		Parameters:
		-----------
		key : string
			The name of the data you want to add
		values : any
			The values that go with the new key. Can be
			single value or list of values.
		"""

		self.data[key] = [values]
		pass

	def add_data(self, fname):
		"""
		Adds a row of data to the spectrum dataframe.
		This is useful if you want to add more data to
		an existing dataframe.
		"""
		# if self.folder != None:
		# 	fname = self.folder+"/"+fname
		# # Read in the data from the file
		# spectrum = self.get_spectrum(fname)
		# # Gets the keys from the file name
		# values = self.parse_fname(fname)
		# values.append(spectrum)
		# # Adds values to the existing data
		# for key,value in zip(list(self.data.keys()), values):
		# 	self.data[key].append(value)

		pass



	def plot_spectrum(self, index):
		"""
		Plots the spectrum of an isotope given a particular
		index (row number).

		Parameters:
		-----------
		index : int
			The row number of the spectrum you want to plot.
		"""

		row = self.df.iloc[index, :]
		label = self.df['Isotopes'][index]
		counts = self.df['Spectra'][index]
		# label = self.data['Isotope'][index]
		# counts = self.data['Spectra'][index]
		channels = np.arange(0, len(counts), 1)
		plt.plot(channels, counts, label=label)
		plt.show()
		pass

	def get_spectrum(self, filename):
		"""
		Retrieves the spectrum data from a file.

		Parameters:
		-----------
		filename : string
			The path to the file.

		Returns:
		--------
		spectrum : numpy array, dtype=float
			The raw counts of the spectrum.
		"""
		if self.folder != None:
			filename = self.folder+"/"+filename

		spectrum = []
		file = open(filename)
		for line in file:
			spectrum.append(line.rstrip())
		file.close()

		spectrum = spectrum[9:1033]
		spectrum = [float(count) for count in spectrum]
		spectrum = np.array(spectrum)

		return spectrum


	def parse_fname(self, filename):
		"""
		Separates the name of a single file into key
		identifiers based on the standard annsa naming
		convention.

		Parameters:
		-----------
		filename : string
			Name of file containing raw spectrum data.

		Returns:
		--------
		parameters : list of strings
			This list of strings will become the
			data for the preset column headers
			based on the standardized file naming
			convention.
		"""
		if self.folder != None:
			filename = self.folder+"/"+filename
		# gets rid of the file extension
		parameters = os.path.splitext(filename)[0]

		# isolates the file name from the path
		parameters = parameters.split('/')[-1]

		# splits into values by '_'
		parameters = parameters.split('_')

		# converts strings to floats if appropriate.
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

	def dict_to_csv(self, output_name):
		"""
		Writes the dictionary to a csv file.
		"""
		if not output_name.endswith(".csv"):
			output_name = output_name + ".csv"
		with open(output_name, 'w') as file:
			for key in self.data.keys():
				file.write("%s,%s\n"%(key, self.data[key]))

		pass


	# def sample_dataframe(self, **kwargs):
	# 	"""
	# 	Returns a single row of data from the dataframe
	# 	by creating a subset of the dataframe based on
	# 	keyword arguments and then randomly selecting
	# 	a row from that subset.
	# 	"""
	#
	# 	#pd.dataframe.sample picks a random row
	# 	pass
	#
	# def sample_spectrum(self):
	#
	# 	"""
	# 	1. rebinning --> This allows you to simulate detector
	# 	calibrations like the gain of a detector which may
	# 	change based on pmt_voltage or ambient temperature.
	# 	This function already exists from Mark.
	# 	2. lower level discriminator (LLD) --> typically the
	# 	first few channels are simply noise and thus should
	# 	be set to zero. ~10 channels.
	# 	3. normalize the function by area --> divides the
	# 	value of each channel by the sum of the counts.
	# 	4. scale by the number of counts. This allows you to
	# 	simulate a dataset with a different number of counts
	# 	than the dataset you are sampling from.
	#
	#
	# 	Can simply use import make_random_spectrum.
	# 	"""
	#
	# 	pass

#========================Non-member functions===================
def plot_spectrum(spectrum, title):
	"""
	Plots the spectrum of an isotope given a particular
	index (row number).

	Parameters:
	-----------
	index : int
		The row number of the spectrum you want to plot.
	"""
	print("inside the plot function")
	channels = np.arange(0, len(spectrum), 1)
	plt.plot(channels, spectrum)
	plt.title(title)
	plt.show()
	pass

def get_spectrum(filename):
	"""
	Retrieves the spectrum data from a file.

	Parameters:
	-----------
	filename : string
		The path to the file.

	Returns:
	--------
	spectrum : numpy array, dtype=float
		The raw counts of the spectrum.
	"""
	spectrum = []
	file = open(filename)
	for line in file:
		spectrum.append(line.rstrip())
	file.close()

	spectrum = spectrum[9:1033]
	spectrum = [float(count) for count in spectrum]
	spectrum = np.array(spectrum)

	return spectrum

if __name__ == '__main__':

	spectrum_folder = "/home/samgdotson/Research/annsa/annsa/spectra_templates/shielded-templates-200keV"
	test_folder = "/home/samgdotson/Research/annsa/annsa/test_data"
	single_spectrum = "/home/samgdotson/Research/annsa/annsa/spectra_templates/shielded-templates-200keV/99MTc_75.0_50.0_none_0_9.0.spe"
	second_spectrum = "/home/samgdotson/Research/annsa/annsa/spectra_templates/shielded-templates-200keV/235U_100.0_175.0_none_0_7.0.spe"
	# data = SpectrumData()
	# data.add_data(single_spectrum)
	# data.add_data(second_spectrum)
	# print(data.data)


	# test_data = SpectrumData(folder=test_folder, csv=True,
	# output_name = "/home/samgdotson/Research/data_test.csv")
	#
	# test_data.plot_spectrum(2)

	# all_data = SpectrumData(folder=spectrum_folder, csv=True,
	# output_name = "/home/samgdotson/Research/all_data.csv")

	background_spectrum = "/home/samgdotson/Research/spectra_templates/background-templates/albuquerque_6.0_0.spe"

	mtc099 = get_spectrum(second_spectrum)
	newmex = get_spectrum(background_spectrum)
	combined = mtc099 + newmex

	t1 = "background only"
	t2 = "spectrum only"
	t3 = "spectrum + background"

	plot_spectrum(newmex, t1)
	plot_spectrum(mtc099, t2)
	plot_spectrum(combined, t3)








	pass
