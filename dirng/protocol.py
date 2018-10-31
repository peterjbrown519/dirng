import numpy as np
from math import log, exp
from copy import deepcopy
from tabulate import tabulate


class Protocol:
	"""
	Protocol Class

	Description:
				Object representing a device-independent randomness expansion protocol.
				Stores the chosen protocol parameters and when combined with an untrusted device object,
				can be used to calculate the relevant entropy accumulation rates.

				Initialisation requires a parameters dictionary containing initial values for
				the protocol parameters. These values can be changed later.

				Example:
					parameters = {
									'number_of_rounds' 		: 		number of rounds in the protocol
									'test_probability'		: 		probability that any given round is a test round
									'eps_smooth'			:		smoothing epsilon for Hmin
									'eps_eat'				:		EAT epsilon (maximum probability with which the bound on the accumulated entropy could be wrong)
					}

	Attributes:
				n				-			number of rounds
				y				-			probability that any given round is a test round
				eps_s			-			Hmin smoothing epsilon
				eps_eat 		-			EAT epsilon
				fmin			-			Dictionary containing the dual solution corresponding to the min-tradeoff function choice.
											Key is the devices' name.

	PUBLIC Methods:
					Name 			-		completeness
					Arguments 		- 		device 		(Class: Devices)
					Description 	-		Calculates the completeness error of the protocol.
					Returns			-		completeness error

					Name 			-		setFmin
					Arguments 		- 		device	- Untrusted device object
											v 		- OVG score vector
					Description 	-		Updates the protocol's fmin dictionary. Setting the device.name entry
											to the dual solution to the GP program for the vector v.
					Returns			-		--

	"""
	def __init__(self, parameters):
		# General settings
		self.n = parameters['number_of_rounds']
		self.y = parameters['test_probability']
		self.eps_s = parameters['eps_smooth']
		self.eps_eat = parameters['eps_eat']

		# Dictionary in which the variables indicating the choice of min-tradeoff function are stored.
		# The dictionary keys are the device objects themselves
		self.fmin = {}

	def completeness(self, device):
		"""
		Calculates the completeness error of the protocol for a specified device.
		"""
		total = 0
		d = device.delta()
		w = device.score()
		for i in range(len(d)):
			total += 2*exp(-self.y*(d[i]**2)*self.n/(3 * w[i]))
		return np.min([total,1.0])

	def setFmin(self, device, v_choice = None):
		"""
		Sets parameters indicating min-tradeoff function choice
		"""
		# Check if v choice was specified
		if v_choice is None:
			v_choice = device.dualSolution()
		elif isinstance(v_choice[-1], float):
			# If only v was passed then get the rest of the solution
			v_choice = device.dualSolution(v_choice[:])

		self.fmin[device.name] = v_choice[:]

	# String produced when print is called
	def __repr__(self):
		print_string = "\n\nPROTOCOL SETTINGS:\n"
		print_string += tabulate([	["Number of rounds", 		self.n],
						  			["Test probability", 		self.y],
						  			['epsilon_s',				self.eps_s],
						  			['epsilon_eat',				self.eps_eat]
						  			],numalign='right')
		#
		# print_string += "\n\nDevices:\n"
		#
		# for name, dsol in self.devices.items():
		# 	if name[-4:] != '-COPY':
		# 		print_string += tabulate([['Name', 					name],
		# 								  ['Hmin',					device.hmin],
		# 								  ['EAT Rate (per round)',	self.eatRate(name)],
		# 								  ['Completeness error',	self.completeness(name)]
		# 								  ],numalign='right',floatfmt=(None,'.3f','.3f'))
		#
		# print_string += '\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
		return print_string
