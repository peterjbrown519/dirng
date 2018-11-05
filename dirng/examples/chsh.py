"""
Script demonstrating the construction of the G_CHSH protocol
"""
import dirng as di

# Path to a sdp solver (only sdpa family solvers currently supported)
SOLVER = '/scratch/pjb519/sdpa-7.3.8/'

"""
STEP 1.
Creating the devices requires us to specify some of its attributes.

Of particular importance are:
	- io_config 		--  	The input/output configuration. This is a list of the form
	 				[[|A_1|, ..., |A_m|], [|B_1|, ..., |B_n|]]
					where |A_i| is the number of outputs for the i-th measurment
					on device A. For G_CHSH the io_config is [[2,2], [2,2,2]].
	- relaxation_level 	--  	Level of the NPA hierarchy to relax to.
	- generation_inputs 	--	Inputs to be used on generation rounds.
	- sdpa_path		--	Path to the sdpa solver.
"""
# Device settings
io_config = [[2,2], [2,2,2]]
device_settings = 	{'name' : 'chsh',
			'io_config' : io_config,
			'relaxation_level' : 2,
			'generation_inputs' : [1,2],
			'solver' : SOLVER}

# Initialise the device object
dev = di.Devices(device_settings)
"""
STEP 2.
Now we need to specify what games the devices will play. To do this we must indicate
the coefficients within the different Bell-expressions.

We write the coefficients of the probabilities in a table of the form
					|	s00|00  s01|00  s00|01  s01|01	s00|02  s01|02  |
			expression =	|	s10|00  s11|00  s10|01  s11|01	s10|02  s11|02	|
					|	s00|10  s01|10  s00|11  s01|11	s00|12  s01|12	|
					|	s10|10  s11|10  s10|11  s11|11	s10|12  s11|12	|
"""
chsh_coefficients = 		[[ 0.25, 0.00, 0.25, 0.00, 0.00, 0.00],
				[ 0.00, 0.25, 0.00, 0.25, 0.00, 0.00],
				[ 0.25, 0.00, 0.00, 0.25, 0.00, 0.00],
				[ 0.00, 0.25, 0.25, 0.00, 0.00, 0.00]]
# When (X,Y) = (0,2) we should have A=B
alignment_coefficients = 	[[ 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
				[ 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
				[ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
				[ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]]

"""
Now we create the game objects that we will supply to the device.
"""
chsh = di.Game(name = 'chsh', matrix = chsh_coefficients, score = 0.845, delta=0.001)
align = di.Game(name = 'align', matrix = alignment_coefficients, score = 0.98, delta=0.001)

# Adding the games
dev.games = [chsh, align]

# Let's have a look at the device.
print(dev)

"""
STEP 3.
Create the protocol.
As before we specify the settings in a dictionary and then initialise the protocol object.
"""
# Protocol setup
protocol_settings =	{'n' 				: 1e10,
					 'y'				: 5e-3,
					 'eps_smooth'		: 1e-8,
					 'eps_eat'			: 1e-8}
protocol = di.Protocol(protocol_settings)
print(protocol)
print('The completeness error is: ', protocol.completeness(dev))

"""
STEP 4.
Calculating the randomness accumulation rates.
"""
initial_rate = protocol.eatRate(dev)
print('Starting eat rate is {}'.format(initial_rate))

# Let's optimise the choice of min-tradeoff function
optimised_rate = protocol.optimiseFminChoice(dev)
print('Optimised eat rate is {}'.format(optimised_rate))

# Note that the protocol object has now stored the optimised choice of fmin for this device.
# Any subsequent calls of eatRate() will use this improved choice and not the default which is the score vector.
# We can also try a random choice of min-tradeoff function (although this is unlikely to yield anything good)
import numpy as np
# This is the score vector saved within the device
w = dev.score
# Let's pick a random point nearby and see if that works as a good min-tradeoff function index.
v = np.random.normal(w, scale = 0.002)
print('Randomly chosen f_v: v=', v)
random_rate = protocol.eatRate(dev, v_choice = v)
print('Random eat rate is {}'.format(random_rate))
