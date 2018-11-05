"""
Script demonstrating the construction of an EB22 protocol and use of the qubit methods.
"""
import dirng as di

# Path to a sdp solver (only sdpa family solvers currently supported)
SOLVER = '/scratch/pjb519/sdpa-7.3.8/'


"""
STEP 1.
Create devices
"""
# Device settings
io_config = [[2,2], [2,2]]
device_settings = 	{'name' : 'eb22',
					'io_config' : io_config,
					'relaxation_level' : 2,
					'generation_inputs' : [0,0],
					'solver' : SOLVER}

# Initialise the device object
dev = di.Devices(device_settings)



"""
STEP 2.
Specify the games played. For EB protocols there is a function
that can be used to generate these.
"""
eb_games = di.EBGames(io_config = io_config, distribution = None, delta = 0.0001)
dev.games  = eb_games

# Note that we have not yet specified a distribution. This is fine, our devices
# will just produce trivial rates.
print(dev)



"""
With games as complex as the EB family, it may be difficult to know a priori what
is a good expected score vector. However, by specifying a state and measurements
we can compute an expected score vector and set the devices' scores accordingly.

We consider the single parameter family of states
 	- psi_theta = cos(theta)|00> + sin(theta)|11>

And local projective measurements in the x-z plane of the Bloch sphere
 	P(phi) = | cos^2(phi/2) 			cos(phi/2)sin(phi/2) |
			 | cos(phi/2)sin(phi/2)		sin^2(phi/2)		 |
angle phi is w.r.t. sigma(z). We only need to specify the angle for outcome 0.

Noise such as inefficient detectors (no-clicks mapped to outcome 0), and
a visibility parameter can be passed to the function if required.
"""
from math import pi
# As an example we choose the state and measurements which achieve T'sirelson's bound
# for chsh.
# The state angle
theta = pi/4
# Measurement angles
A_angles = [0.0, pi/2]
B_angles = [pi/4, -pi/4]

system_angles = [theta, A_angles, B_angles]

# Now we create the score vector (we can change eta (detection efficiency) or vis (visibility))
w = di.angles2Score(dev, system_angles, eta = 1.0, vis = 1.0)
# We can now set this score vector
dev.score = w
# Let's see how this has affected the devices.
print('\nDevices after setting the score...\n', dev)





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





"""
STEP 4.
Calculating the randomness accumulation rates.
"""
initial_rate = protocol.eatRate(dev)
print('Starting eat rate is {}'.format(initial_rate))

# Let's optimise the choice of min-tradeoff function
optimised_rate = protocol.optimiseFminChoice(dev)
print('Optimised eat rate is {}'.format(optimised_rate))
