"""
We extend the eb22.py example and look at optimising the qubit system modelling the
devices.
"""
import dirng as di
from math import pi, log

SOLVER = '/scratch/pjb519/sdpa-7.3.8/'


# Quickly setting everything up again
io_config = [[2,2], [2,2]]
device_settings = 	{'name' : 'eb22',
					'io_config' : io_config,
					'relaxation_level' : 2,
					'generation_inputs' : [0,0],
					'solver' : SOLVER}

# Initializing the device
dev = di.Devices(device_settings)

# Adding the EB games
dev.games = di.EBGames(io_config = io_config, distribution = None, delta = 0.001)

# Choosing initial qubit system angles (as before).
system_angles = [pi/4, [0.0, pi/2], [pi/4, -pi/4]]
# Let's modify the noise level this time
eta = 0.9
vis = 0.99
dev.score = di.angles2Score(dev, system_angles, eta = eta, vis = vis)

# Let's see how this has affected the devices.
print('Before optimisation the devices\' min-entropy is {:.5f} bits'.format(dev.hmin))
print('Using system angles:')
print('theta = {:.5f}'.format(system_angles[0]))
print('A0 = {:.5f}'.format(system_angles[1][0]))
print('A1 = {:.5f}'.format(system_angles[1][1]))
print('B0 = {:.5f}'.format(system_angles[2][0]))
print('B1 = {:.5f}'.format(system_angles[2][1]), end = '\n\n')





"""
Given the detection efficiency, visibility (and the fact that we are not playing chsh),
there may be a better choice of state and measurement angles that we could use.

Using the method:
 					optimiseQubitGP()
We can look for better angles for our setup.
"""

new_gp, new_angles = di.optimiseQubitGP(dev, system_angles, eta = eta, vis = vis)
print('After optimising system angles, the devices\' min-entropy is now {:.5f} bits'.format(-log(new_gp,2)))
print('theta = {:.5f}'.format(new_angles[0]))
print('A0 = {:.5f}'.format(new_angles[1][0]))
print('A1 = {:.5f}'.format(new_angles[1][1]))
print('B0 = {:.5f}'.format(new_angles[2][0]))
print('B1 = {:.5f}'.format(new_angles[2][1]), end = '\n')
