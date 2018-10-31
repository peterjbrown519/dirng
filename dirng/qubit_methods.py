
import numpy as np
from copy import deepcopy
from math import cos, sin, pi, exp, log
from scipy.optimize import minimize
from numpy.random import uniform, normal
import ncpol2sdpa as ncp

########################################
#		  Detection efficiencies	   #
########################################
"""
A collection of functions to help model how inefficient detectors and state Visibility
affects the rates of the protocols.

We model our system as:
	State:				cos(theta)|00> + sin(theta)|11>
	Measurements:
		Only considering 2 outcome projective measurements, given by projectors
		in the zx plane of the Bloch-sphere. Specified by the
		angle they make with the sigma_z axis, i.e. taking the form
					(cos^2(a/2) 			cos(a/2)sin(a/2))
					(cos(a/2)sin(a/2)		sin^2(a/2)		)
	No-clicks:
		We count no-click events, which happen with probability (1-eta), as the
		outcome zero.
	Visibility:
		A single parameter vis in [0,1] that induces the state transformation
		(in density matrix formalism):
					rho -> vis*rho + (1-vis)*Id/4
"""
def zx_projector(x):
	"""
	Given an angle x (with respect to the sigma_z axis), returns the corresponding
	projector in the z-x plane of the Bloch-sphere.
	I.e.
					(cos^2(x/2) 			cos(x/2)sin(x/2))
					(cos(x/2)sin(x/2)		sin^2(x/2)		)
	"""
	return np.array([[cos(x/2)**2, cos(x/2)*sin(x/2)],[cos(x/2)*sin(x/2), sin(x/2)**2]])

def measurementPair(a = None, b = None, eta=1.0):
	"""
	Given two measurement angles, one for Alice and one for Bob, construct their
	joint measurement operators [All four outcomes].
	If in addition a detection efficiency (eta) is specified,
	then apply the induced transformation.
	"""
	eye = np.identity(2)

	if a is None and b is None:
		measurements = [np.identity(4)]
	elif a is None and b is not None:
		measurements = [eta*np.kron(eye ,zx_projector(b)) + (1-eta)*np.identity(4),
						eta*np.kron(eye ,zx_projector(b+pi))]
	elif a is not None and b is None:
		measurements = [eta*np.kron(zx_projector(a), eye) + (1-eta)*np.identity(4),
						eta*np.kron(zx_projector(a+pi), eye)]
	else:
		measurements = [[eta*eta*np.kron(zx_projector(a),zx_projector(b)) +
						eta*(1-eta)*(np.kron(zx_projector(a), eye) +
						np.kron(eye, zx_projector(b))) + ((1-eta)**2)*np.identity(4),
						eta*eta*np.kron(zx_projector(a), zx_projector(b+pi)) + eta*(1-eta)*np.kron(eye, zx_projector(b+pi))],
						[eta*eta*np.kron(zx_projector(a+pi),zx_projector(b)) + eta*(1-eta)*np.kron(zx_projector(a+pi), eye),
						eta*eta*np.kron(zx_projector(a+pi),zx_projector(b+pi))]]

	return measurements

def qubitMeasurement(theta, a = None, b = None, eta = 1.0, vis = 1.0):
	"""
	Given:
		A state angle theta [cos(theta)|00> + sin(theta)|11>]
		Measurement angles a, b
		Optionally a detection efficiency eta and a visibility vis can be given.
	Returns:
		The probabilities of the various outcomes in the form
							[[p(00), p(01)]
							 [p(10), p(11)]]
	"""

	#State of the form given above and mixed with maximally mixed state with weight vis, (1-vis).
	rho = vis*np.array([[cos(theta)**2,           0.0, 0.0,   cos(theta)*sin(theta)],
						[0.0,                     0.0, 0.0,                     0.0],
						[0.0,                     0.0, 0.0,                     0.0],
						[cos(theta)*sin(theta),   0.0, 0.0,           sin(theta)**2]]) + (1-vis)*0.25*np.identity(4)

	measurements = measurementPair(a,b,eta)
	probabilities = np.trace(np.matmul(rho,measurements), axis1 = -2, axis2=-1)
	return probabilities

def cgDistribution(angles, eta = 1.0, vis = 1.0):
	"""
	Given now a collection of angles in the form
					[theta, [a1,a2,...], [b1,b2,...]]
	Returns the behaviour in cg-matrix form.
	"""
	theta = angles[0]
	a_angles = angles[1]
	b_angles = angles[2]

	# First we construct the joint measurement block of the cg-matrix
	joint_block = np.array([[qubitMeasurement(theta, a, b, eta, vis)[0][0] for b in b_angles] for a in a_angles])
	# Then we construct the top row (w/o the element [0,0])
	bob_row = np.array([qubitMeasurement(theta, None, b, eta, vis)[0] for b in b_angles])
	# Then we construct the first column beginning with element 1 for the normalisation.
	alice_col = np.array([[1]] + [[qubitMeasurement(theta, a, None, eta, vis)[0]] for a in a_angles])
	cg_mat = np.hstack((alice_col, np.vstack((bob_row, joint_block))))
	return cg_mat

def angles2Score(device, angles, eta=1.0, vis=1.0):
	"""
	given a set of angles, return the score vector.
	"""
	return device.scoreVectorFromDistribution(cgDistribution(angles, eta, vis))


def angles2DualFunctional(device, angles, eta=1.0, vis=1.0):
	"""
	Given a set of angles, compute the score vector v and return the dual functional gv.
	"""
	# Compute the dual vars
	av, lv, _, _, _ = device.dualSolution(angles2Score(device,angles, eta, vis))
	cg_shift = device._cgShift()
	def gv(w):
		return av + np.dot(lv, w - cg_shift)
	return gv

def angles2GP(device, angles, eta=1.0, vis=1.0):
	"""
	Given a set of angles, compute the GP.
	"""
	# Compute the dual vars
	av, lv, _, _v, _ = device.dualSolution(angles2Score(device, angles, eta, vis))
	return av + np.dot(lv, _v)

def optimiseQubitGP(device, starting_angles, eta=1.0, vis=1.0, tol=1e-6):
	"""
	Iteratively optimise the angle choices by trying to minimise the extracted dual functional
	"""
	a_in, b_in = device.aIn(), device.bIn()
	bounds = [[0,pi/2]] + [[-pi,pi] for k in range(a_in + b_in)]
	old_ang, new_ang = starting_angles[:], starting_angles[:]
	cg_shift = device._cgShift()


	old_gp = 2.0
	new_gp = 1.0

	# Start iteratively optimising
	while new_gp < old_gp - tol:
		old_ang = new_ang[:]
		old_gp = new_gp

		# Get the current dual solution
		av_curr, lv_curr, _, _, status = device.dualSolution(angles2Score(device, old_ang, eta, vis))

		# If this is not a bad solve then proceed with an optimisation.
		if status == 'optimal':
			# function to optimise
			def f0(x):
				ang = [x[0], x[1:a_in + 1], x[1+a_in:]]
				scr = angles2Score(device, ang, eta, vis)
				return av_curr + np.dot(lv_curr, scr - cg_shift)

			# Find minimising angles
			res = minimize(f0, ncp.flatten(old_ang), bounds=bounds)

			# record results
			new_gp = res.fun
			new_ang = [res.x[0], res.x[1:a_in+1].tolist(), res.x[a_in+1:].tolist()]
		else:
			break

	return old_gp, old_ang
