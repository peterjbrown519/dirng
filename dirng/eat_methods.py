"""
EAT tools

We use Depuis and Fawzi's EAT with improved second order.
"""
from math import log, sqrt, ceil, exp
import numpy as np
import copy as cpy
from scipy.optimize import minimize_scalar, minimize
import logging

def f_v(protocol, device, v_choice = None, w = None):
	"""
	fv from the family of min-tradeoff functions F_min

	Optional Args:
		- v_choice = [av,lv,v,_v,status] where v is the OVG score indexing the min-tradeoff function choice
					 NOTE: You can also just pass the v vector and the dual solution will be computed. This is
					 slower though as it needs to run an sdp.
		- w = OVG score underlying the distribution for which we evaluate f_v

	If the guessing probability program is not solved adequately then the return value
	defaults to -1.0e+10
	"""

	# Relevant protocol parameters
	y = protocol.y

	# Relevant device parameters
	cg_shift = device._cgShift()

	# Check if v choice was specified
	if v_choice is None:
		# If not then first check the devices' entry in protocol dictionary.
		if device.name in protocol.fmin:
			v_choice = protocol.fmin[device.name][:]
		else:
			v_choice = device.dualSolution()
	elif isinstance(v_choice[-1], float):
		# If only v was passed then get the rest of the solution
		v_choice = device.dualSolution(v_choice[:])

	# At this point v_choice should be of the form [av,lv,v,_v,status]
	av, lv, v, _v, status = v_choice[:]

	if w is None:
		w = device.score()
	_w = w - cg_shift

	if status == 'optimal':
		Bv = 1/((av + np.dot(lv,_v))*log(2))
		Av = Bv*np.dot(lv,_v) - log(av + np.dot(lv,_v), 2)
		sol = (1-y)*(Av - Bv*np.dot(lv, _w))
	else:
		sol = -1e10

	return sol

def errV(protocol, device, v_choice = None, beta = 0.5):
	"""
	Computes the variance error term epsilon_V

	Lemma 3.3 of [X].

	Args:
		v_choice - As above
		beta - a value in (0,1)
	"""

	# Relevant protocol parameters
	y = protocol.y

	# Relevant device parameters
	AB = np.prod(device.genOutputSize())

	# Check if v choice was specified
	if v_choice is None:
		# If not then first check the devices' entry in protocol dictionary.
		if device.name in protocol.fmin:
			v_choice = protocol.fmin[device.name][:]
		else:
			v_choice = device.dualSolution()
	elif isinstance(v_choice[-1], float):
		# If only v was passed then get the rest of the solution
		v_choice = device.dualSolution(v_choice[:])
	# At this point v_choice should be of the form [av,lv,v,_v,status]
	av, lv, v, _v, status = v_choice[:]

	# Constituent quantities
	lmax = max(lv)
	lmin = min(lv)
	Bv = 1/((av + np.dot(lv,_v))*log(2))

	val = (beta*log(2)/2)*(log(2*AB**2 + 1, 2) + sqrt((1-y)**2 * Bv**2 * (lmax-lmin)**2 / y + 2))**2

	return val

def errK(protocol, device, v_choice = None, beta = 0.5):
	"""
	Computes the variance error term epsilon_K

	Lemma 3.3 of [X].

	Args:
		v_choice - as above
		beta - value in (0,1)
	"""

	# Relevant protocol parameters
	y = protocol.y

	# Relevant device parameters
	AB = np.prod(device.genOutputSize())

	# Check if v choice was specified
	if v_choice is None:
		# If not then first check the devices' entry in protocol dictionary.
		if device.name in protocol.fmin:
			v_choice = protocol.fmin[device.name][:]
		else:
			v_choice = device.dualSolution()
	elif isinstance(v_choice[-1], float):
		# If only v was passed then get the rest of the solution
		v_choice = device.dualSolution(v_choice[:])
	# At this point v_choice should be of the form [av,lv,v,_v,status]
	av, lv, v, _v, status = v_choice[:]

	# Constituent quantities
	lmax = max(lv)
	lmin = min(lv)
	Bv = 1/((av + np.dot(lv,_v))*log(2))
	expo = log(AB, 2) + (1-y) * Bv * (lmax - lmin)

	# Additive part of logarithm term is 2**expo + e**2. 2**expo = d*2**(max-min). We make some negligible overestimations to ease computations with exponents
	# We check if a > b in log(a+b) and if so we write as <= log(a) + log(1+b/a) <= log(a) + log(2)
	if log(exp(2)/AB,2) < (1-y) * Bv * (lmax - lmin):
		logarithm = expo*log(2) + log(2)
	else:
		logarithm = 2 + log(2)

	# Exponent may be causing problems - need efficient numerical method for upper bounding currently just a crude barrier style function.
	if beta <= 10/expo:
		val = (beta**2 / ( 6 * (1- beta)**3 * log(2) )) * 2**(beta * expo) * (logarithm)**3
	else:
		val = 1e10
	return val

def errW(protocol, beta=0.5):
	"""
	Computes the error term epsilon_Omega
	"""

	epeat = protocol.eps_eat
	eps = protocol.eps_s

	return (1 - 2*log(epeat*eps,2))/(beta)

def eatBetaRate(protocol, device, v_choice=None, w=None, beta=0.5):
	"""
	Computes the EAT rate for a specified beta.
	"""
	# Relevant device parameters
	if w is None:
		w = device.score()

	# Check if v choice was specified
	if v_choice is None:
		# If not then first check the devices' entry in protocol dictionary.
		if device.name in protocol.fmin:
			v_choice = protocol.fmin[device.name][:]
		else:
			v_choice = device.dualSolution()
	elif isinstance(v_choice[-1], float):
		# If only v was passed then get the rest of the solution
		v_choice = device.dualSolution(v_choice[:])
	# At this point v_choice should be of the form [av,lv,v,_v,status]
	av, lv, v, _v, status = v_choice[:]


	d = device.delta()
	d_pm = -np.multiply(np.sign(lv), d)

	n = protocol.n

	# First term
	gain = f_v(protocol, device, v_choice, w - d_pm)

	# Second term
	loss = (errV(protocol, device, v_choice, beta) +
		   errK(protocol, device, v_choice, beta) +
		   errW(protocol, beta)/n)

	return gain - loss

def eatRate(protocol, device, v_choice=None, w=None):
	"""
	Computes the EAT rate, optimising choice of beta
	"""
	def f(x):
		return -eatBetaRate(protocol, device, v_choice, w, x)

	res = minimize_scalar(f, bounds = (0,1), method = 'Bounded', options = {'xatol':1e-8})
	return -res.fun


##################################
#       Optimising f_min         #
##################################

def gradEatRate(protocol, device, v_choice = None, w = None, h = None):
	"""
	Computes numerically the gradient vector of the EAT-rate (derivative with respect to indexing score v).

	Numerical derivative calculated as (f(x+h) - f(x-h))/2h
	"""
	# Check if v choice was specified
	if v_choice is None:
		# If not then first check the devices' entry in protocol dictionary.
		if device.name in protocol.fmin:
			v_choice = protocol.fmin[device.name][:]
		else:
			v_choice = device.dualSolution()
	elif isinstance(v_choice[-1], float):
		# If only v was passed then get the rest of the solution
		v_choice = device.dualSolution(v_choice[:])
	# At this point v_choice should be of the form [av,lv,v,_v,status]
	av, lv, v, _v, status = v_choice[:]

	#Machine epsilon
	eps = 7./3 - 4./3 -1

	# Set a sensible numerical derivative parameter
	if h is None:
		h = 10*sqrt(eps)

	# Initialise a gradient vector
	grad = np.zeros(np.size(v))

	# Calculate the starting eat_rate
	base_eat_rate = eatRate(protocol, device, v_choice, w=w)


	for i in range(np.size(v)):
		v_pl = np.array(v[:])
		v_mi = np.array(v[:])

		#Perturb an element of v
		v_pl[i] = v_pl[i] + h
		v_mi[i] = v_mi[i] - h

		v_choice_pl = device.dualSolution(v_pl)
		v_choice_mi = device.dualSolution(v_mi)
		#Compute the rates for each perturbation
		rate_pl = eatRate(protocol, device, v_choice_pl, w=w)
		rate_mi = eatRate(protocol, device, v_choice_mi, w=w)
		# Now we handle various cases of bad sdp solutions
		# If a perturbation results in a bad solution then we try one sided derivatives instead.
		# However, only if they don't push us back towards the bad region.

		if v_choice_pl[-1] == 'optimal' and v_choice_mi[-1] == 'optimal':
			# First case both rates calculated fine.
			grad[i] = (rate_pl - rate_mi)/(2*h)

		elif v_choice_pl[-1] == 'optimal' and not v_choice_mi[-1] == 'optimal':
			# Second case: Negative direction resulted in bad solution.
			grad[i] = (rate_pl - base_eat_rate)/(h)
		elif not v_choice_pl[-1] == 'optimal' and v_choice_mi[-1] == 'optimal':
			# Third case: Positive direction resulted in bad solution.
			grad[i] = (base_eat_rate - rate_mi)/(h)
		else:
			# Final case: Both pertubations resulted in bad solutions.
			grad[i] = 0.0
	return grad

# #
# 	#Optimises fmin using gradient ascent algorithm
# 	#Algorithm has several parameters:
# 		#step_size - initial step size
# 		#min_step_size - minimum step size before algorithm terminates
# 		#tol - minimum gain in rate required, otherwise step_size halved
# 		#h - numerical derivative parameter
#
def entropyRateGA(protocol, device, v_choice = None, w = None, step_size = 0.01, min_step_size = 1.0e-4, tol = 1.0e-6, h=1.0e-6, verbose = 0, update=True):
	"""
	Attempts to optimise the default f_v dictated by the protocol.
	Uses gradient ascent approach to optimisation.
	"""

	if verbose > 0:
		logging.basicConfig(level = logging.INFO)

	logging.info('Starting fv optimisation...')
	step_count = 1

	#Get parameters
	y = protocol.y

	# Check if v choice was specified
	if v_choice is None:
		# If not then first check the devices' entry in protocol dictionary.
		if device.name in protocol.fmin:
			v_choice = protocol.fmin[device.name][:]
		else:
			v_choice = device.dualSolution()
	elif isinstance(v_choice[-1], float):
		# If only v was passed then get the rest of the solution
		v_choice = device.dualSolution(v_choice[:])
	# At this point v_choice should be of the form [av,lv,v,_v,status]
	current_choice = v_choice[:]

	if w is None:
		w = device.score()

	successful_steps = -1

	moved = True
	original_rate = eatRate(protocol, device, v_choice, w)
	current_rate = original_rate

	# Check if we have a bad starting point
	if current_rate < -1.0e+5:
		logging.warning('Bad initial choice - aborting optimisation.')
		return -1.0e+10

	while step_size > min_step_size:
		if moved:
			#Compute gradient at point
			grad = gradEatRate(protocol, device, current_choice, w, h)

			if any(grad != 0.0):
				grad *= 1/np.linalg.norm(grad,ord=1)	#Normalise to avoid problems with large gradient
			successful_steps += 1
			if successful_steps > 5:
				logging.info('Increasing the step size.')
				step_size *= 1.5
				successful_steps = 0
			moved = False

		#Find next point
		new_choice = device.dualSolution(current_choice[2][:] + step_size*grad)
		new_rate = eatRate(protocol, device, new_choice, w=w)

		# if verbose > 1:
		# 	print('Step {}'.format(step_count))
		# 	print('Step size {}'.format(step_size))
		# 	print('Gradient \t\t {}'.format(grad))
		# 	print('(v, v_new) \t=\t (\n{},\n{})'.format(dualVars[2][:],new_dualVars[2]))
		# 	print('Old rate \t\t {}'.format(old_rate))
		# 	print('New rate \t\t {}'.format(new_rate))
		# 	if new_rate - old_rate >=tol:
		# 		print('MOVE!')
		# 	else:
		# 		print('Step more carefully...')
		logging.info(('Step number: {:d}.\t Current rate (bits/n): {:.3e}. Rate change with previous step: {:.3e}\r').format(step_count, max([current_rate,new_rate]), new_rate-current_rate))

		step_count += 1
		if new_rate - current_rate > tol:
			# The last temp_device calculation was the new rate so the dual variables
			# should be stored in the protocols dictionary
			current_choice = new_choice[:]
			current_rate = new_rate
			moved = True
		else:
			#Try a shorter step-size
			successful_steps = 0
			step_size *= 0.5

	return current_rate, current_choice[:]

def optimiseFminChoice(protocol, device, v_choice = None, w = None,
						num_iterations = 0, jump_radius = 0.0005,
						step_size = 0.005, min_step_size = 1.0e-5,
						tol = 1.0e-5, h=1.0e-5, verbose = 0, update = True):
	"""
	Implements a basin-hopping style algorithm to try and optimise the choice of min-tradeoff function.
	Jumps in components of v vector are selected by a normal distribution with sigma = jump_radius
	"""

	if w is None:
		w = device.score()

	current_rate = eatRate(protocol, device, w)
	new_rate, new_choice = entropyRateGA(protocol, device, v_choice, w, step_size = step_size, min_step_size = min_step_size, tol = tol, h=h, update=update)

	if new_rate > current_rate or (device.name not in protocol.fmin):
		protocol.setFmin(device, new_choice)
		current_rate = new_rate

	# Basin hopping style algorithm
	for k in range(num_iterations):
		current_choice = protocol.fmin[device.name]

		# Jump in Fmin space
		v = np.array([np.random.normal(vi, jump_radius) for vi in current_choice[2]])
		new_choice = device.dualSolution(v)
		if new_choice[-1] == 'optimal':
			new_rate, new_choice = entropyRateGA(protocol, device, new_choice, w, step_size=step_size, min_step_size=min_step_size, tol=tol, h=h, verbose=verbose, update=update)
			if new_rate > current_rate:
				protocol.setFmin(device, new_choice)
				current_rate = new_rate
			if verbose > 0:
				logging.info(('OptimiseFvChoice -- Progress: {:.2f}% EAT-rate: {:.2f}-bits.').format(100*(k+1)/num_iterations,eatRate(protocol,device)), end='\r')
	return current_rate
