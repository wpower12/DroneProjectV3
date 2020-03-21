import numpy as np
from . import constants as C

class Wind():
	def __init__(self, rnd_state):
		# Lets us ensure we have the same state between invocations
		self.prng = rnd_state

		self.gusting = False
		self.gust_length = 0 # How many ticks (state-advances) a gust lasts
		self.gust_timer  = 0 # How long a gust has been going
		self.set_non_gust_timer()
		
		# Storing these for debugging, but not strictly needed?
		self.gust_angle  = 0 # Which Direction in x,y plane
		self.gust_angle_zenith = 0 # Which direction in z x (xy) plane
		self.gust_mag    = 0 # How hard

		# The above 2 will resolve into a single 3 vector
		self.gust_vect = None

	def set_seed(self, n):
		np.random.seed = n

	def sample_wind(self):
		self.advance_state()
		return self.get_wind_vec()

	def get_wind_vec(self):
		if self.gusting:
			return self.gust_vect
		else: # Not gusting
			return np.asarray([0, 0, 0])

	def advance_state(self):
		if self.gusting:
			self.gust_timer += 1
			if self.gust_timer > self.gust_length:
				self.gusting = False
				self.gust_timer = 0
				self.set_non_gust_timer()
		else:  # Not Gusting
			self.non_gust_timer -= 1
			if self.non_gust_timer <= 0:
				self.gusting = True
				self.gust_length = self.sample_length()
				self.gust_angle = self.sample_angle()
				self.gust_angle_zenith = self.sample_angle_zenith()
				self.gust_mag = self.sample_mag()
				self.gust_vect = self.resolve_vector()
				self.gust_timer = 1

	def set_non_gust_timer(self):
		self.non_gust_timer = int(self.prng.normal(C.NON_GUST_MEAN, C.NON_GUST_STD))

	def sample_length(self):
		# TODO - Check if this is an ok way to get a 'discrete' normal
		return int(self.prng.normal(C.LENGTH_MEAN, C.LENGTH_VAR))

	def sample_angle(self):
		return self.prng.normal(C.ANGLE_MEAN, C.ANGLE_VAR)

	def sample_angle_zenith(self):
		return self.prng.normal(C.ANGLE_ZENITH_MEAN, C.ANGLE_ZENITH_VAR)

	def sample_mag(self):
		return self.prng.normal(C.MAG_MEAN, C.MAG_VAR)

	def resolve_vector(self):
		x = self.gust_mag * np.cos(self.gust_angle_zenith) * np.cos(self.gust_angle)
		y = self.gust_mag * np.cos(self.gust_angle_zenith) * np.sin(self.gust_angle)
		z = self.gust_mag * np.sin(self.gust_angle_zenith)
		return np.asarray([x, y, z])
