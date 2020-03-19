# Size of time step
DT = 0.05 #0.05

# Drone
DRONE_MASS = 1.0
MAX_VEL    = 3.0
MAX_ACC    = 1.0
MAX_JERK   = 1.0 # Max value that can be applied to acc in a dt

# Swarm
SEPARATION = 0.5 # Similarity matrix sparsification threshold

# Constants for the PID controller
PID_P = 30.0
PID_I = 0.05
PID_D = 125.0

# Parameters for Wind Distribution
LENGTH_MEAN = 6
LENGTH_VAR  = 0.8
ANGLE_MEAN  = -0.5236 # Should be in radians.
ANGLE_VAR   = 0.1309
ANGLE_ZENITH_MEAN = -0.5236 # 1.0472 # In radians # -0.5236
ANGLE_ZENITH_VAR = 0.1309
MAG_MEAN    = 2 #2 #5.0
MAG_VAR     = .3 # 1

NON_GUST_MEAN = 6
NON_GUST_STD = 0.8



# Expansion Procedure Parameters
TEST_VAR_RADIUS = 2.0
TARGET_EPSILON  = 0.1
EXP_OFF       = 0
EXP_HOVER     = 1
EXP_EXPANDING = 2

# Model Parameters
NUM_REGRESSORS = 1
WINDOW_SIZE = 10 # How large each 'temporal window' is