"""
Crazyflie 2.1 Brushless (CF2.1 BL) firmware-compatible default parameters.

All parameters are taken from the crazyflie-firmware source:
  - src/platform/interface/platform_defaults_cf21bl.h
  - src/modules/interface/stabilizer_types.h

These defaults match the real CF2.1 BL to minimize sim2real gap.
Parameters that may need real2sim calibration are documented.
"""
import math

# =============================================================================
# Physical Constants - CF2.1 Brushless
# =============================================================================

# Arm length from center to motor (m)
# NOTE: This is a physical parameter that should match your actual drone
ARM_LENGTH = 0.050  # m (CF2.1 BL uses 0.050m, standard CF2 uses 0.046m)

# Drone mass (kg)
# NOTE: This may vary with battery and attachments. Measure for sim2real.
CF_MASS = 0.0393  # kg (CF2.1 BL with guards, 350mAh battery, LH deck)

# Gravity
GRAVITY = 9.81  # m/s^2

# =============================================================================
# Motor and Thrust Parameters
# =============================================================================

# Number of motors
NUM_MOTORS = 4

# Thrust per motor (N)
# NOTE: May need calibration for sim2real. Measure thrust stand.
THRUST_MIN = 0.02136263065537499  # N per motor
THRUST_MAX = 0.2  # N per motor

# Thrust to torque coefficient (N*m / N)
# This relates propeller thrust to yaw torque
# NOTE: Critical for yaw dynamics. May need sim2real calibration.
THRUST2TORQUE = 0.00569278844371417

# =============================================================================
# Rate Constants
# =============================================================================

# Control loop rates (Hz)
RATE_MAIN_LOOP = 150  # Main stabilizer loop
ATTITUDE_RATE = 150    # Attitude controller rate
POSITION_RATE = 150    # Position controller rate

# Time steps
ATTITUDE_UPDATE_DT = 1.0 / ATTITUDE_RATE  # 2ms
POSITION_UPDATE_DT = 1.0 / POSITION_RATE  # 10ms

# =============================================================================
# Attitude Rate PID Gains (inner loop)
# =============================================================================

# Roll rate PID (tuned down for 0.01s control step)
PID_ROLL_RATE_KP = 250.0
PID_ROLL_RATE_KI = 500.0
PID_ROLL_RATE_KD = 0.05
PID_ROLL_RATE_KFF = 0.0
PID_ROLL_RATE_INTEGRATION_LIMIT = 33.3

# Pitch rate PID
PID_PITCH_RATE_KP = 250.0
PID_PITCH_RATE_KI = 500.0
PID_PITCH_RATE_KD = 0.05
PID_PITCH_RATE_KFF = 0.0
PID_PITCH_RATE_INTEGRATION_LIMIT = 33.3

# Yaw rate PID
PID_YAW_RATE_KP = 120.0
PID_YAW_RATE_KI = 16.7
PID_YAW_RATE_KD = 0.0
PID_YAW_RATE_KFF = 0.0
PID_YAW_RATE_INTEGRATION_LIMIT = 166.7

# =============================================================================
# Attitude PID Gains (outer loop)
# =============================================================================

# Roll attitude PID
PID_ROLL_KP = 15.0
PID_ROLL_KI = 3.0
PID_ROLL_KD = 0.0
PID_ROLL_KFF = 0.0
PID_ROLL_INTEGRATION_LIMIT = 20.0

# Pitch attitude PID
PID_PITCH_KP = 15.0
PID_PITCH_KI = 3.0
PID_PITCH_KD = 0.0
PID_PITCH_KFF = 0.0
PID_PITCH_INTEGRATION_LIMIT = 20.0

# Yaw attitude PID
PID_YAW_KP = 6.0
PID_YAW_KI = 1.0
PID_YAW_KD = 0.35
PID_YAW_KFF = 0.0
PID_YAW_INTEGRATION_LIMIT = 360.0

# Maximum delta for yaw setpoint (0 = disabled)
YAW_MAX_DELTA = 0.0

# =============================================================================
# Position PID Gains
# =============================================================================

# Position X PID
PID_POS_X_KP = 1.9
PID_POS_X_KI = 0.1
PID_POS_X_KD = 0.0
PID_POS_X_KFF = 0.0

# Position Y PID
PID_POS_Y_KP = 1.9
PID_POS_Y_KI = 0.1
PID_POS_Y_KD = 0.0
PID_POS_Y_KFF = 0.0

# Position Z PID
PID_POS_Z_KP = 1.6
PID_POS_Z_KI = 0.5
PID_POS_Z_KD = 0.0
PID_POS_Z_KFF = 0.0

# Position velocity limits (m/s)
PID_POS_VEL_X_MAX = 1.0
PID_POS_VEL_Y_MAX = 1.0
PID_POS_VEL_Z_MAX = 1.0

# =============================================================================
# Velocity PID Gains
# =============================================================================

# Velocity X PID
PID_VEL_X_KP = 25.0
PID_VEL_X_KI = 1.0
PID_VEL_X_KD = 0.0
PID_VEL_X_KFF = 0.0

# Velocity Y PID
PID_VEL_Y_KP = 25.0
PID_VEL_Y_KI = 1.0
PID_VEL_Y_KD = 0.0
PID_VEL_Y_KFF = 0.0

# Velocity Z PID
PID_VEL_Z_KP = 22.0
PID_VEL_Z_KI = 15.0
PID_VEL_Z_KD = 0.0
PID_VEL_Z_KFF = 0.0

# Velocity limits for attitude output
PID_VEL_ROLL_MAX = 20.0   # degrees
PID_VEL_PITCH_MAX = 20.0  # degrees

# Thrust base and minimum (in PWM scale 0-65535)
PID_VEL_THRUST_BASE = 30000.0
PID_VEL_THRUST_MIN = 20000.0

# =============================================================================
# Filter Settings
# =============================================================================

# Attitude low-pass filter
ATTITUDE_LPF_ENABLE = False
ATTITUDE_LPF_CUTOFF_FREQ = 15.0  # Hz

# Attitude rate low-pass filter
ATTITUDE_RATE_LPF_ENABLE = False
ATTITUDE_ROLL_RATE_LPF_CUTOFF_FREQ = 30.0  # Hz
ATTITUDE_PITCH_RATE_LPF_CUTOFF_FREQ = 30.0  # Hz
ATTITUDE_YAW_RATE_LPF_CUTOFF_FREQ = 30.0  # Hz

# Position/velocity low-pass filter
PID_POS_XY_FILT_ENABLE = True
PID_POS_XY_FILT_CUTOFF = 20.0  # Hz
PID_VEL_XY_FILT_ENABLE = True
PID_VEL_XY_FILT_CUTOFF = 20.0  # Hz
PID_POS_Z_FILT_ENABLE = True
PID_POS_Z_FILT_CUTOFF = 20.0  # Hz
PID_VEL_Z_FILT_ENABLE = True
PID_VEL_Z_FILT_CUTOFF = 20.0  # Hz

# =============================================================================
# Motor Configuration (X-configuration)
# =============================================================================

# Motor layout (looking from above, Crazyflie body ENU: +X forward, +Y left, +Z up):
#
#          Front (+X)
#               ^
#               |
#   M4 (CW)     |     M1 (CCW)
#        \      |      /
#         \     |     /
#          \    |    /
#           \   |   /
#            \  |  /
#             \ | /
#       +Y <--- O ---> -Y (right)
#             / | \
#            /  |  \
#           /   |   \
#          /    |    \
#   M3 (CCW)    |     M2 (CW)
#               |
#            Rear (-X)
#
# Motor positions and spin (matching crazyflie-firmware):
#   M1: front-right (x>0, y<0), CCW  -> yaw torque negative
#   M2: rear-right  (x<0, y<0), CW   -> yaw torque positive
#   M3: rear-left   (x<0, y>0), CCW  -> yaw torque negative
#   M4: front-left  (x>0, y>0), CW   -> yaw torque positive
#
# Firmware legacy mixing (power_distribution_quadrotor.c, lines 89-92):
#   M1 = thrust - roll/2 + pitch/2 + yaw
#   M2 = thrust - roll/2 - pitch/2 - yaw
#   M3 = thrust + roll/2 - pitch/2 + yaw
#   M4 = thrust + roll/2 + pitch/2 - yaw
#
# Roll: positive roll (clockwise looking forward) -> right side down -> M1,M2 decrease, M3,M4 increase
# Pitch: positive pitch (clockwise looking left)  -> nose up -> back motors increase, front motors decrease
# Yaw: positive yaw command (matches firmware) -> M1,M3 increase (CCW), M2,M4 decrease (CW)

MOTOR_MIX_ROLL = [-1, -1, 1, 1]   # M1, M2, M3, M4
MOTOR_MIX_PITCH = [-1, 1, 1, -1]
MOTOR_MIX_YAW = [1, -1, 1, -1]

# =============================================================================
# Derived Constants
# =============================================================================

# Arm component for X-configuration (45 degree offset)
ARM_X = ARM_LENGTH * math.sqrt(0.5)  # 0.707 * ARM_LENGTH

# Total maximum thrust (N)
TOTAL_THRUST_MAX = NUM_MOTORS * THRUST_MAX

# Hover thrust per motor (N) - approximate for mass
HOVER_THRUST_PER_MOTOR = CF_MASS * GRAVITY / NUM_MOTORS

# =============================================================================
# Sim2Real Parameters
# NOTE: These parameters may need calibration between sim and real.
# =============================================================================

# Motor response time constant (seconds)
# NOTE: Affects motor dynamics. May need identification.
MOTOR_TIME_CONSTANT = 0.01  # 10ms typical for brushless

# Inertia tensor diagonal [Ixx, Iyy, Izz] (kg*m^2)
# NOTE: Critical for attitude dynamics. Measure or estimate for sim2real.
# These are typical values for CF2.1 BL - may need calibration
INERTIA_XX = 1.5e-5  # kg*m^2
INERTIA_YY = 1.5e-5  # kg*m^2
INERTIA_ZZ = 3.0e-5  # kg*m^2

# Drag coefficient (N/(m/s)^2)
# NOTE: Affects velocity response. Usually small for CF, but matters for sim.
DRAG_COEFFICIENT = 0.0

# Rotor drag coefficient (N*m/(rad/s))
# NOTE: Affects yaw damping. May need sim2real tuning.
ROTOR_DRAG_COEFFICIENT = 0.0
