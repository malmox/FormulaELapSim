import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import csv
import dynamicsFunctions as dynF

#################################################################################
# USER-INPUT CONSTANTS
# TRACK FILE
TRACK = None

# CAR CONSTANTS
mass = None                     # kg - CAR MASS
Af = None                       # m^2 - FRONTAL AREA
mu_rr = None                    # COEFFICIENT OF ROLLING RESISTANCE

# BATTERY CONSTANTS
# DEPENDS ON STARTING CONDITIONS
initial_SoC = None              # % - INITIAL STATE OF CHARGE
starting_voltage = None         # V - INITIAL PACK VOLTAGE
capacity0 = None                # Ah - INITIAL PACK CAPACITY
batteryTemp0 = None             # C - starting temperature of battery pack
# DEPENDS ON BATTERY CHOICE
max_capacity = None             # Ah
n_converter = None              # converter efficiency
cell_max_voltage = None         # V - MAX CELL VOLTAGE
cell_min_voltage = None         # V - MIN CELL VOLTAGE
cell_nominal_voltage = None
num_series_cells = None         # NUMBER OF SERIES ELEMENTS
num_parallel_cells = None       # NUMBER OF PARALLEL CELLS
single_cell_ir = None           # Ohms - CELL INTERNAL RESISTANCE
max_CRate = None                # Max C-Rate
cell_mass = None                # Mass of single cell
battery_cv = None               # Specific heat capacity of battery

# MOTOR CONSTANTS
max_motor_torque = None         # Nm - MAX MOTOR TORQUE
max_motor_rpm = None            # rpm - MAX MOTOR RPM
max_power = None                # W - MAX POWER - AS DEFINED BY USER (FOR POWER LIMITING)

# TRACTION CONSTANTS
max_speed_kmh = None            # km/h - MAX SPEED
traction_speed = None           # km/h - MAX SPEED AROUND RADIUS IN TRACTION TEST
traction_radius = None          # m - RADIUS OF TRACTION TEST

######################################################################
# This allows an external user to have control over the constants without touching the code

# Open .csv and take constants as input:
filename = "LapSimConstants.csv"

# Open the file
with open(filename, 'r', newline='') as infile:
    reader = csv.reader(infile)
    dataList = list(reader)
    dataList.pop(0)             # Remove title row

    # convert to array
    dataArray = np.array(dataList)

    # Take out the valuable columns and convert to floats as necessary
    value_name = dataArray[1:,0]
    value = dataArray[1:,2]
    value = np.asarray(value, dtype = float)

    track_name = dataArray[0,0]
    track = dataArray[0,2]

# Now create variables for everything
for x, y in zip(value_name, value):
    globals()[x] = y

globals()[track_name] = track

################################################################
# CONSTANTS - SHOULD NOT NEED CHANGING
totalTime = 60 * 30         # total time: s
dt = 0.1                # time interval: s
num = int(totalTime / dt)   # number of intervals
g = 9.81                # m/s^2
GR = 4.2                # Gear Ratio
wheel_diameter = 18 * 0.0254    # m
wheel_radius = wheel_diameter / 2
rho_air = 1.23         # air density: kg / m^3
v_air = 0               # air velocity: m/s
radsToRpm = 1 / (2 * math.pi) * 60    # rad/s --> rpm
Cd = 1                  # drag coefficient

# Efficiencies
n_motor_arr = np.array([[0,0.86],
                        [2000,0.9],
                        [1666,0.92],
                        [2160,0.94],
                        [2600,0.95],
                        [3400,0.96],
                        [5100,0.95],
                        [5333,0.94],
                        [5800,0.92],
                        [6200,0.9],
                        [6550,0.86],
                        [7600,0.86]])

# Battery Pack - Calculated Values
num_cells = num_series_cells * num_parallel_cells
total_pack_ir = single_cell_ir / num_parallel_cells * num_series_cells  # ohms
knownTotalEnergy = capacity0 * starting_voltage / 1000  # kWh
pack_min_voltage = cell_min_voltage * num_series_cells  # V
pack_nominal_voltage = cell_nominal_voltage * num_series_cells # V

# Motor
# Motor Power Loss Constants - from excel fit of datasheet graph
A = 0.00002
B = 0.0402
C = 44.083

# Traction Constants
# at 30 km/h, we travelled around a 5 m radius circle
a_centrip = (traction_speed * 1000 / 3600)**2 / traction_radius      # v^2 / r (convert to m/s)
test_mass = 225                             # kg - car mass used in testing
F_friction = test_mass * a_centrip          # calculate the friction force
mu_f = F_friction / (test_mass * g)         # calculate the friction coefficient
max_speed = max_speed_kmh / 3.6             # m/s
# This determines our friction force based on the evaluated car mass.
F_friction = mu_f * mass * g

####################################################################################
# CODE

# Take in a .csv file for the "Track"
infile = TRACK
track_df = pd.read_csv(infile, header=[0])
trackLength = track_df.loc[:,'Cumulative Length'].to_numpy()        # create vector for track length
trackRadius = track_df.loc[:,'Radius'].to_numpy()                   # create vector for track radius

# Create the motor torque curve
motor_torque, motor_rpm = dynF.motorTorqueCurve()

# Vectors for each set of data
# Create a dictionary of values
headers = ['v0',                    # velocity vector (m/s)
           'r0',                    # distance vector (m)
           't0',                    # time vector (s)
           'w_wh',                  # wheel angular velocity (rad/s)
           "w_m",                   # motor angular velocity (rpm)
           "T_m",                   # motor torque (Nm)
           "T_a",                   # Axel torque (Nm)
           "F_trac",                # traction force (N)
           "F_drag",                # drag force (N)
           "F_RR",                  # rolling resistance (N)
           "F_net_tan",             # net force in the tangential direction (N)
           "a_tan0",                # tangential acceleration (m/s^2)
           "a_norm0",               # normal/centripetal acceleration (m/s^2)
           "P_battery",             # battery power (kW)
           "Capacity",              # Battery capacity (Ah)
           "Pack Voltage",          # battery pack voltage (V)
           "Pack Current",          # Battery pack current (A)
           "Energy Use",            # energy use over time (kWh)
           "SoC Energy",            # state of charge - energy based (%)
           "SoC Capacity",          # state of charge - capacity based (%)
           "Dissipated Power",      # Power dissipated from batteries due to internal resistance (W)
           "Battery Temp",          # Temperature of battery pack (C)
           "Max Values"]            # maximum battery power used, total energy used
dataDict = dict.fromkeys(headers)
dataDict['t0'] = np.zeros((1))      # Initialize the time vector separately for reasons that will become apparent later in the code

# add empty zero vectors...
for i in range(0, len(headers)):
    if headers[i] != "t0":
        dataDict[headers[i]] = np.zeros(num)

# Add some starting values
dataDict['Capacity'][0] = capacity0
dataDict['SoC Capacity'][0] = initial_SoC
dataDict['Pack Voltage'][0] = starting_voltage
dataDict['Battery Temp'][0] = batteryTemp0

# CALCULATIONS
for i in range(0,num-1):
    # INITIAL CALCULATIONS

    # Calculates the fastest possible next speed and furthest possible next position
    dataDict = dynF.fastestNextSpeed(dataDict, i)

    #########################################  
    # TRACTION CALCULTIONS

    # Determine track location and current radius:
    trackLocation = np.searchsorted(trackLength, dataDict['r0'][i])    # index of track location

    # break out of loop once we hit the end of the track
    if trackLocation > len(trackRadius) - 1:              
        break
   
    # determine instantaneous track radius
    current_radius = trackRadius[trackLocation]

    # Now, check to find the maximum actually possible speed based on traction considerations
    v_max = dynF.findMaxSpeed(current_radius, dataDict, i)

    # Now determine whether we exceed that speed, and if so, recalculate the possible values
    if dataDict['v0'][i+1] > v_max:
        dataDict = dynF.limit_max_speed(dataDict, v_max, i)

    ###########################################

    # Determine battery power used during the race
    if dataDict['F_trac'][i] > 0:
        dataDict = dynF.batteryPower(dataDict, i)
    else:
        dataDict['P_battery'][i] = 0

    ###########################################
    # Add the further battery calcs here
    dataDict = dynF.batteryPackCalcs(dataDict, i)

    ###########################################

    # Increase time vector
    next_t = np.array([dataDict['t0'][i] + dt])
    dataDict['t0'] = np.append(dataDict['t0'], next_t)

# Trapezoidal approximation for energy used
energy, totalEnergy = dynF.trapezoidApprox(dataDict['P_battery'])
dataDict['Energy Use'] = energy

# dataDict['Max Values'][0] = "Total Energy Used (kWh)"
dataDict['Max Values'][1] = totalEnergy
print('Energy Used: ', totalEnergy, 'kWh')

# Determine SoCe based on this!
dataDict = dynF.SoCenergy(dataDict, knownTotalEnergy)

# Determination of maximum power
dataDict['P_battery'] = dataDict['P_battery'] / 1000        # convert to kW
maxPower = max(dataDict['P_battery'])

# dataDict['Max Values'][2] = "Max power Used (kW)"
dataDict['Max Values'][3] = maxPower
print("Max Power: ", maxPower, "kW")

# Now cut all arrays down to the correct size before inputting into a dataframe
time_size = len(dataDict['t0'])
for i in range(0, len(headers)):
    dataDict[headers[i]] = dataDict[headers[i]][:time_size]

# Now I want to write all the columns to a dictionary and then input it into a dataframe - since it's easier to do column-wise
dfData = pd.DataFrame(dataDict)
dfData.dropna(inplace = True)

# Drop extra columns based on the size of the time vector
outfile = "dynamicsCalcs.csv"
dfData.to_csv(outfile, index=False)

# Create plots
dynF.plotData(dataDict)

print("Completed")
