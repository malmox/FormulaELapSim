import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import csv
import dynamicsFunctions as dynF
#################################################################################
# Mallory Moxham - UBC Formula Electric - September 2023
#################################################################################
# USER-INPUT CONSTANTS
# TRACK FILE
TRACK = None
regen_on = None                 # Regen on - True or False
numLaps = None

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
# !!!
cell_air_area = None            # m^2 - AIR COOLING SURFACE OF CELL
cell_water_area = None          # m^2 - WATER COOLING SURFACE OF CELL
cell_aux_factor = None          # kg/kWh - SEGMENT AUXILLARY MASS/ENERGY

# MOTOR CONSTANTS
max_motor_torque = None         # Nm - MAX MOTOR TORQUE
max_motor_rpm = None            # rpm - MAX MOTOR RPM
max_power = None                # W - MAX POWER - AS DEFINED BY USER (FOR POWER LIMITING)

# TRACTION CONSTANTS
max_speed_kmh = None            # km/h - MAX SPEED
traction_speed = None           # km/h - MAX SPEED AROUND RADIUS IN TRACTION TEST
traction_radius = None          # m - RADIUS OF TRACTION TEST
longitudinal_friction = None    # Coefficient of longitudinal friction

# !!! 
# THERMAL CONSTANTS
heatsink_mass = None            # kg - TOTAL PACK HEATSINK MASS
heatsink_cv = None              # J/C*kg - HEATSINK MATERIAL SPECIFIC HEAT
air_temp = None            # C - CONSTANT ASSUMED AIR TEMP
water_temp = None          # C - CONSTANT ASSUMED WATER TEMP
air_htc = None                  # W/C*m^2 - ASSUMED CONSTANT AIR HTC
water_htc = None                # W/C*m^2 - ASSUMED CONSTANT WATER HTC
air_factor_m = None             # kg/kg AIR COOLING MASS PER BATTERY MASS
water_factor_m = None           # kg/kg WATER COOLING MASS PER BATTERY MASS

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
    value_name = dataArray[2:,0]
    value = dataArray[2:,2]
    value = np.asarray(value, dtype = float)

    track_name = dataArray[0,0]
    track = dataArray[0,2]

    regen_name = dataArray[1,0]
    regen = dataArray[1,2]

# Now create variables for everything
for x, y in zip(value_name, value):
    globals()[x] = y

globals()[track_name] = track
globals()[regen_name] = regen

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

# !!!
# Pack Design Constants
e_spec_aux_mass = 2.51 # kg/kWh of auxillary battery components
enclosure_mass = 4.456 # kg

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
pack_nominal_voltage = cell_nominal_voltage * num_series_cells # V
total_pack_ir = single_cell_ir / num_parallel_cells * num_series_cells  # ohms
# !!! Total known energy is approximately SoC * nominal voltage * max capacity
knownTotalEnergy = initial_SoC * capacity0 * pack_nominal_voltage / 1000  # kWh
pack_min_voltage = cell_min_voltage * num_series_cells  # V

# Braking Data
brakeCsvName = str(max_power / 1000) + "kW_BrakeAndMotor_OL_22.csv"

# !!!
# Car Mass - Calculated Values
total_cell_mass = cell_mass*num_cells # kg
cooled_cell_mass = total_cell_mass*(1 + air_factor_m + water_factor_m) # kg
cell_aux_mass = cell_aux_factor*(capacity0 * pack_nominal_voltage / 1000) # kg
mass = mass + cooled_cell_mass + cell_aux_mass + heatsink_mass # kg

# !!! 
# Thermals - Calculated Values
battery_cooled_hc = battery_cv*cell_mass + (heatsink_mass*heatsink_cv)/num_cells # J/C
air_tc = air_htc*cell_air_area # W/C
water_tc = water_htc*cell_water_area # W/C

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
max_traction_force = mass * g * longitudinal_friction   # N
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

# Take in braking and regen file:
# Read .csv of braking vs distance data
brake_df = pd.read_csv(brakeCsvName, header=[0])
# brake_df = brake_df.drop([0,1])
distanceTravelled = brake_df.loc[:,'elapsedDistance'].to_numpy(dtype=float)
engine_w_vector = brake_df.loc[:,'engineSpeed'].to_numpy(dtype=float)
engine_T_vector = brake_df.loc[:,'torque'].to_numpy(dtype=float)
brakePosition = brake_df.loc[:,'brakePosition'].to_numpy(dtype=float)

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
           "P_battery_OL",          # Optimum Lap battery power (kW)
           "P_battery_regen",       # battery regen power (kW)
           "Capacity",              # Battery capacity (Ah)
           "Pack Current",          # Battery pack current (A)
           "Energy Use",            # energy use over time (kWh)
           "Energy Use OL",         # energy use as per Optimum lap (kWh)
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
    
    # Add the braking
    dataDict = dynF.batteryBrakeAndRegen(dataDict, i)

    ###########################################
    # Add the further battery calcs here:
    # For my simulation battery power
    dataDict = dynF.batteryPackCalcs(dataDict, i)

    ###########################################

    # Increase time vector
    next_t = np.array([dataDict['t0'][i] + dt])
    dataDict['t0'] = np.append(dataDict['t0'], next_t)

# Trapezoidal approximation for energy used
energy, totalEnergy = dynF.trapezoidApprox(dataDict['P_battery'])
dataDict['Energy Use'] = energy

OLenergy, OLTotalEnergy = dynF.trapezoidApprox(dataDict['P_battery_OL'])
dataDict['Energy Use OL'] = OLenergy

dataDict['Max Values'][1] = totalEnergy
print('Energy Used (This Sim): ', totalEnergy, 'kWh')
print('Energy Used (Optimum Lap): ', OLTotalEnergy, 'kWh')

# Determine SoCe based on this!
dataDict = dynF.SoCenergy(dataDict, knownTotalEnergy)

# Determination of maximum power
dataDict['P_battery'] = dataDict['P_battery'] / 1000        # convert to kW
dataDict['P_battery_OL'] = dataDict['P_battery_OL'] / 1000
dataDict['P_battery_regen'] = dataDict['P_battery_regen'] / 1000
maxPower = max(dataDict['P_battery'])
OLMaxPower = max(dataDict['P_battery_OL'])
averagePower = np.mean(dataDict['P_battery'])
averageOLPower = np.mean(dataDict['P_battery_OL'])

# dataDict['Max Values'][2] = "Max power Used (kW)"
dataDict['Max Values'][3] = maxPower
print("Max Power (This Sim): ", maxPower, "kW")
print("Max Power (Optimum Lap): ", OLMaxPower, "kW")
print("Avg Power (This Sim): ", averagePower, "kW")
print("Avg Power (Optimum Lap): ", averageOLPower, "kW")
print("Car Mass: " + str(mass) + " kg")

# Now cut all arrays down to the correct size before inputting into a dataframe
time_size = len(dataDict['t0'])
for i in range(0, len(headers)):
    dataDict[headers[i]] = dataDict[headers[i]][:time_size]

# Additional Outputs
dataDict['v0'] = dataDict['v0'] * 3.6            # convert to km/h
print("Total Time: ", dataDict['t0'][-1], "s")
print("Lap Time: ", dataDict['t0'][-1] / numLaps, "s")
print("Final SoC(c): ", dataDict['SoC Capacity'][-1], "%")

# Now I want to write all the columns to a dictionary and then input it into a dataframe - since it's easier to do column-wise
dfData = pd.DataFrame(dataDict)
dfData.dropna(inplace = True)

# Drop extra columns based on the size of the time vector
outfile = "dynamicsCalcs.csv"
dfData.to_csv(outfile, index=False)

# Create plots
dynF.plotData(dataDict)
print("Completed")