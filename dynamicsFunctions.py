import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import csv

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

with open(filename, 'r', newline='') as infile:
    reader = csv.reader(infile)
    dataList = list(reader)
    dataList.pop(0)

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

#################################################################################################
# Here, I will input all of the functions used in my code so that the code itself gets less messy
# Quadratic formula
def quad_formula(a, b, c):
    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        root1 = 0
        root2 = 0
    else:
        root1 = (-b + discriminant**(1/2)) / (2*a)
        root2 = (-b - discriminant**(1/2)) / (2*a)

    return [root1, root2]

# NEW motor torque curve calculation
def motorTorqueCurve():
    length = 1000

    # create motor rpm vector
    motor_rpm = np.linspace(0, max_motor_rpm, length)

    # create motor torque vector
    motor_torque = np.ones(length) * max_motor_torque

    # create motor power vector
    motor_power = np.zeros(length)

    # iterate:
    switch_index = 0
    for i in range(0, length):
        if switch_index == 0:
            # Determine motor power at this point
            motor_power[i] = motor_torque[i] * motor_rpm[i] / radsToRpm

            # If the motor pomwer at this point is greater than the max power, then we evaluate a diff option.
            if motor_power[i] >= max_power:
                switch_index = i

                # Now make calculations for the linear fit
                dy = 0 - max_motor_torque
                dx = max_motor_rpm - motor_rpm[i]
                slope = dy / dx     # units of Nm / rpm
                y_int = max_motor_torque - slope * motor_rpm[i]     # units of Nm
        else:
            # Check constant power fit
            motor_torque_constP = max_power / (motor_rpm[i] / radsToRpm)
            # Check linear fit
            motor_torque_linFit = slope * motor_rpm[i] + y_int

            # Choose smaller of two options
            motor_torque[i] = min(motor_torque_constP, motor_torque_linFit)
            motor_power[i] = motor_torque[i] * motor_rpm[i] / radsToRpm

    # plt.plot(motor_rpm, motor_torque)
    # plt.grid(True)
    # plt.show()

    return motor_torque, motor_rpm

####
# Call the function to create necessary global variables
motor_torque, motor_rpm = motorTorqueCurve()
####

# Determine motor torque from motor rpm
def motorTorque(w_m, motor_torque, motor_rpm):

    # w_m is delivered in rpm
    # Find the index location of closest rpm
    index = np.searchsorted(motor_rpm, w_m)

    # determine corresponding torque value
    T_m = motor_torque[index]

    return T_m

# the initial calculations to determine a speed and distance
def fastestNextSpeed(dataDict, i):
    # angular frequency of wheel: w_wh
    dataDict['w_wh'][i] = dataDict['v0'][i] / wheel_radius

    # angular frequency of motor: w_m - also convert from rad/s to rpm
    dataDict['w_m'][i] = dataDict['w_wh'][i] * GR * radsToRpm

    # motor torque: T_m
    dataDict['T_m'][i] = motorTorque(dataDict['w_m'][i], motor_torque, motor_rpm)

    # axel torque: T_a
    dataDict['T_a'][i] = dataDict['T_m'][i] * GR

    # Traction force: F_trac
    dataDict['F_trac'][i] = dataDict['T_a'][i] / wheel_radius       # Traction force from two motors - not split between two wheels
    # NOT -> F_trac[i] = T_a[i] / (2 * wheel_radius)

    # Drag force: F_drag
    dataDict['F_drag'][i] = (rho_air * Af * Cd * (dataDict['v0'][i] + v_air)**2) / 2

    # Rolling resistance: F_RR = mu * normal force
    # Only when car is moving:
    if dataDict['v0'][i] == 0:
        dataDict['F_RR'][i] = 0
    else:
        dataDict['F_RR'][i] = mu_rr * mass * g

    # Fnet (tangential)
    dataDict['F_net_tan'][i] = dataDict['F_trac'][i] - (dataDict['F_drag'][i] + dataDict['F_RR'][i])

    # Acceleration (tangential)
    dataDict['a_tan0'][i] = dataDict['F_net_tan'][i] / mass

    # Theoretical max speed - if travelling in a straight line
    # new velocity:
    dataDict['v0'][i+1] = dataDict['v0'][i] + dataDict['a_tan0'][i] * dt

    # new position:
    dataDict['r0'][i+1] = dataDict['r0'][i] + dataDict['v0'][i] * dt + 1/2 * dataDict['a_tan0'][i] * dt**2

    return dataDict

# determining the max speed requirement
def findMaxSpeed(current_radius, dataDict, i):
    if current_radius != 0:
        # What is the max speed around the corner
        dataDict['a_norm0'][i] = F_friction / mass

        v_max = (dataDict['a_norm0'][i] * current_radius)**(1/2)        # Solving for the max velocity around the corner

        if v_max > max_speed:
            v_max = max_speed
    else:
        v_max = max_speed      # The max speed that our car can travel at - in this case 100 km/h

    return v_max

# Based on the new max speed, what are our new variables
def limit_max_speed(dataDict, v_max, i):
    # Check if this force is greater than the maximum friction force. If so, then the car cannot speed up.
    # Then, we need to stay at the previous speed or we need to brake to reach a slower speed
    # So part of this is going to be to determine what the max speed is around the corner and then compare that to our speed.

    # Reset velocity to the max velocity
    dataDict['v0'][i+1] = v_max

    # Back calculate to determine values
    # Now determine the required acceleration at this point.
    dataDict['a_tan0'][i] = (dataDict['v0'][i+1] - dataDict['v0'][i]) / dt

    # Now, what is the net force
    dataDict['F_net_tan'][i] = mass * dataDict['a_tan0'][i]

    # Now based on the net force, what is the traction force sent to the wheels
    dataDict['F_trac'][i] = dataDict['F_net_tan'][i] + (dataDict['F_drag'][i] + dataDict['F_RR'][i])

    # axel torque
    dataDict['T_a'][i] = dataDict['F_trac'][i] * wheel_radius

    # motor torque
    dataDict['T_m'][i] = dataDict['T_a'][i] / GR

    # w_m, w_wh, v0 stay the same

    # Next position is now also different
    dataDict['r0'][i+1] = dataDict['r0'][i] + dataDict['v0'][i] * dt + 1/2 * dataDict['a_tan0'][i] * dt**2

    return dataDict

# EFFICIENCY FUNCTION
def motorEfficiency(w_m):
    # Find efficiency
    index = np.searchsorted(n_motor_arr[:,0], w_m)

    n_transmission = n_motor_arr[index, 1]

    return n_transmission

# BATTERY CALCULATION
# From given traction force, calculates power draw from battery
def batteryPower(dataDict, i):
    P_wheel = dataDict['F_trac'][i] * dataDict['v0'][i]                       # Both wheel power
    # P_wheel2 = dataDict['w_m'][i]  / radsToRpm * dataDict['T_m'][i]           # These are both identical

    # n_transmission calculation
    n_transmission = motorEfficiency(dataDict['w_m'][i])
    
    P_motors = P_wheel / n_transmission
    P_motorloss = A * dataDict['w_m'][i]**2 + B * dataDict['w_m'][i] + C      # Motorloss

    # The traction force is for BOTH motors and both wheels, but we have two motors, so twice the loss
    P_converter = P_motors + 2 * P_motorloss
    dataDict['P_battery'][i] = P_converter / n_converter

    return dataDict

# trapezoidal approximation with energy as result
def trapezoidApprox(P_battery):
    trapezoidal_vector = 2 * np.ones(num)
    trapezoidal_vector[0] = 1
    trapezoidal_vector[-1] = 1
    almost_energy = dt / 2 * trapezoidal_vector * P_battery
    energy = np.cumsum(almost_energy) / 3600000     # convert to kWh
    totalEnergy = energy[-1]

    return [energy, totalEnergy]

# Recursive algorithm for finding SoC in lookup table - should be somewhat faster than looking through all of the data
# could have just used np.searchsorted - however, I did this for my own learning so yayyy
def SoClookup(SoC, thisCapacity):
    if len(SoC) == 1:           # Base Case
        return SoC[0]
    else:                       # Recursive Case
        mid = len(SoC) // 2     # Floor division
        if SoC[mid] > thisCapacity:
            return SoClookup(SoC[mid:], thisCapacity)
        elif SoC[mid] < thisCapacity:
            return SoClookup(SoC[:mid], thisCapacity)
        else:
            return SoC[mid]      # Secondary base case

# This power limiting algorithm has now become obsolete.
# Power Limited
# def powerLimited(dataDict, i):
#     P_converter = dataDict["P_battery"][i] * n_converter

#     # Keep w_m same, keep voltage[i] same, keep vo[i] same 
#     # n_transmission stays same, P_motorloss stays same:
#     n_transmission = motorEfficiency(dataDict['w_m'][i])
#     P_motorloss = A * dataDict['w_m'][i]**2 + B * dataDict['w_m'][i] + C

#     # Motor power
#     P_motors = P_converter - 2 * P_motorloss

#     # Wheel power
#     P_wheel = P_motors * n_transmission

#     # F_trac
#     dataDict["F_trac"][i] = P_wheel / dataDict['v0'][i]

#     # F_drag same, F_RR same
#     # F_net_tan
#     dataDict['F_net_tan'][i] = dataDict['F_trac'][i] - (dataDict['F_drag'][i] + dataDict['F_RR'][i])

#     # Acceleration (tangential)
#     dataDict['a_tan0'][i] = dataDict['F_net_tan'][i] / mass

#     # Theoretical max speed - if travelling in a straight line
#     # new velocity:
#     dataDict['v0'][i+1] = dataDict['v0'][i] + dataDict['a_tan0'][i] * dt

#     # new position:
#     dataDict['r0'][i+1] = dataDict['r0'][i] + dataDict['v0'][i] * dt + 1/2 * dataDict['a_tan0'][i] * dt**2

#     return dataDict

# More battery calculations
def batteryPackCalcs(dataDict, i):

    # Our power limiting step will become obsolete if we change our motor curve to inherently involve the max power.
    # Leave this in for if we want to go back or if we encounter any errors - but at the moment, those errors should never occur.

    # # Power limit may not be more constricting than current limit - calculate limiting current
    # current_limit = dataDict['Pack Voltage'][i] * max_capacity * max_CRate

    # # Now power limiting algorithm
    # if dataDict['P_battery'][i] > max_power:
    #     dataDict['P_battery'][i] = max_power
    #     dataDict = powerLimited(dataDict, i)
    # # Next, power limit from current limit:
    # elif dataDict['P_battery'][i] > current_limit * dataDict['Pack Voltage'][i]:
    #     dataDict['P_battery'][i] = current_limit * dataDict['Pack Voltage'][i]
    #     dataDict = powerLimited(dataDict, i)

    # Determine pack current
    # Note that the effect of the total pack internal resistance is included to make the calculation more accurate
    current = quad_formula(total_pack_ir, pack_nominal_voltage, -1 * dataDict['P_battery'][i])
    dataDict['Pack Current'][i] = max(current)      # ignore the negative result from the quadratic formula

    # Determine Ahr lost at this current based on time interval dt
    Ahr_lost = dataDict['Pack Current'][i] * dt / 3600

    # Determine new capacity
    dataDict['Capacity'][i+1] = dataDict['Capacity'][i] - Ahr_lost

    # Determine new SoCc
    dataDict['SoC Capacity'][i+1] = dataDict["Capacity"][i+1] / capacity0 * 100

    ########################################################################
    # Simple thermal calculations
    # P = I^2 * r
    dataDict['Dissipated Power'][i] = (dataDict['Pack Current'][i] / num_parallel_cells)**2 * single_cell_ir
    
    dataDict['Battery Temp'][i+1] = dataDict['Battery Temp'][i] + dt * dataDict['Dissipated Power'][i] / (cell_mass * battery_cv)

    ########################################################################
    # # Determine which file we should reference for voltage plots
    # # calculate C-rate:
    # CRate = dataDict['Pack Current'][i] / capacity0

    # # determine which file we should look in
    # # options: 0.5, 1, 5, 10
    # actual_options = [0.5, 1, 5, 10]
    # CRate_options = [0, 0, 0, max_CRate]
    # for j in range(0,len(actual_options) - 1):
    #     CRate_options[j] = (actual_options[j+1] - actual_options[j]) / 2 + actual_options[j]     # iffy algorithm to determine which file to look in
    #     # basically just finding the average between each of two points

    # # now determine the location based on the weird thing I made above to deal with the uneven spacing
    # location = np.searchsorted(CRate_options, CRate)

    # # Now find the correct file
    # filename = str(actual_options[location]) + "CDischargeMalloryCSV_out.csv"

    # # Now let's open the file and read it to a dataframe
    # df = pd.read_csv(filename, header=[0])
    # CRateheaders = df.columns.tolist()

    # # Now read specific rows to lists for purposes that become apparent later
    # SoC = df.loc[:,CRateheaders[-1]].to_list()
    # Voltage = df.loc[:,CRateheaders[1]].to_list()

    # # Now that we have our lists, we look for the location of the SoC algorithm
    # searchForSoC = SoClookup(SoC, dataDict['SoC Capacity'][i])
    # SoCIndex = SoC.index(searchForSoC)

    # # Now we determine pack voltage
    # dataDict['Pack Voltage'][i+1] = num_series_cells * Voltage[SoCIndex]      # from that index, we then determine the voltage

    return dataDict         # then we return our favourite dictionary :)

# State of Charge - energy based
def SoCenergy(dataDict, totalEnergy):

    # vector calculation at each point
    dataDict['SoC Energy'] = (totalEnergy - dataDict['Energy Use']) / totalEnergy * 100

    return dataDict

# Plot details to make my life cleaner :))
def plotDetails(x_axis, y_axis, plotTitle, ax):
    ax.set_title(plotTitle)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.grid(True)

    return

# Plot the data
def plotData(dataDict):
    # Plot the data in this function
    ROWS = 2
    COLS = 2
    FIGWIDTH = 20
    FIGHEIGHT = 12

    # create subplots
    fig, ax = plt.subplots(ROWS, COLS, figsize=(FIGWIDTH, FIGHEIGHT))
    supTitle = "Point Mass Vehicle Simulation - " + TRACK.replace(".csv","")
    fig.suptitle(supTitle)

    # Plot 1)
    # Velocity vs Time
    row = 0; col = 0
    x_axis = "Distance (m)"
    y_axis = "Velocity (km/h)"
    plotTitle = "Car Velocity vs Distance"
    # Convert the velocity from m/s to km/h
    dataDict['v0'] = dataDict['v0'] * 3.6
    ax[row][col].plot(dataDict["r0"][0:700], dataDict["v0"][0:700])       # plot the data
    plotDetails(x_axis, y_axis, plotTitle, ax[row][col])  # add the detaila

    # Plot 2)
    # Position vs Time
    row = 0; col = 1
    x_axis = "Time (s)"
    y_axis = "SoC(c) (%)"
    plotTitle = "SoC(c) vs Time"
    ax[row][col].plot(dataDict["t0"], dataDict["SoC Capacity"])
    plotDetails(x_axis, y_axis, plotTitle, ax[row][col])

    # Plot 3)
    # Battery power vs time with Energy use overlay
    row = 1; col = 0
    x_axis = "Time (s)"
    y_axis = "Battery Power (kW)"
    plotTitle = "Battery Power vs Time"
    ax[row][col].plot(dataDict["t0"], dataDict["P_battery"])
    plotDetails(x_axis, y_axis, plotTitle, ax[row][col])

    # Plot 4)
    # SoC energy vs time
    row = 1; col = 1
    x_axis = "Time (s)"
    y_axis = "Temperature (C)"
    plotTitle = "Battery Temperature vs Time"
    ax[row][col].plot(dataDict['t0'], dataDict['Battery Temp'])
    # Will also plot a red line to show the minimum voltage
    # ax[row][col].plot(dataDict['t0'], np.ones_like(dataDict['t0']) * pack_min_voltage, 'r')
    plotDetails(x_axis, y_axis, plotTitle, ax[row][col])

    figTitle = "Point Mass Vehicle Simulation_" + TRACK.replace(".csv","") + ".png"
    plt.savefig(figTitle)

    return
