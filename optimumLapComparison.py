import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Constants
OPTIMUMLAP = "OptimumLapSim61.40.csv"
DYNAMICS = "dynamicsCalcs.csv"

# Import dynamics csv
infile = DYNAMICS
dfD = pd.read_csv(infile, header = [0])
time = dfD.loc[:,'t0'].to_numpy()
velocity = dfD.loc[:,'v0'].to_numpy() * 3.6
distance = dfD.loc[:,'r0'].to_numpy()

# Import actual csv
infile = OPTIMUMLAP
dfO = pd.read_csv(infile, header = [0])
dfO = dfO.drop(index = 0)
distanceO = dfO.loc[:,'elapsedDistance'].to_numpy(dtype = float)
timeO = dfO.loc[:,'elapsedTime'].to_numpy(dtype = float)
velocityO = dfO.loc[:,'speed'].to_numpy(dtype = float)

# plot the graphs on top of each other
plt.plot(distanceO, velocityO, 'b')
plt.plot(distance[0:700], velocity[0:700], 'r')
plt.xlabel('Distance (m)')
plt.ylabel('Velocity (km/h)')
plt.title('Velocity vs Distance over 1 Endurance Lap')
plt.legend(['Optimum Lap Data', 'Python Lap Simulation Data'])
plt.show()