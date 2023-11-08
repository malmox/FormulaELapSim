import numpy as np
import pandas as pd

inFile = "10.0kW_BrakeAndMotor_OL.csv"

brake_df = pd.read_csv(inFile, header=[0])
brake_df = brake_df.drop([0,1])

distanceTravelled0 = brake_df.loc[:,'elapsedDistance'].to_numpy(dtype=float)
engine_w_vector_0 = brake_df.loc[:,'engineSpeed'].to_numpy(dtype=float)
engine_T_vector_0 = brake_df.loc[:,'torque'].to_numpy(dtype=float)
brakePosition_0 = brake_df.loc[:,'brakePosition'].to_numpy(dtype=float)

distanceTravelled = np.zeros(len(distanceTravelled0) // 2)
engine_w_vector_1 = np.zeros(len(distanceTravelled0) // 2)
engine_T_vector_1 = np.zeros(len(distanceTravelled0) // 2)
brakePosition_1 = np.zeros(len(distanceTravelled0) // 2)

# Cut down all vectors by half
j = 0
for i in range(1,len(distanceTravelled0),2):
    distanceTravelled[j] = distanceTravelled0[i]
    engine_w_vector_1[j] = engine_w_vector_0[i]
    engine_T_vector_1[j] = engine_T_vector_0[i]
    brakePosition_1[j] = brakePosition_0[i]

    j += 1

distanceVectorToAdd = np.linspace(distanceTravelled0[-2] + 0.5, distanceTravelled0[-1] * 23, len(distanceTravelled)*22)
distanceTravelled = np.concatenate((distanceTravelled, distanceVectorToAdd))

engine_w_vector = engine_w_vector_1
engine_T_vector = engine_T_vector_1
brakePosition = brakePosition_1

for i in range(0, 22):
    engine_w_vector = np.concatenate((engine_w_vector, engine_w_vector_1))
    engine_T_vector = np.concatenate((engine_T_vector, engine_T_vector_1))
    brakePosition = np.concatenate((brakePosition, brakePosition_1))

# dictionary approach
thisDict = dict.fromkeys(brake_df.columns)
thisDict['elapsedDistance'] = distanceTravelled
thisDict['engineSpeed'] = engine_w_vector
thisDict['torque'] = engine_T_vector
thisDict['brakePosition'] = brakePosition

df = pd.DataFrame(thisDict)

df.to_csv(inFile.replace('.csv','_22.csv'), index=False)