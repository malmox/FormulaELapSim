import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt

INFILE = "10CDischargeMalloryCSV.csv"

element_interval = 100

# Input file and read into pandas dataframe
df = pd.read_csv(INFILE, header=[0])
original_headers = df.columns.tolist()      # convert headers to a list

header_length = len(original_headers)

# requested cols
col_names = [original_headers[0], original_headers[2], original_headers[5]]

# Deal with those columns :)
# we want to search through every tenth element and input that into a new .csv
# Set new columns for the dataFrame
newdf = pd.DataFrame(columns = col_names)

# Now read that original dataframe for every tenth value (for example)
length = len(df.loc[:,col_names[2]])
# max_soc_index = np.searchsorted(n1,n)
max_soc_index =  round(length * 7/12)

for i in range(0, len(col_names)):
    fill_up = []    # empty array to fill up the data frame
    for j in range(max_soc_index, length, element_interval):
        fill_up.append(df.loc[j,col_names[i]])
    
    newdf = newdf.assign(**{col_names[i]: fill_up})

print(newdf)

# Send to a new .csv file
outfile = INFILE.replace(".csv", "_out.csv")
newdf.to_csv(outfile,index=False)

plt.plot(newdf.loc[:,col_names[-1]], newdf.loc[:,col_names[1]])
plt.show()