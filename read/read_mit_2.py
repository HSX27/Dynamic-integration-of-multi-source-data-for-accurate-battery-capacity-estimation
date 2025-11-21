import numpy as np
import pymatreader
import pandas as pd

# Read .mat file
ds = pymatreader.read_mat(r"C:\Users\hsx\Desktop\问题总结\testdata\testdata.mat")
# Print variables in the .mat file
print(ds.keys())

# Assume `myStruct` is one of the variable names, it is a structure
test = ds['testData']
types = list(test.keys())

# Create a dictionary to store each column's data
data_dict = {}
for i in range(5):  # Avoid index out of range
    key = types[i]
    value = test[key]

    # Assume value is a list or array that can be directly used as DataFrame column
    data_dict[key] = value

# Create DataFrame
df = pd.DataFrame(data_dict)
print(df)
## Save as matrix, fill missing values with np.nan if less than 2000
# Read Vdlin and Qdlin
Qdlin = np.full((40, 2000, 1000), np.nan)
Vdlin = np.full((40, 1000), np.nan)

for i in range(40):
    Qdi = test[types[5]][i]['Qdlin']
    ncy = df['cycle_life'][i] - 1
    print('----' + str(ncy) + '----')
    print(len(Qdi))
    for j in range(ncy):
        Qdlin[i, j, :] = Qdi[j]
    Vdlin[i, :] = test['Vdlin'][i]
np.savez('./cycles_Qdlin_Vdlin.npz', inf40=df, Qdlin=Qdlin, Vdlin=Vdlin)

## Save as 40 objects
# Initialize an empty list
Vdlino = []
# Initialize an empty list
Qdlino = []
for i in range(40):
    Qdi = test[types[5]][i]['Qdlin']
    ncy = df['cycle_life'][i] - 1
    print('----' + str(ncy) + '----')
    print(len(Qdi))
    Qdlini = np.full((ncy, 1000), np.nan)
    for j in range(ncy):
        Qdlini[j, :] = Qdi[j]
    Vdlini = test['Vdlin'][i]
    Vdlino.append(Vdlini)
    Qdlino.append(Qdlini)
    del Qdlini, Vdlini
    print(len(Vdlino))
    print(len(Qdlino))
# Convert data to numpy array of object type
df_array = np.array(df, dtype=object)
Qdlin_array = np.array(Qdlino, dtype=object)
Vdlin_array = np.array(Vdlino, dtype=object)
# Save data to npz file
np.savez('cycles_Qdlin_Vdlin.list.npz', inf40=df_array, Qdlin=Qdlin_array, Vdlin=Vdlin_array)

# # Read npz file
data = np.load('cycles_Qdlin_Vdlin.list.npz', allow_pickle=True)

# # Print all keys
print("Keys in the npz file:", data.keys())

# # Access specific data
df = data['inf40']
Qdlin = data['Qdlin']
Vdlin = data['Vdlin']
# Qdlin has 40 objects, select via Qdlin[i], each object is a matrix