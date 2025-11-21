import numpy as np

# Load .npz file and set allow_pickle=True
data = np.load('C:/Users/hsx/Desktop/machine learning-literature/算法/新建文件夹/cycles_Qdlin_Vdlin.npz', allow_pickle=True)

# Print all keys in the file
print("Keys in the npz file:", data.keys())

# Access specific data
# Suppose we want to access the array named 'inf40'
inf40 = data['inf40']

# If you want to access the Qdlin array
Qdlin = data['Qdlin']

# If you want to access the Vdlin array
Vdlin = data['Vdlin']

# Remember to close the file after operation to release resources
data.close()
print(inf40)