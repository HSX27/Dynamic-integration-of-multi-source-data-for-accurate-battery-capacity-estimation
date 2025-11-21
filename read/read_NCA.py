import pandas as pd
import os
import numpy as np

# Create empty dataframe
df_res = pd.DataFrame(columns=['cycle', 'Voltages', 'rate', 'Tem', 'Capacity'])

# Data file path
base_path = r'C:\Users\hsx\Desktop\machine learning-literature\datasets\6405084\Dataset_1_NCA_battery\Dataset_1_NCA_battery'
files = os.listdir(base_path)

for file in files:
    try:
        # Get full file path
        file_path = os.path.join(base_path, file)
        print(f"Processing file: {file_path}")

        # Extract temperature information
        Tem = int(file[2:4])

        # Read data
        data_r = pd.read_csv(file_path)

        # Iterate through cycles
        for i in range(int(np.min(data_r['cycle number'].values)), int(np.max(data_r['cycle number'].values)) + 1):
            data_i = data_r[data_r['cycle number'] == i]
            Ecell = np.array(data_i['Ecell/V'])
            Q_dis = np.array(data_i['Q discharge/mA.h'])
            Current = np.array(data_i['<I>/mA'])
            control = np.array(data_i['control/V/mA'])
            cr = np.array(data_i['control/mA'])[1] / 3500

            print(f"Cycle: {i}, Max Q_dis: {np.max(Q_dis)}")  # Debug discharge capacity

            if np.max(Q_dis) < 2500 or np.max(Q_dis) > 3500:
                continue

            index = np.where(np.abs(control) == 0)
            if len(index[0]) == 0:
                print(f"Cycle: {i}, No point with control=0 found")
                continue

            start = index[0][0]
            end = 13
            for j in range(3):
                if control[start + 3] == 0:
                    break
                else:
                    start = index[0][j + 1]

            print(f"Cycle: {i}, start: {start}, control[start]: {control[start]}")  # Debug starting point

            if Current[start] > 1:
                start = start + 1
                if control[start + 13] != 0:
                    end = 12

            print(f"Cycle: {i}, control[start + end]: {control[start + end]}, Ecell[start + end]: {Ecell[start + end]}")  # Debug conditions

            if control[start + end] == 0 and Ecell[start + end] > 4.0:
                # Create new row
                new_row = pd.DataFrame([{
                    'cycle': i,
                    'Voltages': list(Ecell[start:start + 14]),
                    'rate': cr,
                    'Tem': Tem,
                    'Capacity': np.max(Q_dis)
                }])
                # Add row using pd.concat
                df_res = pd.concat([df_res, new_row], ignore_index=True)
                print(f"Cycle: {i}, Data added")

    except Exception as e:
        print(f"Error processing file {file}: {e}")

# Save to desktop
desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop', 'Dataset_1_NCA_battery_fixed.xlsx')
df_res.to_excel(desktop_path, index=False)
print(f"Features extraction is done. File saved to: {desktop_path}")