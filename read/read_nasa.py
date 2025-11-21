import numpy as np
import scipy.io
from datetime import datetime
import os
import pandas as pd

# Convert time format, convert string to datetime format
def convert_to_time(hmm):
    year, month, day, hour, minute, second = int(hmm[0]), int(hmm[1]), int(hmm[2]), int(hmm[3]), int(hmm[4]), int(hmm[5])
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)

# Load mat file and process impedance data
def loadMat(matfile):
    data = scipy.io.loadmat(matfile)
    filename = matfile.split('\\')[-1].split('.')[0]
    col = data[filename]
    col = col[0][0][0][0]
    size = col.shape[0]

    data = []
    for i in range(size):
        k = list(col[i][3][0].dtype.fields.keys())
        d1, d2 = {}, {}
        if str(col[i][0][0]) == 'impedance':
            for j in range(len(k)):
                t = col[i][3][0][0][j][0]
                l = [t[m] for m in range(len(t))]
                d2[k[j]] = l
        else:
            for j in range(len(k)):
                t = col[i][3][0][0][j][0]
                l = [t[m] for m in range(len(t))]
                d2[k[j]] = l
        d1['type'], d1['temp'], d1['time'], d1['data'] = str(col[i][0][0]), int(col[i][1][0]), str(convert_to_time(col[i][2][0])), d2
        data.append(d1)

    return data

# Extract battery capacity
def getBatteryCapacity(Battery):
    cycle, capacity = [], []
    i = 1
    for Bat in Battery:
        if Bat['type'] == 'discharge':
            capacity.append(Bat['data']['Capacity'][0])
            cycle.append(i)
            i += 1
    return [cycle, capacity]

# Get test data during charging or discharging
def getBatteryValues(Battery, Type='charge'):
    data = []
    for Bat in Battery:
        if Bat['type'] == Type:
            data.append(Bat['data'])
    return data

# Get impedance data
def getImpedanceValues(Battery):
    data = []
    for Bat in Battery:
        if Bat['type'] == 'impedance':
            data.append(Bat['data'])
    return data

# Get desktop path of current user
def get_desktop_path():
    return os.path.join(os.path.expanduser("~"), "Desktop")

# General function to save data to Excel files
def save_data_to_excel(data_dict, filename, sheet_name):
    # Desktop path
    desktop_path = get_desktop_path()
    full_filename = os.path.join(desktop_path, filename)

    # Create an empty DataFrame
    df = pd.DataFrame()

    # Iterate through the dictionary and add data to DataFrame
    for dataset_name, dataset_values in data_dict.items():
        # Assume dataset_values is a list of dictionaries containing 'type', 'temp', 'time', and 'data'
        for item in dataset_values:
            # Append each dictionary to DataFrame
            df = df._append(pd.DataFrame([item]), ignore_index=True)

    # Ensure DataFrame column names are strings
    df.columns = df.columns.astype(str)

    # Save DataFrame to Excel
    with pd.ExcelWriter(full_filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

# Save battery capacity data to desktop
def save_capacity_to_excel(capacity):
    save_data_to_excel(capacity, 'battery_capacity.xlsx', 'Capacity')

# Save charge data to desktop
def save_charge_to_excel(charge):
    save_data_to_excel(charge, 'battery_charge.xlsx', 'Charge')

# Save discharge data to desktop
def save_discharge_to_excel(discharge):
    save_data_to_excel(discharge, 'battery_discharge.xlsx', 'Discharge')

# Save impedance data to desktop
def save_impedance_to_excel(impedance):
    save_data_to_excel(impedance, 'battery_impedance.xlsx', 'Impedance')

# Main program: load data and save
Battery_list = ['B0005', 'B0006', 'B0007', 'B0018']  # Names of 4 datasets
dir_path = 'C:\\Users\\hsx\\Desktop\\machine learning-literature\\datasets\\nasa\\'

capacity, charge, discharge, impedance = {}, {}, {}, {}
for name in Battery_list:
    print('Load Dataset ' + name + '.mat ...')
    path = dir_path + name + '.mat'
    data = loadMat(path)
    capacity[name] = getBatteryCapacity(data)              # Capacity data during discharge
    charge[name] = getBatteryValues(data, 'charge')        # Charge data
    discharge[name] = getBatteryValues(data, 'discharge')  # Discharge data
    impedance[name] = getImpedanceValues(data)             # Impedance data

# Save data to desktop
save_capacity_to_excel(capacity)
save_charge_to_excel(charge)
save_discharge_to_excel(discharge)
save_impedance_to_excel(impedance)