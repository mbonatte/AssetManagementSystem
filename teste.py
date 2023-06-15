import json

import matplotlib.pyplot as plt

from prediction.markov import MarkovContinous
from maintenance.maintenance import ActionEffect
from maintenance.performance import Performance
from maintenance.handle_maintenance import get_IC_through_time_maintenance
from maintenance.handle_maintenance import extract_indicator

import os
path = os.getcwd()

file_path = path+ "/database/ActionsEffects.json"

with open(file_path, 'r') as j:
     maintenance_data = json.loads(j.read())

file_path = path+ "/database/ASFiNAGDataBase.CSV"

with open(file_path, 'r') as j:
     inspections = j.read()

file_path = path+ "/database/road_sections_properties.CSV"

with open(file_path, 'r') as j:
     asset_properties = j.read()

###################################################




maintenance_scenario = {#'15': "Milling (5cm) + TC + RC",
                        }

#"""
a=get_IC_through_time_maintenance(inspections,
                                'ASFiNAG',
                                maintenance_data,
                                maintenance_scenario,
                                5,
                                1,
                                'year',
                                30,
                                asset_properties = asset_properties)

#"""
print(a)
#print(extract_indicator("Eveness", maintenance_data)[0]['name'])

fig, graph = plt.subplots()
plt.plot(a['Cracking_ASFiNAG']['Time'],
         a['Cracking_ASFiNAG']['IC'])
plt.plot(a['Longitudinal_Evenness_ASFiNAG']['Time'],
         a['Longitudinal_Evenness_ASFiNAG']['IC'])
plt.show()
