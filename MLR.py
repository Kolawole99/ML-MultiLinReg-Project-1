#===================================IMPORTING LIBRARIES AND PACKAGES==============================
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
#%matplotlib inline #used in jupyterlab



#============================================DATA=======================================

#===================================Understanding the data================================
# MODELYEAR e.g. 2014
# MAKE e.g. Acura
# MODEL e.g. ILX
# VEHICLE CLASS e.g. SUV
# ENGINE SIZE e.g. 4.7
# CYLINDERS e.g 6
# TRANSMISSION e.g. A6
# FUELTYPE e.g. z
# FUEL CONSUMPTION in CITY(L/100 km) e.g. 9.9
# FUEL CONSUMPTION in HWY (L/100 km) e.g. 8.9
# FUEL CONSUMPTION COMB (L/100 km) e.g. 9.2
# CO2 EMISSIONS (g/km) e.g. 182 --> low --> 0

#=====================================Reading the data==================================
df = pd.read_csv('FuelConsumptionCo2.csv')
# take a look at the dataset
print(df.head())

#================================Project Feature selection================================
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head(9))

#========================plot Emission values with respect to Engine size========================
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()



#================================CREATING TRAIN/TEST SPLIT=================================

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

#====================plot train split of Emission values with respect to Engine size=================
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
