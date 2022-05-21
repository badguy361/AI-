import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

data=pd.read_csv("D:/Important/大四下/AI專題/sensors.csv",header=None,\
                sep=";",names=["time index","X","Y","Z"])

target_data=list(sorted(set(data["time index"])))


result={"time index":[],"X":[],"Y":[],"Z":[]}

for i in tqdm(range(len(target_data))):
# for i in tqdm(range(10)):
    mask=(data["time index"]==target_data[i])
    result["time index"].append(target_data[i])
    result["X"].append(np.mean(data[mask]["X"]))
    result["Y"].append(np.mean(data[mask]["Y"]))
    result["Z"].append(np.mean(data[mask]["Z"]))

result_df=pd.DataFrame(result)

result_df.to_csv("../sensors without double values.csv")
length=1000
fig,ax=plt.subplots(3,1,figsize=(14,7))
ax[0].scatter(data["time index"][:length],data["X"][:length],alpha=0.5)
ax[1].scatter(data["time index"][:length],data["Y"][:length],alpha=0.5)
ax[2].scatter(data["time index"][:length],data["Z"][:length],alpha=0.5)
ax[0].set_title("vibration")

fig,ax=plt.subplots(3,1,figsize=(14,7))
ax[0].scatter(result["time index"][:length],result["X"][:length],alpha=0.5)
ax[1].scatter(result["time index"][:length],result["Y"][:length],alpha=0.5)
ax[2].scatter(result["time index"][:length],result["Z"][:length],alpha=0.5)
ax[0].set_xlim(min(data["time index"][:length]),max(data["time index"][:length]))
ax[1].set_xlim(min(data["time index"][:length]),max(data["time index"][:length]))
ax[2].set_xlim(min(data["time index"][:length]),max(data["time index"][:length]))
ax[0].set_title("removed double value vibration")



