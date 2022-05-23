import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime as dt
import datetime

data=pd.read_csv("D:/Important/大四下/AI專題/sensors.csv",header=None,\
                sep=";",names=["UNIX time (ms)","X","Y","Z"])

counts_df=data["UNIX time (ms)"].value_counts().\
    rename_axis('UNIX time (ms)').reset_index(name='counts')
multiple_counts_mask=(counts_df["counts"]>1)

#相同UNIX Time的table
multiple_counts_df=counts_df[multiple_counts_mask]

result={"UNIX time (ms)":[],"X":[],"Y":[],"Z":[]}

#取平均
for unix_time in tqdm(multiple_counts_df["UNIX time (ms)"]):
    mask=(data["UNIX time (ms)"]==unix_time)
    result["UNIX time (ms)"].append(unix_time)
    result["X"].append(np.mean(data[mask]["X"]))
    result["Y"].append(np.mean(data[mask]["Y"]))
    result["Z"].append(np.mean(data[mask]["Z"]))    

#平均table和總table取差集
counts_mean_df=pd.DataFrame(result)
final_data=data.append(counts_mean_df)
final_data.drop_duplicates(subset=['UNIX time (ms)'],keep=False,inplace=True)
#差集table和平均table結合
final_data=final_data.append(counts_mean_df)
final_data.sort_values(by=['UNIX time (ms)'],inplace=True)
final_data.reset_index(drop=True,inplace=True)

final_data.insert(1,"date time",0)
final_data["root(square(XYZ))"]=np.sqrt(final_data["X"]**2+final_data["Y"]**2+final_data["Z"]**2)

#UNIX time to Date time
for i in tqdm(range(len(final_data))):
    final_data.loc[i,"date time"]=dt.fromtimestamp(final_data["UNIX time (ms)"][i]/1000,\
                                    datetime.timezone(datetime.timedelta(hours=8))) #UTC+8

final_data.to_csv("../sensors without double values.csv")

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



