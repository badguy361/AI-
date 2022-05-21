from scipy.io import loadmat
import wfdb
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#load V4 .mat file (signal data)
file_ID="tr03-0005"
records=loadmat(f"D:/AI_sleep_project/data/physionet.org/files/training/{file_ID}/{file_ID}.mat")
records.keys()
records["val"].shape
for channel in range(records["val"].shape[0]): 
    fig,ax=plt.subplots(figsize=(14,7))
    ax.plot(records["val"][channel,0:1000])

#ECG plot:
fix,ax=plt.subplots(figsize=(14,7))
ax.plot(records["val"][12,0:1000])
ax.set_xlim(0,1000)
ax.set_title(f"ECG, file ID: {file_ID}",fontsize=25)
ax.set_xlabel("Time (sec*200)",fontsize=20)
ax.set_ylabel("mV",fontsize=20)


# ==================load V7 .mat file (label data)==========================
file_ID="tr03-0005"
path=f"D:/AI_sleep_project/data/physionet.org/files/training/{file_ID}/{file_ID}-arousal.mat"

data=h5py.File(path, 'r')
data["data"].keys() # ['arousals', 'sleep_stages']
arousals=data["data"]["arousals"][:]

data["data"]["sleep_stages"].keys() #['nonrem1', 'nonrem2', 'nonrem3', 'rem', 'undefined', 'wake']
sleep_stages=data["data"]["sleep_stages"]

#check sleep stage value
nonrem1=sleep_stages["nonrem1"][:]
for value in sleep_stages["nonrem1"]:
    print(set(value))

#merge V7 file into dataframe
for index,stage_name in enumerate(sleep_stages):
    print(f"stage:{stage_name}")
    if index==0:
        stage=np.array(sleep_stages[stage_name][0])
        stage=np.expand_dims(stage,axis=0)
    else:
        print(f"origin:{stage}")
        to_be_concated_stage=np.expand_dims(sleep_stages[stage_name][0],axis=0)
        print(f"to be concated:{to_be_concated_stage}")
        stage=np.concatenate((stage,to_be_concated_stage),axis=0)
        print(f"result:{stage}")
stage=stage.T

arousal_df=pd.DataFrame(arousals.flatten(),columns=["arousals"])
stage_df=pd.DataFrame(stage,\
                        columns=["nonrem1","nonrem2","nonrem3","rem","undefined","wake"])
merge_df=pd.merge(arousal_df,stage_df,left_index=True, right_index=True)

for column in ["nonrem1","nonrem2","nonrem3","rem","undefined","wake"]:
    sleep_stage_mask=(merge_df[column]==1)
    merge_df.loc[sleep_stage_mask,"sleep_stage"]=column


#total sleep stage labels
output_path=f"D:/AI_sleep_project/data/physionet.org/files/training/{file_ID}"

fig,ax=plt.subplots(figsize=(7,7))
ax.bar(merge_df["sleep_stage"].value_counts().index,merge_df["sleep_stage"].value_counts().values)
ax.set_yscale("log")
ax.set_ylabel("number of points",fontsize=12)
ax.set_title(f"{file_ID} sleep stage labels",fontsize=15)
ax.tick_params(axis='x', labelsize=12 )
fig.savefig(f"{output_path}/{file_ID} sleep stage labels.png",dpi=150)

#arousal, non-arousal, will not be scored compare to sleep stage labels
for num,arousal in zip([-1,0,1],["not be scored","non-arousal","arousal"]):
    arousal_mask=(merge_df["arousals"]==num)
    fig,ax=plt.subplots(figsize=(7,7))
    ax.bar(merge_df[arousal_mask]["sleep_stage"].value_counts().index,\
            merge_df[arousal_mask]["sleep_stage"].value_counts().values)
    ax.set_yscale("log")
    ax.set_ylabel("number of points",fontsize=12)
    ax.tick_params(axis='x', labelsize=12 )
    ax.set_title(f"{file_ID} ({arousal}) sleep stage labels",fontsize=15)
    # fig.savefig(f"{output_path}/{file_ID} ({arousal}) sleep stage labels.png",dpi=150)

# merge_df.to_csv(f"{output_path}/{file_ID}_sleep_stage_labels.csv",index=False)

# =======================================================================

#the structure of V7 file
with h5py.File(path, 'r') as data:
    print("==========database layer========")
    print("first layer: ",data.keys())
    print("second layer: ",data["#refs#"].keys()) #refs# has nothing
    print("second layer: ",data["data"].keys())
    print("third layer: ",data["data"]["arousals"])
    print("third layer: ",data["data"]["sleep_stages"].keys())
    for stage in data["data"]["sleep_stages"]:
        print(data["data"]["sleep_stages"][stage])



#load .hea file
Records=wfdb.rdrecord("D:/AI_sleep_project/data/physionet.org/files/training/tr03-0005/tr03-0005",physical=False)

#read waveform length 10000, two channels:
# Records=wfdb.rdrecord("D:/AI_sleep_project/data/physionet.org/files/training/tr03-0005/tr03-0005",\
#                         sampfrom=0,sampto=10000,channels=[0, 1],physical=False) 

# physical=True or False the value is same, the difference is value format

Records.p_signal #模擬訊號值 shape:(5147000,13) physical=True 才讀入

print(Records.d_signal) #數位訊號值 shape:(5147000,13) physical=False才讀入
sampling_rate=Records.fs # 採樣頻率: 200Hz
part_record = Records.d_signal[0:1000]

for channel in range(part_record.shape[1]):
    fig,ax=plt.subplots(figsize=(14,7))
    ax.plot(part_record[:,channel])


#read .arousal file
wfdb.show_ann_classes() #列印出annotation的後綴以及標註的類型
wfdb.show_ann_labels() #列印出各種符號代表的心率類型

annotation=wfdb.rdann("D:/AI_sleep_project/data/physionet.org/files/training/tr03-0005/tr03-0005",extension="arousal")

annotation.__dict__.keys()

for key in annotation.__dict__:
    print(f"{key}:",annotation.__dict__[key])
    try:
        print(f"{key}:",annotation.__dict__[key].shape)
    except:
        continue

#reference:
# https://kknews.cc/zh-tw/code/amrjjrj.html
#https://www.nature.com/articles/s42003-020-01542-8.pdf


