from scipy.io import loadmat
import wfdb
import matplotlib.pyplot as plt
import h5py
import numpy as np

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


# load V7 .mat file (label data)
path="D:/AI_sleep_project/data/physionet.org/files/training/tr03-0005/tr03-0005-arousal.mat"

data=h5py.File(path, 'r')
data["data"].keys() # ['arousals', 'sleep_stages']
arousals=data["data"]["arousals"][:]

data["data"]["sleep_stages"].keys() #['nonrem1', 'nonrem2', 'nonrem3', 'rem', 'undefined', 'wake']
sleep_stages=data["data"]["sleep_stages"]

#check sleep stage value
nonrem1=sleep_stages["nonrem1"][:]
for value in sleep_stages["nonrem1"]:
    print(set(value))


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


