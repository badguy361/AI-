from scipy.io import loadmat
import wfdb
import matplotlib.pyplot as plt
import h5py
import numpy as np

#load V4 .mat file (signal data)
data=loadmat("D:/AI_sleep_project/data/physionet.org/files/training/tr03-0005/tr03-0005.mat")
data.keys()
data["val"].shape
for channel in range(data["val"].shape[0]):
    fig,ax=plt.subplots(figsize=(14,7))
    ax.plot(data["val"][channel,0:1000])

# load V7 .mat file (label data)
path="D:/AI_sleep_project/data/physionet.org/files/training/tr03-0005/tr03-0005-arousal.mat"
with h5py.File(path, 'r') as data:
    print("==========database layer========")
    print("first layer: ",data.keys())
    print("second layer: ",data["#refs#"].keys())
    print("second layer: ",data["data"].keys())
    print("third layer: ",data["data"]["arousals"])
    print("third layer: ",data["data"]["sleep_stages"].keys())
    for stage in data["data"]["sleep_stages"]:
        print(data["data"]["sleep_stages"][stage])



#load .hea file
Records=wfdb.rdrecord("D:/AI_sleep_project/data/physionet.org/files/training/tr03-0005/tr03-0005",physical=False)

# Records=wfdb.rdrecord("D:/AI_sleep_project/data/physionet.org/files/training/tr03-0005/tr03-0005",\
#                         sampfrom=0,sampto=10000,channels=[0, 1],physical=False) #read waveform length 10000, two channels

Records.p_signal #模擬訊號值 shape:(5147000,13) physical=True 才讀入
print(Records.d_signal.shape) #數位訊號值 shape:(5147000,13) physical=False才讀入
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
    print(annotation.__dict__[key])









#reference:
# https://kknews.cc/zh-tw/code/amrjjrj.html


