
from sys import displayhook
from cv2 import VIDEOWRITER_PROP_QUALITY
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio 
import wfdb

record = wfdb.rdrecord('tr03-0005', sampfrom=800,channels=[1, 3],sampto=3000)
print('record: \n',record)
ecg_record = wfdb.rdheader('tr03-0005')#, channels = [1,3])
print('ecg \n',ecg_record)

signals, fields = wfdb.rdsamp('tr03-0005',
                                  sampfrom=800,
                                  channels =[1,3])
print('sig: \n',signals)
#ann = wfdb.rdann('tr03-0005', extension='atr', sampto=3000)
wfdb.plot_wfdb(record = record,title='plot ECG')
#wfdb.plot_items(signal=record.p_signal,
                    #ann_samp=[ann.sample, ann.sample],
 #                   title='MIT-BIH Record 100', time_units='seconds',
  #                  figsize=(10,4), ecg_grids='all')

#https://limtoinf.com/%E8%A8%8A%E8%99%9F%E8%99%95%E7%90%86-wfdb-python-%E8%AE%80%E5%8F%96%E8%A8%8A%E8%99%9F/
'''
data = scio.loadmat('tr03-0005.mat')
print('data: \n',data)     		    #大致看一下data的结构
print('datatype: \n',type(data)) 	#看一下data的类型
print('keys: \n',data.keys)  		#查看data的键，这里验证一下是否需要加括号
print('keys: \n',data.keys())		#当然也可以用data.values查看值
print('val: \n',data['val'])     		    #查看数据集
a = data['val']
print('a: \n',a)
print('target shape: \n',data['val'].shape)

#reference:https://www.cxyzjd.com/article/weixin_45182000/106771498
'''