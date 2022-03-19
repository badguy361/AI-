
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio 

data = scio.loadmat('tr03-0005.mat')
print('data: \n',data)     		    #大致看一下data的结构
print('datatype: \n',type(data)) 	#看一下data的类型
print('keys: \n',data.keys)  		#查看data的键，这里验证一下是否需要加括号
print('keys: \n',data.keys())		#当然也可以用data.values查看值
print(data['val'])      		    #查看数据集
print('target shape \n',data['val'].shape)

'''
mat = scipy.io.loadmat('tr03-0005.mat')
a = np.array(mat)
print(a)

print(mat)
print(type(mat))
print(mat.keys())
#mat["val"].shape

plt.imshow(a, interpolation='nearest', cmap='bone', origin='lower')
plt.colorbar(shrink=.92)

plt.xticks(())
plt.yticks(())
plt.show()

#x = 
#y = 
plt.show()
'''