# 作者:周子涵
# 2025年05月06日23时30分09秒

'''
data.item() 字典类型<class 'numpy.ndarray'>
()
object
<class 'dict'>

1. 键: max_len
type: <class 'int'>
2. 键: All_train_data
type: <class 'numpy.ndarray'>
(10403, 14, 256)
3. 键: All_train_label
type: <class 'numpy.ndarray'>
(10403,)
4. 键: train_data
type: <class 'numpy.ndarray'>
(9456, 14, 256)
5. 键: train_label
type: <class 'numpy.ndarray'>
(9456,)
6. 键: val_data
type: <class 'numpy.ndarray'>
(947, 14, 256)
7. 键: val_label
type: <class 'numpy.ndarray'>
(947,)
8. 键: test_data
type: <class 'numpy.ndarray'>
(1893, 14, 256)
9. 键: test_label
type: <class 'numpy.ndarray'>
(1893,)
'''

import numpy as np

# 读取 .npy 文件
data = np.load('C:/Users/87391/Desktop/opensource/EEG2Rep/Dataset/Crowdsource/Crowdsource.npy',
               allow_pickle=True)

# 查看内容
# print(data[:10])
print(type(data))  # 通常是 numpy.ndarray
print(data.shape)  # 形状
print(data.dtype)  # 数据类型
obj = data.item()  # 提取其中的Python对象
print(type(obj))  # 看看它是列表、字典、还是别的  <class 'dict'>  字典
for i, (key, value) in enumerate(obj.items()):
    print(f"{i+1}. 键: {key}")
    print("type:", type(value))
    '''
    if isinstance(value, (list, np.ndarray)):
        print("前5项:", value[:5])
    else:
        print("值:", value)
    '''
    if key != 'max_len':
        print(value.shape)
    if key == 'train_data':
        print(value.shape[1])