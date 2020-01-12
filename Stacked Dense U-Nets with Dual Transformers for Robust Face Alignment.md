# Stacked Dense U-Nets with Dual Transformers for Robust Face Alignment
Jia Guo, Jiankang Deng, Niannan Xue, Stefanos Zafeiriou  
## 摘要
设计了一种新颖的尺度聚合网络拓扑结构和一个通道聚合构建块，以提高模型的容量，而且不会牺牲计算复杂性和模型大小。借助于密集堆叠U-Nets内部的可变形卷积和外部数据变换的coherent loss，我们的模型获得了对任意输入面部图像的空间不变的能力  
## 方法  
**Stack Dense U-net**如图所示  
![image](https://raw.githubusercontent.com/terrencewayne/Paper-notes/master/images/stack-unet0.png "structure")  
在不同分辨率保持空间信息，并提高模型容量，绿色表示深度可分离卷积  

**CAB模块**  
![image](https://raw.githubusercontent.com/terrencewayne/Paper-notes/master/images/stack-unet1.png "CAB")  
特征在下采样之前分支，再上采样之前合并。主干网络中的通道压缩可以帮助前后层建模，它包含了通道方向的热图关系，并在局部观测模糊时增强了鲁棒性。  
