# A Discriminative Feature Learning Approach for Deep Face Recognition
Yandong Wen, Kaipeng Zhang, Zhifeng Li, and Yu Qiao
## 摘要
softmax loss用于大多数的神经网络训练任务，本文提出了更discriminative的Center Loss，Center Loss为每一个类别学习一个特征中心，惩罚特征和中心之间的距离，通过结合softmax loss和center loss，可以增大类间距离，减小类内距离。  
## 介绍
对于人脸识别任务，由于实际的类别数远远大于数据所包含的，所以深度特征不仅要求separable，而且需要discriminative。softmax loss只监督了特征的seperability。  
基于mini-batch训练时，网络很难捕捉到深度特征的全局分布，一次送入全部的训练数据又不现实。作为弥补，constrastive loss和triplet loss通过样本对和三元组建立loss，但样本对和元组的选择需要很大的计算量。  
Center Loss为每个类别的深度特征学习一个中心，训练过程中，这个中心不断更新，类内特征与中心的距离得到缩小。联合训练的意义在于，softmax loss强制各类别的特征分开，center loss强制把类内特征集聚。
## 方法
softmax loss可以写作  
<img src="https://latex.codecogs.com/gif.latex?L_S=-\sum_{i=1}^{m}log\frac{e^{W_{y_i}^Tx_i&plus;b_{y_i}}}{\sum&space;_{j=1}^ne^{W_j^Tx_i&plus;b_j}}" title="L_S=-\sum_{i=1}^{m}log\frac{e^{W_{y_i}^Tx_i+b_{y_i}}}{\sum _{j=1}^ne^{W_j^Tx_i+b_j}}" />  
center loss可以写作  
<img src="https://latex.codecogs.com/gif.latex?L_C=\frac{1}{2}\sum_{i=1}^{m}\left&space;\|&space;x_i-c_{y_i}&space;\right&space;\|_2^2" title="L_C=\frac{1}{2}\sum_{i=1}^{m}\left \| x_i-c_{y_i} \right \|_2^2" />  
c表示特征的中心，这个式子表示C类内的特征与特征中心的距离。理想情况是，特征中心在特征改变时得到更新，即在每次迭代时要送入所有的训练数据，这样不现实。因此，这样的center loss不能直接使用。  
文章做了两处修正，其一是**基于每个batch更新特征中心**，其二是**减少网络错误预测引起的扰动，使用α控制中心的学习率**。  
联合训练的loss表示为  
<img src="https://latex.codecogs.com/gif.latex?L=L_S&plus;\lambda&space;L_C" title="L=L_S+\lambda L_C" />  
![img](https://raw.githubusercontent.com/terrencewayne/Paper-notes/master/images/centerloss-1.png "image")
