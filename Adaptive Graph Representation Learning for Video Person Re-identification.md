# Adaptive Graph Representation Learning for Video Person Re-identification
# 面向视频Re-ID的自适应图表示学习
Yiming Wu, Omar El Farouk Bourahla, Xi Li, Fei Wu, and Qi Tian, Fellow, IEEE  
## 摘要
为了解决Re-ID中的遮挡问题，以往基于局部的方法没有注意局部区域之间的联系。本文提出一种自适应的图表示学习机制，使相关的局部特征之间可以相互作用。  
特别地，本文提出姿势对齐连接（pose alignment connection）和特征联结连接（feature afinity connection）来构建图，模拟了图节点之间的本质关系。  
本文在邻接图上传播特征，来迭代地精炼局部特征，邻居节点信息参与了局部特征表示。
## 方法
### 总体框架
通过姿势对齐和特征联结两种连接方式得到图，在通过GNN得到上下文的相互作用，通过特征转播，有意义的局部特征得到加强，噪声部分被减弱。  
方法如图所示
![structure](https://raw.githubusercontent.com/terrencewayne/Paper-notes/master/images/adaptive%20graph%20representation%20learning%20for%20reid-1.png "structure")
### 如何得到Graph
图定义为 *G* = {*V*,*A*}. *V*为点集，*A*为边集。*V* = {*vi*}, *i*∈(*1, T×N*)。  
T为输入的图像数，N为每个图像划分的区域数，划分方法参见上图的Graph branch，共划分7块，每一块认为是原图对应位置的特征。  
*V*的定义：每一条特征向量视为一个顶点；*A*的定义：根据下面两种方法计算(*TN*)×(*TN*)的邻接矩阵。
#### pose alignment connection
![pac](https://raw.githubusercontent.com/terrencewayne/Paper-notes/master/images/adaptive%20graph%20representation%20learning%20for%20reid-2.png "pose alignment connection")  
如图所示，将人划分为头、躯干、腿三部分，包含相同部分的区域将建立一条边，即邻接矩阵对应值为1。 
<img src="https://latex.codecogs.com/gif.latex?A^{p}_{ij}=\left\{\begin{matrix}&space;1&space;&&space;i\neq&space;j&space;,&space;\left&space;|&space;S{i}\cap&space;S_j&space;\neq0&space;\right&space;|\\&space;0&space;&&space;otherwise&space;\end{matrix}\right." title="A^{p}_{ij}=\left\{\begin{matrix} 1 & i\neq j , \left | S{i}\cap S_j \neq0 \right |\\ 0 & otherwise \end{matrix}\right." />  
*S*<sub>*i*</sub> 表示节点*v*<sub>*i*</sub> 所包含的人体结构。
#### feature affinity connection
<img src="https://latex.codecogs.com/gif.latex?A^{f}_{ij}=S\left&space;(&space;x_i,x_j&space;\right&space;)=\frac{2}{e^{\left&space;\|&space;x_i-x_j&space;\right&space;\|_2}&plus;1}" title="A^{f}_{ij}=S\left ( x_i,x_j \right )=\frac{2}{e^{\left \| x_i-x_j \right \|_2}+1}" />  
x<sub>i</sub> 表示节点对应的特征向量。  
两个邻接矩阵加权得到最终的邻接矩阵（论文中直接取平均）
### 在图上传播特征
<img src="https://latex.codecogs.com/gif.latex?x^{\left&space;(&space;l&space;\right&space;)}_i=\left&space;(&space;1-\alpha&space;\right&space;)x^{\left&space;(&space;l-1&space;\right&space;)}_i&plus;\alpha\sum_{j=1}^{T\cdot&space;N}A^{\left&space;(&space;l&space;\right&space;)}_{ij}F^{\left&space;(&space;l&space;\right&space;)}\left&space;(&space;x^{\left&space;(&space;l-1&space;\right&space;)}_j&space;\right&space;)" title="x^{\left ( l \right )}_i=\left ( 1-\alpha \right )x^{\left ( l-1 \right )}_i+\alpha\sum_{j=1}^{T\cdot N}A^{\left ( l \right )}_{ij}F^{\left ( l \right )}\left ( x^{\left ( l-1 \right )}_j \right )" />  
图神经网络GNN共有L层，第l层节点i的特征向量按上式计算，A是根据该层特征计算得到的邻接矩阵，F表示第l全连接层，接收上一层的节点特征向量  
最后对所有的特征向量加权，权重是各特征向量的全局归一化的范数  
### 损失函数
交叉熵和三元组损失  
### 可视化结果
![visual](https://raw.githubusercontent.com/terrencewayne/Paper-notes/master/images/adaptive%20graph%20representation%20learning%20for%20reid-3.png "visual result")  
从图上看，加入图信息后，网络更关注人体部分，减少了图像干扰区域的影响
