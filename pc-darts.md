# PC-DARTS: Partial Channel Connections for Memory-Efficient Architecture Search
Yuhui Xu, Lingxi Xie, Xiaopeng Zhang, Xin Chen, Guo-Jun Qi, Qi Tian, Hongkai Xiong
## Introduction
尽管DARTS可以有效的端到端搜索，但仍然需要一个庞大的网络空间，导致存储和计算问题。  
本文提出了pc-DARTS来解决上述负担，核心思想很直观：不把全部的通道送入op选择中，而是在每一步随机选取一部分，剩下的通过shortcut直接跳过。我们假设在通道
子集上的计算是对所有通道的估计。出了资源消耗的减少，通道选择带来另外一个优势，op搜索被正则化，更不容易陷入局部最优。然而，PC-DARTS也有副作用，在每次迭代
对通道的采样都是不同的，这可能导致通道连接的选择不稳定。因此，我们提出edge-normalization的方法来稳定网络连通性的搜索。具体地是学习一组额外的超参数，称为
edge-selection hyper-parameters。通过在训练过程中共享这些超参，搜索出的网络结构对通道采样不再敏感。  
## Method  
### 1. DARTS  
略
### 2. 部分通道连接  
DARTS中，网络结构的每个节点都要存储|O|个操作和对应的输出，增加了|O|倍内存。  
![img](https://github.com/terrencewayne/Paper-notes/blob/master/images/pc-darts.png "PC-DARTS")  
部分通道连接如上图所示。以x<sub>i</sub>到x<sub>j</sub>的连接为例，定义一个通道选择mask S<sub>i,j</sub>，被选择的通道置1，否则置0.被选择的通道送入
|O|个op的混合计算中，而未选择的通道跳过这些op，也就是说直接复制入输出。  
<img src="https://latex.codecogs.com/gif.latex?f_{i,j}^{PC}(x_i;S_{i,j})=\sum_{o\in&space;O}\frac{exp\left&space;\{&space;\alpha_{i,j}^o&space;\right&space;\}}{\sum_{o'\in&space;O}exp\left&space;\{&space;\alpha_{i,j}^{o'}&space;\right&space;\}}\cdot&space;o(S_{i,j}*x_i)&plus;(1-S_{i,j})*x_i" title="f_{i,j}^{PC}(x_i;S_{i,j})=\sum_{o\in O}\frac{exp\left \{ \alpha_{i,j}^o \right \}}{\sum_{o'\in O}exp\left \{ \alpha_{i,j}^{o'} \right \}}\cdot o(S_{i,j}*x_i)+(1-S_{i,j})*x_i" />  
S代表选择，1-S则是未选择。选择1/K个通道，K为可调整的超参。  
### 3. edge normalization
在DARTS中，每个节点从其祖先节点选择两个作为输入，祖先节点和该节点的连接由权重max<sub>o</sub>α描述，在pc-darts中，这些权重是由其中一小部分的通道计算得到
的，由于通道选择的变化，他们的最优连接就可能不稳定。为了缓解这个问题，为每一条边分配权重，节点的输出计算为  
<img src="https://latex.codecogs.com/gif.latex?x_j^{PC}=\sum_{i<j}\frac{exp\left&space;\{&space;\beta_{i,j}&space;\right&space;\}}{\sum_{i'<j}exp\left&space;\{&space;\beta_{i',j}&space;\right&space;\}}\cdot&space;f_{i,j}(x_i)" title="x_j^{PC}=\sum_{i<j}\frac{exp\left \{ \beta_{i,j} \right \}}{\sum_{i'<j}exp\left \{ \beta_{i',j} \right \}}\cdot f_{i,j}(x_i)" />  
最后搜索完成后，每个节点之间的连接由softmax(α)和softmax(β)乘积计算得到，从中选取最大权重的边。  
