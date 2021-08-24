# MicroNet: Improving Image Recognition with Extremely Low FLOPs
Yunsheng Li, Yinpeng Chen, Xiyang Dai, Dongdong Chen, Mengchen Liu,
Lu Yuan, Zicheng Liu, Lei Zhang, Nuno Vasconcelos

## Introduction
设计计算量极低的网络结构，提出微分解卷积，将卷积矩阵分解为低秩矩阵，制造稀疏性；提出动态激活函数Dynamic Shift Max，通过对特征图和其circular channel shift求取最大的多个动态混合，来提升非线性。  
从两个方面设计计算量极低的网络：节点连接性和非线性。首先证明更低的节点连通性拓宽了网络宽度，提供了更好的折中，其次，通过提升非线性降低网络深度。从而，本文设计了更高效的卷积和激活函数。  
对于卷积，本文提出了Micro-Factorized convolution（MF-Conv），将point-wise卷积转化为两个group-wise卷积，分组数<img src="https://latex.codecogs.com/svg.image?G&space;=&space;\sqrt{C/R}&space;" title="G = \sqrt{C/R} " />，其中C是通道数，R是通道下降比。对于激活函数，本文提出了Dynamic Shift Max（DY-Shift-Max），非线性地将通道以动态相关系数混合。具体地，该激活函数迫使网络学习混合different circular shift，然后选择最优的。基于这两个操作搭建的MicroNet，其12MFLOPs和21MFLOPs版本超过了MobileNetV3 9.6% 和 4.5%
## MF-Conv
MF-Conv的目标是优化通道数与节点连通性之间的折中。连通性E定义为每个输出节点的路径数，一条路径连接一个输入节点和一个输出节点
### Micro-Factorized Pointwise Conv
简单期间，假设卷积核W的输入输出通道数相等，忽略偏置。核矩阵W分解为两个分组自适应卷积，分组数取决于通道数  
<img src="https://latex.codecogs.com/svg.image?W&space;=&space;P\Phi&space;Q^T" title="W = P\Phi Q^T" />  
W是<img src="https://latex.codecogs.com/svg.image?C\times&space;C" title="C\times C" />矩阵，Q是<img src="https://latex.codecogs.com/svg.image?C\times&space;\frac{C}{R}" title="C\times \frac{C}{R}" />矩阵，将通道数压缩，P是<img src="https://latex.codecogs.com/svg.image?\frac{C}{R}\times&space;C" title="\frac{C}{R}\times C" />矩阵，将通道数扩展回去。P与Q都是分块对角矩阵，<img src="https://latex.codecogs.com/svg.image?\Phi" title="\Phi" />是\frac{C}{R} \times \frac{C}{R}的置换矩阵，像shufflenet那样shuffle通道。
