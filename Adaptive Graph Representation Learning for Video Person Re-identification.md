# Adaptive Graph Representation Learning for Video Person Re-identification
# 面向视频Re-ID的自适应图表示学习
Yiming Wu, Omar El Farouk Bourahla, Xi Li*, Fei Wu, and Qi Tian, Fellow, IEEE  
## 摘要
为了解决Re-ID中的遮挡问题，以往基于局部的方法没有注意局部区域之间的联系。本文提出一种自适应的图表示学习机制，使相关的局部特征之间可以相互作用。  
特别地，本文提出姿势对齐连接（pose alignment connection）和特征联结连接（feature afinity connection）来构建图，模拟了图节点之间的本质关系。  
本文在邻接图上传播特征，来迭代地精炼局部特征，邻居节点信息参与了局部特征表示。
## 方法
### 总体框架
通过姿势对齐和特征联结两种连接方式得到图，在通过GNN得到上下文的相互作用，通过特征转播，有意义的局部特征得到加强，噪声部分被减弱。
![structure](images/adaptive graph representation learning for reid-1.png "structure")
