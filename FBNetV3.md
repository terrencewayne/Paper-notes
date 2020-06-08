# FBNetV3: Joint Architecture-Recipe Search using Neural Acquisition Function
Xiaoliang Dai1, AlvinWan, Peizhao Zhang, Bichen Wu, Zijian He, Zhen Wei,
Kan Chen, Yuandong Tian, Matthew Yu, Peter Vajda, and Joseph E. Gonzalez
## Introduction  
提出联合训练方法，既搜索结构，又搜索训练超参。  
## 方法  
目标是在给定资源约束的情况下，寻找最准确的结构和训练策略。  
<img src="https://latex.codecogs.com/gif.latex?\max_{(A,h)\in&space;\Omega}acc(A,h),\qquad&space;subject\quad&space;to&space;\quad&space;g_i(A)\leq&space;C_i&space;\quad&space;for\quad&space;i=1,...,\gamma" title="\max_{(A,h)\in \Omega}acc(A,h),\qquad subject\quad to \quad g_i(A)\leq C_i \quad for\quad i=1,...,\gamma" />  
A,h,Ω 表示网络结构，训练策略和搜索空间。g表示资源约束，比如计算代价，存储代价，时延等。  
设计了两阶段的搜索方法，分别是1）由粗到细的搜索，2）细粒度的搜索  
### 1. Coarse-grained search: Constrained iterative optimization  
本阶段输出一个准确率预测器和具有潜力的候选集  
**Neural Acquisition Function (i.e., Predictor)** 多层感知机结构，包括一个网络结构编码器和两个头1）辅助代理头，同于预训练编码器，根据结构表示预测
结构指标，如FLOPS等。结构使用one-hot类别变量和正数范围变量表示。2）准确率预测器，接收训练策略和网络结构表示，通过有条件的迭代优化训练。  
### 2. Fine-grained search: Predictor-based evolutionary search  
看到这里我才发现v3竟然不是one-shot搜索？？？感觉被骗了，弃坑。
