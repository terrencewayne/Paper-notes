# Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition
Sijie Yan, Yuanjun Xiong, Dahua Lin  
这篇笔记分析空间图卷积的理论推导和代码实现
## Theory
### 普通图像2D卷积
<img src="https://latex.codecogs.com/gif.latex?f_{out}\left&space;(&space;x&space;\right&space;)=\sum_{h=1}^{K}\sum_{w=1}^{K}f_{in}\left&space;(&space;p\left&space;(&space;x,h,w&space;\right&space;)&space;\right&space;)\cdot&space;w\left&space;(&space;h,w&space;\right&space;)" title="f_{out}\left ( x \right )=\sum_{h=1}^{K}\sum_{w=1}^{K}f_{in}\left ( p\left ( x,h,w \right ) \right )\cdot w\left ( h,w \right )" />  
*p*(*x*,*h*,*w*) 表示x位置的邻域，K为卷积核的尺寸，w是对应点的权重  
### 图2D卷积
论文中介绍的空间图卷积，核心思想是把节点看做像素点。
<img src="https://latex.codecogs.com/gif.latex?f_{{}out}\left(v_i&space;\right&space;)=\sum_{v_j\in&space;B(v_j)}\frac{1}{Z_i(v_j)}f_{in}\left&space;(&space;v_j&space;\right&space;)\cdot&space;w\left&space;(&space;l_i\left&space;(&space;v_j&space;\right&space;)&space;\right&space;)" title="f_{{}out}\left(v_i \right )=\sum_{v_j\in B(v_j)}\frac{1}{Z_i(v_j)}f_{in}\left ( v_j \right )\cdot w\left ( l_i\left ( v_j \right ) \right )" />  
其中*B*(*v*<sub>i</sub>) 是*v*<sub>i</sub> 的邻域点集，*l*<sub>i</sub>(*v*<sub>j</sub>) 是*v*<sub>j</sub> 相对于*v*<sub>i</sub> 的距离，图卷积根据距离不同赋予权重，这里不再像普通卷积那样，给每个节点不同的权值，而是根据距离把*B*划分为不同的子集，每个子集共享相同的权值。  

## Code

