# Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition
Sijie Yan, Yuanjun Xiong, Dahua Lin  
这篇笔记分析空间图卷积的理论推导和代码实现
## Theory
### 普通图像2D卷积
<img src="https://latex.codecogs.com/gif.latex?f_{out}\left&space;(&space;x&space;\right&space;)=\sum_{h=1}^{K}\sum_{w=1}^{K}f_{in}\left&space;(&space;p\left&space;(&space;x,h,w&space;\right&space;)&space;\right&space;)\cdot&space;w\left&space;(&space;h,w&space;\right&space;)" title="f_{out}\left ( x \right )=\sum_{h=1}^{K}\sum_{w=1}^{K}f_{in}\left ( p\left ( x,h,w \right ) \right )\cdot w\left ( h,w \right )" />  

*p*(*x*,*h*,*w*) 表示x位置的邻域，K为卷积核的尺寸，w是对应点的权重  、

### 图2D卷积
论文中介绍的空间图卷积，核心思想是把节点看做像素点。  
<img src="https://latex.codecogs.com/gif.latex?f_{{}out}\left(v_i&space;\right&space;)=\sum_{v_j\in&space;B(v_j)}\frac{1}{Z_i(v_j)}f_{in}\left&space;(&space;v_j&space;\right&space;)\cdot&space;w\left&space;(&space;l_i\left&space;(&space;v_j&space;\right&space;)&space;\right&space;)" title="f_{{}out}\left(v_i \right )=\sum_{v_j\in B(v_j)}\frac{1}{Z_i(v_j)}f_{in}\left ( v_j \right )\cdot w\left ( l_i\left ( v_j \right ) \right )" />  

其中*B*(*v*<sub>i</sub>) 是*v*<sub>i</sub> 的邻域点集，*l*<sub>i</sub>(*v*<sub>j</sub>) 是*v*<sub>j</sub> 相对于*v*<sub>i</sub> 的距离，图卷积根据距离不同赋予权重，这里不再像普通卷积那样，给每个节点不同的权值，而是根据距离把*B*划分为不同的子集，每个子集共享相同的权值。Z根据不同子集中节点的数量将每个子集的贡献拉回同一水平。  
#### 实现
上面的阐述实际上涵盖了ST-GCN中不同的划分类型，如图所示。  
![partition](https://raw.githubusercontent.com/terrencewayne/Paper-notes/master/images/st-gcn-1.png "partition")  
实际上对于第一种（也是最普遍的）划分，我们可以用下式来实现  
<img src="https://latex.codecogs.com/gif.latex?f_{{}out}=\Lambda&space;^{-\frac{1}{2}}\left&space;(&space;A&plus;I&space;\right&space;)\Lambda&space;^{-\frac{1}{2}}f_{in}W" title="f_{{}out}=\Lambda ^{-\frac{1}{2}}\left ( A+I \right )\Lambda ^{-\frac{1}{2}}f_{in}W" />  
*A* 是邻接矩阵，*Λ* 是度矩阵，左乘和右乘度矩阵的-1/2次幂实际上对*A*+*I*做了列归一化。  
论文中把输入特征整理成（C通道, V点数, T帧长）的形状，fW是卷积核为（1，t）的2d图像卷积，当不考虑时间尺度时，fW实际上是1X1卷积  
## Code
生成边集  
```
num_node = 18
self_link = [(i, i) for i in range(num_node)]
neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5),
                 (13, 12), (12, 11), (10, 9), (9, 8), (11, 5),
                 (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0),
                 (17, 15), (16, 14)]
edge = self_link + neighbor_link
```
计算邻接矩阵，这里的A实际上是邻接矩阵*A*+*I*  
```
A = np.zeros((num_node, num_node))
for i, j in edge:
    A[j, i] = 1
    A[i, j] = 1
```
计算距离矩阵(不考虑节点之间的差异的时候，这一步省略)  
```
max_hop = 1 #邻域内的节点距离中心节点的最大距离为1
hop_dis = np.zeros((num_node, num_node)) + np.inf #先生成节点数X节点数的距离矩阵，值为无穷大，代表节点之间都没有连接
transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)] #生成邻接矩阵对应的单位阵和自身的列表
arrive_mat = (np.stack(transfer_mat) > 0) #对得到的矩阵，确定其值大于0的位置
for d in range(max_hop, -1, -1):
    hop_dis[arrive_mat[d]] = d #更新距离矩阵，将单位阵的大于0的点位置的距离置为0，将A的大于0的点的位置的距离置为1
dilation = 1
valid_hop = range(0, max_hop + 1, dilation) #dilation表示
adjacency = np.zeros((num_node, num_node))
for hop in valid_hop:
    adjacency[hop_dis == hop] = 1
```
对A做列归一化  
```
Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    
A = np.zeros((1, self.num_node, self.num_node))
A[0] = normalize_adjacency #添加一维待后面使用
```
前向过程  
```
x = conv(x)
x = x.view(n, kernel_size, kc // self.kernel_size, t, v)
x = torch.einsum('nkctv,kvw->nctw', (x, A)) #根据A对输出向量的特征进行局部求和
```

