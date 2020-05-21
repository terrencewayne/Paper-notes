# FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search
Bichen Wu, Xiaoliang Dai, Peizhao Zhang, Yanghan Wang, Fei Sun, Yiming Wu, Yuandong Tian, Peter Vajda, Yangqing Jia, and Kurt Keutzer
## Intorduction
提出一种DNAS搜索算法，避免穷举和单独训练独立的结构  
* 使用一个super net代表搜索空间，super net的每个操作是随机选取的。  
* 使用Gumbel Softmax技术直接训练super net。  
* 测量每种操作的时延，构建查找表来估计总体时延。  
* loss包括交叉熵损失和时延损失  
## Method
搜索问题表示为  

<img src="https://latex.codecogs.com/gif.latex?\min&space;\limits_{a\in&space;\textit{A}}\min&space;\limits_{\omega_a}\textit{L}(a,\omega_a)" title="\min \limits_{a\in \textit{A}}\min \limits_{\omega_a}\textit{L}(a,\omega_a)" />

给定结构空间A，找到其最优结构a，ω<sub>a</sub>是网络参数  
1. 搜索空间  
定义固定的宏结构，每层可选择不同的block  
![img](https://raw.githubusercontent.com/terrencewayne/Paper-notes/master/images/fbnet0.png "block")  
分组卷积的时候使用channel shuffle  
2. 包含时延的loss  
<img src="https://latex.91maths.com/s/?JTVDdGV4dGl0JTdCTCU3RChhJTJDJTIwJTVDb21lZ2FfYSklM0RDRShhJTJDJTIwJTVDb21lZ2FfYSklNUNjZG90JTIwJTVDYWxwaGElMjBsb2coTEFUKCU1Q2FscGhhKSklNUUlN0IlNUNiZXRhJTdE" />  
<img src="https://latex.codecogs.com/gif.latex?\textit{L}(a,&space;\omega_a)=CE(a,&space;\omega_a)\cdot&space;\alpha&space;log(LAT(\alpha))^{\beta}" title="\textit{L}(a, \omega_a)=CE(a, \omega_a)\cdot \alpha log(LAT(\alpha))^{\beta}" />  

第二项(LAT(a))表示结构的时延。β调节大小幅度。  
使用查找表，来通过对每个操作求和来估计整体的时延  
<img src="https://latex.codecogs.com/gif.latex?LAT(a)=\sum&space;_lLAT(b_l^a)" title="LAT(a)=\sum _lLAT(b_l^a)" />  
b<sub>l</sub><sup>a</sup>表示a结构，第l层的block。这样做的好处是可导。  

3. 搜索算法  
构建一个宏结构的supernet。supernet前向时，对每一层，只有一个候选block被采样，采样依据  
<img src="https://latex.codecogs.com/gif.latex?P_{\theta_l}(b_l=b_{l,i})=softmax(\theta_{l,i};\theta_l)=\frac{exp(\theta_{l,i})}{\sum_iexp(\theta_{l,i})}" title="P_{\theta_l}(b_l=b_{l,i})=softmax(\theta_{l,i};\theta_l)=\frac{exp(\theta_{l,i})}{\sum_iexp(\theta_{l,i})}" />  
θ<sub>l</sub>包括确定l层每个block采样概率的参数。第l层的输出可以表示为  
<img src="https://latex.codecogs.com/gif.latex?x_{l&plus;1}=\sum_im_{l,i}\cdot&space;b_{l,i}(x_l)" title="x_{l+1}=\sum_im_{l,i}\cdot b_{l,i}(x_l)" />  
其中m<sub>l,i</sub>是0-1随机变量，如果被选中则为1.给定输入x<sub>l</sub>，l层第i个block的输出为b<sub>l,1</sub>(x<sub>l</sub>)。
让每层独立采样，采样出结构a的概率为  
<img src="https://latex.codecogs.com/gif.latex?P_\theta(a)=\prod&space;_lP_{\theta_l}(b_l=b_{l,i}^a)" title="P_\theta(a)=\prod _lP_{\theta_l}(b_l=b_{l,i}^a)" />  
θ是包含所有θ<sub>l,i</sub>的向量。  
不从离散的搜索空间A中找到最优结构a，我们把问题放松至最优化概率P<sub>θ</sub>来达到最小loss。然而，m<sub>l,i</sub>是离散不可导的，为了避免这一点，使用gumbel softmax函数  
<img src="https://latex.codecogs.com/gif.latex?m_{l,i}=GumbelSoftmax(\theta_{l,i}|\theta_{l})=\frac{exp[(\theta_{l,i}&plus;g_{l,i})/\tau]}{\sum_iexp[(\theta_{l,i}&plus;g_{l,i})/\tau]}" title="m_{l,i}=GumbelSoftmax(\theta_{l,i}|\theta_{l})=\frac{exp[(\theta_{l,i}+g_{l,i})/\tau]}{\sum_iexp[(\theta_{l,i}+g_{l,i})/\tau]}" />  
g<sub>l,i</sub>~Gumbel(0,1)是服从gumbel分布的噪声。这个函数收τ控制，当τ接近0，它近似依据P<sub>θ</sub>(a)分布进行离散类别采样。当τ增大，m<sub>l,i</sub>变成连续随机变量，无论τ的值，m对θ都是可导的。  
所以搜索过程等价于训练随机supernet，训练过程中，loss对ω<sub>a</sub>的导数训练每个操作的权重，当每个操作器经过训练后，整个网络中不同的操作器可能对效率和准确率有不同的贡献  
因此，计算loss对θ的导数来更新每个操作器的采样概率P<sub>θ</sub>，这一步选择更高效的操作器，惩罚相反的。super net训练完成后，依照P<sub>θ</sub>采样得到最优结构。  

4. 实验  
ImageNet：先使用80%的训练集SGD训练ω<sub>a</sub>，再用20%的训练集Adam训练θ，为了控制gumbel softmax函数，使用指数下降的τ。


