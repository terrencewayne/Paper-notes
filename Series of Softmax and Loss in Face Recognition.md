# Series of Softmax and Loss in Face Recognition
## 1. Softmax Loss
softmax归一化到[0, 1]区间，再做交叉熵。  

### softmax  
<img src="https://latex.codecogs.com/gif.latex?f\left&space;(&space;v_k&space;\right&space;)=\frac{e^{v_k}}{\sum&space;_je^{v_j}}" title="f\left ( v_k \right )=\frac{e^{v_k}}{\sum _je^{v_j}}" />  

### Cross Entropy Loss  
由KL散度（相对熵）出发  

<img src="https://latex.codecogs.com/gif.latex?L_{KL}(p||q)=\sum_{i=1}^{N}p\left&space;(&space;x_i&space;\right&space;)log\left&space;(&space;\frac{p\left&space;(&space;x_i&space;\right&space;)}{q\left&space;(&space;x_i&space;\right&space;)}&space;\right&space;)=\sum_{i=1}^{N}p\left&space;(&space;x_i&space;\right&space;)\left&space;(&space;log(p(x_i))-log(q(x_i))&space;\right&space;)" title="L_{KL}(p||q)=\sum_{i=1}^{N}p\left ( x_i \right )log\left ( \frac{p\left ( x_i \right )}{q\left ( x_i \right )} \right )=\sum_{i=1}^{N}p\left ( x_i \right )\left ( log(p(x_i))-log(q(x_i)) \right )" />  

*q*是网络输出的分布，*p*是ground truth分布（one-hot），因此进一步写作  

<img src="https://latex.codecogs.com/gif.latex?L_{KL}(p||q)=-log(q(x_{y}))" title="L_{KL}(p||q)=-log(q(x_{y}))" />  

*y*是ground truth的index，这就是一个样本的交叉熵。  

把softmax的输出代入，得到  

<img src="https://latex.codecogs.com/gif.latex?L_{softmax}(y,v)=-log(f(v_{y}))=-log(\frac{e^{v_y}}{\sum_je^{v_j}})" title="L_{softmax}(y,v)=-log(f(v_{y}))=-log(\frac{e^{v_y}}{\sum_je^{v_j}})" />  

## 2. Large-Margin Softmax Loss (L-Softmax Loss)
Softmax Loss写作  

<img src="https://latex.codecogs.com/gif.latex?L_i=-log\left&space;(&space;\frac{e^{||W_{y_i}||\cdot||x_i||cos(\theta&space;_{y_i})}}{\sum_je^{||W_{y_i}||\cdot||x_i||cos(\theta&space;_j)}}&space;\right&space;)\qquad&space;where\quad0\leq&space;\theta_j\leq\pi" title="L_i=-log\left ( \frac{e^{||W_{y_i}||\cdot||x_i||cos(\theta _{y_i})}}{\sum_je^{||W_{y_i}||\cdot||x_i||cos(\theta _j)}} \right )\qquad where\quad0\leq \theta_j\leq\pi" />  

以2分类为例，<img src="https://latex.codecogs.com/gif.latex?v_k=W_k\cdot&space;X" title="v_k=W_k\cdot X" />是全连接层对第k类的输出。  
对于第一类的样本，我们希望  

<img src="https://latex.codecogs.com/gif.latex?v_1>&space;v_2\quad\rightarrow&space;\quad&space;W_1\cdot&space;X>W_2\cdot&space;X&space;\quad&space;\rightarrow&space;\quad&space;||W_1||\cdot||X||cos\theta_1>||W_2||\cdot||X||cos\theta_2" title="v_1> v_2\quad\rightarrow \quad W_1\cdot X>W_2\cdot X \quad \rightarrow \quad ||W_1||\cdot||X||cos\theta_1>||W_2||\cdot||X||cos\theta_2" />  

这就是softmax的思路，如果把这个要求设置的更加严格  

<img src="https://latex.codecogs.com/gif.latex?||W_1||\cdot||X||cosm\theta_1>||W_2||\cdot||X||cos\theta_2&space;\quad&space;m\geq&space;1,0\leq\theta_j\leq\pi/m" title="||W_1||\cdot||X||cosm\theta_1>||W_2||\cdot||X||cos\theta_2 \quad m\geq 1,0\leq\theta_j\leq\pi/m" />  

![img](https://pic1.zhimg.com/80/v2-78507c4cb3c5532e8eebd74cd4b15f68_hd.jpg "L-softmax")  
如图所示，这样从一个决策边界变成了两个决策边界，类间距离拉大，类内距离减小，样本分类得更好。  
L-Softmax可以这样表示  

<img src="https://latex.codecogs.com/gif.latex?L_i=-log\left(\frac{e^{||W_{y_i}||\cdot||x_i||\psi(\theta&space;_{y_i})}}{e^{||W_{y_i}||\cdot||x_i||\psi(\theta&space;_{y_i})}&plus;\sum_{j\neq&space;y_i}e^{||W_{y_i}||\cdot||x_i||cos(\theta&space;_j)}}&space;\right&space;)" title="L_i=-log\left(\frac{e^{||W_{y_i}||\cdot||x_i||\psi(\theta _{y_i})}}{e^{||W_{y_i}||\cdot||x_i||\psi(\theta _{y_i})}+\sum_{j\neq y_i}e^{||W_{y_i}||\cdot||x_i||cos(\theta _j)}} \right )" />  

<img src="https://latex.codecogs.com/gif.latex?where&space;\quad&space;\psi&space;(\theta)=\left\{\begin{matrix}&space;cos(m\theta),0\leq\theta\leq\pi/m\\&space;D(\theta),\pi/m<\theta\leq\pi&space;\end{matrix}\right." title="where \quad \psi (\theta)=\left\{\begin{matrix} cos(m\theta),0\leq\theta\leq\pi/m\\ D(\theta),\pi/m<\theta\leq\pi \end{matrix}\right." />  

## 3. Angular softmax Loss (A-Softmax Loss、SphereFace)  
对L-softmax做出修改，令||W||=1,b=0，使得预测仅取决于W和X之间的角度。  
为了移除 0≤θ≤π/m 的约束，  

<img src="https://latex.codecogs.com/gif.latex?\psi(\theta)=(-1)^kcos(m\theta_{y_i})-2k&space;\quad&space;where&space;\quad&space;\theta_{y_i}\in&space;\left&space;[&space;\frac{k\pi}{m},\frac{(k&plus;1)\pi}{m}&space;\right&space;]" title="\psi(\theta)=(-1)^kcos(m\theta_{y_i})-2k \quad where \quad \theta_{y_i}\in \left [ \frac{k\pi}{m},\frac{(k+1)\pi}{m} \right ]" />  

## 4. NormFace  
将特征归一化，归一化后L2距离和余弦距离是等价的，使得人脸特征更加集中在夹角上。  
将特征x做L2归一化，再rescale到s，这就是超球面的半径。  
## 5. CosFace
CosFace写作  

<img src="https://latex.codecogs.com/gif.latex?L_{cosface}=-\frac{1}{m}\sum_{i=1}^{m}log\frac{e^{s(cos\theta_{y_i}-m)}}{e^{s(cos\theta_{y_i}-m)}&plus;\sum&space;_{j=1,j\neq&space;y_i}^ne^{scos\theta_j}}" title="L_{cosface}=-\frac{1}{m}\sum_{i=1}^{m}log\frac{e^{s(cos\theta_{y_i}-m)}}{e^{s(cos\theta_{y_i}-m)}+\sum _{j=1,j\neq y_i}^ne^{scos\theta_j}}" />   

角度边际cosmθ → 余弦边际 cosθ-m，有三个优势：1.不需要超参数就可实现 2.更清晰，不需要softmax的辅助 3.性能有提升  
## 6.ArcFace  
ArcFace写作  

<img src="https://latex.codecogs.com/gif.latex?L_{arcface}=-\frac{1}{m}\sum_{i=1}^{m}log\frac{e^{s(cos(\theta_{y_i}&plus;m))}}{e^{s(cos(\theta_{y_i}&plus;m))}&plus;\sum&space;_{j=1,j\neq&space;y_i}^ne^{scos\theta_j}}" title="L_{arcface}=-\frac{1}{m}\sum_{i=1}^{m}log\frac{e^{s(cos(\theta_{y_i}+m))}}{e^{s(cos(\theta_{y_i}+m))}+\sum _{j=1,j\neq y_i}^ne^{scos\theta_j}}" />  

对特征和W的夹角施加惩罚项，而不是对余弦施加惩罚项。有更好的几何解释：在超球面上分类实际上是用角度距离把类别分割开，如果让m和cos值做加减实际上是余弦距离，不如和角度值加减那样直接。

## 7.Combined Margin
直接上公式  

<img src="https://latex.codecogs.com/gif.latex?L_{arcface}=-\frac{1}{m}\sum_{i=1}^{m}log\frac{e^{s(cos(m_1\theta_{y_i}&plus;m_2)-m_3)}}{e^{s(cos(m_1\theta_{y_i}&plus;m_2)-m_3)}&plus;\sum&space;_{j=1,j\neq&space;y_i}^ne^{scos\theta_j}}" title="L_{arcface}=-\frac{1}{m}\sum_{i=1}^{m}log\frac{e^{s(cos(m_1\theta_{y_i}+m_2)-m_3)}}{e^{s(cos(m_1\theta_{y_i}+m_2)-m_3)}+\sum _{j=1,j\neq y_i}^ne^{scos\theta_j}}" />  

这个应该是把理论推到极致了，结合了SphereFace/CosFace/ArcFace。  

## 几点思考
1. Large Margin的loss是显示约束类内距离，让x更靠近W，而不是显示约束类间距离。
2. 训练难度 mθ > cosθ-m > cos(θ+m)，我认为这和m的调节尺度和超球面的分类夹角尺度有关。乘性的m，微小的变化会引起角度的大变化，余弦的变化所对应的角度变化也要大，多以角度空间的夹角是最好调节的。


