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
L-Softmax Loss写作  

<img src="https://latex.codecogs.com/gif.latex?L_i=-log\left&space;(&space;\frac{e^{||W_{y_i}||\cdot||x_i||cos(\theta&space;_{y_i})}}{\sum_je^{||W_{y_i}||\cdot||x_i||cos(\theta&space;_j)}}&space;\right&space;)\qquad&space;where\quad0\leq&space;\theta_j\leq\pi" title="L_i=-log\left ( \frac{e^{||W_{y_i}||\cdot||x_i||cos(\theta _{y_i})}}{\sum_je^{||W_{y_i}||\cdot||x_i||cos(\theta _j)}} \right )\qquad where\quad0\leq \theta_j\leq\pi" />  

以2分类为例，<img src="https://latex.codecogs.com/gif.latex?v_k=W_k\cdot&space;X" title="v_k=W_k\cdot X" />是全连接层对第k类的输出。  
对于第一类的样本，我们希望  

<img src="https://latex.codecogs.com/gif.latex?v_1>&space;v_2\quad\rightarrow&space;\quad&space;W_1\cdot&space;X>W_2\cdot&space;X&space;\quad&space;\rightarrow&space;\quad&space;||W_1||\cdot||X||cos\theta_1>||W_2||\cdot||X||cos\theta_2" title="v_1> v_2\quad\rightarrow \quad W_1\cdot X>W_2\cdot X \quad \rightarrow \quad ||W_1||\cdot||X||cos\theta_1>||W_2||\cdot||X||cos\theta_2" />  

这就是softmax的思路，如果把这个要求设置的更加严格  

<img src="https://latex.codecogs.com/gif.latex?||W_1||\cdot||X||cosm\theta_1>||W_2||\cdot||X||cos\theta_2&space;\quad&space;m\geq&space;1,0\leq\theta_j\leq\pi/m" title="||W_1||\cdot||X||cosm\theta_1>||W_2||\cdot||X||cos\theta_2 \quad m\geq 1,0\leq\theta_j\leq\pi/m" />  

![img](https://pic1.zhimg.com/80/v2-78507c4cb3c5532e8eebd74cd4b15f68_hd.jpg "L-softmax")  
如图所示，这样从一个决策边界变成了两个决策边界，类间距离拉大，类内距离减小，样本分类得更好。  
L-Softmax更具体的定义这样表示  

<img src="https://latex.codecogs.com/gif.latex?L_i=-log\left(\frac{e^{||W_{y_i}||\cdot||x_i||\psi(\theta&space;_{y_i})}}{e^{||W_{y_i}||\cdot||x_i||\psi(\theta&space;_{y_i})}&plus;\sum_{j\neq&space;y_i}e^{||W_{y_i}||\cdot||x_i||cos(\theta&space;_j)}}&space;\right&space;)" title="L_i=-log\left(\frac{e^{||W_{y_i}||\cdot||x_i||\psi(\theta _{y_i})}}{e^{||W_{y_i}||\cdot||x_i||\psi(\theta _{y_i})}+\sum_{j\neq y_i}e^{||W_{y_i}||\cdot||x_i||cos(\theta _j)}} \right )" />  

<img src="https://latex.codecogs.com/gif.latex?where&space;\quad&space;\psi&space;(\theta)=\left\{\begin{matrix}&space;cos(m\theta),0\leq\theta\leq\pi/m\\&space;D(\theta),\pi/m<\theta\leq\pi&space;\end{matrix}\right." title="where \quad \psi (\theta)=\left\{\begin{matrix} cos(m\theta),0\leq\theta\leq\pi/m\\ D(\theta),\pi/m<\theta\leq\pi \end{matrix}\right." />  

## 3. Angular softmax Loss (A-Softmax Loss)  


