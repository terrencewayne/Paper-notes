# ML-SVM
## 支持向量

距离分类超平面最近的样本点。两个异类支持向量到超平面的距离之和为<img src="https://latex.codecogs.com/gif.latex?\gamma&space;=&space;\frac{2}{\left&space;\|&space;\omega&space;\right&space;\|}" title="\gamma = \frac{2}{\left \| \omega \right \|}" />
，称为间隔  

最大化间隔即最大化<img src="https://latex.codecogs.com/gif.latex?\left&space;\|&space;\omega&space;\right&space;\|^{-1}" title="\left \| \omega \right \|^{-1}" />
等价于最小化<img src="https://latex.codecogs.com/gif.latex?\left&space;\|&space;\omega&space;\right&space;\|^{2}" title="\left \| \omega \right \|^{2}" />，即  

<img src="https://latex.codecogs.com/gif.latex?\underset{\omega,b}{min}\frac{1}{2}\left&space;\|&space;\omega&space;\right&space;\|^2" title="\underset{\omega,b}{min}\frac{1}{2}\left \| \omega \right \|^2" />  
<img src="https://latex.codecogs.com/gif.latex?s.t.&space;y_i(\omega^Tx_i&plus;b)\geqslant&space;1,\quad&space;i=1,2,...,m." title="s.t. y_i(\omega^Tx_i+b)\geqslant 1,\quad i=1,2,...,m." />  


## KKT条件
<img src="https://latex.codecogs.com/gif.latex?\left\{\begin{matrix}&space;\alpha_i\geqslant&space;0\\&space;y_if(x_i)-1\geqslant&space;0\\&space;\alpha_i(y_if(x_i)-1)=0&space;\end{matrix}\right." title="\left\{\begin{matrix} \alpha_i\geqslant 0\\ y_if(x_i)-1\geqslant 0\\ \alpha_i(y_if(x_i)-1)=0 \end{matrix}\right." />  

## 核函数
**针对线性不可分样本**  
思路：将样本从原始空间映射到高维特征空间，使得样本在这个特征空间线性可分。  

模型表示为 <img src="https://latex.codecogs.com/gif.latex?f(x)&space;=&space;\omega^T\phi&space;(x)&plus;b" title="f(x) = \omega^T\phi (x)+b" />  

求解时定义 <img src="https://latex.codecogs.com/gif.latex?\kappa&space;(x_i,x_j)=\left&space;\langle&space;\phi&space;(x_i),\phi&space;(x_j)&space;\right&space;\rangle=\phi(x_i)^T\phi(x_j)" title="\kappa (x_i,x_j)=\left \langle \phi (x_i),\phi (x_j) \right \rangle=\phi(x_i)^T\phi(x_j)" />  
k(·)即为核函数。  

模型进一步可表示为  
<img src="https://latex.codecogs.com/gif.latex?f(x)=\omega^T\phi(x)&plus;b\\&space;=\sum_{i=1}^m\alpha_iy_i\phi(x_i)^T\phi(x)&plus;b\\&space;=\sum_{i=1}^m\alpha_iy_i\kappa(x,x_i)&plus;b" title="f(x)=\omega^T\phi(x)+b\\ =\sum_{i=1}^m\alpha_iy_i\phi(x_i)^T\phi(x)+b\\ =\sum_{i=1}^m\alpha_iy_i\kappa(x,x_i)+b" />  

常用的核函数有线性核、多项式核、高斯核、拉普拉斯核、Sigmoid核等。核函数的线性组合、直积也是核函数，<img src="https://latex.codecogs.com/gif.latex?\kappa(x,z)=g(x)\kappa_1(x,z)g(z)" title="\kappa(x,z)=g(x)\kappa_1(x,z)g(z)" /> 也是核函数  
