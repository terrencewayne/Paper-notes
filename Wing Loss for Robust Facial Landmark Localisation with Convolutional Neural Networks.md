# Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks
Zhen-Hua Feng Josef Kittler Muhammad Awais Patrik Huber Xiao-Jun Wu
## Wing Loss
![image](https://raw.githubusercontent.com/terrencewayne/Paper-notes/master/images/wing0.png "wing loss")  
<img src="https://latex.codecogs.com/gif.latex?wing(x)&space;=&space;\left\{\begin{matrix}&space;\omegaln(1&plus;|x|/\varepsilon&space;)&space;&&space;if&space;|x|<&space;\omega\\&space;|x|-C&space;&otherwise&space;\end{matrix}\right." title="wing(x) = \left\{\begin{matrix} \omegaln(1+|x|/\varepsilon ) & if |x|< \omega\\ |x|-C &otherwise \end{matrix}\right." />  

L1/L2损失函数存在问题：其导数分别为1和x，优化步长是x和1，L1导数为常数，优化步长会被较大的error主导；L2的优化步长为常数，但梯度被较大的error主导，对小的error不友好
而wingloss的第一段ln函数对小error友好  
为了避免x较大时，其函数值过大，所以要将Wing loss分段，对于大的那一部分采用另外一个表达式，灵感来自于smoothL1

## two-stage的关键点检测
第一阶段层数浅，速度快，接收64x64x3的输入，第二阶段接收128x128x3的输入
