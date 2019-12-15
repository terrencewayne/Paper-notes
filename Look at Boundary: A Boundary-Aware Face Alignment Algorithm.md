# Look at Boundary: A Boundary-Aware Face Alignment Algorithm
# 考虑边界信息的人脸对齐算法
Wayne Wu, Chen Qian, Shuo Yang, Quan Wang, Yici Cai, Qiang Zhou
## 摘要  
引入面部边缘线来辅助关键点定位，回答了三个问题：为什么使用边缘？怎么使用边缘？边界估计和关键点定位之间有什么联系？  
## 介绍
每一个关键点和脸部边缘是有很强的结构关系的，比如眼眶、鼻梁等等。但是相比较面部边缘，关键点并没有很好的定义。一方面，在特殊的角度、姿势下，被遮挡的点的标注没有很好的语义信息，另一方面，不同数据及对关键点的标注有所不同。  
所以考虑引入边缘信息，相比较关键点，检测边缘要容易得多。算法分为两步，首先预测面部边缘的热力图，然后在这个热力图的辅助下回归关键点。  
## 方法
![image](https://raw.githubusercontent.com/terrencewayne/Paper-notes/master/images/lab-0.png "pipeline")
