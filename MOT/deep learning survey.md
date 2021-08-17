# Deep Learning in Video Multi-Object Tracking: A Survey
Gioele Ciaparrone, Francisco Luque Sánchez, Siham Tabik,
Luigi Troiano, Roberto Tagliaferri, Francisco Herrera
## Introduction
这篇论文为深度学习模型在MOT中的应用做了相应综述。分为下面几个方面：
1. 描述了MOT算法的通用架构、常用指标和数据集
2. 介绍MOT在4个步骤中使用的各种深度学习模型和算法
3. 比较当前各算法的数值性能，梳理共同的趋势、模式，以及局限性和可能的研究方向
## 算法、指标和数据集
### 算法简介
MOT的标准方法是tracking by detection：从视频帧得到的检测boxes被用来描述tracking过程，通常是把他们关联起来，将同一个目标的框给予相同的id。需要注意的是，检测器性能会在很大程度上影响跟踪的结果。  
MOT可以分为batch和online方法。batch方法在决定当前帧中框的归属时允许使用未来信息。oneline方法则只能使用当前和过去的信息——在自动驾驶、机器人导航等场景中，这是必要的。但需要指出的是，大部分online算法并不能达到实时的速度。大部分的MOT算法可分为下面四个步骤，这些stages可以顺序执行，也可以合并起来，异步执行等等。
1. Detection stage：检测出框
2. Feature extraction/motion prediction stage：分析detections/tracklets，提取motion/interaction特征，有时会预测目标的下一个位置
3. Affinity stage：利用上述特征，计算detection pair或tracklet pair之间的距离/相似度
4. Association stage：把相同目标的detections和tracklets关联起来，分配相同的id
### 指标
#### 经典指标
·MT：Mostly Tracked trajectories。在至少80%的帧上被正确跟踪的gt trajectory数量  
·Fragments：覆盖gt trajectory超过80%的trajectory prediction。一个gt可能被多个prediction覆盖  
·ML：Mostly Lost trajectorise。在少于20%的帧上被正确跟踪的gt trajectory数量  
·False trajectories：不对应gt的trajectory  
·ID switches：目标被正确跟踪，但是其id发生改变的次数  
#### CLEAR MOT
·FP、FN  
·Fragm：fragmentation的数量。fragmentation指一个正确的tracking中断又恢复  
·IDSW：ID switch的数量。  
·MOTA：<img src="https://latex.codecogs.com/svg.image?MOTA=1-\frac{FN&plus;FP&plus;IDSW}{GT}&space;\quad&space;\in&space;(-\infty,1]" title="MOTA=1-\frac{FN+FP+IDSW}{GT} \quad \in (-\infty,1]" />  
·MOTP：<img src="https://latex.codecogs.com/svg.image?MOTP&space;=&space;\frac{\sum_{t,i}d_{t,i}}{\sum_tc_t}" title="MOTP = \frac{\sum_{t,i}d_{t,i}}{\sum_tc_t}" />，c_t指t帧的匹配的数量，d_t,i指正确的匹配，这个指标更关注检测性能而非跟踪。  
### ID scores
