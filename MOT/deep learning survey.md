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
·MOTP：<img src="https://latex.codecogs.com/svg.image?MOTP&space;=&space;\frac{\sum_{t,i}d_{t,i}}{\sum_tc_t}" title="MOTP = \frac{\sum_{t,i}d_{t,i}}{\sum_tc_t}" />，c_t指t帧的匹配的数量，d为检测目标i和给它分配的ground truth之间在所有帧中的平均度量距离，在这里是使用bonding box的overlap rate来进行度量（在这里MOTP是越大越好，但对于使用欧氏距离进行度量的就是MOTP越小越好，这主要取决于度量距离d的定义方式），这个指标更关注检测性能而非跟踪。  
### ID scores
不重要
### 数据集
#### MOTChallenge
MOTChallenge是最常用的数据集。它提供了最大的行人跟踪公开数据集。它的每个数据集提供训练集的gt，训练集和测试集的检测框（为了消除检测器性能的影响）  
 ·MOT15. 22条视频（训练、检测各一半），总共包含各种分辨率的11283帧图像，1221个目标id和101345个框。其检测框由ACF detector给出  
 ·MOT16/17. 14条视频（训练、检测各一半），总共包括11235帧，1342个目标id和292733个框。其检测框由Deformable Part-based Model (DPM) v5给出。  
 ·MOT19（MOT20）. 8条视频（训练、检测各一半），平均每帧有245个人，总共包括13410帧，6869条轨迹，2259143个框。其检测框由Faster-RCNN w ResNet-101给出  
 #### KITTI
 KITTI关注人和车。包括21训练视频和29测试视频，总共大约有19000帧，其检测框由DPM和RegionLets给出。
 #### 其他数据集
 UA-DETRAC、TUD、PETS2009等等老数据集，其中一些是MOTChallenge的一部分。
 ## MOT中的深度学习
 文章的Appendix A提供了一个表格，总结了各方法在4个stage中使用了什么样的深度学习方法。
 ### detection step的深度学习（略）
 ### feature extraction和motion prediction中的深度学习
 最常用的方法是使用CNN提取视觉表征。此外，也有使用对比损失函数寻找能够区分目标的最好的feature，也有使用CNN来预测目标的运动
 #### autoencoder
 使用2层的autoencoder来得到自然场景下得到的视觉特征，然后使用SVM进行affinity计算，关联任务被建模为最小生成树算法。实验证明提升比较大，但数据集不常用，难以比较
 #### CNN提取视觉表征
 reid特征或者分类任务的特征，结合卡尔曼滤波器的运动特征。DeepSORT使用余弦距离。  
 [a] 使用使用判别器评估feature之间的关系，其得分结合时空关系得分，最终得分用作高斯混合概率假设密度滤波器中的似然。  
 [b] 复用了检测器的特征，按照反向最近邻关联  
 [c] 区分高速细胞和低速细胞，低速细胞仅使用运动特征，高速细胞结合视觉特征和运动特征。同时包含优化跟踪的过程，通过组合误中断的track来减少FP和FN  
 [d] 使用CNN和AlphaPose提取视觉特征和姿态特征，结合tracklet历史信息送入LSTM来计算相似度  
   
 [a] Z. Fu, F. Angelini, S. M. Naqvi, J. A. Chambers, Gm-phd filter based online multiple human tracking using deep discriminative correlation
matching, in: 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), IEEE, 2018, pp. 4299–4303.  
[b] F. Pernici, F. Bartoli, M. Bruni, A. Del Bimbo, Memory based online learning of deep representations from video streams, in: Proceedings of
the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 2324–2334  
 [c] H. Hu, L. Zhou, Q. Guan, Q. Zhou, S. Chen, An automatic tracking method for multiple cells based on multi-feature fusion, IEEE Access 6
(2018) 69782–69793.  
[d] N. Ran, L. Kong, Y. Wang, Q. Liu, A robust multi-athlete tracking algorithm by exploiting discriminant features and long-term dependencies,
in: International Conference on Multimedia Modeling, Springer, 2019, pp. 411–423.  
#### 孪生网络
使用对比学习训练  
[a]  设计了新的CNN结构Quad-CNN，接受4个image patches作为输入，前三个来自同一个目标，时间上升序，后一个来自另外的目标。网络使用自定义的loss训练，结合detection之间的时序距离信息，视觉特征和bbox位置。测试阶段接收两个detection，预测两者属于同一目标的概率  
[b] 向网络输入三个样本，两个来自相同目标（positive pair），一个来自另外的样本（negtiave pair），使用triplet loss训练。推理阶段，基于视觉表征的相似度同运动稳定性和潜在空间位置结合，运动稳定性基于预测下一个位置的得分，假设是线性运动，关联匹配通过计算3维张量的相似度解决  

[a]  J. Son, M. Baek, M. Cho, B. Han, Multi-object tracking with quadruplet convolutional neural networks, in: Proceedings of the IEEE
conference on computer vision and pattern recognition, 2017, pp. 5620–5629  
[b] Z. Zhou, J. Xing, M. Zhang, W. Hu, Online multi-target tracking with tensor-based high-order graph matching, in: 2018 24th International
Conference on Pattern Recognition (ICPR), IEEE, 2018, pp. 1809–1814.  
#### 其他复杂方式
[a] 将检测器的分类结果和ROI pooling结果结合在一起作为特征，送入LSTM去学习detections之间的联系  
[b] 使用3个RNN，第一个reid，第二个预测轨迹，第三个学习同一场景下不同目标的交互，最后这三个RNN的输出作为一个LSTM的输入进行affinity计算。  
[c] 为每条track训练一个CNN, [d] 为每条track训练一个random ferns分类器，区分同目标和其他目标  
[e] 使用隐马尔科夫模型计算motion

[a] Y. Lu, C. Lu, C.-K. Tang, Online video object detection using association lstm, in: Proceedings of the IEEE International Conference on
Computer Vision, 2017, pp. 2344–2352.  
[b] A. Sadeghian, A. Alahi, S. Savarese, Tracking the untrackable: Learning to track multiple cues with long-term dependencies, in: Proceedings
of the IEEE International Conference on Computer Vision, 2017, pp. 300–311.  
[c] Q. Chu, W. Ouyang, H. Li, X. Wang, B. Liu, N. Yu, Online multi-object tracking using cnn-based single object tracker with spatial-temporal
attention mechanism, in: Proceedings of the IEEE International Conference on Computer Vision, 2017, pp. 4836–4845.  
[d] S. J. Kim, J.-Y. Nam, B. C. Ko, Online tracker optimization for multi-pedestrian tracking using a moving vehicle camera, IEEE Access 6
(2018) 48675–48687.  
[e] M. Ullah, F. Alaya Cheikh, A directed sparse graphical model for multi-target tracking, in: Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition Workshops, 2018, pp. 1816–1823.
#### CNN预测motion：相关滤波器
没啥东西
### Affinity计算中的深度学习
#### RNN与LSTM
用LSTM计算tracklets和detections之间的affinity score，输入VGG16特征等
#### MHT框架下使用LSTM
在多假设跟踪方法（MHT）中，首先为每个候选目标的潜在track假设建立一棵树，然后计算每条track的似然，具有最大似然的track组合作为解。这些方法大多使用LSTM来对树进行操作
#### 其它循环网络
[a] 使用两层MLP来计算tracklet和detection之间的得分

[a] H. Kieritz, W. Hubner, M. Arens, Joint detection and online multi-object tracking, in: Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition Workshops, 2018, pp. 1459–1467.  
#### CNN计算affinity得分
[a] 把关联任务建模为最小切割问题：可被视为图聚类问题，每一个簇代表一个被跟踪的目标。边的代价则是两个detection之间的相似度。此相似度是reid/相关匹配/时空关系的组合。另一特点是其reid用两张图片及其pose分割的相应图作为输入，输出真或假  
[b] 输出detection与轨迹预测box的affinity得分，轨迹预测box由深度连续条件随机场得到。最高得分的detection连到tracklet上，如果发生冲突则使用匈牙利匹配。  


[a] S. Tang, M. Andriluka, B. Andres, B. Schiele, Multiple people tracking by lifted multicut and person re-identification, in: Proceedings of the
IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 3539–3548.  
[b] H. Zhou, W. Ouyang, J. Cheng, X. Wang, H. Li, Deep continuous conditional random fields with asymmetric inter-object constraints for
online multi-object tracking, IEEE Transactions on Circuits and Systems for Video Technology.  
#### 孪生CNN
[a] 在解决短track连成长track的过程中，使用孪生网络来计算代价。其输出两幅图片的相似度、以及额外的两层去预测身份。在测试序列上进一步进行无监督训练，从local短track中选择正负样本对

[a] L. Ma, S. Tang, M. J. Black, L. V. Gool, Customized multi-person tracker, in: Computer Vision – ACCV 2018, Springer International
Publishing, 2018.  
### Association中的深度学习
#### RNN
[a] 使用RNN来预测每个track在每帧中的存在性，帮助决策什么时候开始和结束
[b] 使用GRU为断开的tracklet提取特征，通过特征距离决定是否连接两条track

[a] A. Milan, S. H. Rezatofighi, A. Dick, I. Reid, K. Schindler, Online multi-target tracking using recurrent neural networks, in: Thirty-First
AAAI Conference on Artificial Intelligence, 2017.  
[b] C. Ma, C. Yang, F. Yang, Y. Zhuang, Z. Zhang, H. Jia, X. Xie, Trajectory factory: Tracklet cleaving and re-connection by deep siamese bi-gru
for multiple object tracking, in: 2018 IEEE International Conference on Multimedia and Expo (ICME), IEEE, 2018, pp. 1–6.  
#### MLP
没啥东西
#### 强化学习
[a] 使用多个深度RL代理来管理多个跟踪目标，决定何时开始和终止track以及影响卡尔曼滤波器的决策，这些代理使用3层隐藏层的MLP实现  
[b] 使用合作环境下的多个深度RL agents来管理。主要由两部分组成：预测网络和决策网络。预测网路接收新图片、目标、目标最近轨迹信息，预测目标在新一帧的运动。决策网络包括每个目标的代理，每个agent基于自身，邻居和环境做出决策，agents与环境的交互由最大化效益函数来实现。每个agent由trajectory，表观特征和当前位置实现，环境由新一帧的detections实现。决策网路的输入包括每个track目标的预测位置，最近邻目标和最近邻检测，根据检测的可靠性和目标的遮挡状态来采取如下决策之一：使用prediction和detection更新轨迹和表观特征、忽略检测仅使用预测来更新track、检测到遮挡或者删除track。agents使用MDNet特抽取特征部分加上三层卷积实现，实验证明比线性运动模型和匈牙利匹配效果好，但ids比较大

[a] P. Rosello, M. J. Kochenderfer, Multi-agent reinforcement learning for multi-object tracking, in: Proceedings of the 17th International
Conference on Autonomous Agents and MultiAgent Systems, International Foundation for Autonomous Agents and Multiagent Systems,
2018, pp. 1397–1404.  
[b] L. Ren, J. Lu, Z. Wang, Q. Tian, J. Zhou, Collaborative deep reinforcement learning for multi-object tracking, in: Proceedings of the European
Conference on Computer Vision (ECCV), 2018, pp. 586–602.  
