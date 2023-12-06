my implementation of NinNet (also from the d2l textbook)
# 完成Ninnet的实现
- Nin使用Nin块进行主要的模型设计
- 主要思想是使用1x1的卷积层取代全连接层
- 节省了很多参数的空间，可以防止过拟合
- 可能导致模型训练比较慢
# 主要结构图
![nin](https://github.com/blameitonme1/NinNet/assets/113235913/301aa609-0250-4576-9a1a-bb92785f3600)# NinNet
