the first attempt to infer light occlusion in the SH formulation directly

infer not only an albedo map, illumination but also a light transport map that encodes occlusion as nine SH coefficients per pixel.

##### architecture

![image-20220819093118355](https://chaosharmonix.github.io\Notebook\assets\images\image-20220819093118355.png)

为什么light transport map $\in \R^{NX9}$,N是pixel数量

15 L1 loss functions??

light transport map是什么

和SfSNet很接近

encoder有六层卷积，输出为{64, 128, 256, 512, 512, 512}，步幅stride为2

Each decoder has a residual block (consisting of two convolutional layers with 512 channels) and six deconvolutional layers (output channels are { 512, 512, 256, 128, 64, 9 or 3 } and the stride is also two).

Encoder、Albedo decoder的ResNet block、Transport decoder的ResNet block的结果连接起来传入四层卷积中，产生27维向量。

每个卷积层和反卷积层（除了第一个和最后一个）后都要做batch normalization和(leaky) ReLU

前三个反卷积层dropout概率为0.5

optimizer用Adam

lr为0.0002

batch size 1??

GTX 1080上跑一个epoch需要三个小时。。。，共60个epoch

##### 数据集

###### Synthetic human image dataset

一部分来源于公开的BUFF，一部分买来。

从BUFF中挑出来74个代表性的模型

买了271个模型

276用于训练，69进行测试

aligned 3D models(脸朝前，垂直方向大小一致，padding一致（上下padding都为图片的5%）)。

只用了战力的人物

图片分辨率1024*1024

###### Illumination dataset

Laval Indoor HDR dataset

![image-20220819100442181](https://chaosharmonix.github.io\Notebook\assets\images\image-20220819100442181.png)