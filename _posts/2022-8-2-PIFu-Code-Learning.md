### App

---

crop_img.py

* 处理预处理过的图片（有明显的边界线），生成mask，并对原图像和msk进行resize
* resize之后对msk进行了图像腐蚀，不是很明白为什么要这么做

---

eval.py

* 含Evaluator类，用于根据配置和训练好的模型，载入测试目录下的所有非mask图像并据此生成mesh

---

prt_util.py

* PRT（precomputed radiance transfer）相关的函数，与实时渲染有关

* $$
  factratio(N, D)=
  \begin{cases}
  (D+1) * (D+2) * ... *N, &\text{if N}\ge D\\
  \frac{1}{(N+1) * (N+2) *...*D}, &\text{if N}<D
  \end{cases}
  $$

* $$
  KVal(M, L)=\sqrt{\frac{(2L+1)}{4\pi}*factratio(L-M, L+M)}
  $$

* AssociatedLegendre 连带勒让德函数

* SphericalHarmonic 球谐函数 SH以及球采样

---

render_data.py

* 渲染相关的函数，包括旋转和prt正交渲染，没细看

---

train_color.py

* 训练netC（ResBlkPIFuNet）、netG（HGPIFuNet）
* 批量、学习率等超参数由opt控制（BaseOptions().parse())
* 用到的优化算法为Adam（还不了解）
* 由opt.freq_save控制保存的频率，根据这个设置checkpoints，训练开始的时候会先试图读取checkpoints
* 根据代码训练时还有保存.ply文件
* 每训练完一批量都会进行测试

---

train_shape.py

* 训练netG（HGPIFuNet）
* 优化算法为RMSprop
* 动态调整学习率

---

### lib

---

colab_util.py

* 感觉和我们关系不大，就没看

---

ext_transform.py

* 提供对图像的处理类

* RandomVerticalFlip：0.5的概率翻转     img.transpose(Image.FLIP_TOP_BOTTOM)
* DeNormalize： normalize的逆运算
* MaskToTensor： 好像就算把float转long
* FreeScale： 带插值的resize
* FlipChannels：图像的其余不变，channel逆序
* RandomGaussianBlur：返回经随机sigma模糊后的图像
* Lighting：看不懂

---

geometry.py

* 透视投影和正交投影
* index似乎是生成颜色？根据net_util.py
> **index** 从代码来看，index是uv贴图到mesh位置的映射。用处是把得到的颜色贴图通过位置映射得到具体每个点的颜色信息。

---

mesh_util.py

* reconstruction：通过训练好的网络查询采样点是否在面内外，通过marching_cube方法重构面
* save_obj_mesh、save_obj_mesh_with_color、save_obj_mesh_with_uv用于保存生成好的mesh
* save_obj_mesh_with_uv并没有被调用过，我试着调用了一下，好像并不能生成uv？

---

net_util.py

* reshape_multiview_tensors：转化tensor的维度，[B, num_views, C, W, H] --> [B*num_views, C, W, H]
* 这里似乎透露出B就是Batch
* gen_mesh
* compute_acc：计算IOU（重叠度）、precision、recall
* compute_err
* 初始化网络
  * 四种初始化方式：normal、xavier、kaiming、orthogonal
* 卷积
* cal_gradient_penalty：选择图片源（真实数据、虚假数据、真实虚假插值成的数据）计算梯度惩罚gp，不知道用来做什么，而且也没有调用该函数的地方。可能是想比较自己实现的梯度下降与torch内实现的梯度下降的差别？
* get_norm_layer：返回归一化的层？不理解是做什么的
* ConvBlock：一个激活函数用RELu，内含多个卷积网络的网络类

---

options.py

* BaseOptions类：含网络的各种超参数如num_views、gpu_id、num_threads等，具体看add_argument的--后面的内容。这些配置不是从文件中读取的，而是直接由default指定，或者传参指定

---

sample_util.py

* 以可视化的形式保存采样点的错误情况到.ply文件中，红为正确预测，绿为错误预测

---

sdf.py

* signed distance field：有向距离场。通过描述空间内任意一点到几何体表面的最小距离的空间几何体表达方式

* create_grid：根据指定的分辨率和BBox，生成该空间划分出的小格子和对应的变换矩阵。调用时的eval_func参数实际上就是由模型判断是否在几何体内。
* eval_grid：调用eval_func（也就是用模型判断）来估计grid是否在几何体内部

---

train_util.py

* 该文件内容与net_util.py一模一样

---

#### data

---

BaseDataset.py

* 继承自Dataset
* 初始化时为训练状态、正交投影
* 长度始终为0，不明白为什么
* 重载了[]操作符，返回一个字典，但字典内部的元素都为None
* 并没有地方用到这个类。。。

---

EvalDataset.py

* 也没有用到

---

TrainDataset.py

* 对图像的亮度、对比度、饱和度进行调节以生成更多的数据
* get_subjects：如果在训练阶段，返回还没有用于训练过的数据；如果不在，返回所有数据
* 采样方法

---

#### model

---

BasePIFuNet.py

* 没有实现filter和query的基类

---

ConvFilters.py

* 定义如下三个imagefilter
  * MultiConv：多个卷积组成的卷积
  * Vgg16：4、6、8、8、8的五层结构
  * ResNet：根据传入model字符串决定是resnet18还是34还是50

---

ConvPIFuNet.py

* 没有用到
* 继承自BasePIFuNet

---

DepthNormaizer.py

* 对z进行归一化的层

---

HGFilters.py

* HourGlass：3*level + 1层ConvBlock
* HGFilter：根据不同参数有不同的网络结构，且网络结构较复杂

---

HGPIFuNet.py

* HGPIFuNet：继承自BasePIFuNet，用到了HGFilter、SurfaceClassifier、DepthNormalizer

---

ResBlkPIFuNet.py

* ResBlkPIFuNet：继承自BasePIFuNet，用到了ResnetFilter、SurfaceClassifier、DeNormalizer
* ResnetBlock：比较复杂的一个网络结构
* ResnetFilter：含ResnetBlock的一个复杂网络结构

----

SurfaceClassifier.py

* 大量一维卷积组成的网络

---

VhullPIFuNet.py

* 仅用于展示和调试的简单网络

#### renderer

---

camera.py

* 与相机、视角相关的各种矩阵

---

glm.py

* 图形学中用到的简单的数学库，如vec3、叉乘、旋转、投影等。

---

mesh.py

* mesh相关的一系列函数
* 读写mesh、纹理
* 计算法向量

##### gl

---

调用OpenGL库进行渲染