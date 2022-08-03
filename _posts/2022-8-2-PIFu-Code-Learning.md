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
* cal_gradient_penalty：选择图片源（真实数据、虚假数据、真实虚假插值成的数据）计算梯度惩罚gp，不知道用改做什么，而且也没有调用该函数的地方。可能是想必将自己实现的梯度下降与torch内实现的梯度下降的差别？
* get_norm_layer：返回归一化的层？不理解是做什么的
* ConvBlock：一个激活函数用RELu，内涵多个卷积网络的网络类

---

options.py

* BaseOptions类：含网络的各种超参数如num_views、gpu_id、num_threads等，具体看add_argument的--后面的内容。这些配置不是从文件中读取的，而是直接由default指定，或者传参指定

---

sample_util.py

---

sdf.py

---

train_util.py

---

#### data

---

BaseDataset.py

---

EvalDataset.py

---

TrainDataset.py

---

#### model

---

BasePIFuNet.py

---

ConvFilters.py

---

ConvPIFuNet.py

---

DepthNormaizer.py

---

HGFilters.py

---

HGPIFuNet.py

---

ResBlkPIFuNet.py

----

SurfaceClassifier.py

---

VhullPIFuNet.py

#### renderer

---

camera.py

---

glm.py

---

mesh.py

