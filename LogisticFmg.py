import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## 多元线性回归

# 从CSV文件中读取数据，并返回2个数组。分别是自变量x和因变量y。方便TF计算模型。
def zc_read_csv():
    zc_dataframe = pd.read_csv("data_fmg.csv", sep=",")
    x = []
    y = []
    for zc_index in zc_dataframe.index:
        zc_row = zc_dataframe.loc[zc_index]
        x.append(zc_row["m"])
        y.append(zc_row["F"])
    return (x,y)

x, y = zc_read_csv()

zc_x = []
for item in x:
    zc_x.append([1., item])
zc_x4tf = np.array(zc_x).astype(np.float32)

zc_y = []
for item in y:
    zc_y.append([item])
zc_y4tf = np.array(zc_y).astype(np.float32)

# 存放 L2 损失的数组
loss_arr = []
# 训练的步数。即训练的迭代次数。
training_steps = 55
# 在梯度下降算法中，控制梯度步长的大小。
learning_rate = 0.01

# 开启TF会话，在TF 会话的上下文中进行 TF 的操作。
with tf.Session() as sess:
    # 设置 tf 张量（tensor）。注意：TF会话中的注释里面提到的常量和变量是针对TF设置而言，不是python语法。

    # 因为在TF运算过程中，x作为特征值，y作为标签
    # 是不会改变的，所以分别设置成input 和 target 两个常量。
    # 这是 x 取值的张量。设一共有m条数据，可以把input理解成是一个m行2列的矩阵。矩阵第一列都是1，第二列是x取值。
    input = tf.constant(zc_x4tf)
    # 设置 y 取值的张量。target可以被理解成是一个m行1列的矩阵。 有些文章称target为标签。
    target = tf.constant(zc_y4tf)

    # 设置权重变量。因为在每次训练中，都要改变权重，来寻找L2损失最小的权重，所以权重是变量。
    # 可以把权重理解成一个2行1列的矩阵。初始值是随机的。[2,1] 表示2行1列。
    weights = tf.Variable(tf.random_normal([2, 1], 0, 0.1))

    # 初始化上面所有的 TF 常量和变量。
    tf.global_variables_initializer().run()
    # input 作为特征值和权重做矩阵乘法。m行2列矩阵乘以2行1列矩阵，得到m行1列矩阵。
    # yhat是新矩阵，yhat中的每个数 yhat' = w0 * 1 + w1 * x。 
    # yhat是预测值，随着每次TF调整权重，yhat都会变化。
    yhat = tf.matmul(input, weights)
    # tf.subtract计算两个张量相减，当然两个张量必须形状一样。 即 yhat - target。
    yerror = tf.subtract(yhat, target)
    # 计算L2损失，也就是方差。
    loss = tf.nn.l2_loss(yerror)
    # 梯度下降算法。
    zc_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # 注意：为了安全起见，我们还会通过 clip_gradients_by_norm 将梯度裁剪应用到我们的优化器。
    # 梯度裁剪可确保梯度大小在训练期间不会变得过大，梯度过大会导致梯度下降法失败。
    zc_optimizer = tf.contrib.estimator.clip_gradients_by_norm(zc_optimizer, 5.0)
    zc_optimizer = zc_optimizer.minimize(loss)
    for _ in range(training_steps):
        # 重复执行梯度下降算法，更新权重数值，找到最合适的权重数值。
        sess.run(zc_optimizer)
#         print(weights.eval())
        # 每次循环都记录下损失loss的值，病放到数组loss_arr中。
        loss_arr.append(loss.eval())
    zc_weight_arr = weights.eval()
    zc_yhat = yhat.eval()

print("weights", zc_weight_arr)


# 画出原始数据的散点图和数学模型的直线图。
def paint_module(fig):
    ax = fig.add_subplot(1, 2, 1)  # 整个绘图区分成一行两列，当前图是第一个。
    # 画出原始数据的散点图。
    ax.set_title("计算 力（F）与 质量（m）之间的关系")
    ax.set_xlabel("m")
    ax.set_ylabel("F")
    ax.scatter(x, y)
    # 画出预测值的散点图。
    p_yhat = [a[0] for a in zc_yhat]
    ax.scatter(x, p_yhat, c="red", alpha=.6)
    # 画出线性回归计算出的直线模型。
    line_x_arr = [1, 40]
    line_y_arr = []
    for item in line_x_arr:
        line_y_arr.append(zc_weight_arr[0] + zc_weight_arr[1] * item)
    ax.plot(line_x_arr, line_y_arr, "g", alpha=0.6)

# 画出训练过程中的损失变化
def paint_loss(fig):
    print("loss", loss_arr)
    ax = fig.add_subplot(1, 2, 2)  # 整个绘图区分成一行两列，当前图是第二个。
    ax.plot(range(0, training_steps), loss_arr)


# 获得画图对象。
fig = plt.figure()
fig.set_size_inches(10, 5)   # 整个绘图区域的宽度10和高度4
paint_module(fig)
paint_loss(fig)

plt.show()
plt.savefig
