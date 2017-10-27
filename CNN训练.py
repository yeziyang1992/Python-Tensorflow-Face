import tensorflow as tf
import cv2
import numpy as np
import os
import random
import sys
import one_hot
from sklearn.model_selection import train_test_split


my_faces_path = './faces/'
size = 64

np_img = []
np_lab = []
lab_name = []
lab_full = []
num_tot = None


def get_padding_size(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0, 0, 0, 0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        # //表示整除符号
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right


def read_data(path, h=size, w=size):
    for filename in os.listdir(path):

        for img_name in os.listdir(path+filename):
            if img_name.endswith('.jpg'):
                path_name = path+filename + '/' + img_name
                img = cv2.imread(path_name)

                top, bottom, left, right = get_padding_size(img)
                # 将图片放大， 扩充图片边缘部分
                img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                img = cv2.resize(img, (h, w))
                np_lab.append(filename)
                np_img.append(img)


read_data(my_faces_path)


# 将图片数据与标签转换成数组
np_img = np.array(np_img)

for lab in np_lab:
    if lab not in lab_name:
        lab_name.append(lab)
for lab in np_lab:
    max_num = len(lab_name)
    num_tot = max_num
    for num in range(max_num):
        if lab == lab_name[num]:
            lab = one_hot.y_one_hot(num, max_num)
            lab_full.append(lab)

print(lab_full)

# 随机划分测试集与训练集
train_x, test_x, train_y, test_y = train_test_split(np_img, lab_full, test_size=0.15,
                                                    random_state=random.randint(0, 100))

# 参数：图片数据的总数，图片的高、宽、通道
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)
# 将数据转换成小于1的数
train_x = train_x.astype('float32') / 255.0
test_x = test_x.astype('float32') / 255.0
print('train size:%s, test size:%s' % (len(train_x), len(test_x)))

# 图片块，每次取100张图片
batch_size = 128
num_batch = len(train_x) // batch_size

x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, num_tot])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)


# 初始化权值
def weight_variable(shape):
    init = tf.random_normal(shape, stddev=0.01)  # 生成一个截断的正态分布
    return tf.Variable(init)


# 初始化偏置
def bias_variable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)


# 卷积层
def conv_2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


# 池化层
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# keep_prob用来表示神经元的输出概率
def dropout(x, keep):
    return tf.nn.dropout(x, keep)


def cnnlayer():
    # 第一层
    W1 = weight_variable([5, 5, 3, 64])  # 卷积核大小(5,5)， 输入通道(3)， 输出通道(64)
    b1 = bias_variable([64])
    # 卷积
    conv1 = tf.nn.relu(conv_2d(x, W1) + b1)
    # 池化
    pool1 = max_pool(conv1)
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5)

    # 第二层，此时输入图片大小为30*30
    W2 = weight_variable([5, 5, 64, 128])
    b2 = bias_variable([128])
    conv2 = tf.nn.relu(conv_2d(drop1, W2) + b2)
    pool2 = max_pool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    # 第三层，此时输入图片大小为14*14
    W3 = weight_variable([5, 5, 128, 256])
    b3 = bias_variable([256])
    conv3 = tf.nn.relu(conv_2d(drop2, W3) + b3)
    pool3 = max_pool(conv3)
    drop3 = dropout(pool3, keep_prob_5)

    # 全连接层，此时输入图片大小为5*5
    Wf = weight_variable([5*5*256, 1024])
    bf = bias_variable([1024])
    drop_flat = tf.reshape(drop3, [-1, 5*5*256])
    dense = tf.nn.relu(tf.matmul(drop_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    print('num_tot: '+str(num_tot))
    Wout = weight_variable([1024, num_tot])
    bout = weight_variable([num_tot])
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out


def cnnTrain():
    out = cnnlayer()

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # 比较标签是否相等，再求的所有数的平均值，tf.cast(强制转换类型)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))
    # 将loss与accuracy保存以供tensorboard使用
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    # 数据保存器的初始化
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('./tmp', graph=tf.get_default_graph())

        for n in range(500):
            # 每次取128(batch_size)张图片
            for i in range(num_batch):
                batch_x = train_x[i*batch_size: (i+1)*batch_size]
                batch_y = train_y[i*batch_size: (i+1)*batch_size]
                # 开始训练数据，同时训练三个变量，返回三个数据
                _,loss,summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                           feed_dict={x: batch_x,y_: batch_y, keep_prob_5: 0.40, keep_prob_75: 0.70})
                summary_writer.add_summary(summary, n*num_batch+i)
                # 打印损失
                # print(n*num_batch+i, loss)

                if (n*num_batch+i) % 100 == 0:
                    # 获取测试数据的准确率
                    acc = accuracy.eval({x: test_x, y_: test_y, keep_prob_5: 1.0, keep_prob_75: 1.0})
                    print(n*num_batch+i, acc)
                    # 准确率大于0.98时保存并退出
                    if acc > 0.998 and n > 200:
                        saver.save(sess, 'model/train_faces.model')
                        sys.exit(0)
        saver.save(sess, 'model/train_faces.model')
        print('保存成功')

cnnTrain()

