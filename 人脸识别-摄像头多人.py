import tensorflow as tf
import cv2
import dlib
import numpy as np
import os
import sys
import one_hot

my_faces_path = './faces/'

size = 64
np_img = []
np_lab = []
lab_name = []
lab_full = []
num_tot = None
names = []


# 读取文件夹中的图片目录
def read_pic_names(path):
    for filename in os.listdir(path):
        print(filename)
        names.append(filename.split('_')[-1])


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
read_pic_names(my_faces_path)
print(names)
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

# 输入参数初始化
x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, num_tot])
# 神经元输出权重初始化
keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)


# 初始化权值
def weight_variable(shape):
    init = tf.random_normal(shape, stddev=0.01)  # 生成一个截断的正态分布
    return tf.Variable(init)


# 初始化偏置
def bias_variable(shape):
    init = tf.random_normal(shape)
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

output = cnnlayer()

predict = tf.argmax(output, 1)

   
saver = tf.train.Saver()  
sess = tf.Session()  
saver.restore(sess, 'model/train_faces.model')


def is_my_face(image):
    res = sess.run(output, feed_dict={x: [image / 255.0], keep_prob_5: 1.0, keep_prob_75: 1.0})
    p = sess.run(tf.nn.softmax(res))
    pre = sess.run(tf.argmax(p, 1))
    p = p[0]
    if pre[0] == 0:
        return names[0], p[0]
    elif pre[0] == 1:
        return names[1], p[1]
    elif pre[0] == 2:
        return names[2], p[2]
    elif pre[0] == 3:
        return names[3], p[3]
    elif pre[0] == 4:
        return names[4], p[4]
    elif pre[0] == 5:
        return names[5], p[5]
    elif pre[0] == 6:
        return names[6], p[6]
    elif pre[0] == 7:
        return names[7], p[7]
    elif pre[0] == 8:
        return names[8], p[8]
    elif pre[0] == 9:
        return names[9], p[9]


# 使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()

cam = cv2.VideoCapture(0)  
   
while True:
    ret, img = cam.read()
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)

    for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        x_mid = (x1 + y1)//2
        y_mid = (x2 + y2) // 2
        face = img[x1:y1, x2:y2]
        # 调整图片的尺寸
        face = cv2.resize(face, (size, size))
        Id, Probability = is_my_face(face)

        cv2.rectangle(img, (x2, x1), (y2, y1), (255, 255, 0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(Id) + " : " + str(Probability), (x_mid, y_mid), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('image', img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            sys.exit(0)

sess.close()
