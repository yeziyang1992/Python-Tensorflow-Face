import cv2
import dlib
import os
import sys
import random

output_dir = './faces'
input_dir = './video'
file_names = []
pic_names = []
size = 64
pic_num = 0
# 如果不存在目录 就创造目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 读取文件夹中的视频目录
def read_video_names(path):
    for filename in os.listdir(path):
        if filename.endswith('.avi'):
            filename = path + '/' + filename
            print(filename)
            file_names.append(filename)


# 读取文件夹中的图片目录
def read_pic_names(path, num):
    for filename in os.listdir(path):
        print(filename)
        pic_names.append(filename.split('_')[-1])
        num = num + 1
    return num


# 改变图片的亮度与对比度
def relight(img, light=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    # image = []
    for i in range(0,w):
        for j in range(0,h):
            for c in range(3):
                tmp = int(img[j,i,c]*light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j,i,c] = tmp
    return img


# 旋转图片
def reangle(img, angle=0):
    w = img.shape[1]
    h = img.shape[0]
    # 第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    # 第三个参数：变换后的图像大小
    img = cv2.warpAffine(img, M, (w, h))
    return img


def video_read(path, num):
    name_file = str(num) + '_'+path.split('/')[-1].split('.')[0]
    name_file = output_dir+'/' + name_file
    # 如果不存在目录 就创造目录
    if not os.path.exists(name_file):
        os.makedirs(name_file)
    # 打开摄像头 参数为输入流，可以为摄像头或视频文件
    camera = cv2.VideoCapture(path)
    index = 1
    while True:
        # 从摄像头读取照片
        success, img = camera.read()
        if not success:
            camera.release()
            print('finish')
            break
        else:
            # 转为灰度图片
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # # 旋转图片
            # gray_img = reangle(gray_img, random.randint(-65, 65))
            # 使用detector进行人脸检测
            dets = detector(gray_img, 1)
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0

                face = img[x1:y1, x2:y2]
                # 调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性
                face = relight(face, random.uniform(0.5, 1.5), random.randint(-75, 75))
                # 旋转图片
                face = reangle(face, random.randint(-65, 65))
                face = cv2.resize(face, (size, size))

                cv2.imwrite(name_file+'/' + str(index) + '.jpg', face)

                index += 1


# 使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()
# 读取文件夹中的视频目录
read_video_names(input_dir)
pic_num = read_pic_names(output_dir, pic_num)
print(pic_num)
print(pic_names)
for file_name in file_names:
    name = file_name.split('/')[-1].split('.')[0]
    if name not in pic_names:
        print(file_name+"正在处理")
        video_read(file_name, pic_num)
        pic_num = pic_num + 1
    else:
        pass
print('Done')





