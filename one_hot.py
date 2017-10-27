# 该函数的作用是输入一个小于10的数字 返回一个一维one_hot向量
# 例如：num=2 max_num=3 时：
# 返回[0, 0, 1]
# 注意：max_num不可大于10


def y_one_hot(num, max_num):
    if max_num == 2:
        if num == 0:
            return [1, 0]
        if num == 1:
            return [0, 1]
        else:
            return None

    if max_num == 3:
        if num == 0:
            return [1, 0, 0]
        if num == 1:
            return [0, 1, 0]
        if num == 2:
            return [0, 0, 1]
        else:
            return None

    if max_num == 4:
        if num == 0:
            return [1, 0, 0, 0]
        if num == 1:
            return [0, 1, 0, 0]
        if num == 2:
            return [0, 0, 1, 0]
        if num == 3:
            return [0, 0, 0, 1]
        else:
            return None

    if max_num == 5:
        if num == 0:
            return [1, 0, 0, 0, 0]
        if num == 1:
            return [0, 1, 0, 0, 0]
        if num == 2:
            return [0, 0, 1, 0, 0]
        if num == 3:
            return [0, 0, 0, 1, 0]
        if num == 4:
            return [0, 0, 0, 0, 1]
        else:
            return None

    if max_num == 6:
        if num == 0:
            return [1, 0, 0, 0, 0, 0]
        if num == 1:
            return [0, 1, 0, 0, 0, 0]
        if num == 2:
            return [0, 0, 1, 0, 0, 0]
        if num == 3:
            return [0, 0, 0, 1, 0, 0]
        if num == 4:
            return [0, 0, 0, 0, 1, 0]
        if num == 5:
            return [0, 0, 0, 0, 0, 1]
        else:
            return None

    if max_num == 7:
        if num == 0:
            return [1, 0, 0, 0, 0, 0, 0]
        if num == 1:
            return [0, 1, 0, 0, 0, 0, 0]
        if num == 2:
            return [0, 0, 1, 0, 0, 0, 0]
        if num == 3:
            return [0, 0, 0, 1, 0, 0, 0]
        if num == 4:
            return [0, 0, 0, 0, 1, 0, 0]
        if num == 5:
            return [0, 0, 0, 0, 0, 1, 0]
        if num == 6:
            return [0, 0, 0, 0, 0, 0, 1]
        else:
            return None

    if max_num == 8:
        if num == 0:
            return [1, 0, 0, 0, 0, 0, 0, 0]
        if num == 1:
            return [0, 1, 0, 0, 0, 0, 0, 0]
        if num == 2:
            return [0, 0, 1, 0, 0, 0, 0, 0]
        if num == 3:
            return [0, 0, 0, 1, 0, 0, 0, 0]
        if num == 4:
            return [0, 0, 0, 0, 1, 0, 0, 0]
        if num == 5:
            return [0, 0, 0, 0, 0, 1, 0, 0]
        if num == 6:
            return [0, 0, 0, 0, 0, 0, 1, 0]
        if num == 7:
            return [0, 0, 0, 0, 0, 0, 0, 1]
        else:
            return None

    if max_num == 9:
        if num == 0:
            return [1, 0, 0, 0, 0, 0, 0, 0, 0]
        if num == 1:
            return [0, 1, 0, 0, 0, 0, 0, 0, 0]
        if num == 2:
            return [0, 0, 1, 0, 0, 0, 0, 0, 0]
        if num == 3:
            return [0, 0, 0, 1, 0, 0, 0, 0, 0]
        if num == 4:
            return [0, 0, 0, 0, 1, 0, 0, 0, 0]
        if num == 5:
            return [0, 0, 0, 0, 0, 1, 0, 0, 0]
        if num == 6:
            return [0, 0, 0, 0, 0, 0, 1, 0, 0]
        if num == 7:
            return [0, 0, 0, 0, 0, 0, 0, 1, 0]
        if num == 8:
            return [0, 0, 0, 0, 0, 0, 0, 0, 1]
        else:
            return None

    if max_num == 10:
        if num == 0:
            return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if num == 1:
            return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        if num == 2:
            return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        if num == 3:
            return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        if num == 4:
            return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        if num == 5:
            return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        if num == 6:
            return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        if num == 7:
            return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        if num == 8:
            return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        if num == 9:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        else:
            return None
