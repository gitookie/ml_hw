import numpy as np
import cv2 as cv
import math

# 测试了没问题
def get_integral_image_without_channels(img):
    """获取给定图像的积分图(忽略多通道, 取个平均简化一下)
    要求输入前先把图像转换成单通道图"""

    # 将图像转换为浮点数类型
    img = img.astype(float)
    
    # 获取图像的高度和宽度


    m, n = img.shape
    img = img.astype(int)
    # print(img.shape)

    # 初始化积分图像数组
    integral_image = np.zeros((m, n))
    
    # 计算积分图像
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                integral_image[i, j] = img[i, j]
            elif i == 0 and j != 0:
                integral_image[i, j] = integral_image[i, j-1] + img[i, j]
            elif i != 0 and j == 0:
                integral_image[i, j] = integral_image[i-1, j] + img[i, j]
            else:
                integral_image[i, j] = (img[i, j] +
                                            integral_image[i-1, j] +
                                            integral_image[i, j-1] -
                                            integral_image[i-1, j-1])
    
    return integral_image
    
# 测试了没问题
def get_integral_image(img):
    """获取给定图像的积分图(三个通道)"""

    # 将图像转换为浮点数类型
    img = img.astype(float)
    
    # 获取图像的高度和宽度
    m, n, channels = img.shape
    
    # 初始化积分图像数组
    integral_image = np.zeros((m, n, channels))
    
    # 计算积分图像
    for i in range(m):
        for j in range(n):
            for c in range(channels):
                if i == 0 and j == 0:
                    integral_image[i, j, c] = img[i, j, c]
                elif i == 0 and j != 0:
                    integral_image[i, j, c] = integral_image[i, j-1, c] + img[i, j, c]
                elif i != 0 and j == 0:
                    integral_image[i, j, c] = integral_image[i-1, j, c] + img[i, j, c]
                else:
                    integral_image[i, j, c] = (img[i, j, c] +
                                                integral_image[i-1, j, c] +
                                                integral_image[i, j-1, c] -
                                                integral_image[i-1, j-1, c])
    
    return integral_image

# 单通道的测试了，三通道的还没过
def get_haar_horizontal(img, integral_img, size = (32, 30), one_channel = 1, step = 3):
    """获得一副图像的水平方向的haar特征, 此时是矩形模板的左半边像素之和减去右半边像素之和
    size存放了模板的大小, 分别表示x, y方向上的长度"""

    img = img.astype(np.int64)  # 这里好像一定要赋值，不然改不了数据类型，还是uint8，后面归一化的时候就会溢出
    square_img = img ** 2   # 创建一个每个元素都平方过的矩阵，用于对haar特征进行归一化
    # print(square_img)
    if one_channel == 0:
        result = [[], [], []]
    else:
        result = []
    size_x = int(size[0])
    size_y = int(size[1])
    if size_x % 2 != 0:
        size_x -= 1
    if size_y % 2 != 0:
        size_y -= 1
    # mask = np.zeros((size_y, size_x))
    # mask[:, :size_x / 2] = 1
    # mask[:, size_x / 2:] = -1
    if one_channel == 0:
        rows, cols, channels= img.shape
        for i in range(size_y - 1, rows):
            for j in range(size_x - 1, cols):
                for k in range(channels):
                    if i != size_y and j != size_x:
                        pass

    else:
        square_integral = get_integral_image_without_channels(square_img)
        rows, cols= img.shape
        for i in range(size_y - 1, rows, step):   # 这个滑动方式没问题，是在一个行里从左到右滑动，滑动完了再到下一行
            for j in range(size_x - 1, cols, step):
                        left_area = get_window_sum_without_channels(integral_img, (int(j - size_x / 2), i), \
                                                                    int(size_x / 2), size_y)
                        
                        right_area = get_window_sum_without_channels(integral_img, (j, i), int(size_x / 2)\
                                                                     , size_y)
                        # print('left_area', left_area)
                        # print('right area', right_area)
                        square_sum = get_window_sum_without_channels(square_integral, (j, i), size_x, size_y)
                        # print('square sum:', square_sum)
                        mean = (left_area + right_area) / (size_x * size_y)
                        # print('mean:', mean)
                        square_mean = square_sum / (size_x * size_y)
                        # print('square mean:', square_mean)
                        if square_mean > mean ** 2:
                            norm_factor = math.sqrt(square_mean - mean ** 2)
                            # print('yes')
                        else:
                            norm_factor = 1
                            # print('no')
                        tmp = (left_area - right_area) / norm_factor
                        result.append(tmp)
    return result

# 单通道的测试了，三通道压根没改
def get_haar_vertical(img, integral_img, size = (32, 30), one_channel = 1, step = 3):
    """获得一幅图像竖直方向上的haar特征, 此时特征是上半部分的像素之和减去下半部分的像素之和"""

    img = img.astype(np.int64)  # 这里好像一定要赋值，不然改不了数据类型，还是uint8，后面归一化的时候就会溢出
    square_img = img ** 2   # 创建一个每个元素都平方过的矩阵，用于对haar特征进行归一化
    # print(square_img)
    if one_channel == 0:
        result = [[], [], []]
    else:
        result = []
    size_x = int(size[0])
    size_y = int(size[1])
    if size_x % 2 != 0:
        size_x -= 1
    if size_y % 2 != 0:
        size_y -= 1
    # mask = np.zeros((size_y, size_x))
    # mask[:, :size_x / 2] = 1
    # mask[:, size_x / 2:] = -1
    if one_channel == 0:
        rows, cols, channels= img.shape
        for i in range(size_y - 1, rows):
            for j in range(size_x - 1, cols):
                for k in range(channels):
                    if i != size_y and j != size_x:
                        pass

    else:
        square_integral = get_integral_image_without_channels(square_img)
        rows, cols= img.shape
        for i in range(size_y - 1, rows, step):   # 这个滑动方式没问题，是在一个行里从左到右滑动，滑动完了再到下一行
            for j in range(size_x - 1, cols, step):
                        up_area = get_window_sum_without_channels(integral_img, (j, int(i - size_y / 2)), \
                                                                    size_x, int(size_y / 2))
                        
                        down_area = get_window_sum_without_channels(integral_img, (j, i), size_x\
                                                                     , int(size_y / 2))
                        # print('up_area', up_area)
                        # print('down area', down_area)
                        square_sum = get_window_sum_without_channels(square_integral, (j, i), size_x, size_y)
                        # print('square sum:', square_sum)
                        mean = (up_area + down_area) / (size_x * size_y)
                        # print('mean:', mean)
                        square_mean = square_sum / (size_x * size_y)
                        # print('square mean:', square_mean)
                        if square_mean > mean ** 2:
                            norm_factor = math.sqrt(square_mean - mean ** 2)
                            # print('yes')
                        else:
                            norm_factor = 1
                            # print('no')
                        tmp = (up_area - down_area) / norm_factor
                        # tmp = up_area - down_area
                        result.append(tmp)
    return result

# 单通道的测试了没问题，三通道的还没改
def get_haar_centered(img, integral_img, size_large = (32, 30), epsilon = (8, 8), one_channel = 1, step = 3):
    """获得一幅图像的中心形状的haar特征, 此时特征是方框的外部的像素之和减去下半部分的像素之和
    size_large指的是大框的大小, 里面的元素分别指示x方向, y方向的长度
    epsilon指的是内部的小框的边界和大框的边界之间的距离, 里面的元素分别指示x方向, y方向的距离"""

    img = img.astype(np.int64)  # 这里好像一定要赋值，不然改不了数据类型，还是uint8，后面归一化的时候就会溢出
    square_img = img ** 2   # 创建一个每个元素都平方过的矩阵，用于对haar特征进行归一化
    # print(square_img)
    if one_channel == 0:
        result = [[], [], []]
    else:
        result = []
    size_x_large = int(size_large[0])
    size_y_large = int(size_large[1])
    epsilon_x = int(epsilon[0])
    epsilon_y = int(epsilon[1])
    """if size_x_large % 2 != 0:
        size_x_large -= 1
    if size_y_large % 2 != 0:
        size_y_large -= 1"""
    # mask = np.zeros((size_y, size_x))
    # mask[:, :size_x / 2] = 1
    # mask[:, size_x / 2:] = -1
    if one_channel == 0:
        rows, cols, channels= img.shape
        for i in range(size_y_large - 1, rows):
            for j in range(size_x_large - 1, cols):
                for k in range(channels):
                    if i != size_y_large and j != size_x_large:
                        pass

    else:
        square_integral = get_integral_image_without_channels(square_img)
        rows, cols= img.shape
        for i in range(size_y_large - 1, rows, step):   # 这个滑动方式没问题，是在一个行里从左到右滑动，滑动完了再到下一行
            for j in range(size_x_large - 1, cols, step):
                        all_area = get_window_sum_without_channels(integral_img, (j, i), \
                                                                    size_x_large, size_y_large)
                        
                        inner_area = get_window_sum_without_channels(integral_img, (j - epsilon_x, i - epsilon_y), \
                                                                     size_x_large - 2 * epsilon_x, \
                                                                        size_y_large - 2 * epsilon_y)
                        outter_area = all_area - inner_area
                        # print('outter_area', outter_area)
                        # print('inner area', inner_area)
                        square_sum = get_window_sum_without_channels(square_integral, (j, i), size_x_large, size_y_large)
                        # print('square sum:', square_sum)
                        mean = (outter_area + inner_area) / (size_x_large * size_y_large)
                        # print('mean:', mean)
                        square_mean = square_sum / (size_x_large * size_y_large)
                        # print('square mean:', square_mean)
                        if square_mean > mean ** 2:
                            norm_factor = math.sqrt(square_mean - mean ** 2)
                            # print('yes')
                        else:
                            norm_factor = 1
                            # print('no')
                        tmp = (outter_area - inner_area) / norm_factor
                        # tmp = outter_area - inner_area
                        result.append(tmp)
    return result

def get_window_sum(integral_img, right_bottom, size_x, size_y, k):
    """计算指定窗口内的像素和, 其中right_bottom是一个元组, 存储了窗口的右下角的坐标(x, y)
    size_x, size_y分别表示窗口的x, y方向的长度, k是通道数, 表明是在第几个通道上进行"""
    
    left_top = (right_bottom[0] - size_x, right_bottom[1] - size_y)     
    # 左上角顶点的坐标

    A = integral_img[k][left_top[0], left_top[1]] if left_top[0] >= 0 and left_top[1] \
    >= 0 else 0     # 这是左上角最小的那块区域的像素之和
    B = integral_img[k][left_top[0], right_bottom[1]] if left_top[0] >= 0 else 0
    # 这是左下角顶点对应的那部分区域的像素之和
    C = integral_img[k][right_bottom[0], left_top[1]] if left_top[1] >= 0 else 0
    # 这是右上角顶点对应的那部分区域的像素之和
    D = integral_img[k][right_bottom[0], right_bottom[1]]
    return D + A - B - C

# 测试了没问题
def get_window_sum_without_channels(integral_img, right_bottom, size_x, size_y):
    """此时right_bottom里存储的点是按(x, y)的形式来存储的, 进行索引的时候不太一样, 要反过来"""

    left_top = (right_bottom[0] - size_x, right_bottom[1] - size_y)     
    # 左上角顶点的坐标(同样是按x, y的形式)

    A = integral_img[left_top[1], left_top[0]] if left_top[0] >= 0 and left_top[1] \
    >= 0 else 0     # 这是左上角最小的那块区域的像素之和
    
    B = integral_img[right_bottom[1], left_top[0]] if left_top[0] >= 0 else 0
    # 这是左下角顶点对应的那部分区域的像素之和
    
    C = integral_img[left_top[1], right_bottom[0]] if left_top[1] >= 0 else 0
    # 这是右上角顶点对应的那部分区域的像素之和
    
    D = integral_img[right_bottom[1], right_bottom[0]]
    """print('A:', A)
    print('B:', B)
    print('C:', C)
    print('D:', D)"""
    return D + A - B - C

# 测试了没问题
def shuffle_simultaneously(imgs, y, integrals):
    """输入存储了图像的numpy数组和存储了标签的numpy数组, 同步打乱并返回"""

    random_seed = np.random.randint(0, 101)
    rng = np.random.default_rng(random_seed)
    random_indices = rng.permutation(len(imgs))
    imgs_shuffled = imgs[random_indices]
    y_shuffled = y[random_indices]
    integrals_shuffled = integrals[random_indices]
    return imgs_shuffled, y_shuffled, integrals_shuffled

# test 1
"""img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
img = np.expand_dims(img, axis = 2)
img = np.repeat(img, repeats = 3, axis = 2)
print(img)
integral_img = get_integral_image(img)
print(integral_img)"""


# test 2 测试各种提取特征的函数

# img = cv.imread('D:\Develop\ml\ml_hw\Caltech_WebFaces\pic00001.jpg')

"""tmp = np.zeros((img.shape[0], img.shape[1]))
for i in range(3):
    tmp += img[:, :, i]
tmp /= 3
tmp.astype(int)
img = tmp"""
"""img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
integral_img = get_integral_image_without_channels(img)
print(integral_img)
haar = np.array(get_haar_centered(img, integral_img, size_large=(3, 3), epsilon=(1, 1), step = 1))
print(haar)"""




# test 3
"""sum = get_window_sum(integral_img, (2, 2), 2, 2, 2)
print(sum)"""
"""print(integral_img)
cv.imshow('integral', integral_img)
cv.waitKey()
cv.destroyAllWindows()"""
"""img = cv.imread('D:\Develop\ml\ml_hw\Caltech_WebFaces\pic00001.jpg')
integral_img = get_integral_image(img)
cv.imshow('integral', integral_img)
cv.waitKey()
cv.destroyAllWindows()"""
"""cv.imshow('img', img)
cv.waitKey()
cv.destroyAllWindows()"""