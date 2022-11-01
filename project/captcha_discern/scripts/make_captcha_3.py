
import os
import random
import base64
from io import BytesIO
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import cv2
import numpy as np

TRAIN_DATASET_PATH = '../data/train'
# '../data/val'
# '../data/train'

def random_color():
    c1 = random.randint(0, 255)
    c2 = random.randint(0, 255)
    c3 = random.randint(0, 255)
    return c1, c2, c3


def generate_picture(width=100*4, height=40*4):
    image = Image.new('RGB', (width, height), (255,255,255))#random_color()

    color = random_color()
    #xian
    draw = ImageDraw.Draw(image)
    line_count = 1
    rd = random.random()
    if rd<0.5:
        line_count = 2
    elif rd<0.8:
        line_count = 3

    for i in range(line_count):
        x1 = random.randint(0, width)
        x2 = random.randint(0, width)
        y1 = random.randint(0, height)
        y2 = random.randint(0, height)
        draw.line((x1, y1, x2, y2), fill=color, width=random.randint(1*4,3*4))

    return image,color

def random_str_mean():
    '''
    获取一个随机字符, 数字或小写字母
    :return:
    '''
    chars = "0123456789qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"

    random_char = ''
    for _ in range(4):
        random_char += ''.join(random.sample(chars, 1))
    return random_char

def random_str():
    '''
    获取一个随机字符, 数字或小写字母
    :return:
    '''
    chars = "0123456789qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"
    chars_hard = "0127wWtTyYuiIoOSs5zZjJkKlxXcCvVwWQpP4A"
    chars_hardest = "0oOQilI"

    random_char = ''
    if random.random()<0.2:
        for _ in range(4):
            random_char += ''.join(random.sample(chars, 1))
    elif random.random()<0.8:
        for _ in range(4):
            if random.random()<0.5:
                random_char += ''.join(random.sample(chars_hardest, 1))
            elif random.random()<0.8:
                random_char += ''.join(random.sample(chars_hard, 1))
            else:
                random_char += ''.join(random.sample(chars, 1))
    else:
        for _ in range(4):
            random_char += ''.join(random.sample(chars_hard, 1))

    return random_char

def random_str2():
    '''
    获取一个随机字符, 数字或小写字母
    :return:
    '''
    chars = "0123456789qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"
    chars_new = "Lt23jJyYIls9o"

    random_char = ''
    if random.random()<0.5:
        if random.random()<0.5:
            random_char += ''.join(random.sample(chars, 1))
        else:
            random_char += ''.join(random.sample(chars_new, 1))
        random_char += ''.join(random.sample(chars_new, 1))
        if random.random()<0.5:
            random_char += ''.join(random.sample(chars, 1))
        else:
            random_char += ''.join(random.sample(chars_new, 1))
        random_char += ''.join(random.sample(chars_new, 1))
    else:
        random_char += ''.join(random.sample(chars_new, 1))
        if random.random()<0.5:
            random_char += ''.join(random.sample(chars, 1))
        else:
            random_char += ''.join(random.sample(chars_new, 1))
        random_char += ''.join(random.sample(chars_new, 1))
        if random.random()<0.5:
            random_char += ''.join(random.sample(chars, 1))
        else:
            random_char += ''.join(random.sample(chars_new, 1))

    return random_char

def draw_str(count, image, font_size,color):
    """
    在图片上写随机字符
    :param count: 字符数量
    :param image: 图片对象
    :param font_size: 字体大小
    :return:
    """
    draw = ImageDraw.Draw(image)
    # 获取一个font字体对象参数是ttf的字体文件的目录，以及字体的大小
    font_file = os.path.join(random.choice(['taile.ttf','ntailu.ttf']))
    font = ImageFont.truetype(font_file, size=font_size)

    random_char = random_str2()
    
    draw.text((random.randint(30,50), random.randint(-5,5)), random_char, color, font=font)


    return random_char, image, color


def noise(image, color, width=100, height=40, point_count=40):
    '''

    :param image: 图片对象
    :param width: 图片宽度
    :param height: 图片高度
    :param line_count: 线条数量
    :param point_count: 点的数量
    :return:
    '''
    draw = ImageDraw.Draw(image)

    

    # 画点
    for i in range(point_count):
        color = random.randint(60,150)
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        re = random.randint(1,3)
        for j in range(re):
            draw.point([x+j,y+j], fill=(color,color,color))
        # x = random.randint(0, width)
        # y = random.randint(0, height)
        # draw.arc((x, y, x + 4, y + 4), 0, 90, fill=(0,0,0))


    for i in range(10):
        color = random_color()
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        re = random.randint(1,3)
        for j in range(re):
            draw.point([x+j,y+j], fill=color)

    return image


def rotate(image, angle, center=None, scale=1.0): #1
    (h, w) = image.shape[:2] #2
    if center is None: #3
        center = (w // 2, h // 2) #4
 
    M = cv2.getRotationMatrix2D(center, angle, scale) #5
 
    rotated = cv2.warpAffine(image, M, (w, h),borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255)) #6
    return rotated #7


def contour(image, color):

    img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("ttt.png",imgray)
    ret, thresh = cv2.threshold(imgray, 250, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(contours))
    for c in contours:
        area = cv2.contourArea(c)
        if area<100*40*4*4/4*3:
            re = random.randint(70,100)
            cv2.drawContours(img,[c],0,(max(0,color[2]-re),max(0,color[1]-re),max(0,color[0]-re)),2)

    img = cv2.resize(img, (100,40))
    img = rotate(img, random.randint(-13,13))
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    return img


def main(save_dir, count):
    """
    生成图片验证码,并对图片进行base64编码
    :return:
    """
    for i in range(count):
        image,color = generate_picture()
        valid_str, image, color = draw_str(4, image, random.randint(28*4,32*4), color)
        #print(valid_str)

        image = contour(image, color)
        image = noise(image,color)

        draw = ImageDraw.Draw(image)
        #random line
        if random.random()<0.1:
            y1=random.randint(0,2)
            x1=random.randint(0,100)
            y2=random.randint(0,2)
            x2=random.randint(0,100)
            draw.line((x1, y1, x2, y2), fill=color, width=random.randint(2,3))
        elif random.random()<0.2:
            y1=random.randint(38,39)
            x1=random.randint(0,100)
            y2=random.randint(38,39)
            x2=random.randint(0,100)
            draw.line((x1, y1, x2, y2), fill=color, width=random.randint(2,3))

        image.save(os.path.join(save_dir,valid_str+"_"+str(random.random())[3:8]+".png"))


if __name__ == '__main__':
    save_dir = TRAIN_DATASET_PATH
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    count = 15
    main(save_dir, count)