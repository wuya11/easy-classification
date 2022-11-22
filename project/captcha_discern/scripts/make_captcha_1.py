# -*- coding: UTF-8 -*-
from myimage import ImageCaptcha  # pip install captcha
# from captcha.image import ImageCaptcha
from PIL import Image
import random
import time
# import captcha_setting
import os




# -*- coding: UTF-8 -*-
import os
# 验证码中的字符
# string.digits + string.ascii_uppercase
NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

ALL_CHAR_SET = NUMBER + ALPHABET
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 4

# 图像大小
# IMAGE_HEIGHT = 40
# IMAGE_WIDTH = 100

TRAIN_DATASET_PATH = '../data/val'
# '../data/val'
# '../data/train'




# def random_captcha():
#     captcha_text = []
#     for i in range(MAX_CAPTCHA):
#         c = random.choice(ALL_CHAR_SET)
#         captcha_text.append(c)
#     return ''.join(captcha_text)

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
    chars_new = "5SsvV8B0Oog9xXiIjlLwWpPzZ4A7"
    chars_new2 = "5Ss0Oog9xXiIjl"
    chars_new3 = "vVCcxXIl0Oio7T"


    def _gen1():
        if random.random()<0.3:
            c = random.choice(chars)
        elif random.random()<0.6:
            c = random.choice(chars_new)
        elif random.random()<0.8:
            c = random.choice(chars_new2)
        else:
            c = random.choice(chars_new3)
        return c

    def _gen2():
        if random.random()<0.7:
            c = random.choice(chars_new)
        else:
            c = random.choice(chars_new2)
        return c

    random_char = ''
    if random.random()<0.5:
        random_char += _gen1()

        random_char += _gen2()

        random_char += _gen1()

        random_char += _gen2()
    else:
        random_char += _gen2()

        random_char += _gen1()

        random_char += _gen2()

        random_char += _gen1()

    return random_char

# 生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha(width=100, height=40, fonts=['taile.ttf','ntailu.ttf'], font_sizes=(26,28,30,32,34,36,38,40,42,44,46,48,50,52,54))#
    captcha_text = random_str2()
    captcha_image = Image.open(image.generate(captcha_text))
    return captcha_text, captcha_image

if __name__ == '__main__':
    count = 5000
    path = TRAIN_DATASET_PATH    #通过改变此处目录，以生成 训练、测试和预测用的验证码集
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(count):
        now = str(random.random())[3:8]#str(int(time.time()))
        text, image = gen_captcha_text_and_image()
        filename = text+'_'+now+'.png'
        image.save(path  + os.path.sep +  filename)
        print('saved %d : %s' % (i+1,filename))

