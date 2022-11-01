# coding: utf-8
"""
    captcha.image
    ~~~~~~~~~~~~~

    Generate Image CAPTCHAs, just the normal image CAPTCHAs you are using.
"""

import os
import random
from PIL import Image
from PIL import ImageFilter, ImageEnhance
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype
import numpy as np

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
try:
    from wheezy.captcha import image as wheezy_captcha
except ImportError:
    wheezy_captcha = None

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
DEFAULT_FONTS = [os.path.join(DATA_DIR, 'DroidSansMono.ttf')]

if wheezy_captcha:
    __all__ = ['ImageCaptcha', 'WheezyCaptcha']
else:
    __all__ = ['ImageCaptcha']


table  =  []
for  i  in  range( 256 ):
    table.append( int(i * 1.97) )


class _Captcha(object):
    def generate(self, chars, format='png'):
        """Generate an Image Captcha of the given characters.

        :param chars: text to be generated.
        :param format: image file format
        """
        im = self.generate_image(chars)
        out = BytesIO()
        im.save(out, format=format)
        out.seek(0)
        return out

    def write(self, chars, output, format='png'):
        """Generate and write an image CAPTCHA data to the output.

        :param chars: text to be generated.
        :param output: output destination.
        :param format: image file format
        """
        im = self.generate_image(chars)
        return im.save(output, format=format)


class WheezyCaptcha(_Captcha):
    """Create an image CAPTCHA with wheezy.captcha."""
    def __init__(self, width=200, height=75, fonts=None):
        self._width = width
        self._height = height
        self._fonts = fonts or DEFAULT_FONTS

    def generate_image(self, chars):
        text_drawings = [
            wheezy_captcha.warp(),
            wheezy_captcha.rotate(),
            wheezy_captcha.offset(),
        ]
        fn = wheezy_captcha.captcha(
            drawings=[
                wheezy_captcha.background(),
                wheezy_captcha.text(fonts=self._fonts, drawings=text_drawings),
                wheezy_captcha.curve(),
                wheezy_captcha.noise(),
                wheezy_captcha.smooth(),
            ],
            width=self._width,
            height=self._height,
        )
        return fn(chars)


class ImageCaptcha(_Captcha):
    """Create an image CAPTCHA.

    Many of the codes are borrowed from wheezy.captcha, with a modification
    for memory and developer friendly.

    ImageCaptcha has one built-in font, DroidSansMono, which is licensed under
    Apache License 2. You should always use your own fonts::

        captcha = ImageCaptcha(fonts=['/path/to/A.ttf', '/path/to/B.ttf'])

    You can put as many fonts as you like. But be aware of your memory, all of
    the fonts are loaded into your memory, so keep them a lot, but not too
    many.

    :param width: The width of the CAPTCHA image.
    :param height: The height of the CAPTCHA image.
    :param fonts: Fonts to be used to generate CAPTCHA images.
    :param font_sizes: Random choose a font size from this parameters.
    """
    def __init__(self, width=160, height=60, fonts=None, font_sizes=None):
        self._width = width
        self._height = height
        self._fonts = fonts #or DEFAULT_FONTS
        self._font_sizes = font_sizes #or (42, 50, 56)
        self._truefonts = []

    @property
    def truefonts(self):
        if self._truefonts:
            return self._truefonts
        self._truefonts = tuple([
            truetype(n, s)
            for n in self._fonts
            for s in self._font_sizes
        ])
        return self._truefonts

    @staticmethod
    def create_noise_curve(image):
        draw = Draw(image)

        
        w, h = image.size

        rd = random.random()
        if rd<0.4:
            x1 = random.randint(0, w//2)
            y1 = random.randint(0, h)
            x2 = min(w-1,x1+random.randint(w//3, w//2))
            y2 = min(h-1,y1+random.randint(h//3, h//2))
            points = [x1, y1, x2, y2]
            # if random.random()<0.5:
            #     end = random.randint(70, 100)
            #     start = random.randint(0, 30)
            # else:
            start = random.randint(200, 240)
            end = random.randint(0, 30)
            draw.arc(points, start, end, fill=fontColors())
        elif rd<0.8:
            x1 = random.randint(0, w//2)
            y1 = random.randint(0, h)
            x2 = min(w-1,x1+random.randint(w//3, w//2))
            y2 = min(h-1,y1+random.randint(h//3, h//2))
            points = [x1, y1, x2, y2]
            # if random.random()<0.5:
            #     end = random.randint(70, 100)
            #     start = random.randint(0, 30)
            # else:
            start = random.randint(0, 30)
            end = random.randint(200, 240)
            draw.arc(points, start, end, fill=fontColors())

        line1_count = random.randint(0,3)
        for _ in range(line1_count):
            # x1 = random.randint(0, int(w / 5))
            # x2 = random.randint(w - int(w / 5), w)
            # y1 = random.randint(int(h / 5), h - int(h / 5))
            # y2 = random.randint(y1, h - int(h / 5))
            # x1 = random.randint(0, w//2)
            # y1 = random.randint(0, h)
            # x2 = min(w-1,x1+random.randint(w//3, w//2))
            # y2 = min(h-1,y1+random.randint(h//3, h//2))
            x1 = random.randint(0, w-1)
            y1 = random.randint(0, h-1)

            x2 = x1+random.randint(w//3, w//2)
            y2 = y1+random.randint(1, (x2-x1)//2)
            points = [x1, y1, x2, y2]
            draw.line(((x1, y1), (x2, y2)), fill=fontColors(), width=random.randint(1,2))

        
        return image

    @staticmethod
    def create_noise_dots(image,  width=1, number=30):
        draw = Draw(image)
        w, h = image.size
        while number:
            x1 = random.randint(0, w-2)
            y1 = random.randint(0, h-2)
            # draw.line(((x1, y1), (x1+1, y1+1)), fill=pointColors(), width=width)
            draw.point((x1, y1), pointColors())
            number -= 1
        return image

    def create_captcha_image(self, chars, image):
        """Create the CAPTCHA image itself.

        :param chars: text to be generated.
        :param color: color of the text.
        :param background: color of the background.

        The color should be a tuple of 3 numbers, such as (0, 255, 255).
        """
        
        draw = Draw(image)
        font_rd= random.random()
        # if font_rd<0.25:
        #     font = truetype('taile.ttf', random.randint(120,160))
                
        # elif font_rd<0.5:
        #     # font = 'ebrima.ttf'
        #     font = truetype('ebrima.ttf', random.randint(120,160))
                
        # else:
            # font = 'indieflower.ttf'
            
        if font_rd<0.3:
            font = truetype(random.choice(['taile.ttf',
                            'ebrima.ttf']), random.randint(120,150))
        elif font_rd<0.6:
            font = truetype('arial.ttf', random.randint(120,150))
        else:
            font = truetype('indieflower.ttf', random.randint(80,150))


        def _draw_character(c, color):
            #font = random.choice(self.truefonts)

            w, h = draw.textsize(c, font=font)

            dx = 0#random.randint(10, 40)
            dy = 0#200#random.randint(0, 6)
            # if random.random()<0.7:
            im = Image.new('RGB', (w + dx, h + dy))
            # else:
            #     im = Image.new('RGBA', (w + dx, h + dy))

            Draw(im).text((dx, dy), c, font=font, fill=color)

            ### rotate
            im = im.crop(im.getbbox())

            #c = random.choice([Image.BILINEAR, Image.LINEAR, Image.BICUBIC, Image.NEAREST])
            angle = random.uniform(-35, 35)
            im = im.rotate(angle, Image.BILINEAR, expand=1)#BICUBIC NEAREST

            if random.random()<0.7:
                color = fontColors()

            return im,color,angle

        images = []
        color = fontColors()

        for c in chars:
            # if random.random() > 0.5:
            #     images.append(_draw_character(" "))
            im,color,angle = _draw_character(c, color)
            images.append(im)

        text_width = sum([im.size[0] for im in images])

        width = max(text_width, self._width)
        image = image.resize((width, self._height))

        average = int(text_width / len(chars))
        rand = int(0.5 * average)
        offset = random.randint(int(average * 0.1), int(average * 0.2))

        def _imTh(x):
            if x<1:
                return 0
            else:
                return 255

        one_flag = False
        #print(len(images))
        for i,im in enumerate(images):
            w, h = im.size
            mask = im.convert('L').point(_imTh)

            image.paste(im, (offset, int((self._height - h) / 2)), mask)
            #im.save(str(i)+"_t.png")
            #print(width,w,offset,int((self._height - h) / 2))
            #break
            if i==len(images)-2:
                offset = min(width-w-int(abs(angle)*0.7)-5, offset+ w + random.randint(0, int(average * 0.1)))
                #print("---",width-w-abs(angle),offset+ w + random.randint(0, int(average * 0.1)))
            else:
                if one_flag:
                    offset = offset+ w + random.randint(0, int(average * 0.1))
                else:
                    if random.random()<0.5:
                        offset = offset+ w + random.randint(-rand, int(average * 0.1))
                        one_flag = True
                    else:
                        offset = offset+ w + random.randint(0, int(average * 0.1))

        if width > self._width:
            image = image.resize((self._width, self._height))

        #print(chars)
        # if chars=="Vop5":
        #     b
        return image

    def generate_image(self, chars):
        """Generate the image of the given characters.

        :param chars: text to be generated.
        """
        # background = random_color(238, 255)
        background = bgColors()
        #color = random_color(10, 200, random.randint(220, 255))
        im = Image.new('RGB', (self._width, self._height), background)
        
        im = im.resize((50, 20))
        self.create_noise_curve(im)
        im = im.resize((100, 40))
        self.create_noise_dots(im, number=random.randint(100,150))
        # im = im.filter(ImageFilter.SMOOTH)

        im = im.resize((400, 160))
        im = self.create_captcha_image(chars, im)
        im = im.resize((100, 40))

        # enh_con = ImageEnhance.Contrast(im)
        # radio = 2
        # im = enh_con.enhance(radio)

        return im

# def randomBG():
#     r = random.randint(220,250)
#     g = random.randint(220,250)
#     b = random.randint(220,250)
#     return tuple([r,g,b])


def fontColors():
    #RGB
    r = random.randint(1,127)
    g = random.randint(1,127)
    b = random.randint(1,127)
    return tuple([r,g,b,255])

def lineColors():
    #RGB
    r = random.randint(50,200)
    g = random.randint(50,200)
    b = random.randint(50,200)
    return tuple([r,g,b,255])

def pointColors():
    #RGB
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    return tuple([r,g,b,255])

def bgColors():
    #RGB
    r = random.randint(210,245)
    g = random.randint(210,245)
    b = random.randint(210,245)
    return tuple([r,g,b,255])

# def random_color(start, end, opacity=None):
#     red = random.randint(start, end)
#     green = random.randint(start, end)
#     blue = random.randint(start, end)
#     if opacity is None:
#         return (red, green, blue)
#     return (red, green, blue, opacity)


# def fontColors():
#     #RGB
#     colors = [(95,10,42),(32,89,43),(13,103,16),(65,45,117),(47,56,44),
#             (8,5,72)]
#     return random.choice(colors)

# def bgColors():
#     #RGB
#     colors = [(202,221,248),(201,222,211),(201,222,233),(201,211,201),
#             (241,20),(),(),(),
#             (),(),(),(),(),(),(),(),]
#     return random.choice(colors)

