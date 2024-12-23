import cv2
import numpy as np
from numpy import random
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class Perspective(object):
    def __init__(self,
                 max_ratio_translation=(0.2, 0.2, 0),
                 max_rotation=(10, 10, 360),
                 max_scale=(0.1, 0.1, 0.2),
                 max_shearing=(15, 15, 5)):

        self.max_ratio_translation = np.array(max_ratio_translation)
        self.max_rotation = np.array(max_rotation)
        self.max_scale = np.array(max_scale)
        self.max_shearing = np.array(max_shearing)

    def __call__(self, X):#, Y):

        # get the height and the width of the image
        h, w = X.shape[:2]
        max_translation = self.max_ratio_translation * np.array([w, h, 1])
        # get the values on each axis
        t_x, t_y, t_z = np.random.uniform(-1, 1, 3) * max_translation
        r_x, r_y, r_z = np.random.uniform(-1, 1, 3) * self.max_rotation
        sc_x, sc_y, sc_z = np.random.uniform(-1, 1, 3) * self.max_scale + 1
        sh_x, sh_y, sh_z = np.random.uniform(-1, 1, 3) * self.max_shearing

        # convert degree angles to rad
        theta_rx = np.deg2rad(r_x)
        theta_ry = np.deg2rad(r_y)
        theta_rz = np.deg2rad(r_z)
        theta_shx = np.deg2rad(sh_x)
        theta_shy = np.deg2rad(sh_y)
        theta_shz = np.deg2rad(sh_z)


        # compute its diagonal
        diag = (h ** 2 + w ** 2) ** 0.5
        # compute the focal length
        f = diag
        if np.sin(theta_rz) != 0:
            f /= 2 * np.sin(theta_rz)

        # set the image from cartesian to projective dimension
        H_M = np.array([[1, 0, -w / 2],
                        [0, 1, -h / 2],
                        [0, 0,      1],
                        [0, 0,      1]])
        # set the image projective to carrtesian dimension
        Hp_M = np.array([[f, 0, w / 2, 0],
                         [0, f, h / 2, 0],
                         [0, 0,     1, 0]])
        # adjust the translation on z
        t_z = (f - t_z) / sc_z ** 2
        # translation matrix to translate the image
        T_M = np.array([[1, 0, 0, t_x],
                        [0, 1, 0, t_y],
                        [0, 0, 1, t_z],
                        [0, 0, 0,  1]])

        # calculate cos and sin of angles
        sin_rx, cos_rx = np.sin(theta_rx), np.cos(theta_rx)
        sin_ry, cos_ry = np.sin(theta_ry), np.cos(theta_ry)
        sin_rz, cos_rz = np.sin(theta_rz), np.cos(theta_rz)
        # get the rotation matrix on x axis
        R_Mx = np.array([[1,      0,       0, 0],
                         [0, cos_rx, -sin_rx, 0],
                         [0, sin_rx,  cos_rx, 0],
                         [0,      0,       0, 1]])
        # get the rotation matrix on y axis
        R_My = np.array([[cos_ry, 0, -sin_ry, 0],
                         [     0, 1,       0, 0],
                         [sin_ry, 0,  cos_ry, 0],
                         [     0, 0,       0, 1]])
        # get the rotation matrix on z axis
        R_Mz = np.array([[cos_rz, -sin_rz, 0, 0],
                         [sin_rz,  cos_rz, 0, 0],
                         [     0,       0, 1, 0],
                         [     0,       0, 0, 1]])
        # compute the full rotation matrix
        R_M = np.dot(np.dot(R_Mx, R_My), R_Mz)

        # get the scaling matrix
        Sc_M = np.array([[sc_x,     0,    0, 0],
                         [   0,  sc_y,    0, 0],
                         [   0,     0, sc_z, 0],
                         [   0,     0,    0, 1]])

        # get the tan of angles
        tan_shx = np.tan(theta_shx)
        tan_shy = np.tan(theta_shy)
        tan_shz = np.tan(theta_shz)
        # get the shearing matrix on x axis
        Sh_Mx = np.array([[      1, 0, 0, 0],
                          [tan_shy, 1, 0, 0],
                          [tan_shz, 0, 1, 0],
                          [      0, 0, 0, 1]])
        # get the shearing matrix on y axis
        Sh_My = np.array([[1, tan_shx, 0, 0],
                          [0,       1, 0, 0],
                          [0, tan_shz, 1, 0],
                          [0,       0, 0, 1]])
        # get the shearing matrix on z axis
        Sh_Mz = np.array([[1, 0, tan_shx, 0],
                          [0, 1, tan_shy, 0],
                          [0, 0,       1, 0],
                          [0, 0,       0, 1]])
        # compute the full shearing matrix
        Sh_M = np.dot(np.dot(Sh_Mx, Sh_My), Sh_Mz)

        Identity = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

        # compute the full transform matrix
        M = Identity
        M = np.dot(Sh_M, M)
        M = np.dot(R_M,  M)
        M = np.dot(Sc_M, M)
        M = np.dot(T_M,  M)
        M = np.dot(Hp_M, np.dot(M, H_M))
        # apply the transformation
        X = cv2.warpPerspective(X, M, (w, h), \
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=WHITE)
        #Y = cv2.warpPerspective(Y, M, (w, h))
        return X#, Y

class Vignetting(object):
    def __init__(self,
                 ratio_min_dist=0.2,
                 range_vignette=(0.2, 0.8),
                 random_sign=False):
        self.ratio_min_dist = ratio_min_dist
        self.range_vignette = np.array(range_vignette)
        self.random_sign = random_sign

    def __call__(self, X):#, Y):
        h, w = X.shape[:2]
        min_dist = np.array([h, w]) / 2 * np.random.random() * self.ratio_min_dist

        # create matrix of distance from the center on the two axis
        x, y = np.meshgrid(np.linspace(-w/2, w/2, w), np.linspace(-h/2, h/2, h))
        x, y = np.abs(x), np.abs(y)

        # create the vignette mask on the two axis
        x = (x - min_dist[0]) / (np.max(x) - min_dist[0])
        x = np.clip(x, 0, 1)
        y = (y - min_dist[1]) / (np.max(y) - min_dist[1])
        y = np.clip(y, 0, 1)

        # then get a random intensity of the vignette
        vignette = (x + y) / 2 * np.random.uniform(*self.range_vignette)
        # vignette = np.tile(vignette[..., None], [1, 1, 3])

        sign = 2 * (np.random.random() < 0.5) * (self.random_sign) - 1
        X = X * (1 + sign * vignette)
        X = X.astype(np.uint8)
        return X#, Y
    
class GaussianBlur(object):
    def __init__(self, max_kernel=(7, 7)):
        tmp = ((max_kernel[0] + 1) // 2, (max_kernel[1] + 1) // 2)
        self.max_kernel = tmp#(tmp // 2)

    def __call__(self, X):#, Y):
        kernel_size = (
            np.random.randint(1, self.max_kernel[0]) * 2 + 1,
            np.random.randint(1, self.max_kernel[1]) * 2 + 1,
        )
        X = cv2.GaussianBlur(X, kernel_size, 0)
        return X#, Y

class Brightness(object):
    def __call__(self, X):#, Y):
        X = X * (0.3 + np.random.uniform()) #scale channel V uniformly
        X[X > 255] = 255 #reset out of range values
        X = X.astype(np.uint8)

        return X

class Contrast(object):
    def __init__(self, range_contrast=(-50, 50)):
        self.range_contrast = range_contrast

    def __call__(self, X):#, Y):
        contrast = np.random.randint(*self.range_contrast)
        X = X * (contrast / 127 + 1) - contrast
        X[X > 255] = 255 #reset out of range values
        X[X < 0] = 0     #reset out of range values
        X = X.astype(np.uint8)
        return X

class GaussianNoise(object):
    def __init__(self, center=0, std=50):
        self.center = center
        self.std = std

    def __call__(self, X):#, Y):
        noise = np.random.normal(self.center, self.std, X.shape)
        X = X + noise
        X = 255*(X - X.min())/(X.max() - X.min())
        X = X.astype(np.uint8)
        return X#, Y
    
class DisturbLines(object):
    """Disturb Lines: underline, crossline, topline, sidelines
    """

    def __call__(self, img):
        r, c = img.shape[:2]

        # random thickness
        thickness = np.random.randint(1, 3)

        # random color, gray level [0, 50)
        v = np.random.randint(0, 50)
        color = [v*i for i in (1,1,1)]

        # random gap
        gap = np.random.randint(1, 20)

        # random position
        position = np.random.choice(['top', 'bottom', 'left', 'right'], 1, p=[0.25, 0.25, 0.25, 0.25])

        # line points
        if position == "top":
            pt1 = (np.random.randint(0, 2), np.random.randint(0,2))
            pt2 = (np.random.randint(c-2, c), np.random.randint(0,2))

        if position == "bottom":
            pt1 = (np.random.randint(0, 2), np.random.randint(r-2,r))
            pt2 = (np.random.randint(c-2, c), np.random.randint(r-2,r))

        if position == "left":
            pt1 = (np.random.randint(0, 5), 0)
            pt2 = (pt1[0], r)

        if position == "right":
            pt1 = (np.random.randint(c-5, c), 0)
            pt2 = (pt1[0], r)

        # random stype
        if position not in ["top", "bottom"]:
            style = "dashed"
            gap = 1
        else:
            style = np.random.choice(['dotted', 'dashed'], 1, p=[0.2, 0.8])

        # distance between points
        dist = ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2) ** .5
        # print('-------------', dist, pt1, pt2, gap)

        # compute coords of points
        pts = []
        for i in  np.arange(0, dist, gap):
            if dist==0:
                print(pt1, pt2, i, dist)
            r = i/dist
            x = int((pt1[0]*(1-r) + pt2[0]*r) + .5)
            y = int((pt1[1]*(1-r) + pt2[1]*r) + .5)
            p = (x, y)
            pts.append(p)

        if style=='dotted':
            # draw cirles
            for p in pts:
                cv2.circle(img,p,thickness,color,-1)
        else:
            s=pts[0]
            e=pts[0]
            i=0
            # draw lines
            for p in pts:
                s=e
                e=p
                if i%2==1:
                    cv2.line(img,s,e,color,thickness)
                i+=1

        return img


class ImageAugmenter(object):
    """Image Augmentation Pipeline
    """
    def __init__(self):
        # max_ratio_translation=(0.2, 0.2, 0)
        # max_rotation=(10, 10, 360)
        # max_scale=(0.1, 0.1, 0.2)
        # max_shearing=(15, 15, 5))

        max_ratio_translation = (0.01, 0.1, 0)
        max_rotation = (30, 0, 0)    # 10-50
        max_scale = (0, 0, 0)        # 0.1-0.15
        max_shearing = (0, 0, 0)     # 5-15

        p = Perspective(max_ratio_translation=max_ratio_translation, \
            max_rotation=max_rotation,
            max_scale=max_scale,
            max_shearing=max_shearing)
        self.p = p
        self.v = Vignetting()
        self.g = GaussianBlur(max_kernel=(5,5))
        self.n = GaussianNoise(std=5)
        self.b = Brightness()
        self.c = Contrast(range_contrast=(-50, 50))
        self.d = DisturbLines()

    def image_augmentation(self, img_aug, show=False):
        # 干扰线
        r = np.random.uniform(0, 1.0)
        if r < 0.05:
            img_aug = self.d(img_aug)

        # 透视变换
        r = np.random.uniform(0, 1.0)
        if r < 0.5:
            img_aug = self.p(img_aug)
            if show:
                cv2.imshow('perspective', img_aug)
                cv2.waitKey()

        # 渐晕
        r = np.random.uniform(0, 1.0)
        if r < 0.05:
            img_aug = self.v(img_aug)
            if show:
                cv2.imshow('vignetting', img_aug)
                cv2.waitKey()

        # 模糊
        r = np.random.uniform(0, 1.0)
        if r < 0.5:
            img_aug = self.g(img_aug)
            if show:
                cv2.imshow('gaussian blur', img_aug)
                cv2.waitKey()

        # 噪声
        r = np.random.uniform(0, 1.0)
        if r < 0.5:
            img_aug = self.n(img_aug)
            if show:
                cv2.imshow('gaussian noise', img_aug)
                cv2.waitKey()

        # 亮度变化
        r = np.random.uniform(0, 1.0)
        if r < 0.2:
            img_aug = self.b(img_aug)
            if show:
                cv2.imshow('brightness', img_aug)
                cv2.waitKey()

        # 对比度
        r = np.random.uniform(0, 1.0)
        if r < 0.1:
            img_aug = self.c(img_aug)
            if show:
                cv2.imshow('contrast', img_aug)
                cv2.waitKey()

        return img_aug


if __name__ == "__main__":
    img = cv2.imread('./tmp_imgs/test.png', 0)
    print(img.shape)
    augmenter = ImageAugmenter()
    img_aug = augmenter.image_augmentation(img, show=True)
    cv2.imshow('final', img_aug)
    cv2.waitKey()
