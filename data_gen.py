from easydict import EasyDict
import siannodel.img_process.opencv_extend as mycv
import cv2
import numpy as np
import os
import sys
sys.path.append('../../MyLibrary/')


def get_image_paths(image_dirs):
    image_paths = []
    for image_dir in image_dirs:
        for name in os.listdir(image_dir):
            if (name.endswith('.png')
                or (name.endswith('.jpg')
                    and not name.endswith('.lines.jpg'))):
                image_paths.append(os.path.join(image_dir, name))
    return image_paths


class PhotoGenerator():
    def __init__(self, config):
        self.warp_scale = config.warp_scale
        self.image_size = config.image_size
        self.img_paths = get_image_paths(config.img_dirs)
        self.bg_img_paths = get_image_paths(config.bg_img_dirs)

    def warp_and_addbg(self, img, bg_img, M):
        '''
        对img进行透视变换并添加背景bg_img.
        Args:
            img:文档图像
            bg_img:背景图像,尺寸需要等于img
            M:透视变换矩阵
        Returns:
            result:img经过透视变换及添加背景的结果
            mask:对应的mask图
        '''
        h, w = img.shape[:2]
        warp_img = cv2.warpPerspective(
            img, M, (w, h), borderValue=(255, 255, 255))
        mask = np.full((h, w, 3), 0, np.uint8)
        mask = cv2.warpPerspective(
            mask, M, (w, h), borderValue=(255, 255, 255))
        tmp = cv2.bitwise_not(cv2.bitwise_and(mask,
                                              cv2.bitwise_not(bg_img)))
        result = cv2.bitwise_and(tmp, warp_img)
        return result, cv2.bitwise_not(mask)

    def get_warp_point_and_M(self, h, w):
        '''
        获取变换后的文档图像的4个坐标以及变换矩阵
        '''
        max_x = w*0.1
        max_y = h*0.1
        src_pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]])
        dst_pts = np.float32([[np.random.randint(max_x),
                               np.random.randint(max_y)],
                              [np.random.randint(max_x),
                               h-np.random.randint(max_y)],
                              [w-np.random.randint(max_x),
                               h-np.random.randint(max_y)],
                              [w-np.random.randint(max_x),
                               np.random.randint(max_y)]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return dst_pts, M

    def img_preprocess(self, img):
        img = cv2.resize(img, tuple(self.image_size))
        return img

    def bg_img_preprocess(self, bg_img):
        bg_img = cv2.resize(bg_img, tuple(self.image_size))
        return bg_img

    def gen(self):

        while(True):
            img_path = np.random.choice(self.img_paths)
            bg_img_path = np.random.choice(self.bg_img_paths)

            img = cv2.imread(img_path)
            img = self.img_preprocess(img)

            bg_img = cv2.imread(bg_img_path)
            bg_img = self.bg_img_preprocess(bg_img)

            print(img.shape[0], img.shape[1])
            warp_pts, M = self.get_warp_point_and_M(img.shape[0], img.shape[1])
            warp_img, mask = self.warp_and_addbg(img, bg_img, M)
            yield warp_pts, M, warp_img, mask

    def train_input_fn(self):
        dataset = tf.data.Dataset.from_generator(
            self.gen,
            (tf.float32, tf.float32,
             tf.float32, tf.float32),
            (tf.TensorShape([8]),
             tf.TensorShape([8]),
             tf.TensorShape([self.image_size[1], self.image_size[0], 3]),
             tf.TensorShape([self.image_size[1], self.image_size[0], 1])))
        
        dataset = dataset.batch(batch_size=self.config.batch_size)
        dataset = dataset.repeat(self.num_epochs)
        train_iterator = dataset.make_one_shot_iterator()
        warp_pts, M, warp_img, mask = train_iterator.get_next()

        samples = {}
        samples['warp_pts'] = warp_pts
        samples['M'] = M
        samples['mask'] = mask
        samples['warp_img'] = warp_img
        
        return samples
        
    def test_input_fn():
        pass

