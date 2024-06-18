import numpy as np
import random
import math
from PIL import Image

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision.transforms import ColorJitter
import torch.nn.functional as F

import pdb

class ImageFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob: # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
        
        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img1, img2, flow

    def __call__(self, img1, img2, flow):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        return img1, img2, flow


class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        
    def eraser_transform(self, img1, img2):
        channels = img1.shape[-1]
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, channels), axis=0)
            for _ in range(np.random.randint(1, channels)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob: # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
        
        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img1, img2, flow

    def spatial_transform_no_resize(self, img1, img2, flow):
        # # randomly sample scale
        # ht, wd = img1.shape[:2]
        # min_scale = np.maximum(
        #     (self.crop_size[0] + 8) / float(ht), 
        #     (self.crop_size[1] + 8) / float(wd))

        # scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        # scale_x = scale
        # scale_y = scale
        # if np.random.rand() < self.stretch_prob:
        #     scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        #     scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        # scale_x = np.clip(scale_x, min_scale, None)
        # scale_y = np.clip(scale_y, min_scale, None)

        # if np.random.rand() < self.spatial_aug_prob:
        #     # rescale the images
        #     img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        #     img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        #     flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        #     flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob: # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        # y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        # x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
        
        # img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        # img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        # flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img1, img2, flow

    def __call__(self, img1, img2, flow, without_resize=False):
        # img1, img2 = self.eraser_transform(img1, img2)
        if(without_resize):
            img1, img2, flow= self.spatial_transform_no_resize(img1, img2, flow)
        else:
            img1, img2, flow= self.spatial_transform(img1, img2, flow)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        return img1, img2, flow

class FlowAugmentor_imglist:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        
    def spatial_transform(self, img_list, flow):
        # randomly sample scale
        img1 = img_list[0]
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        # if np.random.rand() < self.spatial_aug_prob:
        #     # rescale the images
        #     img_list = [cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR) for img in img_list]

        #     flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        #     flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                img_list = [img[:, ::-1] for img in img_list]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob: # v-flip
                img_list = [img[::-1, :] for img in img_list]
                flow = flow[::-1, :] * [1.0, -1.0]

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
        
        img_list = [img[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] for img in img_list]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img_list, flow

    def __call__(self, img_list, flow):
        img_list, flow= self.spatial_transform(img_list, flow)

        img_list = [np.ascontiguousarray(img) for img in img_list]
        flow = np.ascontiguousarray(flow)

        return img_list, flow

class DenseSparseAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        
    def eraser_transform(self, img1, img2, dimg1, dimg2):
        channels = img1.shape[-1]
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, channels), axis=0)
            for _ in range(np.random.randint(1, channels)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color
                dimg2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2, dimg1, dimg2

    def spatial_transform(self, img1, img2, dimg1, dimg2, flow):
        # randomly sample scale
        # ht, wd = img1.shape[:2]
        # min_scale = np.maximum(
        #     (self.crop_size[0] + 8) / float(ht), 
        #     (self.crop_size[1] + 8) / float(wd))

        # scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        # scale_x = scale
        # scale_y = scale
        # if np.random.rand() < self.stretch_prob:
        #     scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        #     scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        # scale_x = np.clip(scale_x, min_scale, None)
        # scale_y = np.clip(scale_y, min_scale, None)

        # if np.random.rand() < self.spatial_aug_prob:
        #     # rescale the images
        #     img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        #     img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        #     dimg1 = cv2.resize(dimg1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        #     dimg2 = cv2.resize(dimg2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        #     flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        #     flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                dimg1 = dimg1[:, ::-1]
                dimg2 = dimg2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob: # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                dimg1 = dimg1[::-1, :]
                dimg2 = dimg2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        if(img1.shape[0] == self.crop_size[0]):
            y0 = 0
        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        
        if(img1.shape[1] == self.crop_size[1]):
            x0 = 0
        else:
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
        
        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        dimg1 = dimg1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        dimg2 = dimg2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img1, img2, dimg1, dimg2, flow

    def __call__(self, img1, img2, dimg1, dimg2, flow):
        # img1, img2, dimg1, dimg2 = self.eraser_transform(img1, img2, dimg1, dimg2)
        img1, img2, dimg1, dimg2 ,flow= self.spatial_transform(img1, img2, dimg1, dimg2, flow)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        dimg1 = np.ascontiguousarray(dimg1)
        dimg2 = np.ascontiguousarray(dimg2)
        flow = np.ascontiguousarray(flow)

        return img1, img2, dimg1, dimg2, flow

class EventAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1
    
    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=1]
        flow0 = flow[valid>=1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, events1, events2, flow, valid):
        # randomly sample scale
        ht, wd = flow.shape[:2]
        # pdb.set_trace()
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        # if np.random.rand() < self.spatial_aug_prob:
        #     # rescale the images
        #     events1[:,0] = events1[:,0] * scale_x
        #     events1[:,1] = events1[:,1] * scale_y

        #     events2[:,0] = events2[:,0] * scale_x
        #     events2[:,1] = events2[:,1] * scale_y

        #     # flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        #     # flow = flow * [scale_x, scale_y]
        #     flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                events1[:,0] = events1[:,0].max() - events1[:,0]
                events2[:,0] = events2[:,0].max() - events2[:,0]
                flow = flow[:, ::-1] * [-1.0, 1.0] # x
                valid = valid[:, ::-1]

            if np.random.rand() < self.v_flip_prob: # v-flip
                events1[:,1] = events1[:,1].max() - events1[:,1]
                events2[:,1] = events2[:,1].max() - events2[:,1]
                flow = flow[::-1, :] * [1.0, -1.0] # y
                valid = valid[::-1, :]

        if(flow.shape[0]==self.crop_size[0]):
            y0=0
        else:
            y0 = np.random.randint(0, flow.shape[0] - self.crop_size[0])

        if(flow.shape[1]==self.crop_size[1]):
            x0=0
        else:
            x0 = np.random.randint(0, flow.shape[1] - self.crop_size[1])
        
        # img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        # img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        # coord1 = (events1[:,1] >= y0 and events1[:,1] <= y0+self.crop_size[0]) and \
        #     (events1[:,0] >= x0 and events1[:,0] <= x0+self.crop_size[1])

        # coord2 = (events2[:,1] >= y0 and events2[:,1] <= y0+self.crop_size[0]) and \
        #     (events2[:,0] >= x0 and events2[:,0] <= x0+self.crop_size[1])
        
        # pdb.set_trace()

        coord1 = np.logical_and(np.logical_and(np.logical_and(events1[:,1] >= y0, events1[:,1] < y0+self.crop_size[0]), \
            events1[:,0] >= x0), events1[:,0] < x0+self.crop_size[1])
        coord2 = np.logical_and(np.logical_and(np.logical_and(events2[:,1] >= y0, events2[:,1] < y0+self.crop_size[0]), \
            events2[:,0] >= x0), events2[:,0] < x0+self.crop_size[1])
        
        events1 = events1[coord1]
        if events1.shape[0] > 0:
            events1[:,0] = events1[:, 0] - np.min(events1[:, 0])
            events1[:,1] = events1[:, 1] - np.min(events1[:, 1])

        events2 = events2[coord2]
        if events2.shape[0] > 0:
            events2[:,0] = events2[:, 0] - np.min(events2[:, 0])
            events2[:,1] = events2[:, 1] - np.min(events2[:, 1])
        
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return events1, events2, flow, valid

    def __call__(self, events1, events2, flow, valid):
        events1, events2, flow, valid = self.spatial_transform(events1, events2, flow, valid)

        events1 = np.ascontiguousarray(events1)
        events2 = np.ascontiguousarray(events2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return events1, events2, flow, valid

class MixEventVolumeAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1
    
    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=1]
        flow0 = flow[valid>=1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, events, volume, flow, valid):
        # randomly sample scale
        ht, wd = flow.shape[:2]
        # pdb.set_trace()
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        # if np.random.rand() < self.spatial_aug_prob:
        #     # rescale the images
        #     events[:,0] = events[:,0] * scale_x
        #     events[:,1] = events[:,1] * scale_y

        #     volume = cv2.resize(volume, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

        #     # flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        #     # flow = flow * [scale_x, scale_y]
        #     flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                events[:,0] = events[:,0].max() - events[:,0]
                volume = volume[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                valid = valid[:, ::-1]

            if np.random.rand() < self.v_flip_prob: # v-flip
                events[:,1] = events[:,1].max() - events[:,1]
                volume = volume[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]
                valid = valid[::-1, :]

        if(flow.shape[0]==self.crop_size[0]):
            y0=0
        else:
            y0 = np.random.randint(0, flow.shape[0] - self.crop_size[0])

        if(flow.shape[1]==self.crop_size[1]):
            x0=0
        else:
            x0 = np.random.randint(0, flow.shape[1] - self.crop_size[1])
        
        # img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        # img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        # coord1 = (events1[:,1] >= y0 and events1[:,1] <= y0+self.crop_size[0]) and \
        #     (events1[:,0] >= x0 and events1[:,0] <= x0+self.crop_size[1])

        # coord2 = (events2[:,1] >= y0 and events2[:,1] <= y0+self.crop_size[0]) and \
        #     (events2[:,0] >= x0 and events2[:,0] <= x0+self.crop_size[1])
        
        # pdb.set_trace()

        coord1 = np.logical_and(np.logical_and(np.logical_and(events[:,1] >= y0, events[:,1] < y0+self.crop_size[0]), \
            events[:,0] >= x0), events[:,0] < x0+self.crop_size[1])

        
        events = events[coord1]
        if events.shape[0] > 0:
            events[:,0] = events[:, 0] - np.min(events[:, 0])
            events[:,1] = events[:, 1] - np.min(events[:, 1])

        volume = volume[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return events, volume, flow, valid

    def __call__(self, events, volume, flow, valid):
        events, volume, flow, valid = self.spatial_transform(events, volume, flow, valid)

        events = np.ascontiguousarray(events)
        volume = np.ascontiguousarray(volume)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return events, volume, flow, valid
