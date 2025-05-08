import torch.optim as optim
import imageio
import cv2
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn.init import xavier_normal, kaiming_normal
from torch.utils.data import Dataset
import pickle
import argparse
import collections
import random
from shutil import rmtree
import time
import zipfile
import png
import array
import warnings
import shutil
from PIL import Image
import re
from collections import Iterable
from matplotlib.colors import hsv_to_rgb

try:
    import nori2 as nori
except:
    print('can not import nori')

if torch.__version__ in ['1.1.0', ]:
    from torch.utils.data.dataloader import _DataLoaderIter, DataLoader
elif torch.__version__ in ['1.2.0', '1.4.0', '1.5.1', '1.6.0', '1.7.0','1.10.0','1.12.0']:
    from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter as _DataLoaderIter
    from torch.utils.data.dataloader import DataLoader
else:
    raise ValueError('torch version error: %s' % torch.__version__)


class tools():
    class abs_database():
        def __init__(self):
            self.data_ls = {'train': [], 'val': [], 'test': []}
            self.len = {'train': 0, 'val': 0, 'test': 0}
            self.len_train = 0
            self.len_val = 0
            self.len_test = 0

        def sample(self, index, split):
            pass

        def _init_len(self):
            self.len = {'train': len(self.data_ls['train']), 'val': len(self.data_ls['val']), 'test': len(self.data_ls['test'])}
            self.len_train = self.len['train']
            self.len_val = self.len['train']
            self.len_test = self.len['train']

    class abstract_config():
        @classmethod
        def _check_length_of_file_name(cls, file_name):
            if len(file_name) >= 255:
                return False
            else:
                return True

        @classmethod
        def _check_length_of_file_path(cls, filepath):
            if len(filepath) >= 4096:
                return False
            else:
                return True

        @property
        def to_dict(self):
            def dict_class(obj):
                temp = {}
                k = dir(obj)
                for name in k:
                    if not name.startswith('_') and name != 'to_dict':
                        value = getattr(obj, name)
                        if callable(value):
                            pass
                        else:
                            temp[name] = value
                return temp

            s_dict = dict_class(self)
            return s_dict

        @property
        def _key_list(self):
            k_list = list(self.to_dict.keys())
            return k_list

        def update(self, data: dict):

            t_key = list(data.keys())
            for i in self._key_list:
                if i in t_key:
                    setattr(self, i, data[i])
                    print('set param ====  %s:   %s' % (i, data[i]))

        def __contains__(self, item):
            '''  use to check something in config '''
            if item in self._key_list:
                return True
            else:
                return False

        def print_defaut_dict(self):
            d = self.to_dict
            l = self._key_list
            l = sorted(l)
            for i in l:
                value = d[i]
                if type(value) == str:
                    temp = "'%s'" % value
                else:
                    temp = value
                print("'%s':%s," % (i, temp))

        @classmethod
        def __demo(cls):
            class temp(tools.abstract_config):
                def __init__(self, **kwargs):
                    self.if_gpu = True
                    self.eval_batch_size = 1
                    self.eval_name = 'flyingchairs'
                    self.eval_datatype = 'nori'  # or base
                    self.if_print_process = False

                    self.update(kwargs)

            a = temp(eval_batch_size=8, eval_name='flyingchairs', eval_datatype='nori', if_print_process=False)

    class abstract_model(nn.Module):

        def save_model(self, save_path):
            torch.save(self.state_dict(), save_path)

        def load_model(self, load_path, if_relax=False, if_print=True):
            if if_print:
                print('loading protrained model from %s' % load_path)
            if if_relax:
                model_dict = self.state_dict()
                pretrained_dict = torch.load(load_path)
                # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                pretrained_dict_v2 = {}
                for k, v in pretrained_dict.items():
                    if k in model_dict:
                        if v.shape == model_dict[k].shape:
                            pretrained_dict_v2[k] = v
                model_dict.update(pretrained_dict_v2)
                self.load_state_dict(model_dict)
            else:
                self.load_state_dict(torch.load(load_path))

        def load_from_model(self, model: nn.Module, if_relax=False):
            if if_relax:
                model_dict = self.state_dict()
                pretrained_dict = model.state_dict()
                # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                pretrained_dict_v2 = {}
                for k, v in pretrained_dict.items():
                    if k in model_dict:
                        if v.shape == model_dict[k].shape:
                            pretrained_dict_v2[k] = v
                model_dict.update(pretrained_dict_v2)
                self.load_state_dict(model_dict)
            else:
                self.load_state_dict(model.state_dict())

        def choose_gpu(self, gpu_opt=None):
            # choose gpu
            if gpu_opt is None:
                # gpu=0
                model = self.cuda()
                # torch.cuda.set_device(gpu)
                # model.cuda(gpu)
                # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
                # print('torch.cuda.device_count()  ',torch.cuda.device_count())
                # model=torch.nn.parallel.DistributedDataParallel(model,device_ids=range(torch.cuda.device_count()))
                model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))  # multi gpu
            elif gpu_opt == 0:
                model = self.cuda()
            else:
                if type(gpu_opt) != int:
                    raise ValueError('wrong gpu config:  %s' % (str(gpu_opt)))
                torch.cuda.set_device(gpu_opt)
                model = self.cuda(gpu_opt)
            return model

        @classmethod
        def save_model_gpu(cls, model, path):
            name_dataparallel = torch.nn.DataParallel.__name__
            if type(model).__name__ == name_dataparallel:
                model = model.module
            model.save_model(path)

    class abs_test_model(abstract_model):
        def __init__(self):
            super(tools.abs_test_model, self).__init__()
            self.result_save_dir = None
            self.some_save_results = False
            self.some_ids = None  # only save some results
            self.id_cnt = -1
            self.eval_id_scores = {}
            self.id_cnt_save_dir = ''

        def prepare_eval(self):
            self.id_cnt += 1
            if self.result_save_dir is not None:
                save_flag = True
                self.id_cnt_save_dir = os.path.join(self.result_save_dir, '%s' % self.id_cnt)
                if self.some_save_results and self.id_cnt not in self.some_ids:
                    eval_flag = False
                else:
                    file_tools.check_dir(self.id_cnt_save_dir)
                    eval_flag = True
            else:
                save_flag = False
                eval_flag = True
            return eval_flag, save_flag

        def eval_forward(self, im1, im2, *args, **kwargs):  # do model forward and cache forward results
            self.id_cnt += 1
            if self.some_ids is None:
                pass
            else:
                pass
            return 0

        def do_save_results(self, result_save_dir=None, some_save_results=False):
            self.result_save_dir = result_save_dir
            if result_save_dir is not None:
                file_tools.check_dir(result_save_dir)
            self.some_save_results = some_save_results
            # define some id
            if result_save_dir is not None:
                self.some_ids = [7 * 7 * i + 1 for i in range(24)]  # [1,50,99,148, ..., 981, 1030, 1079, 1128]

        def record_eval_score(self, eval_score):
            self.eval_id_scores[self.id_cnt] = eval_score
            if os.path.isdir(self.id_cnt_save_dir):
                new_dir_name = '%s_EPE_%.2f' % (self.id_cnt, float(eval_score))
                os.renames(self.id_cnt_save_dir, os.path.join(self.result_save_dir, new_dir_name))

        def save_record(self):
            print('eval results saved at: %s' % self.result_save_dir)
            file_tools.pickle_saver.save_pickle(self.eval_id_scores, os.path.join(self.result_save_dir, 'scores.pkl'))
            self.id_cnt = -1

    class data_prefetcher():

        def __init__(self, dataset, gpu_opt=None, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False, batch_gpu_index=0):
            self.dataset = dataset
            loader = DataLoader(dataset=self.dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, pin_memory=pin_memory)
            # self.loader = iter(loader)
            self.loader = _DataLoaderIter(loader)
            self.stream = torch.cuda.Stream()
            self.gpu_opt = gpu_opt

            self.batch_size = batch_size
            self.shuffle = shuffle
            self.num_workers = num_workers
            self.pin_memory = pin_memory
            self.drop_last = drop_last
            self.batch_gpu_index = batch_gpu_index

        def build(self):
            loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, drop_last=self.drop_last, pin_memory=self.pin_memory)
            self.loader = _DataLoaderIter(loader)
            # self.loader = iter(loader)

        def next(self):
            try:
                # batch = next(self.loader)
                batch = self.loader.next()
            except StopIteration:
                self.build()
                return None
            # print('self.batch',type(self.batch))
            # for i in range(len(self.batch)):
            #     print('i',i,type(self.batch[i]))
            with torch.cuda.stream(self.stream):
                cpu_batch, gpu_batch = batch[:self.batch_gpu_index], batch[self.batch_gpu_index:]
                gpu_batch = tensor_tools.tensor_gpu(*gpu_batch, check_on=True, non_blocking=True, gpu_opt=self.gpu_opt)
                batch = cpu_batch + gpu_batch
                # self.next_img = self.next_img.cuda(non_blocking=True).float()
                # self.next_seg = self.next_seg.cuda(non_blocking=True).float()
                # self.next_weight = self.next_weight.cuda(non_blocking=True)
                # self.mask2 = self.mask2.cuda(non_blocking=True).float()
                # self.mask3 = self.mask3.cuda(non_blocking=True).float()

                # With Amp, it isn't necessary to manually convert data to half.
                # if args.fp16:
                #     self.next_input = self.next_input.half()
                # else:
                # self.next_input = self.next_input.float()
                # self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            return batch

    class data_prefetcher_dict():

        def __init__(self, dataset, gpu_keys, gpu_opt=None, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False):
            self.dataset = dataset
            loader = DataLoader(dataset=self.dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, pin_memory=pin_memory)
            self.gpu_keys = gpu_keys  # keys in batches to be loaded to gpu, e.g. gpu_keys=('im1', 'im2')
            # self.loader = iter(loader)
            self.loader = _DataLoaderIter(loader)
            self.stream = torch.cuda.Stream()
            self.gpu_opt = gpu_opt

            self.batch_size = batch_size
            self.shuffle = shuffle
            self.num_workers = num_workers
            self.pin_memory = pin_memory
            self.drop_last = drop_last

        def build(self):
            loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, drop_last=self.drop_last, pin_memory=self.pin_memory)
            self.loader = _DataLoaderIter(loader)
            # self.loader = iter(loader)

        def next(self):
            try:
                # batch = next(self.loader)
                batch = self.loader.next()
            except StopIteration:
                self.build()
                return None
            with torch.cuda.stream(self.stream):
                for i in self.gpu_keys:
                    batch[i] = self.check_on_gpu(batch[i], non_blocking=True)
            return batch

        def check_on_gpu(self, tensor_, non_blocking=True):
            if type(self.gpu_opt) == int:
                tensor_g = tensor_.cuda(self.gpu_opt, non_blocking=non_blocking)
            else:
                tensor_g = tensor_.cuda()
            return tensor_g

    class DataProvider:

        def __init__(self, dataset, batch_size, shuffle=True, num_worker=4, drop_last=True, pin_memory=True):
            self.batch_size = batch_size
            self.dataset = dataset
            self.dataiter = None
            self.iteration = 0  #
            self.epoch = 0  #
            self.shuffle = shuffle
            self.pin_memory = pin_memory
            self.num_worker = num_worker
            self.drop_last = drop_last

        def build(self):
            dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_worker,
                                    pin_memory=self.pin_memory,
                                    drop_last=self.drop_last)
            self.dataiter = _DataLoaderIter(dataloader)

        def next(self):
            if self.dataiter is None:
                self.build()
            try:
                batch = self.dataiter.next()
                self.iteration += 1

                # if self.is_cuda:
                #     batch = [batch[0].cuda(), batch[1].cuda(), batch[2].cuda()]
                return batch

            except StopIteration:  # ??epoch???reload
                self.epoch += 1
                self.build()
                self.iteration = 1  # reset and return the 1st batch

                batch = self.dataiter.next()
                # if self.is_cuda:
                #     batch = [batch[0].cuda(), batch[1].cuda(), batch[2].cuda()]
                return batch

    class AverageMeter():

        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, num):
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count

    class Avg_meter_ls():
        def __init__(self):
            self.data_ls = {}
            self.short_name_ls = {}

        def update(self, name, val, num, short_name=None):
            if name not in self.data_ls.keys():
                self.data_ls[name] = tools.AverageMeter()
                if short_name is None:
                    short_name = name
                self.short_name_ls[name] = short_name
            self.data_ls[name].update(val=val, num=num)

        def print_loss(self, name):
            a = ' %s %.4f(%.4f)' % (self.short_name_ls[name], self.data_ls[name].val, self.data_ls[name].avg)
            return a

        def print_avg_loss(self, name):
            a = ' %s: %.4f' % (self.short_name_ls[name], self.data_ls[name].avg)
            return a

        def print_all_losses(self):
            a = ''
            for i in sorted(self.data_ls.keys()):
                a += ' %s %.4f(%.4f)' % (self.short_name_ls[i], self.data_ls[i].val, self.data_ls[i].avg)
            return a

        def print_all_losses_final(self):
            a = ''
            for i in sorted(self.data_ls.keys()):
                a += ' %s=%.4f' % (self.short_name_ls[i], self.data_ls[i].avg)
            return a

        def get_all_losses_final(self):
            a = {}
            for i in sorted(self.data_ls.keys()):
                a[i] = self.data_ls[i].avg
            return a

        def reset(self):
            for name in self.data_ls.keys():
                self.data_ls[name].reset()

    class TimeClock():

        def __init__(self):
            self.st = 0
            self.en = 0
            self.start_flag = False

        def start(self):
            self.reset()
            self.start_flag = True
            self.st = time.time()

        def reset(self):
            self.start_flag = False
            self.st = 0
            self.en = 0

        def end(self):
            self.en = time.time()

        def get_during(self):
            return self.en - self.st

    # 研究一下图像加字体展示结果
    class Text_img():
        def __init__(self, **kwargs):
            self.font = 'simplex'
            self.my_font_type = 'black_white'
            self.__update(kwargs)
            self.font_ls = {
                'simplex': cv2.FONT_HERSHEY_SIMPLEX,
                'plain': cv2.FONT_HERSHEY_PLAIN,
                'complex': cv2.FONT_HERSHEY_COMPLEX,
                'trplex': cv2.FONT_HERSHEY_TRIPLEX,
                # 'complex_small': cv2.FONT_HERSHEY_COMPLEX_SMALL,
                'italic': cv2.FONT_ITALIC,
            }
            self.my_font_type_ls = {
                'black_white': self._black_white,
            }
            self.show_func = self.my_font_type_ls[self.my_font_type]

        def __update(self, data: dict):
            def dict_class(obj):
                temp = {}
                k = dir(obj)
                for name in k:
                    if not name.startswith('_'):
                        value = getattr(obj, name)
                        if callable(value):
                            pass
                        else:
                            temp[name] = value
                return temp

            s_dict = dict_class(self)
            k_list = list(s_dict.keys())
            t_key = list(data.keys())
            for i in k_list:
                if i in t_key:
                    setattr(self, i, data[i])
                    # print('set param ====  %s:   %s' % (i, data[i]))

        def _black_white(self, img, text, scale, row=0):
            # params
            color_1 = (10, 10, 10)
            thick_1 = 5
            color_2 = (255, 255, 255)
            thick_2 = 2

            # get position: Bottom-left
            t_w, t_h, t_inter = self._check_text_size(text=text, scale=scale, thick=thick_1)
            pw = t_inter
            ph = t_h + t_inter + row * (t_h + t_inter)

            # put text
            img_ = img.copy()
            img_ = cv2.putText(img_, text, (pw, ph), fontFace=self.font_ls[self.font], fontScale=scale, color=color_1, thickness=thick_1)
            img_ = cv2.putText(img_, text, (pw, ph), fontFace=self.font_ls[self.font], fontScale=scale, color=color_2, thickness=thick_2)
            return img_

        def _check_text_size(self, text: str, scale=1, thick=1):
            textSize, baseline = cv2.getTextSize(text, self.font_ls[self.font], scale, thick)
            twidth, theight = textSize
            return twidth, theight, baseline // 2

        def put_text(self, img, text=None, scale=1):
            if text is not None:
                if type(text) == str:
                    img = self.show_func(img, text, scale, 0)
                elif isinstance(text, Iterable):
                    for i, t in enumerate(text):
                        img = self.show_func(img, t, scale, i)
            return img

        def draw_cross(self, img, point_wh, cross_length=5, color=(0, 0, 255)):  #
            thick = cross_length // 2
            new_img = img.copy()
            new_img = cv2.line(new_img, (point_wh[0] - cross_length, point_wh[1]), (point_wh[0] + cross_length, point_wh[1]), color, thick)
            new_img = cv2.line(new_img, (point_wh[0], point_wh[1] - cross_length), (point_wh[0], point_wh[1] + cross_length), color, thick)
            return new_img

        def draw_cross_black_white(self, img, point_wh, cross_length=5):  #
            if cross_length <= 5:
                cross_length = 5
            thick = cross_length // 2
            new_img = img.copy()
            new_img = cv2.line(new_img, (point_wh[0] - cross_length, point_wh[1]), (point_wh[0] + cross_length, point_wh[1]), (0, 0, 0), thick)
            new_img = cv2.line(new_img, (point_wh[0], point_wh[1] - cross_length), (point_wh[0], point_wh[1] + cross_length), (0, 0, 0), thick)
            new_img = cv2.line(new_img, (point_wh[0] - cross_length, point_wh[1]), (point_wh[0] + cross_length, point_wh[1]), (250, 250, 250), thick // 2)
            new_img = cv2.line(new_img, (point_wh[0], point_wh[1] - cross_length), (point_wh[0], point_wh[1] + cross_length), (250, 250, 250), thick // 2)
            return new_img

        def draw_x(self, img, point_wh, cross_length=5, color=(0, 0, 255)):
            thick = cross_length // 2
            new_img = img.copy()
            new_img = cv2.line(new_img, (point_wh[0] - cross_length, point_wh[1] - cross_length), (point_wh[0] + cross_length, point_wh[1] + cross_length), color, thick)
            new_img = cv2.line(new_img, (point_wh[0] - cross_length, point_wh[1] + cross_length), (point_wh[0] + cross_length, point_wh[1] - cross_length), color, thick)
            return new_img

        def draw_x_black_white(self, img, point_wh, cross_length=5):
            if cross_length <= 5:
                cross_length = 5
            thick = cross_length // 2
            new_img = img.copy()
            new_img = cv2.line(new_img, (point_wh[0] - cross_length, point_wh[1] - cross_length), (point_wh[0] + cross_length, point_wh[1] + cross_length), (0, 0, 0), thick)
            new_img = cv2.line(new_img, (point_wh[0] - cross_length, point_wh[1] + cross_length), (point_wh[0] + cross_length, point_wh[1] - cross_length), (0, 0, 0), thick)
            new_img = cv2.line(new_img, (point_wh[0] - cross_length, point_wh[1] - cross_length), (point_wh[0] + cross_length, point_wh[1] + cross_length), (250, 250, 250), thick // 2)
            new_img = cv2.line(new_img, (point_wh[0] - cross_length, point_wh[1] + cross_length), (point_wh[0] + cross_length, point_wh[1] - cross_length), (250, 250, 250), thick // 2)
            return new_img

        def demo(self):
            im = np.ones((500, 500, 3), dtype='uint8') * 50
            imshow = self.put_text(im, text=list('demo show sample text'.split(' ')), scale=1)
            cv2.imshow('im', imshow)
            cv2.waitKey()

    # TODO 展示gpu cpu等情况，但目前怀疑会降低速度所以暂时没有弄
    class system_monitor(abstract_config):
        def __init__(self, **kwargs):
            self.update(kwargs)
            import pynvml
            pynvml.nvmlInit()  # init pynvml

    @classmethod
    def clear(cls):
        os.system("clear")  # 清屏

    @classmethod
    def random_flag(cls, threshold_0_1=0.5):
        a = random.random() < threshold_0_1
        return a

    @classmethod
    def show_compare_img_ls(cls, img_ls):
        flag = True
        assert len(img_ls) > 1
        N = len(img_ls)
        ind = 0
        while flag:
            cv2.imshow('show_compare_two_img', img_ls[ind % N])
            k = cv2.waitKey()
            if k == ord('q'):
                flag = False
            elif k == ord('a'):
                ind -= 1
            else:
                ind += 1


class file_tools():
    class Nori_tools():
        # need id, shape and type to read nori data
        class Fetcher():

            def __init__(self):
                self.fetcher = nori.Fetcher()

            def get(self, id=0, shape=(320, 640, 3), data_type=np.uint8, **kwargs):
                byte_data = self.fetcher.get(id)
                img = np.frombuffer(byte_data, data_type)
                img = np.reshape(img, shape)
                return img
            
            def get_compressed(self, id=0, shape=(320, 640, 3), data_type=np.uint8, **kwargs):
                byte_data = self.fetcher.get(id)
                from balls.imgproc import imdecode
                img = imdecode(byte_data)
                img = np.reshape(img, shape)
                return img

            def getEvent(self, id=0, data_type=np.float64, **kwargs):
                byte_data = self.fetcher.get(id)
                event = np.frombuffer(byte_data, data_type)
                return event

        # speed up nori
        @classmethod
        def nori_speedup_file(cls, file_path):
            print(file_path + ' speed up!!!!!!!!!!!')
            os.system('nori speedup ' + file_path + ' --on' + ' --replica 6')

        class Nori_saver():
            '''
            how to use：
            1. initialize an instance by giving out_nori_path, which is the dir to save nori file
            2. use 'start' to do some initialization
            3. prepare your image(numpy) and name(str)
            4. use 'end' to finish saving nori file (note that: if you do not use this function, nori file cannot be correctly saved)
            5. you can optionally speed up your nori file

            '''

            recorder_nr = None

            def __init__(self, out_nori_path='', if_remote=False):
                self.out_nori_path = out_nori_path
                self.if_remote = if_remote

            def start(self):
                if self.if_remote:
                    self.recorder_nr = nori.remotewriteopen(self.out_nori_path)
                else:
                    pass
                self.recorder_nr = nori.open(self.out_nori_path, "w")
                print('=' * 3 + ' start making nori file: %s' % self.out_nori_path)

            def end(self):
                self.recorder_nr.close()

            def put_file(self, img, name):
                img_temp = img.tobytes()
                data_id1 = self.recorder_nr.put(img_temp, filename=name)
                data = {'id': data_id1, 'shape': img.shape, 'data_type': img.dtype, 'name': name}
                return data
        
            def put_compressed_file(self, img, name):
                from balls.imgproc import imencode
                _,img_temp=imencode('.np4', img)
                data_id1 = self.recorder_nr.put(img_temp, filename=name)
                data = {'id': data_id1, 'shape': img.shape, 'data_type': img.dtype, 'name': name}
                return data

            def upload_oss_and_speedup(self, dataset_name):
                oss_path = file_tools.Nori_tools.save_filedir_to_oss(self.out_nori_path, dataset_name)
                file_tools.Nori_tools.nori_speedup_file(oss_path)

            def nori_speedup(self):
                pcmd = 'nori speedup ' + self.out_nori_path + ' --on' + ' --replica 2'
                print('===')
                print(pcmd)
                print('===')
                os.system(pcmd)

        @classmethod
        def demo(cls):
            # === save nori files direcctly to oss
            nori_path = ''
            data_ls = [np.zeros((10, 10, 3)), ]
            nsaver = file_tools.Nori_tools.Nori_saver(out_nori_path=nori_path, if_remote=True)
            nsaver.start()
            new_data_ls = []
            for ind, data in enumerate(data_ls):
                name = '%s' % ind
                nori_id = nsaver.put_file(img=data, name=name)
                new_data_ls.append(nori_id)
            nsaver.end()
            # === end save nori file, and speed up
            nsaver.nori_speedup()
            # === wait a minute to speed up nori file
            import time
            for i in range(10):
                print('waite a minute...[%i|10]' % i)
                time.sleep(6)

            # === test read nori files
            nori_fetcher = file_tools.Nori_tools.Fetcher()
            for ind in new_data_ls:
                data = nori_fetcher.get(**ind)
                tensor_tools.check_tensor_np(data, ind['name'])

    class npz_saver():

        @classmethod
        def save_npz(cls, files, npz_save_path):
            np.savez(npz_save_path, files=[files, 0])

        @classmethod
        def load_npz(cls, npz_save_path):
            with np.load(npz_save_path) as fin:
                files = fin['files']
                files = list(files)
                return files[0]

    class pickle_saver():

        @classmethod
        def save_pickle(cls, files, file_path):
            with open(file_path, 'wb') as data:
                pickle.dump(files, data)

        @classmethod
        def load_picke(cls, file_path):
            with open(file_path, 'rb') as data:
                data = pickle.load(data)
            return data

    class txt_read_write():
        @classmethod
        def read(cls, path):
            with open(path, "r") as f:
                data = f.readlines()
            return data

        @classmethod
        def write(cls, path, data_ls):
            # file_write_obj = open(path, 'a')
            file_write_obj = open(path, 'w')
            for i in data_ls:
                file_write_obj.writelines(i)
            file_write_obj.close()

        @classmethod
        def demo(cls):
            txt_path = r'E:\research\unsupervised_optical_flow\projects\Ric-master\Ric-master\data\MPI-Sintel\frame_0001_match.txt'
            data = file_tools.txt_read_write.read(txt_path)
            print(' ')
            write_txt_path = txt_path = r'E:\research\unsupervised_optical_flow\projects\Ric-master\Ric-master\data\MPI-Sintel\temp.txt'
            file_tools.txt_read_write.write(write_txt_path, data[:10])
            print(' ')

    class flow_read_write():

        @classmethod
        def write_flow_png(cls, filename, uv, v=None, mask=None):

            if v is None:
                assert (uv.ndim == 3)
                assert (uv.shape[2] == 2)
                u = uv[:, :, 0]
                v = uv[:, :, 1]
            else:
                u = uv

            assert (u.shape == v.shape)

            height_img, width_img = u.shape
            if mask is None:
                valid_mask = np.ones([height_img, width_img], dtype=np.uint16)
            else:
                valid_mask = mask

            flow_u = np.clip((u * 64 + 2 ** 15), 0.0, 65535.0).astype(np.uint16)
            flow_v = np.clip((v * 64 + 2 ** 15), 0.0, 65535.0).astype(np.uint16)

            output = np.stack((flow_u, flow_v, valid_mask), axis=-1)

            with open(filename, 'wb') as f:
                # writer = png.Writer(width=width_img, height=height_img, bitdepth=16)
                # temp = np.reshape(output, (-1, width_img * 3))
                # writer.write(f, temp)

                png_writer = png.Writer(width=width_img, height=height_img, bitdepth=16, compression=3, greyscale=False)
                # png_writer.write_array(f, output)
                temp = np.reshape(output, (-1, width_img * 3))
                png_writer.write(f, temp)

        @classmethod
        def write_kitti_png_file(cls, flow_fn, flow_data, mask_data=None):
            flow_img = np.zeros((flow_data.shape[0], flow_data.shape[1], 3),
                                dtype=np.uint16)
            if mask_data is None:
                mask_data = np.ones([flow_data.shape[0], flow_data.shape[1]], dtype=np.uint16)
            flow_img[:, :, 2] = (flow_data[:, :, 0] * 64.0 + 2 ** 15).astype(np.uint16)
            flow_img[:, :, 1] = (flow_data[:, :, 1] * 64.0 + 2 ** 15).astype(np.uint16)
            flow_img[:, :, 0] = mask_data[:, :]
            cv2.imwrite(flow_fn, flow_img)

        @classmethod
        def read_flo(cls, filename):
            with open(filename, 'rb') as f:
                magic = np.fromfile(f, np.float32, count=1)
                if 202021.25 != magic:
                    print('Magic number incorrect. Invalid .flo file')
                else:
                    w = np.fromfile(f, np.int32, count=1)
                    h = np.fromfile(f, np.int32, count=1)
                    data = np.fromfile(f, np.float32, count=int(2 * w * h))
                    # Reshape data into 3D array (columns, rows, bands)
                    data2D = np.resize(data, (h[0], w[0], 2))
                    return data2D

        @classmethod
        def write_flo(cls, flow, filename):
            """
            write optical flow in Middlebury .flo format
            :param flow: optical flow map
            :param filename: optical flow file path to be saved
            :return: None
            """
            f = open(filename, 'wb')
            magic = np.array([202021.25], dtype=np.float32)
            (height, width) = flow.shape[0:2]
            w = np.array([width], dtype=np.int32)
            h = np.array([height], dtype=np.int32)
            magic.tofile(f)
            w.tofile(f)
            h.tofile(f)
            flow.tofile(f)
            f.close()

        @classmethod
        def point_vec(cls, img, flow, valid=None):
            meshgrid = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
            rate_x = 1
            rate_y = 1
            dispimg = cv2.resize(img, None, fx=rate_x, fy=rate_y)
            # colorflow = tools.flow_to_image(flow).astype(int)
            colorflow = tensor_tools.flow_to_image_ndmax(flow)
            if valid is None:
                valid = np.ones((img.shape[0], img.shape[1]), dtype=flow.dtype)
            for i in range(img.shape[1]):  # x
                for j in range(img.shape[0]):  # y
                    # if flow[j, i, 2] != 1: continue
                    if valid[j, i] != 1: continue
                    if j % 20 != 0 or i % 20 != 0: continue
                    xend = int((meshgrid[0][j, i] + flow[j, i, 0]) * rate_x)
                    yend = int((meshgrid[1][j, i] + flow[j, i, 1]) * rate_y)
                    leng = np.linalg.norm(flow[j, i, :2])
                    if leng < 1: continue
                    dispimg = cv2.arrowedLine(dispimg, (meshgrid[0][j, i] * rate_y, meshgrid[1][j, i] * rate_x), (xend, yend),
                                            (int(colorflow[j, i, 0]), int(colorflow[j, i, 1]), int(colorflow[j, i, 2])), 1,
                                            tipLength=8 / leng, line_type=cv2.LINE_AA)
            return dispimg

    @classmethod
    def check_dir(cls, path):
        if not os.path.exists(path):
            os.makedirs(path)

    @classmethod
    def tryremove(cls, name, file=False):
        try:
            if file:
                os.remove(name)
            else:
                rmtree(name)
        except OSError:
            pass

    @classmethod
    def extract_zip(cls, zip_path, extract_dir):
        print('unzip file: %s' % zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)


class tensor_tools():

    class meshflow():
        @classmethod
        def check_out_bounds(cls, point_i,point_j,HEIGHT,WIDTH):
            if(point_i>=HEIGHT):
                point_i=HEIGHT-1
            elif(point_i<0):
                point_i=0
            if(point_j>=WIDTH):
                point_j=WIDTH-1
            elif(point_j<0):
                point_j=0
            return point_i,point_j

        @classmethod
        def motion_propagate(cls, fflow, MESH=16, RADIUS=1, filter_size = 5):
            from scipy.signal import medfilt2d

            if(fflow.shape[0]==2):
                fflow.transpose(1,2,0)
            u = fflow[...,0]
            v = fflow[...,1]

            HEIGHT, WIDTH = fflow.shape[0], fflow.shape[1]

            # spreads motion over the mesh for the old_frame    
            mesh_cols, mesh_rows = WIDTH//MESH, HEIGHT//MESH
            x_motion = {}
            y_motion = {}
            for i in range(MESH):
                for j in range(MESH):
                    x_motion.update({(i,j):[]})
                    y_motion.update({(i,j):[]})

                    for r in range(RADIUS):
                        offect_x = r*mesh_rows//2
                        offect_y = r*mesh_cols//2
                        point_i = mesh_rows*i+offect_x
                        point_j = mesh_cols*j+offect_y
                        point_i,point_j=cls.check_out_bounds(point_i,point_j,HEIGHT,WIDTH)
                        x_motion[i, j].append(u[point_i,point_j])
                        y_motion[i, j].append(v[point_i,point_j])
                        point_i = mesh_rows*i+offect_x
                        point_j = mesh_cols*j-offect_y
                        point_i,point_j=cls.check_out_bounds(point_i,point_j,HEIGHT,WIDTH)
                        x_motion[i, j].append(u[point_i,point_j])
                        y_motion[i, j].append(v[point_i,point_j])
                        point_i = mesh_rows*i-offect_x
                        point_j = mesh_cols*j+offect_y
                        point_i,point_j=cls.check_out_bounds(point_i,point_j,HEIGHT,WIDTH)
                        x_motion[i, j].append(u[point_i,point_j])
                        y_motion[i, j].append(v[point_i,point_j])
                        point_i = mesh_rows*i-offect_x
                        point_j = mesh_cols*j-offect_y
                        point_i,point_j=cls.check_out_bounds(point_i,point_j,HEIGHT,WIDTH)
                        x_motion[i, j].append(u[point_i,point_j])
                        y_motion[i, j].append(v[point_i,point_j])

            # apply median filter (f-1) on obtained motion for each vertex
            x_motion_mesh = np.zeros((MESH, MESH), dtype=float)
            y_motion_mesh = np.zeros((MESH, MESH), dtype=float)
            for key in x_motion.keys():
                if(len(x_motion[key])>0):
                    x_motion[key].sort()
                    x_motion_mesh[key] = x_motion[key][len(x_motion[key])//2]
                if(len(y_motion[key])>0):
                    y_motion[key].sort()
                    y_motion_mesh[key] = y_motion[key][len(y_motion[key])//2]

            # apply second median filter (f-2) over the motion mesh for outliers
            
            pad_size = (filter_size - 1) // 2
            x_motion_mesh_ = cv2.copyMakeBorder(x_motion_mesh,pad_size,pad_size,pad_size,pad_size,cv2.BORDER_REPLICATE)
            y_motion_mesh_ = cv2.copyMakeBorder(y_motion_mesh,pad_size,pad_size,pad_size,pad_size,cv2.BORDER_REPLICATE)
            x_motion_mesh_ = medfilt2d(x_motion_mesh_, [filter_size, filter_size])
            y_motion_mesh_ = medfilt2d(y_motion_mesh_, [filter_size, filter_size])

            x_mesh = x_motion_mesh_[pad_size:(pad_size+MESH), pad_size:(pad_size+MESH)]
            y_mesh = y_motion_mesh_[pad_size:(pad_size+MESH), pad_size:(pad_size+MESH)]
            
            return  np.stack((x_mesh, y_mesh), axis=-1)

    class nianjin_warp():

        @classmethod
        def get_grid(cls, batch_size, H, W, start):
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
            ones = torch.ones_like(xx)
            grid = torch.cat((xx, yy, ones), 1).float()
            if torch.cuda.is_available():
                grid = grid.cuda()
            # print("grid",grid.shape)
            # print("start", start)
            grid[:, :2, :, :] = grid[:, :2, :, :] + start  # 加上patch在原图内的偏移量

            return grid

        @classmethod
        def transformer(cls, I, vgrid, train=True):
            # I: Img, shape: batch_size, 1, full_h, full_w
            # vgrid: vgrid, target->source, shape: batch_size, 2, patch_h, patch_w
            # outsize: (patch_h, patch_w)

            def _repeat(x, n_repeats):

                rep = torch.ones([n_repeats, ]).unsqueeze(0)
                rep = rep.int()
                x = x.int()

                x = torch.matmul(x.reshape([-1, 1]), rep)
                return x.reshape([-1])

            def _interpolate(im, x, y, out_size, scale_h):
                # x: x_grid_flat
                # y: y_grid_flat
                # out_size: same as im.size
                # scale_h: True if normalized
                # constants
                num_batch, num_channels, height, width = im.size()

                out_height, out_width = out_size[0], out_size[1]
                # zero = torch.zeros_like([],dtype='int32')
                zero = 0
                max_y = height - 1
                max_x = width - 1
                if scale_h:
                    # scale indices from [-1, 1] to [0, width or height]
                    # print('--Inter- scale_h:', scale_h)
                    x = (x + 1.0) * (height) / 2.0
                    y = (y + 1.0) * (width) / 2.0

                # do sampling
                x0 = torch.floor(x).int()
                x1 = x0 + 1
                y0 = torch.floor(y).int()
                y1 = y0 + 1

                x0 = torch.clamp(x0, zero, max_x)  # same as np.clip
                x1 = torch.clamp(x1, zero, max_x)
                y0 = torch.clamp(y0, zero, max_y)
                y1 = torch.clamp(y1, zero, max_y)

                dim1 = torch.from_numpy(np.array(width * height))
                dim2 = torch.from_numpy(np.array(width))

                base = _repeat(torch.arange(0, num_batch) * dim1, out_height * out_width)  # 其实就是单纯标出batch中每个图的下标位置
                # base = torch.arange(0,num_batch) * dim1
                # base = base.reshape(-1, 1).repeat(1, out_height * out_width).reshape(-1).int()
                # 区别？expand不对数据进行拷贝 .reshape(-1,1).expand(-1,out_height * out_width).reshape(-1)
                if torch.cuda.is_available():
                    dim2 = dim2.cuda()
                    dim1 = dim1.cuda()
                    y0 = y0.cuda()
                    y1 = y1.cuda()
                    x0 = x0.cuda()
                    x1 = x1.cuda()
                    base = base.cuda()
                base_y0 = base + y0 * dim2
                base_y1 = base + y1 * dim2
                idx_a = base_y0 + x0
                idx_b = base_y1 + x0
                idx_c = base_y0 + x1
                idx_d = base_y1 + x1

                # use indices to lookup pixels in the flat image and restore
                # channels dim
                im = im.permute(0, 2, 3, 1)
                im_flat = im.reshape([-1, num_channels]).float()

                idx_a = idx_a.unsqueeze(-1).long()
                idx_a = idx_a.expand(out_height * out_width * num_batch, num_channels)
                Ia = torch.gather(im_flat, 0, idx_a)

                idx_b = idx_b.unsqueeze(-1).long()
                idx_b = idx_b.expand(out_height * out_width * num_batch, num_channels)
                Ib = torch.gather(im_flat, 0, idx_b)

                idx_c = idx_c.unsqueeze(-1).long()
                idx_c = idx_c.expand(out_height * out_width * num_batch, num_channels)
                Ic = torch.gather(im_flat, 0, idx_c)

                idx_d = idx_d.unsqueeze(-1).long()
                idx_d = idx_d.expand(out_height * out_width * num_batch, num_channels)
                Id = torch.gather(im_flat, 0, idx_d)

                # and finally calculate interpolated values
                x0_f = x0.float()
                x1_f = x1.float()
                y0_f = y0.float()
                y1_f = y1.float()

                wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
                wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
                wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
                wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
                output = wa * Ia + wb * Ib + wc * Ic + wd * Id

                return output

            def _transform(I, vgrid, scale_h):

                C_img = I.shape[1]
                B, C, H, W = vgrid.size()

                x_s_flat = vgrid[:, 0, ...].reshape([-1])
                y_s_flat = vgrid[:, 1, ...].reshape([-1])
                out_size = vgrid.shape[2:]
                input_transformed = _interpolate(I, x_s_flat, y_s_flat, out_size, scale_h)

                output = input_transformed.reshape([B, H, W, C_img])
                return output

            # scale_h = True
            output = _transform(I, vgrid, scale_h=False)
            if train:
                output = output.permute(0, 3, 1, 2)
            return output

        @classmethod
        def warp_im(cls, I_nchw, flow_nchw, start_n211):
            batch_size, _, img_h, img_w = I_nchw.size()
            _, _, patch_size_h, patch_size_w = flow_nchw.size()
            patch_indices = cls.get_grid(batch_size, patch_size_h, patch_size_w, start_n211)
            vgrid = patch_indices[:, :2, ...]
            # grid_warp = vgrid - flow_nchw
            grid_warp = vgrid + flow_nchw
            pred_I2 = cls.transformer(I_nchw, grid_warp)
            return pred_I2

    class occ_check_model():

        def __init__(self, occ_type='for_back_check', occ_alpha_1=1.0, occ_alpha_2=0.05, obj_out_all='all'):
            self.occ_type_ls = ['for_back_check', 'forward_warp', 'for_back_check&forward_warp']
            assert occ_type in self.occ_type_ls
            assert obj_out_all in ['obj', 'out', 'all']
            self.occ_type = occ_type
            self.occ_alpha_1 = occ_alpha_1
            self.occ_alpha_2 = occ_alpha_2
            self.sum_abs_or_squar = False
            self.obj_out_all = obj_out_all

        def __call__(self, flow_f, flow_b, scale=1):
            # 输入进来是可使用的光流

            if self.obj_out_all == 'all':
                if self.occ_type == 'for_back_check':
                    occ_1, occ_2 = self._forward_backward_occ_check(flow_fw=flow_f, flow_bw=flow_b, scale=scale)
                elif self.occ_type == 'forward_warp':
                    raise ValueError('not implemented')
                elif self.occ_type == 'for_back_check&forward_warp':
                    raise ValueError('not implemented')
                else:
                    raise ValueError('occ type should be in %s, get %s' % (self.occ_type_ls, self.occ_type))
                return occ_1, occ_2
            elif self.obj_out_all == 'obj':
                if self.occ_type == 'for_back_check':
                    occ_1, occ_2 = self._forward_backward_occ_check(flow_fw=flow_f, flow_bw=flow_b, scale=scale)
                elif self.occ_type == 'forward_warp':
                    raise ValueError('not implemented')
                elif self.occ_type == 'for_back_check&forward_warp':
                    raise ValueError('not implemented')
                else:
                    raise ValueError('occ type should be in %s, get %s' % (self.occ_type_ls, self.occ_type))
                out_occ_fw = self.torch_outgoing_occ_check(flow_f)
                out_occ_bw = self.torch_outgoing_occ_check(flow_b)
                obj_occ_fw = self.torch_get_obj_occ_check(occ_mask=occ_1, out_occ=out_occ_fw)
                obj_occ_bw = self.torch_get_obj_occ_check(occ_mask=occ_2, out_occ=out_occ_bw)
                return obj_occ_fw, obj_occ_bw
            elif self.obj_out_all == 'out':
                out_occ_fw = self.torch_outgoing_occ_check(flow_f)
                out_occ_bw = self.torch_outgoing_occ_check(flow_b)
                return out_occ_fw, out_occ_bw
            else:
                raise ValueError("obj_out_all should be in ['obj','out','all'], but get: %s" % self.obj_out_all)

        def _forward_backward_occ_check(self, flow_fw, flow_bw, scale=1):
            """
            In this function, the parameter alpha needs to be improved
            """

            def length_sq_v0(x):
                # torch.sum(x ** 2, dim=1, keepdim=True)
                # temp = torch.sum(x ** 2, dim=1, keepdim=True)
                # temp = torch.pow(temp, 0.5)
                return torch.sum(torch.pow(x ** 2, 0.5), dim=1, keepdim=True)
                # return temp

            def length_sq(x):
                # torch.sum(x ** 2, dim=1, keepdim=True)
                temp = torch.sum(x ** 2, dim=1, keepdim=True)
                temp = torch.pow(temp, 0.5)
                # return torch.sum(torch.pow(x ** 2, 0.5), dim=1, keepdim=True)
                return temp

            if self.sum_abs_or_squar:
                sum_func = length_sq_v0
            else:
                sum_func = length_sq
            mag_sq = sum_func(flow_fw) + sum_func(flow_bw)
            flow_bw_warped = tensor_tools.torch_warp(flow_bw, flow_fw)  # torch_warp(img,flow)
            flow_fw_warped = tensor_tools.torch_warp(flow_fw, flow_bw)
            flow_diff_fw = flow_fw + flow_bw_warped
            flow_diff_bw = flow_bw + flow_fw_warped
            occ_thresh = self.occ_alpha_1 * mag_sq + self.occ_alpha_2 / scale
            occ_fw = sum_func(flow_diff_fw) < occ_thresh  # 0 means the occlusion region where the photo loss we should ignore
            occ_bw = sum_func(flow_diff_bw) < occ_thresh
            # if IF_DEBUG:
            #     temp_ = sum_func(flow_diff_fw)
            #     tools.check_tensor(data=temp_, name='check occlusion mask sum_func flow_diff_fw')
            #     temp_ = sum_func(flow_diff_bw)
            #     tools.check_tensor(data=temp_, name='check occlusion mask sum_func flow_diff_bw')
            #     tools.check_tensor(data=mag_sq, name='check occlusion mask mag_sq')
            #     tools.check_tensor(data=occ_thresh, name='check occlusion mask occ_thresh')
            return occ_fw.float(), occ_bw.float()

        def forward_backward_occ_check(self, flow_fw, flow_bw, alpha1, alpha2, obj_out_all='obj'):
            """
            In this function, the parameter alpha needs to be improved
            """

            def length_sq_v0(x):
                # torch.sum(x ** 2, dim=1, keepdim=True)
                # temp = torch.sum(x ** 2, dim=1, keepdim=True)
                # temp = torch.pow(temp, 0.5)
                return torch.sum(torch.pow(x ** 2, 0.5), dim=1, keepdim=True)
                # return temp

            def length_sq(x):
                # torch.sum(x ** 2, dim=1, keepdim=True)
                temp = torch.sum(x ** 2, dim=1, keepdim=True)
                temp = torch.pow(temp, 0.5)
                # return torch.sum(torch.pow(x ** 2, 0.5), dim=1, keepdim=True)
                return temp

            if self.sum_abs_or_squar:
                sum_func = length_sq_v0
            else:
                sum_func = length_sq
            mag_sq = sum_func(flow_fw) + sum_func(flow_bw)
            flow_bw_warped = tensor_tools.torch_warp(flow_bw, flow_fw)  # torch_warp(img,flow)
            flow_fw_warped = tensor_tools.torch_warp(flow_fw, flow_bw)
            flow_diff_fw = flow_fw + flow_bw_warped
            flow_diff_bw = flow_bw + flow_fw_warped
            occ_thresh = alpha1 * mag_sq + alpha2
            occ_fw = sum_func(flow_diff_fw) < occ_thresh  # 0 means the occlusion region where the photo loss we should ignore
            occ_bw = sum_func(flow_diff_bw) < occ_thresh
            occ_fw = occ_fw.float()
            occ_bw = occ_bw.float()
            # if IF_DEBUG:
            #     temp_ = sum_func(flow_diff_fw)
            #     tools.check_tensor(data=temp_, name='check occlusion mask sum_func flow_diff_fw')
            #     temp_ = sum_func(flow_diff_bw)
            #     tools.check_tensor(data=temp_, name='check occlusion mask sum_func flow_diff_bw')
            #     tools.check_tensor(data=mag_sq, name='check occlusion mask mag_sq')
            #     tools.check_tensor(data=occ_thresh, name='check occlusion mask occ_thresh')
            if obj_out_all == 'obj':
                out_occ_fw = self.torch_outgoing_occ_check(flow_fw)
                out_occ_bw = self.torch_outgoing_occ_check(flow_bw)
                occ_fw = self.torch_get_obj_occ_check(occ_mask=occ_fw, out_occ=out_occ_fw)
                occ_bw = self.torch_get_obj_occ_check(occ_mask=occ_bw, out_occ=out_occ_bw)
            return occ_fw, occ_bw

        def _forward_warp_occ_check(self, flow_bw):  # TODO
            return 0

        @classmethod
        def torch_outgoing_occ_check(cls, flow):

            B, C, H, W = flow.size()
            # mesh grid
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1).float()
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1).float()
            flow_x, flow_y = torch.split(flow, 1, 1)
            if flow.is_cuda:
                xx = xx.cuda()
                yy = yy.cuda()
            # tools.check_tensor(flow_x, 'flow_x')
            # tools.check_tensor(flow_y, 'flow_y')
            # tools.check_tensor(xx, 'xx')
            # tools.check_tensor(yy, 'yy')
            pos_x = xx + flow_x
            pos_y = yy + flow_y
            # tools.check_tensor(pos_x, 'pos_x')
            # tools.check_tensor(pos_y, 'pos_y')
            # print(' ')
            # check mask
            outgoing_mask = torch.ones_like(pos_x)
            outgoing_mask[pos_x > W - 1] = 0
            outgoing_mask[pos_x < 0] = 0
            outgoing_mask[pos_y > H - 1] = 0
            outgoing_mask[pos_y < 0] = 0
            return outgoing_mask.float()

        @classmethod
        def torch_get_obj_occ_check(cls, occ_mask, out_occ):
            outgoing_mask = torch.zeros_like(occ_mask)
            if occ_mask.is_cuda:
                outgoing_mask = outgoing_mask.cuda()
            outgoing_mask[occ_mask == 1] = 1
            outgoing_mask[out_occ == 0] = 1
            return outgoing_mask

    # Part of the code from https://github.com/visinf/irr/blob/master/augmentations.py  ## Portions of Code from, copyright 2018 Jochen Gast
    class Interpolation():
        @classmethod
        def _bchw2bhwc(cls, tensor):
            return tensor.transpose(1, 2).transpose(2, 3)

        @classmethod
        def _bhwc2bchw(cls, tensor):
            return tensor.transpose(2, 3).transpose(1, 2)

        @classmethod
        def resize2D(cls, inputs, size_targets, mode="bilinear"):
            size_inputs = [inputs.size(2), inputs.size(3)]

            if all([size_inputs == size_targets]):
                return inputs  # nothing to do
            elif any([size_targets < size_inputs]):
                resized = F.adaptive_avg_pool2d(inputs, size_targets)  # downscaling
            else:
                resized = F.upsample(inputs, size=size_targets, mode=mode)  # upsampling

            # correct scaling
            return resized

        @classmethod
        def resize2D_as(cls, inputs, output_as, mode="bilinear"):
            size_targets = [output_as.size(2), output_as.size(3)]
            return tensor_tools.Interpolation.resize2D(inputs, size_targets, mode=mode)

        class Meshgrid(nn.Module):
            def __init__(self):
                super(tensor_tools.Interpolation.Meshgrid, self).__init__()
                self.width = 0
                self.height = 0
                self.register_buffer("xx", torch.zeros(1, 1))
                self.register_buffer("yy", torch.zeros(1, 1))
                self.register_buffer("rangex", torch.zeros(1, 1))
                self.register_buffer("rangey", torch.zeros(1, 1))

            def _compute_meshgrid(self, width, height):
                torch.arange(0, width, out=self.rangex)
                torch.arange(0, height, out=self.rangey)
                self.xx = self.rangex.repeat(height, 1).contiguous()
                self.yy = self.rangey.repeat(width, 1).t().contiguous()

            def forward(self, width, height):
                if self.width != width or self.height != height:
                    self._compute_meshgrid(width=width, height=height)
                    self.width = width
                    self.height = height
                return self.xx, self.yy

        class BatchSub2Ind(nn.Module):
            def __init__(self):
                super(tensor_tools.Interpolation.BatchSub2Ind, self).__init__()
                self.register_buffer("_offsets", torch.LongTensor())

            def forward(self, shape, row_sub, col_sub, out=None):
                batch_size = row_sub.size(0)
                height, width = shape
                ind = row_sub * width + col_sub
                torch.arange(batch_size, out=self._offsets)
                self._offsets *= (height * width)

                if out is None:
                    return torch.add(ind, self._offsets.view(-1, 1, 1))
                else:
                    torch.add(ind, self._offsets.view(-1, 1, 1), out=out)

        class Interp2(nn.Module):
            def __init__(self, clamp=False):
                super(tensor_tools.Interpolation.Interp2, self).__init__()
                self._clamp = clamp
                self._batch_sub2ind = tensor_tools.Interpolation.BatchSub2Ind()
                self.register_buffer("_x0", torch.LongTensor())
                self.register_buffer("_x1", torch.LongTensor())
                self.register_buffer("_y0", torch.LongTensor())
                self.register_buffer("_y1", torch.LongTensor())
                self.register_buffer("_i00", torch.LongTensor())
                self.register_buffer("_i01", torch.LongTensor())
                self.register_buffer("_i10", torch.LongTensor())
                self.register_buffer("_i11", torch.LongTensor())
                self.register_buffer("_v00", torch.FloatTensor())
                self.register_buffer("_v01", torch.FloatTensor())
                self.register_buffer("_v10", torch.FloatTensor())
                self.register_buffer("_v11", torch.FloatTensor())
                self.register_buffer("_x", torch.FloatTensor())
                self.register_buffer("_y", torch.FloatTensor())

            def forward(self, v, xq, yq):
                batch_size, channels, height, width = v.size()

                # clamp if wanted
                if self._clamp:
                    xq.clamp_(0, width - 1)
                    yq.clamp_(0, height - 1)

                # ------------------------------------------------------------------
                # Find neighbors
                #
                # x0 = torch.floor(xq).long(),          x0.clamp_(0, width - 1)
                # x1 = x0 + 1,                          x1.clamp_(0, width - 1)
                # y0 = torch.floor(yq).long(),          y0.clamp_(0, height - 1)
                # y1 = y0 + 1,                          y1.clamp_(0, height - 1)
                #
                # ------------------------------------------------------------------
                self._x0 = torch.floor(xq).long().clamp(0, width - 1)
                self._y0 = torch.floor(yq).long().clamp(0, height - 1)

                self._x1 = torch.add(self._x0, 1).clamp(0, width - 1)
                self._y1 = torch.add(self._y0, 1).clamp(0, height - 1)

                # batch_sub2ind
                self._batch_sub2ind([height, width], self._y0, self._x0, out=self._i00)
                self._batch_sub2ind([height, width], self._y0, self._x1, out=self._i01)
                self._batch_sub2ind([height, width], self._y1, self._x0, out=self._i10)
                self._batch_sub2ind([height, width], self._y1, self._x1, out=self._i11)

                # reshape
                v_flat = tensor_tools.Interpolation._bchw2bhwc(v).contiguous().view(-1, channels)
                torch.index_select(v_flat, dim=0, index=self._i00.view(-1), out=self._v00)
                torch.index_select(v_flat, dim=0, index=self._i01.view(-1), out=self._v01)
                torch.index_select(v_flat, dim=0, index=self._i10.view(-1), out=self._v10)
                torch.index_select(v_flat, dim=0, index=self._i11.view(-1), out=self._v11)

                # local_coords
                torch.add(xq, - self._x0.float(), out=self._x)
                torch.add(yq, - self._y0.float(), out=self._y)

                # weights
                w00 = torch.unsqueeze((1.0 - self._y) * (1.0 - self._x), dim=1)
                w01 = torch.unsqueeze((1.0 - self._y) * self._x, dim=1)
                w10 = torch.unsqueeze(self._y * (1.0 - self._x), dim=1)
                w11 = torch.unsqueeze(self._y * self._x, dim=1)

                def _reshape(u):
                    return tensor_tools.Interpolation._bhwc2bchw(u.view(batch_size, height, width, channels))

                # values
                values = _reshape(self._v00) * w00 + _reshape(self._v01) * w01 \
                         + _reshape(self._v10) * w10 + _reshape(self._v11) * w11

                if self._clamp:
                    return values
                else:
                    #  find_invalid
                    invalid = ((xq < 0) | (xq >= width) | (yq < 0) | (yq >= height)).unsqueeze(dim=1).float()
                    # maskout invalid
                    transformed = invalid * torch.zeros_like(values) + (1.0 - invalid) * values

                return transformed

    class SP_transform():
        @classmethod
        def denormalize_coords(cls, xx, yy, width, height):
            """ scale indices from [-1, 1] to [0, width/height] """
            xx = 0.5 * (width - 1.0) * (xx.float() + 1.0)
            yy = 0.5 * (height - 1.0) * (yy.float() + 1.0)
            return xx, yy

        @classmethod
        def normalize_coords(cls, xx, yy, width, height):
            """ scale indices from [0, width/height] to [-1, 1] """
            xx = (2.0 / (width - 1.0)) * xx.float() - 1.0
            yy = (2.0 / (height - 1.0)) * yy.float() - 1.0
            return xx, yy

        @classmethod
        def apply_transform_to_params(cls, theta0, theta_transform):
            a1 = theta0[:, 0]
            a2 = theta0[:, 1]
            a3 = theta0[:, 2]
            a4 = theta0[:, 3]
            a5 = theta0[:, 4]
            a6 = theta0[:, 5]
            #
            b1 = theta_transform[:, 0]
            b2 = theta_transform[:, 1]
            b3 = theta_transform[:, 2]
            b4 = theta_transform[:, 3]
            b5 = theta_transform[:, 4]
            b6 = theta_transform[:, 5]
            #
            c1 = a1 * b1 + a4 * b2
            c2 = a2 * b1 + a5 * b2
            c3 = b3 + a3 * b1 + a6 * b2
            c4 = a1 * b4 + a4 * b5
            c5 = a2 * b4 + a5 * b5
            c6 = b6 + a3 * b4 + a6 * b5
            #
            new_theta = torch.stack([c1, c2, c3, c4, c5, c6], dim=1)
            return new_theta

        class _IdentityParams(nn.Module):
            def __init__(self):
                super(tensor_tools.SP_transform._IdentityParams, self).__init__()
                self._batch_size = 0
                self.register_buffer("_o", torch.FloatTensor())
                self.register_buffer("_i", torch.FloatTensor())

            def _update(self, batch_size):
                torch.zeros([batch_size, 1], out=self._o)
                torch.ones([batch_size, 1], out=self._i)
                return torch.cat([self._i, self._o, self._o, self._o, self._i, self._o], dim=1)

            def forward(self, batch_size):
                if self._batch_size != batch_size:
                    self._identity_params = self._update(batch_size)
                    self._batch_size = batch_size
                return self._identity_params

        class RandomMirror(nn.Module):
            def __init__(self, vertical=True, p=0.5):
                super(tensor_tools.SP_transform.RandomMirror, self).__init__()
                self._batch_size = 0
                self._p = p
                self._vertical = vertical
                self.register_buffer("_mirror_probs", torch.FloatTensor())

            def update_probs(self, batch_size):
                torch.ones([batch_size, 1], out=self._mirror_probs)
                self._mirror_probs *= self._p

            def forward(self, theta_list):
                batch_size = theta_list[0].size(0)
                if batch_size != self._batch_size:
                    self.update_probs(batch_size)
                    self._batch_size = batch_size

                # apply random sign to a1 a2 a3 (these are the guys responsible for x)
                sign = torch.sign(2.0 * torch.bernoulli(self._mirror_probs) - 1.0)
                i = torch.ones_like(sign)
                horizontal_mirror = torch.cat([sign, sign, sign, i, i, i], dim=1)
                theta_list = [theta * horizontal_mirror for theta in theta_list]

                # apply random sign to a4 a5 a6 (these are the guys responsible for y)
                if self._vertical:
                    sign = torch.sign(2.0 * torch.bernoulli(self._mirror_probs) - 1.0)
                    vertical_mirror = torch.cat([i, i, i, sign, sign, sign], dim=1)
                    theta_list = [theta * vertical_mirror for theta in theta_list]

                return theta_list

        class RandomAffineFlow(nn.Module):
            def __init__(self, cfg, addnoise=True):
                super(tensor_tools.SP_transform.RandomAffineFlow, self).__init__()
                self.cfg = cfg
                self._interp2 = tensor_tools.Interpolation.Interp2(clamp=False)
                self._flow_interp2 = tensor_tools.Interpolation.Interp2(clamp=False)
                self._meshgrid = tensor_tools.Interpolation.Meshgrid()
                self._identity = tensor_tools.SP_transform._IdentityParams()
                self._random_mirror = tensor_tools.SP_transform.RandomMirror(cfg.vflip) if cfg.hflip else tensor_tools.SP_transform.RandomMirror(p=1)
                self._addnoise = addnoise

                self.register_buffer("_noise1", torch.FloatTensor())
                self.register_buffer("_noise2", torch.FloatTensor())
                self.register_buffer("_xbounds", torch.FloatTensor([-1, -1, 1, 1]))
                self.register_buffer("_ybounds", torch.FloatTensor([-1, 1, -1, 1]))
                self.register_buffer("_x", torch.IntTensor(1))
                self.register_buffer("_y", torch.IntTensor(1))

            def inverse_transform_coords(self, width, height, thetas, offset_x=None,
                                         offset_y=None):
                xx, yy = self._meshgrid(width=width, height=height)

                xx = torch.unsqueeze(xx, dim=0).float()
                yy = torch.unsqueeze(yy, dim=0).float()

                if offset_x is not None:
                    xx = xx + offset_x
                if offset_y is not None:
                    yy = yy + offset_y

                a1 = thetas[:, 0].contiguous().view(-1, 1, 1)
                a2 = thetas[:, 1].contiguous().view(-1, 1, 1)
                a3 = thetas[:, 2].contiguous().view(-1, 1, 1)
                a4 = thetas[:, 3].contiguous().view(-1, 1, 1)
                a5 = thetas[:, 4].contiguous().view(-1, 1, 1)
                a6 = thetas[:, 5].contiguous().view(-1, 1, 1)

                xx, yy = tensor_tools.SP_transform.normalize_coords(xx, yy, width=width, height=height)
                xq = a1 * xx + a2 * yy + a3
                yq = a4 * xx + a5 * yy + a6
                xq, yq = tensor_tools.SP_transform.denormalize_coords(xq, yq, width=width, height=height)
                return xq, yq

            def transform_coords(self, width, height, thetas):
                xx1, yy1 = self._meshgrid(width=width, height=height)
                xx, yy = tensor_tools.SP_transform.normalize_coords(xx1, yy1, width=width, height=height)

                def _unsqueeze12(u):
                    return torch.unsqueeze(torch.unsqueeze(u, dim=1), dim=1)

                a1 = _unsqueeze12(thetas[:, 0])
                a2 = _unsqueeze12(thetas[:, 1])
                a3 = _unsqueeze12(thetas[:, 2])
                a4 = _unsqueeze12(thetas[:, 3])
                a5 = _unsqueeze12(thetas[:, 4])
                a6 = _unsqueeze12(thetas[:, 5])
                #
                z = a1 * a5 - a2 * a4
                b1 = a5 / z
                b2 = - a2 / z
                b4 = - a4 / z
                b5 = a1 / z
                #
                xhat = xx - a3
                yhat = yy - a6
                xq = b1 * xhat + b2 * yhat
                yq = b4 * xhat + b5 * yhat

                xq, yq = tensor_tools.SP_transform.denormalize_coords(xq, yq, width=width, height=height)
                return xq, yq

            def find_invalid(self, width, height, thetas):
                x = self._xbounds
                y = self._ybounds
                #
                a1 = torch.unsqueeze(thetas[:, 0], dim=1)
                a2 = torch.unsqueeze(thetas[:, 1], dim=1)
                a3 = torch.unsqueeze(thetas[:, 2], dim=1)
                a4 = torch.unsqueeze(thetas[:, 3], dim=1)
                a5 = torch.unsqueeze(thetas[:, 4], dim=1)
                a6 = torch.unsqueeze(thetas[:, 5], dim=1)
                #
                z = a1 * a5 - a2 * a4
                b1 = a5 / z
                b2 = - a2 / z
                b4 = - a4 / z
                b5 = a1 / z
                #
                xhat = x - a3
                yhat = y - a6
                xq = b1 * xhat + b2 * yhat
                yq = b4 * xhat + b5 * yhat
                xq, yq = tensor_tools.SP_transform.denormalize_coords(xq, yq, width=width, height=height)
                #
                invalid = (
                                  (xq < 0) | (yq < 0) | (xq >= width) | (yq >= height)
                          ).sum(dim=1, keepdim=True) > 0

                return invalid

            def apply_random_transforms_to_params(self,
                                                  theta0,
                                                  max_translate,
                                                  min_zoom, max_zoom,
                                                  min_squeeze, max_squeeze,
                                                  min_rotate, max_rotate,
                                                  validate_size=None):
                max_translate *= 0.5
                batch_size = theta0.size(0)
                height, width = validate_size

                # collect valid params here
                thetas = torch.zeros_like(theta0)

                zoom = theta0.new(batch_size, 1).zero_()
                squeeze = torch.zeros_like(zoom)
                tx = torch.zeros_like(zoom)
                ty = torch.zeros_like(zoom)
                phi = torch.zeros_like(zoom)
                invalid = torch.ones_like(zoom).byte()

                while invalid.sum() > 0:
                    # random sampling
                    zoom.uniform_(min_zoom, max_zoom)
                    squeeze.uniform_(min_squeeze, max_squeeze)
                    tx.uniform_(-max_translate, max_translate)
                    ty.uniform_(-max_translate, max_translate)
                    phi.uniform_(-min_rotate, max_rotate)

                    # construct affine parameters
                    sx = zoom * squeeze
                    sy = zoom / squeeze
                    sin_phi = torch.sin(phi)
                    cos_phi = torch.cos(phi)
                    b1 = cos_phi * sx
                    b2 = sin_phi * sy
                    b3 = tx
                    b4 = - sin_phi * sx
                    b5 = cos_phi * sy
                    b6 = ty

                    theta_transform = torch.cat([b1, b2, b3, b4, b5, b6], dim=1)
                    theta_try = tensor_tools.SP_transform.apply_transform_to_params(theta0, theta_transform)
                    thetas = invalid.float() * theta_try + (1 - invalid.float()).float() * thetas

                    # compute new invalid ones
                    invalid = self.find_invalid(width=width, height=height, thetas=thetas)

                # here we should have good thetas within borders
                return thetas

            def transform_image(self, images, thetas):
                batch_size, channels, height, width = images.size()
                xq, yq = self.transform_coords(width=width, height=height, thetas=thetas)
                transformed = self._interp2(images, xq, yq)
                return transformed

            def transform_flow(self, flow, theta1, theta2):
                batch_size, channels, height, width = flow.size()
                u = flow[:, 0, :, :]
                v = flow[:, 1, :, :]

                # inverse transform coords
                x0, y0 = self.inverse_transform_coords(
                    width=width, height=height, thetas=theta1)

                x1, y1 = self.inverse_transform_coords(
                    width=width, height=height, thetas=theta2, offset_x=u, offset_y=v)

                # subtract and create new flow
                u = x1 - x0
                v = y1 - y0
                new_flow = torch.stack([u, v], dim=1)

                # transform coords
                xq, yq = self.transform_coords(width=width, height=height, thetas=theta1)

                # interp2
                transformed = self._flow_interp2(new_flow, xq, yq)
                return transformed

            def forward(self, data):
                # 01234 flow 12 21 23 32
                imgs = data['imgs']
                flows_f = data['flows_f']
                masks_f = data['masks_f']

                batch_size, _, height, width = imgs[0].size()

                # identity = no transform
                theta0 = self._identity(batch_size)

                # global transform
                theta_list = [self.apply_random_transforms_to_params(
                    theta0,
                    max_translate=self.cfg.trans[0],
                    min_zoom=self.cfg.zoom[0], max_zoom=self.cfg.zoom[1],
                    min_squeeze=self.cfg.squeeze[0], max_squeeze=self.cfg.squeeze[1],
                    min_rotate=self.cfg.rotate[0], max_rotate=self.cfg.rotate[1],
                    validate_size=[height, width])
                ]

                # relative transform
                for i in range(len(imgs) - 1):
                    theta_list.append(
                        self.apply_random_transforms_to_params(
                            theta_list[-1],
                            max_translate=self.cfg.trans[1],
                            min_zoom=self.cfg.zoom[2], max_zoom=self.cfg.zoom[3],
                            min_squeeze=self.cfg.squeeze[2], max_squeeze=self.cfg.squeeze[3],
                            min_rotate=-self.cfg.rotate[2], max_rotate=self.cfg.rotate[2],
                            validate_size=[height, width])
                    )

                # random flip images
                theta_list = self._random_mirror(theta_list)

                # 01234
                imgs = [self.transform_image(im, theta) for im, theta in zip(imgs, theta_list)]

                if len(imgs) > 2:
                    theta_list = theta_list[1:-1]
                # 12 23
                flows_f = [self.transform_flow(flo, theta1, theta2) for flo, theta1, theta2 in
                           zip(flows_f, theta_list[:-1], theta_list[1:])]

                masks_f = [self.transform_image(mask, theta) for mask, theta in
                           zip(masks_f, theta_list)]

                if self._addnoise:
                    '''
                    im1 <class 'torch.Tensor'> (3, 320, 1152) max 0.5885537 min -0.4305366 mean -0.040912468
                    im1 <class 'torch.Tensor'> (3, 320, 1152) max 0.5885537 min -0.4305366 mean -0.03847942
                    im1 <class 'torch.Tensor'> (3, 320, 1152) max 0.5885537 min -0.4187718 mean -0.011021424
                    '''
                    stddev = np.random.uniform(0.0, 0.04)
                    for im in imgs:
                        noise = torch.zeros_like(im)
                        noise.normal_(std=stddev)
                        im.add_(noise)
                        im.clamp_(-1.0, 1.0)

                data['imgs'] = imgs
                data['flows_f'] = flows_f
                data['masks_f'] = masks_f
                return data

            class config():
                def __init__(self):
                    self.add_noise = False
                    self.hflip = True
                    self.rotate = [-0.01, 0.01, -0.01, 0.01]
                    self.squeeze = [1.0, 1.0, 1.0, 1.0]
                    self.trans = [0.04, 0.005]
                    self.vflip = True
                    self.zoom = [1.0, 1.4, 0.99, 1.01]

            model = tensor_tools.SP_transform.RandomAffineFlow(config(), addnoise=False)
            input = {'imgs': [im0 / 255, im1 / 255], 'flows_f': [flow], 'masks_f': [mask]}
            data = model(input)
            imgs0, imgs1 = data['imgs']
            flows_f = data['flows_f'][0]
            # show
            tensor_tools.check_tensor(imgs0, 'imgs0 out')

            def process_image(tens, ind=0):
                tens_, = tensor_tools.tensor_gpu(tens, check_on=False)
                im = tens_[ind, :, :, :]
                im = np.transpose(im, (1, 2, 0))
                return im

            im0_ori = tensor_tools.im_norm(process_image(im0))
            im0_res = tensor_tools.im_norm(process_image(imgs0))
            flow_ori = tensor_tools.flow_to_image_ndmax(process_image(flow))
            flow_res = tensor_tools.flow_to_image_ndmax(process_image(flows_f))
            tensor_tools.cv2_show_dict(im0_ori=im0_ori, im0_res=im0_res, flow_ori=flow_ori, flow_res=flow_res)

    class SP_transform_no_buffer():
        @classmethod
        def denormalize_coords(cls, xx, yy, width, height):
            """ scale indices from [-1, 1] to [0, width/height] """
            xx = 0.5 * (width - 1.0) * (xx.float() + 1.0)
            yy = 0.5 * (height - 1.0) * (yy.float() + 1.0)
            return xx, yy

        @classmethod
        def normalize_coords(cls, xx, yy, width, height):
            """ scale indices from [0, width/height] to [-1, 1] """
            xx = (2.0 / (width - 1.0)) * xx.float() - 1.0
            yy = (2.0 / (height - 1.0)) * yy.float() - 1.0
            return xx, yy

        @classmethod
        def apply_transform_to_params(cls, theta0, theta_transform):
            a1 = theta0[:, 0]
            a2 = theta0[:, 1]
            a3 = theta0[:, 2]
            a4 = theta0[:, 3]
            a5 = theta0[:, 4]
            a6 = theta0[:, 5]
            #
            b1 = theta_transform[:, 0]
            b2 = theta_transform[:, 1]
            b3 = theta_transform[:, 2]
            b4 = theta_transform[:, 3]
            b5 = theta_transform[:, 4]
            b6 = theta_transform[:, 5]
            #
            c1 = a1 * b1 + a4 * b2
            c2 = a2 * b1 + a5 * b2
            c3 = b3 + a3 * b1 + a6 * b2
            c4 = a1 * b4 + a4 * b5
            c5 = a2 * b4 + a5 * b5
            c6 = b6 + a3 * b4 + a6 * b5
            #
            new_theta = torch.stack([c1, c2, c3, c4, c5, c6], dim=1)
            return new_theta

        class IdentityParams(nn.Module):
            def __init__(self):
                super(tensor_tools.SP_transform_no_buffer.IdentityParams, self).__init__()
                self._batch_size = 0
                self._o = torch.FloatTensor()
                self._i = torch.FloatTensor()

            def _update(self, batch_size):
                torch.zeros([batch_size, 1], out=self._o)
                torch.ones([batch_size, 1], out=self._i)
                return torch.cat([self._i, self._o, self._o, self._o, self._i, self._o], dim=1)

            def forward(self, batch_size):
                if self._batch_size != batch_size:
                    self._identity_params = self._update(batch_size)
                    self._batch_size = batch_size
                return self._identity_params

        class RandomMirror(nn.Module):
            def __init__(self, vertical=True, p=0.5):
                super(tensor_tools.SP_transform_no_buffer.RandomMirror, self).__init__()
                self._batch_size = 0
                self._p = p
                self._vertical = vertical
                self._mirror_probs = torch.FloatTensor()

            def update_probs(self, batch_size):
                torch.ones([batch_size, 1], out=self._mirror_probs)
                self._mirror_probs *= self._p

            def forward(self, theta_list):
                batch_size = theta_list[0].size(0)
                if batch_size != self._batch_size:
                    self.update_probs(batch_size)
                    self._batch_size = batch_size

                # apply random sign to a1 a2 a3 (these are the guys responsible for x)
                sign = torch.sign(2.0 * torch.bernoulli(self._mirror_probs) - 1.0)
                i = torch.ones_like(sign)
                horizontal_mirror = torch.cat([sign, sign, sign, i, i, i], dim=1)
                theta_list = [theta * horizontal_mirror for theta in theta_list]

                # apply random sign to a4 a5 a6 (these are the guys responsible for y)
                if self._vertical:
                    sign = torch.sign(2.0 * torch.bernoulli(self._mirror_probs) - 1.0)
                    vertical_mirror = torch.cat([i, i, i, sign, sign, sign], dim=1)
                    theta_list = [theta * vertical_mirror for theta in theta_list]

                return theta_list

        class RandomAffineFlow(tools.abstract_model):
            def __init__(self, cfg, addnoise=True):
                super(tensor_tools.SP_transform_no_buffer.RandomAffineFlow, self).__init__()
                self.cfg = cfg
                self._interp2 = tensor_tools.Interpolation.Interp2(clamp=False)
                self._flow_interp2 = tensor_tools.Interpolation.Interp2(clamp=False)
                self._meshgrid = tensor_tools.Interpolation.Meshgrid()
                self._identity = tensor_tools.SP_transform_no_buffer.IdentityParams()
                self._random_mirror = tensor_tools.SP_transform_no_buffer.RandomMirror(cfg.vflip) if cfg.hflip else tensor_tools.SP_transform_no_buffer.RandomMirror(p=1)
                self._addnoise = addnoise
                self._xbounds = torch.FloatTensor([-1, -1, 1, 1])
                self._ybounds = torch.FloatTensor([-1, 1, -1, 1])
                self._x = torch.IntTensor(1)
                self._y = torch.IntTensor(1)

            def inverse_transform_coords(self, width, height, thetas, offset_x=None,
                                         offset_y=None):
                xx, yy = self._meshgrid(width=width, height=height)

                xx = torch.unsqueeze(xx, dim=0).float()
                yy = torch.unsqueeze(yy, dim=0).float()

                if offset_x is not None:
                    xx = xx + offset_x
                if offset_y is not None:
                    yy = yy + offset_y

                a1 = thetas[:, 0].contiguous().view(-1, 1, 1)
                a2 = thetas[:, 1].contiguous().view(-1, 1, 1)
                a3 = thetas[:, 2].contiguous().view(-1, 1, 1)
                a4 = thetas[:, 3].contiguous().view(-1, 1, 1)
                a5 = thetas[:, 4].contiguous().view(-1, 1, 1)
                a6 = thetas[:, 5].contiguous().view(-1, 1, 1)

                xx, yy = tensor_tools.SP_transform_no_buffer.normalize_coords(xx, yy, width=width, height=height)
                xq = a1 * xx + a2 * yy + a3
                yq = a4 * xx + a5 * yy + a6
                xq, yq = tensor_tools.SP_transform_no_buffer.denormalize_coords(xq, yq, width=width, height=height)
                return xq, yq

            def transform_coords(self, width, height, thetas):
                xx1, yy1 = self._meshgrid(width=width, height=height)
                xx, yy = tensor_tools.SP_transform_no_buffer.normalize_coords(xx1, yy1, width=width, height=height)

                def _unsqueeze12(u):
                    return torch.unsqueeze(torch.unsqueeze(u, dim=1), dim=1)

                a1 = _unsqueeze12(thetas[:, 0])
                a2 = _unsqueeze12(thetas[:, 1])
                a3 = _unsqueeze12(thetas[:, 2])
                a4 = _unsqueeze12(thetas[:, 3])
                a5 = _unsqueeze12(thetas[:, 4])
                a6 = _unsqueeze12(thetas[:, 5])
                #
                z = a1 * a5 - a2 * a4
                b1 = a5 / z
                b2 = - a2 / z
                b4 = - a4 / z
                b5 = a1 / z
                #
                xhat = xx - a3
                yhat = yy - a6
                xq = b1 * xhat + b2 * yhat
                yq = b4 * xhat + b5 * yhat

                xq, yq = tensor_tools.SP_transform_no_buffer.denormalize_coords(xq, yq, width=width, height=height)
                return xq, yq

            def find_invalid(self, width, height, thetas):
                x = self._xbounds
                y = self._ybounds
                #
                a1 = torch.unsqueeze(thetas[:, 0], dim=1)
                a2 = torch.unsqueeze(thetas[:, 1], dim=1)
                a3 = torch.unsqueeze(thetas[:, 2], dim=1)
                a4 = torch.unsqueeze(thetas[:, 3], dim=1)
                a5 = torch.unsqueeze(thetas[:, 4], dim=1)
                a6 = torch.unsqueeze(thetas[:, 5], dim=1)
                #
                z = a1 * a5 - a2 * a4
                b1 = a5 / z
                b2 = - a2 / z
                b4 = - a4 / z
                b5 = a1 / z
                #
                xhat = x - a3
                yhat = y - a6
                xq = b1 * xhat + b2 * yhat
                yq = b4 * xhat + b5 * yhat
                xq, yq = tensor_tools.SP_transform_no_buffer.denormalize_coords(xq, yq, width=width, height=height)
                #
                invalid = (
                                  (xq < 0) | (yq < 0) | (xq >= width) | (yq >= height)
                          ).sum(dim=1, keepdim=True) > 0

                return invalid

            def apply_random_transforms_to_params(self,
                                                  theta0,
                                                  max_translate,
                                                  min_zoom, max_zoom,
                                                  min_squeeze, max_squeeze,
                                                  min_rotate, max_rotate,
                                                  validate_size=None):
                max_translate *= 0.5
                batch_size = theta0.size(0)
                height, width = validate_size

                # collect valid params here
                thetas = torch.zeros_like(theta0)

                zoom = theta0.new(batch_size, 1).zero_()
                squeeze = torch.zeros_like(zoom)
                tx = torch.zeros_like(zoom)
                ty = torch.zeros_like(zoom)
                phi = torch.zeros_like(zoom)
                invalid = torch.ones_like(zoom).byte()

                while invalid.sum() > 0:
                    # random sampling
                    zoom.uniform_(min_zoom, max_zoom)
                    squeeze.uniform_(min_squeeze, max_squeeze)
                    tx.uniform_(-max_translate, max_translate)
                    ty.uniform_(-max_translate, max_translate)
                    phi.uniform_(-min_rotate, max_rotate)

                    # construct affine parameters
                    sx = zoom * squeeze
                    sy = zoom / squeeze
                    sin_phi = torch.sin(phi)
                    cos_phi = torch.cos(phi)
                    b1 = cos_phi * sx
                    b2 = sin_phi * sy
                    b3 = tx
                    b4 = - sin_phi * sx
                    b5 = cos_phi * sy
                    b6 = ty

                    theta_transform = torch.cat([b1, b2, b3, b4, b5, b6], dim=1)
                    theta_try = tensor_tools.SP_transform_no_buffer.apply_transform_to_params(theta0, theta_transform)
                    thetas = invalid.float() * theta_try + (1 - invalid).float() * thetas

                    # compute new invalid ones
                    invalid = self.find_invalid(width=width, height=height, thetas=thetas)

                # here we should have good thetas within borders
                return thetas

            def transform_image(self, images, thetas):
                batch_size, channels, height, width = images.size()
                xq, yq = self.transform_coords(width=width, height=height, thetas=thetas)
                transformed = self._interp2(images, xq, yq)
                return transformed

            def transform_flow(self, flow, theta1, theta2):
                batch_size, channels, height, width = flow.size()
                u = flow[:, 0, :, :]
                v = flow[:, 1, :, :]

                # inverse transform coords
                x0, y0 = self.inverse_transform_coords(
                    width=width, height=height, thetas=theta1)

                x1, y1 = self.inverse_transform_coords(
                    width=width, height=height, thetas=theta2, offset_x=u, offset_y=v)

                # subtract and create new flow
                u = x1 - x0
                v = y1 - y0
                new_flow = torch.stack([u, v], dim=1)

                # transform coords
                xq, yq = self.transform_coords(width=width, height=height, thetas=theta1)

                # interp2
                transformed = self._flow_interp2(new_flow, xq, yq)
                return transformed

            def forward(self, data):
                # 01234 flow 12 21 23 32
                imgs = data['imgs']
                flows_f = data['flows_f']
                masks_f = data['masks_f']

                batch_size, _, height, width = imgs[0].size()

                # identity = no transform
                theta0 = self._identity(batch_size)

                # global transform
                theta_list = [self.apply_random_transforms_to_params(
                    theta0,
                    max_translate=self.cfg.trans[0],
                    min_zoom=self.cfg.zoom[0], max_zoom=self.cfg.zoom[1],
                    min_squeeze=self.cfg.squeeze[0], max_squeeze=self.cfg.squeeze[1],
                    min_rotate=self.cfg.rotate[0], max_rotate=self.cfg.rotate[1],
                    validate_size=[height, width])
                ]

                # relative transform
                for i in range(len(imgs) - 1):
                    theta_list.append(
                        self.apply_random_transforms_to_params(
                            theta_list[-1],
                            max_translate=self.cfg.trans[1],
                            min_zoom=self.cfg.zoom[2], max_zoom=self.cfg.zoom[3],
                            min_squeeze=self.cfg.squeeze[2], max_squeeze=self.cfg.squeeze[3],
                            min_rotate=-self.cfg.rotate[2], max_rotate=self.cfg.rotate[2],
                            validate_size=[height, width])
                    )

                # random flip images
                theta_list = self._random_mirror(theta_list)

                # 01234
                imgs = [self.transform_image(im, theta) for im, theta in zip(imgs, theta_list)]

                if len(imgs) > 2:
                    theta_list = theta_list[1:-1]
                # 12 23
                flows_f = [self.transform_flow(flo, theta1, theta2) for flo, theta1, theta2 in
                           zip(flows_f, theta_list[:-1], theta_list[1:])]

                masks_f = [self.transform_image(mask, theta) for mask, theta in
                           zip(masks_f, theta_list)]

                if self._addnoise:
                    '''
                    im1 <class 'torch.Tensor'> (3, 320, 1152) max 0.5885537 min -0.4305366 mean -0.040912468
                    im1 <class 'torch.Tensor'> (3, 320, 1152) max 0.5885537 min -0.4305366 mean -0.03847942
                    im1 <class 'torch.Tensor'> (3, 320, 1152) max 0.5885537 min -0.4187718 mean -0.011021424
                    '''
                    stddev = np.random.uniform(0.0, 0.04)
                    for im in imgs:
                        noise = torch.zeros_like(im)
                        noise.normal_(std=stddev)
                        im.add_(noise)
                        im.clamp_(-1.0, 1.0)

                data['imgs'] = imgs
                data['flows_f'] = flows_f
                data['masks_f'] = masks_f
                return data

            class config():
                def __init__(self):
                    self.add_noise = False
                    self.hflip = True
                    self.rotate = [-0.01, 0.01, -0.01, 0.01]
                    self.squeeze = [1.0, 1.0, 1.0, 1.0]
                    self.trans = [0.04, 0.005]
                    self.vflip = True
                    self.zoom = [1.0, 1.4, 0.99, 1.01]

            model = tensor_tools.SP_transform.RandomAffineFlow(config(), addnoise=False)
            input = {'imgs': [im0 / 255, im1 / 255], 'flows_f': [flow], 'masks_f': [mask]}
            data = model(input)
            imgs0, imgs1 = data['imgs']
            flows_f = data['flows_f'][0]
            # show
            tensor_tools.check_tensor(imgs0, 'imgs0 out')

            def process_image(tens, ind=0):
                tens_, = tensor_tools.tensor_gpu(tens, check_on=False)
                im = tens_[ind, :, :, :]
                im = np.transpose(im, (1, 2, 0))
                return im

            im0_ori = tensor_tools.im_norm(process_image(im0))
            im0_res = tensor_tools.im_norm(process_image(imgs0))
            flow_ori = tensor_tools.flow_to_image_ndmax(process_image(flow))
            flow_res = tensor_tools.flow_to_image_ndmax(process_image(flows_f))
            tensor_tools.cv2_show_dict(im0_ori=im0_ori, im0_res=im0_res, flow_ori=flow_ori, flow_res=flow_res)

    @classmethod
    def MSE(cls, img1, img2):
        img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite("L1_.jpg",img1gray-img2gray)
        rows, cols = img1gray.shape[:2]
        loss = 0.0
        pixel_nums = 0
        for row in range(30, rows - 30):
            for col in range(60, cols - 60):
                if img1gray[row][col] == 0 or img2gray[row][col] == 0:
                    continue
                else:
                    pixel_nums += 1
                    loss += np.square(np.abs(img1gray[row][col] - img2gray[row][col]))

        loss /= pixel_nums

        return loss

    @classmethod
    def torch_warp_mask(cls, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """

        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        # print(grid.shape,flo.shape,'...')
        vgrid = grid + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
        output = nn.functional.grid_sample(x, vgrid, padding_mode='zeros')
        mask = torch.autograd.Variable(torch.ones(x.size()))
        if x.is_cuda:
            mask = mask.cuda()
        mask = nn.functional.grid_sample(mask, vgrid, padding_mode='zeros')

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1
        output = output * mask
        # # nchw->>>nhwc
        # if x.is_cuda:
        #     output = output.cpu()
        # output_im = output.numpy()
        # output_im = np.transpose(output_im, (0, 2, 3, 1))
        # output_im = np.squeeze(output_im)
        return output, mask

    @classmethod
    def torch_warp(cls, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """

        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        # print(grid.shape,flo.shape,'...')
        vgrid = grid + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
        # tools.check_tensor(x, 'x')
        # tools.check_tensor(vgrid, 'vgrid')
        output = nn.functional.grid_sample(x, vgrid, padding_mode='zeros')
        # mask = torch.autograd.Variable(torch.ones(x.size()))
        # if x.is_cuda:
        #     mask = mask.cuda()
        # mask = nn.functional.grid_sample(mask, vgrid, padding_mode='zeros')
        #
        # mask[mask < 0.9999] = 0
        # mask[mask > 0] = 1
        # output = output * mask
        # # nchw->>>nhwc
        # if x.is_cuda:
        #     output = output.cpu()
        # output_im = output.numpy()
        # output_im = np.transpose(output_im, (0, 2, 3, 1))
        # output_im = np.squeeze(output_im)
        return output

    @classmethod
    def torch_warp_boundary(cls, x, flo, start_point):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        start_point: [B,2,1,1]
        """

        _, _, Hx, Wx = x.size()
        B, C, H, W = flo.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        # print(grid.shape,flo.shape,'...')
        vgrid = grid + flo + start_point

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(Wx - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(Hx - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
        # tools.check_tensor(x, 'x')
        # tools.check_tensor(vgrid, 'vgrid')
        output = nn.functional.grid_sample(x, vgrid, padding_mode='zeros')
        # mask = torch.autograd.Variable(torch.ones(x.size()))
        # if x.is_cuda:
        #     mask = mask.cuda()
        # mask = nn.functional.grid_sample(mask, vgrid, padding_mode='zeros')
        #
        # mask[mask < 0.9999] = 0
        # mask[mask > 0] = 1
        # output = output * mask
        # # nchw->>>nhwc
        # if x.is_cuda:
        #     output = output.cpu()
        # output_im = output.numpy()
        # output_im = np.transpose(output_im, (0, 2, 3, 1))
        # output_im = np.squeeze(output_im)
        return output

    @classmethod
    def weights_init(cls, m):
        classname = m.__class__.__name__
        if classname.find('conv') != -1:
            # torch.nn.init.xavier_normal(m.weight)
            torch.nn.init.kaiming_normal(m.weight)

            torch.nn.init.constant(m.bias, 0)

    @classmethod
    def create_gif(cls, image_list, gif_name, duration=0.5):
        frames = []
        for image_name in image_list:
            frames.append(image_name)
        imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
        return

    @classmethod
    def warp_cv2(cls, img_prev, flow):
        # calculate mat
        w = int(img_prev.shape[1])
        h = int(img_prev.shape[0])
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        coords = np.float32(np.dstack([x_coords, y_coords]))
        pixel_map = coords + flow
        new_frame = cv2.remap(img_prev, pixel_map, None, cv2.INTER_LINEAR)
        return new_frame

    @classmethod
    def flow_to_image_dmax(cls, flow, display=False):
        """

        :param flow: H,W,2
        :param display:
        :return: H,W,3
        """

        def compute_color(u, v):
            def make_color_wheel():
                """
                Generate color wheel according Middlebury color code
                :return: Color wheel
                """
                RY = 15
                YG = 6
                GC = 4
                CB = 11
                BM = 13
                MR = 6

                ncols = RY + YG + GC + CB + BM + MR

                colorwheel = np.zeros([ncols, 3])

                col = 0

                # RY
                colorwheel[0:RY, 0] = 255
                colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
                col += RY

                # YG
                colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
                colorwheel[col:col + YG, 1] = 255
                col += YG

                # GC
                colorwheel[col:col + GC, 1] = 255
                colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
                col += GC

                # CB
                colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
                colorwheel[col:col + CB, 2] = 255
                col += CB

                # BM
                colorwheel[col:col + BM, 2] = 255
                colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
                col += + BM

                # MR
                colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
                colorwheel[col:col + MR, 0] = 255

                return colorwheel

            """
            compute optical flow color map
            :param u: optical flow horizontal map
            :param v: optical flow vertical map
            :return: optical flow in color code
            """
            [h, w] = u.shape
            img = np.zeros([h, w, 3])
            nanIdx = np.isnan(u) | np.isnan(v)
            u[nanIdx] = 0
            v[nanIdx] = 0

            colorwheel = make_color_wheel()
            ncols = np.size(colorwheel, 0)

            rad = np.sqrt(u ** 2 + v ** 2)

            a = np.arctan2(-v, -u) / np.pi

            fk = (a + 1) / 2 * (ncols - 1) + 1

            k0 = np.floor(fk).astype(int)

            k1 = k0 + 1
            k1[k1 == ncols + 1] = 1
            f = fk - k0

            for i in range(0, np.size(colorwheel, 1)):
                tmp = colorwheel[:, i]
                col0 = tmp[k0 - 1] / 255
                col1 = tmp[k1 - 1] / 255
                col = (1 - f) * col0 + f * col1

                idx = rad <= 1
                col[idx] = 1 - rad[idx] * (1 - col[idx])
                notidx = np.logical_not(idx)

                col[notidx] *= 0.75
                img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

            return img

        UNKNOWN_FLOW_THRESH = 1e7
        """
        Convert flow into middlebury color code image
        :param flow: optical flow map
        :return: optical flow image in middlebury color
        """
        u = flow[:, :, 0]
        v = flow[:, :, 1]

        maxu = -999.
        maxv = -999.
        minu = 999.
        minv = 999.

        idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
        u[idxUnknow] = 0
        v[idxUnknow] = 0

        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))

        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))

        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(-1, np.max(rad))

        if display:
            print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu, maxu, minv, maxv))

        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)

        img = compute_color(u, v)

        idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
        img[idx] = 0

        return np.uint8(img)

    @classmethod
    def flow_to_image_ndmax(cls, flow, max_flow=None):
        # flow shape (H, W, C)
        if max_flow is not None:
            max_flow = max(max_flow, 1.)
        else:
            max_flow = np.max(flow)

        n = 8
        u, v = flow[:, :, 0], flow[:, :, 1]
        mag = np.sqrt(np.square(u) + np.square(v))
        angle = np.arctan2(v, u)
        im_h = np.mod(angle / (2 * np.pi) + 1, 1)
        im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
        im_v = np.clip(n - im_s, a_min=0, a_max=1)
        im = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))
        return (im * 255).astype(np.uint8)

    @classmethod
    def flow_error_image_np(cls, flow_pred, flow_gt, mask_occ, mask_noc=None, log_colors=True):
        """Visualize the error between two flows as 3-channel color image.
        Adapted from the KITTI C++ devkit.
        Args:
            flow_pred: prediction flow of shape [ height, width, 2].
            flow_gt: ground truth
            mask_occ: flow validity mask of shape [num_batch, height, width, 1].
                Equals 1 at (occluded and non-occluded) valid pixels.
            mask_noc: Is 1 only at valid pixels which are not occluded.
        """
        # mask_noc = tf.ones(tf.shape(mask_occ)) if mask_noc is None else mask_noc
        mask_noc = np.ones(mask_occ.shape) if mask_noc is None else mask_noc
        diff_sq = (flow_pred - flow_gt) ** 2
        # diff = tf.sqrt(tf.reduce_sum(diff_sq, [3], keep_dims=True))
        diff = np.sqrt(np.sum(diff_sq, axis=2, keepdims=True))
        if log_colors:
            height, width, _ = flow_pred.shape
            # num_batch, height, width, _ = tf.unstack(tf.shape(flow_1))
            colormap = [
                [0, 0.0625, 49, 54, 149],
                [0.0625, 0.125, 69, 117, 180],
                [0.125, 0.25, 116, 173, 209],
                [0.25, 0.5, 171, 217, 233],
                [0.5, 1, 224, 243, 248],
                [1, 2, 254, 224, 144],
                [2, 4, 253, 174, 97],
                [4, 8, 244, 109, 67],
                [8, 16, 215, 48, 39],
                [16, 1000000000.0, 165, 0, 38]]
            colormap = np.asarray(colormap, dtype=np.float32)
            colormap[:, 2:5] = colormap[:, 2:5] / 255
            # mag = tf.sqrt(tf.reduce_sum(tf.square(flow_2), 3, keep_dims=True))
            tempp = np.square(flow_gt)
            # temp = np.sum(tempp, axis=2, keep_dims=True)
            # mag = np.sqrt(temp)
            mag = np.sqrt(np.sum(tempp, axis=2, keepdims=True))
            # error = tf.minimum(diff / 3, 20 * diff / mag)
            error = np.minimum(diff / 3, 20 * diff / (mag + 1e-7))
            im = np.zeros([height, width, 3])
            for i in range(colormap.shape[0]):
                colors = colormap[i, :]
                cond = np.logical_and(np.greater_equal(error, colors[0]), np.less(error, colors[1]))
                # temp=np.tile(cond, [1, 1, 3])
                im = np.where(np.tile(cond, [1, 1, 3]), np.ones([height, width, 1]) * colors[2:5], im)
            # temp=np.cast(mask_noc, np.bool)
            # im = np.where(np.tile(np.cast(mask_noc, np.bool), [1, 1, 3]), im, im * 0.5)
            im = np.where(np.tile(mask_noc == 1, [1, 1, 3]), im, im * 0.5)
            im = im * mask_occ
        else:
            error = (np.minimum(diff, 5) / 5) * mask_occ
            im_r = error  # errors in occluded areas will be red
            im_g = error * mask_noc
            im_b = error * mask_noc
            im = np.concatenate([im_r, im_g, im_b], axis=2)
            # im = np.concatenate(axis=2, values=[im_r, im_g, im_b])
        return im[:, :, ::-1]

    @classmethod
    def tensor_gpu(cls, *args, check_on=True, gpu_opt=None, non_blocking=True):
        def check_on_gpu(tensor_):
            if type(gpu_opt) == int:
                tensor_g = tensor_.cuda(gpu_opt, non_blocking=non_blocking)
            else:
                tensor_g = tensor_.cuda()
            return tensor_g

        def check_off_gpu(tensor_):
            if tensor_.is_cuda:
                tensor_c = tensor_.cpu()
            else:
                tensor_c = tensor_
            tensor_c = tensor_c.detach().numpy()
            # tensor_c = cv2.normalize(tensor_c.detach().numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            return tensor_c

        if torch.cuda.is_available():
            if check_on:
                data_ls = [check_on_gpu(a) for a in args]
            else:
                data_ls = [check_off_gpu(a) for a in args]
        else:
            if check_on:
                data_ls = args
            else:
                # data_ls = args
                data_ls = [a.detach().numpy() for a in args]
                # data_ls = [cv2.normalize(a.detach().numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) for a in args]
                # data_ls = args
        return data_ls

    @classmethod
    def cv2_show_dict(cls, **kwargs):
        for i in kwargs.keys():
            cv2.imshow(i, kwargs[i])
        cv2.waitKey()

    @classmethod
    def hist_match_np_hw3(cls, img, ref):
        '''need BGR image input'''
        # channels = ['blue', 'green', 'red']
        out = np.zeros_like(img)
        _, _, colorChannel = img.shape
        for i in range(colorChannel):
            # print(channels[i])
            hist_img, _ = np.histogram(img[:, :, i], 256)  # get the histogram
            hist_ref, _ = np.histogram(ref[:, :, i], 256)
            cdf_img = np.cumsum(hist_img)  # get the accumulative histogram
            cdf_ref = np.cumsum(hist_ref)

            for j in range(256):
                tmp = abs(cdf_img[j] - cdf_ref)
                tmp = tmp.tolist()
                idx = tmp.index(min(tmp))  # find the smallest number in tmp, get the index of this number
                out[:, :, i][img[:, :, i] == j] = idx
        return out
        # cv2.imwrite('0.jpg', out)
        # print('Done')

    @classmethod
    def hist_match_np_3hw(cls, img, ref):
        '''need BGR image input'''
        # channels = ['blue', 'green', 'red']
        out = np.zeros_like(img)
        colorChannel, _, _ = img.shape
        for i in range(colorChannel):
            # print(channels[i])
            hist_img, _ = np.histogram(img[i, :, :], 256)  # get the histogram
            hist_ref, _ = np.histogram(ref[i, :, :], 256)
            cdf_img = np.cumsum(hist_img)  # get the accumulative histogram
            cdf_ref = np.cumsum(hist_ref)

            for j in range(256):
                tmp = abs(cdf_img[j] - cdf_ref)
                tmp = tmp.tolist()
                idx = tmp.index(min(tmp))  # find the smallest number in tmp, get the index of this number
                out[i, :, :][img[i, :, :] == j] = idx
        return out
        # cv2.imwrite('0.jpg', out)
        # print('Done')

    @classmethod
    def compute_model_size(cls, model, *args):
        from thop import profile
        flops, params = profile(model, inputs=args, verbose=False)
        print('flops: %.3f G, params: %.3f M' % (flops / 1000 / 1000 / 1000, params / 1000 / 1000))

    @classmethod
    def count_parameters(cls, model):
        a = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return a

    @classmethod
    def im_norm(cls, img):
        eps = 1e-6
        a = np.max(img)
        b = np.min(img)
        if a - b <= 0:
            img = (img - b) / (a - b + eps)
        else:
            img = (img - b) / (a - b)
        img = img * 255
        img = img.astype('uint8')
        return img

    @classmethod
    def check_tensor(cls, data, name, print_data=False, print_in_txt=None):
        if data.is_cuda:
            temp = data.detach().cpu().numpy()
        else:
            temp = data.detach().numpy()
        a = len(name)
        name_ = name + ' ' * 100
        name_ = name_[0:max(a, 10)]
        print_str = '%s, %s, %s, %s,%s,%s,%s,%s' % (name_, temp.shape, data.dtype, ' max:%.2f' % np.max(temp), ' min:%.2f' % np.min(temp),
                                                    ' mean:%.2f' % np.mean(temp), ' sum:%.2f' % np.sum(temp), data.device)
        if print_in_txt is None:
            print(print_str)
        else:
            print(print_str, file=print_in_txt)
        if print_data:
            print(temp)
        return print_str

    @classmethod
    def check_tensor_np(cls, data, name, print_data=False, print_in_txt=None):
        temp = data
        a = len(name)
        name_ = name + ' ' * 100
        name_ = name_[0:max(a, 10)]
        print_str = '%s, %s, %s, %s,%s,%s,%s' % (name_, temp.shape, data.dtype, ' max:%.2f' % np.max(temp), ' min:%.2f' % np.min(temp),
                                                 ' mean:%.2f' % np.mean(temp), ' sum:%.2f' % np.sum(temp))
        if print_in_txt is None:
            print(print_str)
        else:
            print(print_str, file=print_in_txt)
        if print_data:
            print(temp)
        return print_str


class frame_utils():
    '''  borrowed from RAFT '''
    TAG_CHAR = np.array([202021.25], np.float32)

    @classmethod
    def readFlow(cls, fn):
        """ Read .flo file in Middlebury format"""
        # Code adapted from:
        # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

        # WARNING: this will work on little-endian architectures (eg Intel x86) only!
        # print 'fn = %s'%(fn)
        with open(fn, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print('Magic number incorrect. Invalid .flo file')
                return None
            else:
                w = np.fromfile(f, np.int32, count=1)
                h = np.fromfile(f, np.int32, count=1)
                # print 'Reading %d x %d flo file\n' % (w, h)
                data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
                # Reshape data into 3D array (columns, rows, bands)
                # The reshape here is for visualization, the original code is (w,h,2)
                return np.resize(data, (int(h), int(w), 2))

    @classmethod
    def readPFM(cls, file):
        file = open(file, 'rb')

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data

    @classmethod
    def writeFlow(cls, filename, uv, v=None):
        """ Write optical flow to file.

        If v is None, uv is assumed to contain both u and v channels,
        stacked in depth.
        Original code by Deqing Sun, adapted from Daniel Scharstein.
        """
        nBands = 2

        if v is None:
            assert (uv.ndim == 3)
            assert (uv.shape[2] == 2)
            u = uv[:, :, 0]
            v = uv[:, :, 1]
        else:
            u = uv

        assert (u.shape == v.shape)
        height, width = u.shape
        f = open(filename, 'wb')
        # write the header
        f.write(cls.TAG_CHAR)
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        # arrange into matrix form
        tmp = np.zeros((height, width * nBands))
        tmp[:, np.arange(width) * 2] = u
        tmp[:, np.arange(width) * 2 + 1] = v
        tmp.astype(np.float32).tofile(f)
        f.close()

    @classmethod
    def readFlowKITTI(cls, filename):
        flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        flow = flow[:, :, ::-1].astype(np.float32)
        flow, valid = flow[:, :, :2], flow[:, :, 2]
        flow = (flow - 2 ** 15) / 64.0
        return flow, valid

    @classmethod
    def read_png_flow(cls, fpath):
        """
        Read KITTI optical flow, returns u,v,valid mask

        """

        R = png.Reader(fpath)
        width, height, data, _ = R.asDirect()
        # This only worked with python2.
        # I = np.array(map(lambda x:x,data)).reshape((height,width,3))
        gt = np.array([x for x in data]).reshape((height, width, 3))
        flow = gt[:, :, 0:2]
        flow = (flow.astype('float64') - 2 ** 15) / 64.0
        flow = flow.astype(np.float)
        mask = gt[:, :, 2:3]
        mask = np.uint8(mask)
        return flow, mask

    @classmethod
    def readDispKITTI(cls, filename):
        disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
        valid = disp > 0.0
        flow = np.stack([-disp, np.zeros_like(disp)], -1)
        return flow, valid

    @classmethod
    def writeFlowKITTI(cls, filename, uv):
        uv = 64.0 * uv + 2 ** 15
        valid = np.ones([uv.shape[0], uv.shape[1], 1])
        uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
        cv2.imwrite(filename, uv[..., ::-1])

    @classmethod
    def read_gen(cls, file_name, read_mask=False):
        ext = os.path.splitext(file_name)[-1]
        if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
            if read_mask:
                return imageio.imread(file_name)
            else:
                return Image.open(file_name)
        elif ext == '.bin' or ext == '.raw':
            return np.load(file_name)
        elif ext == '.flo':
            return cls.readFlow(file_name).astype(np.float32)
        elif ext == '.pfm':
            flow = cls.readPFM(file_name).astype(np.float32)
            if len(flow.shape) == 2:
                return flow
            else:
                return flow[:, :, :-1]
        else:
            raise ValueError('wrong file type: %s' % ext)


class DIS_test(tools.abs_test_model):

    def proce(self, im):
        if im.is_cuda:
            im = im.cpu()
        b = im.numpy()
        b = np.squeeze(b[0, :, :, :])
        b = np.transpose(b, (1, 2, 0))
        b = tensor_tools.im_norm(b)
        b = b.astype(np.uint8)
        return b

    def eval_forward(self, im1, im2, *args, **kwargs):
        n, _, h, w = im1.shape
        if n > 1:
            flow_ls = []
            for i in range(n):
                im1_ = im1.narrow(0, i, 1)  # dim, start, length
                im2_ = im2.narrow(0, i, 1)
                # tensor_tools.check_tensor(im1_, 'im1_')
                flow = self.single_im(im1_, im2_)
                flow_ls.append(flow)
            flow_torch = torch.cat(flow_ls, dim=0)
            return flow_torch
        else:
            return self.single_im(im1, im2)

    def single_im(self, im1, im2):
        n, _, h, w = im1.shape
        im1_np = self.proce(im1)
        im2_np = self.proce(im2)
        gray0 = cv2.cvtColor(im1_np, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(im2_np, cv2.COLOR_BGR2GRAY)

        dis = cv2.DISOpticalFlow().create(preset=2)
        dis.setUseSpatialPropagation(True)
        flow = dis.calc(gray0, gray1, None)
        flow_ = np.transpose(flow, (2, 0, 1))
        flow_ = np.expand_dims(flow_, 0)
        flow_torch = torch.from_numpy(flow_).float()  # .cuda()
        if im1.is_cuda:
            flow_torch = flow_torch.cuda()
        return flow_torch

# functions to compute loss
class Loss_tools():

    @classmethod
    def weighted_ssim(cls, x, y, weight, c1=float('inf'), c2=9e-6, weight_epsilon=0.01):
        """Computes a weighted structured image similarity measure.
        Args:
          x: a batch of images, of shape [B, C, H, W].
          y:  a batch of images, of shape [B, C, H, W].
          weight: shape [B, 1, H, W], representing the weight of each
            pixel in both images when we come to calculate moments (means and
            correlations). values are in [0,1]
          c1: A floating point number, regularizes division by zero of the means.
          c2: A floating point number, regularizes division by zero of the second
            moments.
          weight_epsilon: A floating point number, used to regularize division by the
            weight.

        Returns:
          A tuple of two pytorch Tensors. First, of shape [B, C, H-2, W-2], is scalar
          similarity loss per pixel per channel, and the second, of shape
          [B, 1, H-2. W-2], is the average pooled `weight`. It is needed so that we
          know how much to weigh each pixel in the first tensor. For example, if
          `'weight` was very small in some area of the images, the first tensor will
          still assign a loss to these pixels, but we shouldn't take the result too
          seriously.
        """

        def _avg_pool3x3(x):
            # tf kernel [b,h,w,c]
            return F.avg_pool2d(x, (3, 3), (1, 1))
            # return tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 1, 1, 1], 'VALID')

        if c1 == float('inf') and c2 == float('inf'):
            raise ValueError('Both c1 and c2 are infinite, SSIM loss is zero. This is '
                             'likely unintended.')
        average_pooled_weight = _avg_pool3x3(weight)
        weight_plus_epsilon = weight + weight_epsilon
        inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)

        def weighted_avg_pool3x3(z):
            wighted_avg = _avg_pool3x3(z * weight_plus_epsilon)
            return wighted_avg * inverse_average_pooled_weight

        mu_x = weighted_avg_pool3x3(x)
        mu_y = weighted_avg_pool3x3(y)
        sigma_x = weighted_avg_pool3x3(x ** 2) - mu_x ** 2
        sigma_y = weighted_avg_pool3x3(y ** 2) - mu_y ** 2
        sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y
        if c1 == float('inf'):
            ssim_n = (2 * sigma_xy + c2)
            ssim_d = (sigma_x + sigma_y + c2)
        elif c2 == float('inf'):
            ssim_n = 2 * mu_x * mu_y + c1
            ssim_d = mu_x ** 2 + mu_y ** 2 + c1
        else:
            ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
            ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        result = ssim_n / ssim_d
        return torch.clamp((1 - result) / 2, 0, 1), average_pooled_weight

    @classmethod
    def edge_aware_smoothness_order1(cls, img, pred, constant=1.0, weight_type='gauss', error_type='L1'):
        def gradient_x(img):
            gx = img[:, :, :-1, :] - img[:, :, 1:, :]
            return gx

        def gradient_y(img):
            gy = img[:, :, :, :-1] - img[:, :, :, 1:]
            return gy

        def weight_fn(x):
            if weight_type == 'gauss':
                y = x ** 2
            elif weight_type == 'exp':
                y = torch.abs(x)
            else:
                raise ValueError('')
            return y

        def error_fn(x):
            if error_type == 'L1':
                y = torch.abs(x)
            elif error_type == 'abs_robust':
                y = (torch.abs(x) + 0.01).pow(0.4)
            else:
                raise ValueError('')
            return y

        pred_gradients_x = gradient_x(pred)
        pred_gradients_y = gradient_y(pred)

        image_gradients_x = gradient_x(img)
        image_gradients_y = gradient_y(img)

        weights_x = torch.exp(-torch.mean(weight_fn(constant * image_gradients_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(weight_fn(constant * image_gradients_y), 1, keepdim=True))

        smoothness_x = error_fn(pred_gradients_x) * weights_x
        smoothness_y = error_fn(pred_gradients_y) * weights_y
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    @classmethod
    def edge_aware_smoothness_order2(cls, img, pred, constant=1.0, weight_type='gauss', error_type='L1'):
        def gradient_x(img, stride=1):
            gx = img[:, :, :-stride, :] - img[:, :, stride:, :]
            return gx

        def gradient_y(img, stride=1):
            gy = img[:, :, :, :-stride] - img[:, :, :, stride:]
            return gy

        def weight_fn(x):
            if weight_type == 'gauss':
                y = x ** 2
            elif weight_type == 'exp':
                y = torch.abs(x)
            else:
                raise ValueError('')
            return y

        def error_fn(x):
            if error_type == 'L1':
                y = torch.abs(x)
            elif error_type == 'abs_robust':
                y = (torch.abs(x) + 0.01).pow(0.4)
            else:
                raise ValueError('')
            return y

        pred_gradients_x = gradient_x(pred)
        pred_gradients_xx = gradient_x(pred_gradients_x)
        pred_gradients_y = gradient_y(pred)
        pred_gradients_yy = gradient_y(pred_gradients_y)

        image_gradients_x = gradient_x(img, stride=2)
        image_gradients_y = gradient_y(img, stride=2)

        weights_x = torch.exp(-torch.mean(weight_fn(constant * image_gradients_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(weight_fn(constant * image_gradients_y), 1, keepdim=True))

        smoothness_x = error_fn(pred_gradients_xx) * weights_x
        smoothness_y = error_fn(pred_gradients_yy) * weights_y
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    @classmethod
    def flow_smooth_delta(cls, flow, if_second_order=False):
        def gradient(x):
            D_dy = x[:, :, 1:] - x[:, :, :-1]
            D_dx = x[:, :, :, 1:] - x[:, :, :, :-1]
            return D_dx, D_dy

        dx, dy = gradient(flow)
        # dx2, dxdy = gradient(dx)
        # dydx, dy2 = gradient(dy)
        if if_second_order:
            dx2, dxdy = gradient(dx)
            dydx, dy2 = gradient(dy)
            smooth_loss = dx.abs().mean() + dy.abs().mean() + dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()
        else:
            smooth_loss = dx.abs().mean() + dy.abs().mean()
        # smooth_loss = dx.abs().mean() + dy.abs().mean()  # + dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()
        # 暂时不上二阶的平滑损失，似乎加上以后就太猛了，无法降低photo loss TODO
        return smooth_loss

    @classmethod
    def photo_loss_multi_type(cls, x, y, occ_mask, photo_loss_type='abs_robust',  # abs_robust, charbonnier,L1, SSIM
                              photo_loss_use_occ=False,
                              ):
        occ_weight = occ_mask
        if photo_loss_type == 'abs_robust':
            photo_diff = x - y
            loss_diff = (torch.abs(photo_diff) + 0.01).pow(0.4)
        elif photo_loss_type == 'charbonnier':
            photo_diff = x - y
            loss_diff = ((photo_diff) ** 2 + 1e-6).pow(0.4)
        elif photo_loss_type == 'L1':
            photo_diff = x - y
            loss_diff = torch.abs(photo_diff + 1e-6)
        elif photo_loss_type == 'SSIM':
            loss_diff, occ_weight = cls.weighted_ssim(x, y, occ_mask)
        else:
            raise ValueError('wrong photo_loss type: %s' % photo_loss_type)

        if photo_loss_use_occ:
            photo_loss = torch.sum(loss_diff * occ_weight) / (torch.sum(occ_weight) + 1e-6)
        else:
            photo_loss = torch.mean(loss_diff)
        return photo_loss

    @classmethod
    def photo_loss_function(cls, diff, mask, q, charbonnier_or_abs_robust, if_use_occ, averge=True):
        if charbonnier_or_abs_robust:
            if if_use_occ:
                p = ((diff) ** 2 + 1e-6).pow(q)
                p = p * mask
                if averge:
                    p = p.mean()
                    ap = mask.mean()
                else:
                    p = p.sum()
                    ap = mask.sum()
                loss_mean = p / (ap * 2 + 1e-6)
            else:
                p = ((diff) ** 2 + 1e-8).pow(q)
                if averge:
                    p = p.mean()
                else:
                    p = p.sum()
                return p
        else:
            if if_use_occ:
                diff = (torch.abs(diff) + 0.01).pow(q)
                diff = diff * mask
                diff_sum = torch.sum(diff)
                loss_mean = diff_sum / (torch.sum(mask) * 2 + 1e-6)
            else:
                diff = (torch.abs(diff) + 0.01).pow(q)
                if averge:
                    loss_mean = diff.mean()
                else:
                    loss_mean = diff.sum()
        return loss_mean

    @classmethod
    def census_loss_torch(cls, img1, img1_warp, mask, q, charbonnier_or_abs_robust, if_use_occ, averge=True, max_distance=3):
        patch_size = 2 * max_distance + 1

        def _ternary_transform_torch(image):
            R, G, B = torch.split(image, 1, 1)
            intensities_torch = (0.2989 * R + 0.5870 * G + 0.1140 * B)  # * 255  # convert to gray
            # intensities = tf.image.rgb_to_grayscale(image) * 255
            out_channels = patch_size * patch_size
            w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))  # h,w,1,out_c
            w_ = np.transpose(w, (3, 2, 0, 1))  # 1,out_c,h,w
            weight = torch.from_numpy(w_).float()
            if image.is_cuda:
                weight = weight.cuda()
            patches_torch = torch.conv2d(input=intensities_torch, weight=weight, bias=None, stride=[1, 1], padding=[max_distance, max_distance])
            transf_torch = patches_torch - intensities_torch
            transf_norm_torch = transf_torch / torch.sqrt(0.81 + transf_torch ** 2)
            return transf_norm_torch

        def _hamming_distance_torch(t1, t2):
            dist = (t1 - t2) ** 2
            dist = torch.sum(dist / (0.1 + dist), 1, keepdim=True)
            return dist

        def create_mask_torch(tensor, paddings):
            shape = tensor.shape  # N,c, H,W
            inner_width = shape[2] - (paddings[0][0] + paddings[0][1])
            inner_height = shape[3] - (paddings[1][0] + paddings[1][1])
            inner_torch = torch.ones([shape[0], shape[1], inner_width, inner_height]).float()
            if tensor.is_cuda:
                inner_torch = inner_torch.cuda()
            mask2d = F.pad(inner_torch, [paddings[0][0], paddings[0][1], paddings[1][0], paddings[1][1]])
            return mask2d

        img1 = _ternary_transform_torch(img1)
        img1_warp = _ternary_transform_torch(img1_warp)
        dist = _hamming_distance_torch(img1, img1_warp)
        transform_mask = create_mask_torch(mask, [[max_distance, max_distance],
                                                  [max_distance, max_distance]])
        census_loss = cls.photo_loss_function(diff=dist, mask=mask * transform_mask, q=q,
                                              charbonnier_or_abs_robust=charbonnier_or_abs_robust, if_use_occ=if_use_occ, averge=averge)
        return census_loss

    @classmethod
    def upsample2d_flow_as(cls, inputs, target_as, mode="bilinear", if_rate=False):
        _, _, h, w = target_as.size()
        res = F.interpolate(inputs, [h, w], mode=mode, align_corners=True)
        if if_rate:
            _, _, h_, w_ = inputs.size()
            # inputs[:, 0, :, :] *= (w / w_)
            # inputs[:, 1, :, :] *= (h / h_)
            u_scale = (w / w_)
            v_scale = (h / h_)
            u, v = res.chunk(2, dim=1)
            u *= u_scale
            v *= v_scale
            res = torch.cat([u, v], dim=1)
        return res


if __name__ == '__main__':
    a_np = np.ones((1, 3, 100, 100))
    m = np.ones((1, 1, 100, 100))
    print(a_np.shape)
    a = torch.from_numpy(a_np)
    b = torch.from_numpy(a_np)
    m = torch.from_numpy(m)
    loss_diff, occ_weight = Loss_tools.weighted_ssim(a, b, m)
    ssim = torch.sum(loss_diff * occ_weight) / (torch.sum(occ_weight) + 1e-6)
    print(ssim)
