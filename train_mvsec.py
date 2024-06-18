import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import utils
from utils.helper_functions import *
import utils.visualization as visualization
import utils.filename_templates as TEMPLATES
import utils.helper_functions as helper
import utils.logger as logger
from utils import image_utils
from matplotlib import colors
from utils.gmflownet_loss import compute_supervision_coarse, compute_coarse_loss, backwarp

import os
import errno
import shutil
import cv2

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100

import pdb

class train(object):
    """
    train class

    """

    def __init__(self, model, config, args,
                 data_loader, visualizer, train_logger=None, save_path=None, additional_args=None, visualizer_map=False, return_epe=False):
        self.downsample = False # Downsampling for Rebuttal
        self.model = model
        self.config = config
        self.data_loader = data_loader
        self.additional_args = additional_args

        self.lr = config["train"]["lr"]
        self.wdecay = config["train"]["wdecay"]
        self.eps = config["train"]["epsilon"]
        self.num_steps = config["train"]["num_steps"]
        self.mixed_precision = config["train"]["mixed_precision"]
        self.gamma = config["train"]["gamma"]
        self.clip = config["train"]["clip"]
        self.image_size = config["train_img_size"]
        self.bestepe = args.best_epe
        self.visualize_map = visualizer_map
        self.return_epe = return_epe

        self.gpu = torch.device('cuda')
        # self.gpu = config["device"]

        if save_path is None:
            self.save_path = helper.create_save_path(config['save_dir'].lower(),
                                           config['name'].lower())
        else:
            self.save_path=save_path
        if logger is None:
            self.logger = logger.Logger(self.save_path)
        else:
            self.logger = train_logger
        if isinstance(self.additional_args, dict) and 'name_mapping_train' in self.additional_args.keys():
            visu_add_args = {'name_mapping' : self.additional_args['name_mapping_train']}
        else:
            visu_add_args = None
        self.visualizer = visualizer(data_loader, self.save_path, additional_args=visu_add_args)
    
    def get_estimation_and_target(self, batch):
        if not self.downsample:
            if 'valid' in batch.keys():
                return batch['flow_est'], (batch['flow'], batch['valid'])
        else:
            f_est = batch['flow_est']
            f_gt = torch.nn.functional.interpolate(batch['flow'], scale_factor=0.5)
            if 'valid' in batch.keys():
                f_mask = torch.nn.functional.interpolate(batch['valid'], scale_factor=0.5)
                return f_est, (f_gt, f_mask)

    def get_key_map(self, batch):
        if not self.downsample:
            map1 = batch['map_list'][0].cpu().data
            map2 = batch['map_list'][1].cpu().data
        else:
            map1 = torch.nn.functional.interpolate(batch['map_list'][0].cpu().data, scale_factor=0.5)
            map2 = torch.nn.functional.interpolate(batch['map_list'][1].cpu().data, scale_factor=0.5)
        return map1, map2 

    def vis_map(self, map, name):
        map = map[0].numpy()
        map_img = np.squeeze(np.sum(map, axis=0))
        if(map_img.mean() != 0):
            map_img = (map_img - map_img.min()) / (map_img.max() - map_img.min())
        map_img = np.asarray(map_img * 255, dtype=np.uint8)
        save_map_path = os.path.join(self.save_path, 'train', 'input')
        if not os.path.exists(save_map_path):
            os.makedirs(save_map_path)
        cv2.imwrite(os.path.join(save_map_path, name), map_img)
        return 
    
    def vis_map_RGB(self, map, name):
        map = map[0].numpy()
        channel,h,w = map.shape
        if(channel==5): # events
            map = np.concatenate([map, np.zeros((1,h,w))], axis=0)
            map_img1 = map[:3, ...]
            map_img2 = map[3:, ...]
            for c in range(3):
                if(map_img1[c].mean() != 0):
                    map_img1[c] = (map_img1[c] - map_img1[c].min()) / (map_img1[c].max() - map_img1[c].min()) * 255
                if(map_img2[c].mean() != 0):
                    map_img2[c] = (map_img2[c] - map_img2[c].min()) / (map_img2[c].max() - map_img2[c].min()) * 255

            map_img_sum = np.concatenate([map_img1, map_img2], axis=2) # 叠加在
            map_img_sum = np.asarray(map_img_sum.transpose(1,2,0), dtype=np.uint8)
        elif(channel==3):
            map_img = (map - np.min(map)) / (np.max(map) - np.min(map)) * 255
            map_img_sum = np.asarray(map_img.transpose(1,2,0), dtype=np.uint8)

        else:
            map_img = np.squeeze(np.sum(map, axis=0))
            if(map_img.mean() != 0):
                map_img = (map_img - map_img.min()) / (map_img.max() - map_img.min())
            map_img_sum = np.asarray(map_img * 255, dtype=np.uint8)

        save_map_path = os.path.join(self.save_path, 'train')
        if not os.path.exists(save_map_path):
            os.makedirs(save_map_path)
        cv2.imwrite(os.path.join(save_map_path, name), map_img_sum)
        return 

    def visualize_optical_flow(self, flow, name=None):
        # out = flow_to_image(flow)
            # flow = flow.transpose(1,2,0)
        flow[np.isinf(flow)]=0
        # Use Hue, Saturation, Value colour model
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=float)

        # The additional **0.5 is a scaling factor
        mag = np.sqrt(flow[...,0]**2+flow[...,1]**2)**0.5

        ang = np.arctan2(flow[...,1], flow[...,0])
        ang[ang<0]+=np.pi*2
        hsv[..., 0] = ang/np.pi/2.0 # Scale from 0..1
        hsv[..., 1] = 1
        hsv[..., 2] = (mag-mag.min())/(mag-mag.min()).max() # Scale from 0..1
        rgb = colors.hsv_to_rgb(hsv)
        # This all seems like an overkill, but it's just to exactly match the cv2 implementation
        bgr = np.stack([rgb[...,2],rgb[...,1],rgb[...,0]], axis=2)
        out = bgr*255
        save_flow_path = os.path.join(self.save_path, 'train')
        if not os.path.exists(save_flow_path):
            os.makedirs(save_flow_path)
        cv2.imwrite(os.path.join(save_flow_path, name), out)
        return

    def fetch_optimizer(self, model):
        """ Create the optimizer and learning rate scheduler """
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr, weight_decay=self.wdecay, eps=self.eps)

        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, self.lr, self.num_steps + 100,
                                                pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def summary(self):
        self.logger.write_line("====================================== train SUMMARY ======================================", True)
        self.logger.write_line("Model:\t\t\t" + self.model.__class__.__name__, True)
        self.logger.write_line("trainer:\t\t" + self.__class__.__name__, True)
        self.logger.write_line("train Set:\t" + self.data_loader.dataset.__class__.__name__, True)
        self.logger.write_line("\t-Dataset length:\t"+str(len(self.data_loader)), True)
        self.logger.write_line("\t-Batch size:\t\t" + str(self.data_loader.batch_size), True)
        self.logger.write_line("\t-Parameter Count:\t\t" + str(self.count_parameters(self.model)), True)
        self.logger.write_line("==========================================================================================", True)

    def move_batch_to_cuda(self, batch):
        return move_dict_to_cuda(batch, self.gpu)   
    
    def sequence_loss(self, flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
        """ Loss function defined over sequence of flow predictions """

        n_predictions = len(flow_preds)
        flow_loss = 0.0
        # pdb.set_trace()
        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
        
        valid = (valid >= 0.5) & (mag < max_flow)

        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()

        epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }

        return flow_loss, metrics
    
    def train_iters(self, model, start_epoch=0, val_iters=500):
        model.module.change_imagesize(self.image_size)
        # model.cuda()
        model.train()       

        print("No freeze bn!")
        # model.module.freeze_bn()

        scaler = GradScaler(enabled=self.mixed_precision)
        iters = 0
        total_loss = 0.0
        total_epe = 0.0
        for batch_idx, batch in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            # Move Data to GPU
            batch = self.move_batch_to_cuda(batch)
            # Network Forward Pass
            self.run_network(model,batch)
            f_est, f_flow_mask = self.get_estimation_and_target(batch)
            f_gt = f_flow_mask[0]
            f_mask = f_flow_mask[1]

            loss, metrics = self.sequence_loss(f_est, f_gt, f_mask, self.gamma)

            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
            scaler.step(self.optimizer)
            self.scheduler.step()
            scaler.update()
            total_loss += loss.item()
            total_epe += metrics['epe']
            iters += 1
            istr = 'iters{:02d}  {:05d} / {:05d}  Training Loss:{:2.6f}  AEE: {:2.6f}'.format(iters+start_epoch*val_iters, iters, val_iters, total_loss/iters, total_epe/iters)
            if iters % 10 == 0:
                self.logger.write_line(istr, True)
            if self.visualize_map:
                if iters % 100 ==0:
                        # idx = batch['idx'].cpu().data.numpy()[0]
                        idx = iters + start_epoch*val_iters
                        flow_est = f_est[-1].cpu().data.numpy()[0].transpose(1,2,0)
                        self.visualize_optical_flow(flow_est, str(idx)+'_flow_est.jpg')
                        flow_gt = f_gt.cpu().data.numpy()[0].transpose(1,2,0)
                        self.visualize_optical_flow(flow_gt, str(idx)+'_flow_gt.jpg')

                        map1, map2 = self.get_key_map(batch)
                        self.vis_map_RGB(map1, str(idx)+'_map1.jpg')
                        self.vis_map_RGB(map2, str(idx)+'_map2.jpg') 
            if(iters >= val_iters):
                break
        mloss = total_loss / iters
        mepe = total_epe / iters
        istr = '{:d}th {:d}iters:  Mean Loss:{:2.6f}  Mean AEE: {:2.6f}'.format(start_epoch+1, val_iters, mloss, mepe)
        self.logger.write_line(istr, True)

        if self.return_epe:
            return model, mepe
        return model

    def _train(self, model, epoch=0):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        model.module.change_imagesize(self.image_size)
        model.cuda()
        model.train()

        model.module.freeze_bn()

        scaler = GradScaler(enabled=self.mixed_precision)
        iters = 0
        total_loss = 0.0
        total_epe = 0.0
        for batch_idx, batch in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            # Move Data to GPU
            batch = self.move_batch_to_cuda(batch)
            # Network Forward Pass
            self.run_network(model,batch)
            f_est, f_flow_mask = self.get_estimation_and_target(batch)
            f_gt = f_flow_mask[0]
            f_mask = f_flow_mask[1]
            loss, metrics = self.sequence_loss(f_est, f_gt, f_mask, self.gamma)

            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)                
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), self.clip)
            scaler.step(self.optimizer)
            self.scheduler.step()
            scaler.update()
            total_loss += loss.item()
            total_epe += metrics['epe']
            iters += 1
            istr = 'epoch{:02d}  iters{:02d}  {:05d} / {:05d}  Training Loss:{:2.6f}  AEE: {:2.6f}'.format(epoch, iters, batch_idx + 1, len(self.data_loader), loss.item(), metrics['epe'])
            if iters % 10 == 0:
                self.logger.write_line(istr, True)
            if self.visualize_map:
                if iters % 100 ==0:
                        # idx = batch['idx'].cpu().data.numpy()[0]
                        idx = iters + epoch * len(self.data_loader)
                        flow_est = f_est[-1].cpu().data.numpy()[0].transpose(1,2,0)
                        self.visualize_optical_flow(flow_est, str(idx)+'_flow_est.jpg')
                        flow_gt = f_gt.cpu().data.numpy()[0].transpose(1,2,0)
                        self.visualize_optical_flow(flow_gt, str(idx)+'_flow_gt.jpg')

                        map1, map2 = self.get_key_map(batch)
                        self.vis_map_RGB(map1, str(idx)+'_map1.jpg')
                        self.vis_map_RGB(map2, str(idx)+'_map2.jpg')            
        mloss = total_loss / iters
        mepe = total_epe / iters
        istr = 'epoch{:02d}   Mean Loss:{:2.6f}  Mean AEE: {:2.6f}'.format(epoch, mloss, mepe)
        self.logger.write_line(istr, True)

        if self.return_epe:
            return model, mepe
        return model
    

class TrainSteRaft(train):

    def get_key_map(self, batch):
        if not self.downsample:
            event_volumn = batch['event_volume'].cpu().data
        else:
            event_volumn = torch.nn.functional.interpolate(batch['event_volume'].cpu().data, scale_factor=0.5)
        print(event_volumn.shape)
        event1 = event_volumn[...,0]
        event3 = event_volumn[...,4]
        return event1, event3  

    def run_network(self, model, batch):
        # RAFT just expects two images as input. cleanest. code. ever.
        if not self.downsample:
            event_volume = batch['event_volume']
        else:
            event_volume = torch.nn.functional.interpolate(batch['event_volume'], scale_factor=0.5)
        print(batch['event_volume'].shape)
        batch['map_list'], batch['flow_list'] = model(event_volume)
        batch['flow_est'] = batch['flow_list']


class TrainRaftEvents(train): 

    def run_network(self, model, batch):
        # RAFT just expects two images as input. cleanest. code. ever.
        if not self.downsample:
                im1 = batch['event_volume_old']
                im2 = batch['event_volume_new']
        else:
            im1 = torch.nn.functional.interpolate(batch['event_volume_old'], scale_factor=0.5)
            im2 = torch.nn.functional.interpolate(batch['event_volume_new'], scale_factor=0.5)
        batch['map_list'], batch['flow_list'] = model(im1,im2)
        batch['flow_est'] = batch['flow_list']


class TrainRaftSegEvents(train):
    def move_batch_to_cuda(self, batch):
        return move_dict_to_cuda(batch, self.gpu)    
    
    def get_estimation_and_target(self, batch):
        if not self.downsample:
            if 'valid' in batch.keys():
                return batch['flow_est'], (batch['flow'], batch['valid'])
            return batch['flow_est'], batch['flow']
        else:
            f_est = batch['flow_est']
            f_gt = torch.nn.functional.interpolate(batch['flow'], scale_factor=0.5)
            if 'valid' in batch.keys():
                f_mask = torch.nn.functional.interpolate(batch['valid'], scale_factor=0.5)
                return f_est, (f_gt, f_mask)
            return f_est, f_gt

    def run_network(self, model, batch):
        # RAFT just expects two images as input. cleanest. code. ever.
        event_seg = batch['event_raw']
        event_volume = batch['event_volume']
        batch['map_list'], batch['flow_list'] = model(event_seg, event_volume)
        batch['flow_est'] = batch['flow_list']


class TrainDenseSparse(train):
    def move_batch_to_cuda(self, batch):
        return move_dict_to_cuda(batch, self.gpu)    
    
    def get_estimation_and_target(self, batch):
        if not self.downsample:
            if 'valid' in batch.keys():
                return batch['flow_est'], (batch['flow'], batch['valid'])
            return batch['flow_est'], batch['flow']
        else:
            f_est = batch['flow_est']
            f_gt = torch.nn.functional.interpolate(batch['flow'], scale_factor=0.5)
            if 'valid' in batch.keys():
                f_mask = torch.nn.functional.interpolate(batch['valid'], scale_factor=0.5)
                return f_est, (f_gt, f_mask)
            return f_est, f_gt

    def get_events(self, batch):
        if not self.downsample:
            events0 = batch['event_volume_old'].cpu().data
            events1 = batch['event_volume_new'].cpu().data
        else:
            events0 = torch.nn.functional.interpolate(batch['event_volume_old'].cpu().data, scale_factor=0.5)
            events1 = torch.nn.functional.interpolate(batch['event_volume_new'].cpu().data, scale_factor=0.5)
        return events0, events1
    
    def get_dense_events(self, batch):
        if not self.downsample:
            events0 = batch['d_event_volume_old'].cpu().data
            events1 = batch['d_event_volume_new'].cpu().data
        else:
            events0 = torch.nn.functional.interpolate(batch['d_event_volume_old'].cpu().data, scale_factor=0.5)
            events1 = torch.nn.functional.interpolate(batch['d_event_volume_old'].cpu().data, scale_factor=0.5)
        return events0, events1

    def get_dense_events_batch(self, batch):
        if not self.downsample:
            d_event0 = batch['d_event_volume_old']
            d_event1 = batch['d_event_volume_new']
        else:
            d_event0 = torch.nn.functional.interpolate(batch['d_event_volume_old'], scale_factor=0.5)
            d_event1 = torch.nn.functional.interpolate(batch['d_event_volume_new'], scale_factor=0.5)
        return torch.cat((d_event0, d_event1), dim=0)
    
    def get_unet_out_batch(self, batch):
        if not self.downsample:
            map1 = batch['map_list'][0]
            map2 = batch['map_list'][1]
        else:
            map1 = torch.nn.functional.interpolate(batch['map_list'], scale_factor=0.5)
            map2 = torch.nn.functional.interpolate(batch['map_list'], scale_factor=0.5)
        return torch.cat((map1, map2), dim=0)
        # return map1, map2
    
    def get_mimo_unet_out_batch(self, batch):
        map_list = []
        for maps in batch['map_list']:
            if not self.downsample:
                map1 = maps[0]
                map2 = maps[1]
            else:
                map1 = torch.nn.functional.interpolate(maps[0], scale_factor=0.5)
                map2 = torch.nn.functional.interpolate(maps[1], scale_factor=0.5)
            map_list.append(torch.cat((map1, map2), dim=0))
        return map_list

    def run_unet(self, model, batch):
        # RAFT just expects two images as input. cleanest. code. ever.
        if not self.downsample:
                im1 = batch['event_volume_old']
                im2 = batch['event_volume_new']
        else:
            im1 = torch.nn.functional.interpolate(batch['event_volume_old'], scale_factor=0.5)
            im2 = torch.nn.functional.interpolate(batch['event_volume_new'], scale_factor=0.5)
        batch['map_list'] = model.module.run_unet(im1,im2)


    def run_network(self, model, batch):
        # RAFT just expects two images as input. cleanest. code. ever.
        if not self.downsample:
                im1 = batch['event_volume_old']
                im2 = batch['event_volume_new']
        else:
            im1 = torch.nn.functional.interpolate(batch['event_volume_old'], scale_factor=0.5)
            im2 = torch.nn.functional.interpolate(batch['event_volume_new'], scale_factor=0.5)
        batch['map_list'], batch['flow_list'] = model(im1,im2)
        batch['flow_est'] = batch['flow_list']

    def run_network_with_dense(self, model, batch):
        # RAFT just expects two images as input. cleanest. code. ever.
        if not self.downsample:
                im1 = batch['event_volume_old']
                im2 = batch['event_volume_new']
                d_im1 = batch['d_event_volume_old']
                d_im2 = batch['d_event_volume_new']
        else:
            im1 = torch.nn.functional.interpolate(batch['event_volume_old'], scale_factor=0.5)
            im2 = torch.nn.functional.interpolate(batch['event_volume_new'], scale_factor=0.5)
            d_im1 = torch.nn.functional.interpolate(batch['d_event_volume_old'], scale_factor=0.5)
            d_im2 = torch.nn.functional.interpolate(batch['d_event_volume_new'], scale_factor=0.5)
        batch['map_list'], batch['flow_list'] = model(im1,im2,d_im1,d_im2)
        batch['flow_est'] = batch['flow_list']
    
    def dice_reg(self, x, target):
        # Average of Dice coefficient for all batches, or for a single mask
        regress = 0.
        batch_size = x.shape[0]
        for i in range(batch_size):
            x_i = x[i].reshape(-1)
            t_i = target[i].reshape(-1)
            regress += torch.sum((x_i - t_i) ** 2, dim=0).sqrt()
        
        return regress / batch_size

    def compute_loss(self, batch):
        d_event = self.get_dense_events_batch(batch)
        d_event_pre = self.get_unet_out_batch(batch)

        reg_loss = self.dice_reg(d_event, d_event_pre)

        f_est, f_flow_mask = self.get_estimation_and_target(batch)
        f_gt = f_flow_mask[0]
        f_mask = f_flow_mask[1]
        flow_loss, metrics = self.sequence_loss(f_est, f_gt, f_mask, self.gamma)

        den = 100.0
        loss = reg_loss  + flow_loss * den

        return loss, metrics, reg_loss, flow_loss
    
    def compute_ctx_loss(self, batch):
        event1, d_event1 = self.get_unet_out_batch(batch)
        # pdb.set_trace()
        reg_loss = self.dice_reg(event1, d_event1)

        f_est, f_flow_mask = self.get_estimation_and_target(batch)
        f_gt = f_flow_mask[0]
        f_mask = f_flow_mask[1]
        flow_loss, metrics = self.sequence_loss(f_est, f_gt, f_mask, self.gamma)

        den = 100.0
        loss = reg_loss / 100 + flow_loss * den

        return loss, metrics, reg_loss, flow_loss
    
    def compute_mimounet_loss(self, batch):
        # criterion = torch.nn.L1Loss()

        d_event_pre = self.get_mimo_unet_out_batch(batch)

        d_event = self.get_dense_events_batch(batch)

        # print(d_event.shape)
        # print(d_event_pre[-1].shape)
        d_event2 = torch.nn.functional.interpolate(d_event, scale_factor=0.5, mode='bilinear')
        d_event4 = torch.nn.functional.interpolate(d_event, scale_factor=0.25, mode='bilinear')

        l1 = self.dice_reg(d_event_pre[0], d_event4)
        l2 = self.dice_reg(d_event_pre[1], d_event2)
        l3 = self.dice_reg(d_event_pre[2], d_event)
        reg_loss = l1+l2+l3

        # l1 = criterion(d_event_pre[0], d_event4)
        # l2 = criterion(d_event_pre[1], d_event2)
        # l3 = criterion(d_event_pre[2], d_event)
        # loss_content = l1+l2+l3

        # label_fft1 = torch.rfft(d_event4, signal_ndim=2, normalized=False, onesided=False)
        # pred_fft1 = torch.rfft(d_event_pre[0], signal_ndim=2, normalized=False, onesided=False)
        # label_fft2 = torch.rfft(d_event2, signal_ndim=2, normalized=False, onesided=False)
        # pred_fft2 = torch.rfft(d_event_pre[1], signal_ndim=2, normalized=False, onesided=False)
        # label_fft3 = torch.rfft(d_event, signal_ndim=2, normalized=False, onesided=False)
        # pred_fft3 = torch.rfft(d_event_pre[2], signal_ndim=2, normalized=False, onesided=False)
        # f1 = criterion(pred_fft1, label_fft1)
        # f2 = criterion(pred_fft2, label_fft2)
        # f3 = criterion(pred_fft3, label_fft3)
        # loss_fft = f1+f2+f3

        # reg_loss = 10 * loss_content + loss_fft

        f_est, f_flow_mask = self.get_estimation_and_target(batch)
        f_gt = f_flow_mask[0]
        f_mask = f_flow_mask[1]
        flow_loss, metrics = self.sequence_loss(f_est, f_gt, f_mask, self.gamma)

        den = 100.0
        loss = reg_loss  + flow_loss * den

        return loss, metrics, reg_loss, flow_loss
    
    def compute_density_loss(self, d_event_pre, d_event):
        def compute_density(event):
            c,h,w = event.shape
            value_th = 0
            event_sum = torch.sum(torch.abs(event), dim=0)
            # event_map = torch.where(event_sum>value_th)
            density = torch.sum(event_sum>value_th)/ (h*w)
            return density
        criterion = torch.nn.L1Loss()
        loss = 0.
        batch_size = d_event.shape[0]
        for i in range(batch_size):    
            x_density = compute_density(d_event_pre[i])
            t_density = compute_density(d_event[i])
            
            loss += criterion(x_density, t_density)

        return loss

    def compute_mimounet_loss_with_density(self, batch):
        # criterion = torch.nn.L1Loss()

        d_event_pre = self.get_mimo_unet_out_batch(batch)

        d_event = self.get_dense_events_batch(batch)

        # print(d_event.shape)
        # print(d_event_pre[-1].shape)
        d_event2 = torch.nn.functional.interpolate(d_event, scale_factor=0.5, mode='bilinear')
        d_event4 = torch.nn.functional.interpolate(d_event, scale_factor=0.25, mode='bilinear')

        l1 = self.dice_reg(d_event_pre[0], d_event4)
        l2 = self.dice_reg(d_event_pre[1], d_event2)
        l3 = self.dice_reg(d_event_pre[2], d_event)
        reg_loss = l1+l2+l3

        density_loss = self.compute_density_loss(d_event_pre[2], d_event)

        reg_loss += density_loss * 50

        f_est, f_flow_mask = self.get_estimation_and_target(batch)
        f_gt = f_flow_mask[0]
        f_mask = f_flow_mask[1]
        flow_loss, metrics = self.sequence_loss(f_est, f_gt, f_mask, self.gamma)

        den = 200.0
        loss = reg_loss  + flow_loss * den 

        return loss, metrics, reg_loss, flow_loss
    
    def compute_mimounet_loss_just_density(self, batch):
        # criterion = torch.nn.L1Loss()

        d_event_pre = self.get_mimo_unet_out_batch(batch)
        d_event = self.get_dense_events_batch(batch)

        density_loss = self.compute_density_loss(d_event_pre[2], d_event)

        reg_loss = density_loss * 50
        print(density_loss)
        f_est, f_flow_mask = self.get_estimation_and_target(batch)
        f_gt = f_flow_mask[0]
        f_mask = f_flow_mask[1]
        flow_loss, metrics = self.sequence_loss(f_est, f_gt, f_mask, self.gamma)

        den = 200.0
        loss = reg_loss  + flow_loss * den 

        return loss, metrics, reg_loss, flow_loss
    
    def compute_fix_density_loss(self, d_event_pre, fix_density=0.6):
        def compute_density(event):
            c,h,w = event.shape
            value_th = 0.5
            event_sum = torch.sum(torch.abs(event), dim=0)
            # event_map = torch.where(event_sum>value_th)
            density = torch.sum(event_sum>value_th)/ (h*w)
            return density
        criterion = torch.nn.L1Loss()
        loss = 0.
        batch_size = d_event_pre.shape[0]

        for i in range(batch_size):    

            x_density = compute_density(d_event_pre[i])

            t_density = torch.tensor(fix_density).to(device=x_density.device)
            loss += criterion(x_density, t_density)
        
        return loss

    def compute_mimounet_fix_density(self, batch):
        # criterion = torch.nn.L1Loss()

        d_event_pre = self.get_mimo_unet_out_batch(batch)

        d_event = self.get_dense_events_batch(batch)

        d_event2 = torch.nn.functional.interpolate(d_event, scale_factor=0.5, mode='bilinear')
        d_event4 = torch.nn.functional.interpolate(d_event, scale_factor=0.25, mode='bilinear')

        l1 = self.dice_reg(d_event_pre[0], d_event4)
        l2 = self.dice_reg(d_event_pre[1], d_event2)
        l3 = self.dice_reg(d_event_pre[2], d_event)
        reg_loss = l1+l2+l3

        density_loss = self.compute_fix_density_loss(d_event_pre[2], fix_density=0.6)
        print(density_loss)
        reg_loss += density_loss * 50

        f_est, f_flow_mask = self.get_estimation_and_target(batch)
        f_gt = f_flow_mask[0]
        f_mask = f_flow_mask[1]
        flow_loss, metrics = self.sequence_loss(f_est, f_gt, f_mask, self.gamma)

        den = 200.0
        loss = reg_loss  + flow_loss * den 

        return loss, metrics, reg_loss, flow_loss

    def train_iters(self, model, start_epoch=0, val_iters=500):
        model.module.change_imagesize(self.image_size)
        model.cuda()
        model.train()       

        model.module.freeze_bn()

        scaler = GradScaler(enabled=self.mixed_precision)
        iters = 0
        total_loss = 0.0
        total_epe = 0.0
        total_reg_loss = 0.0
        total_flow_loss = 0.0
        for batch_idx, batch in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            # Move Data to GPU
            batch = self.move_batch_to_cuda(batch)
            # Network Forward Pass
            self.run_network(model,batch)
            f_est, f_flow_mask = self.get_estimation_and_target(batch)
            f_gt = f_flow_mask[0]
            f_mask = f_flow_mask[1]
            loss, metrics, reg_loss, flow_loss = self.compute_loss(batch)
            # loss, metrics, reg_loss, flow_loss = self.compute_ctx_loss(batch)

            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
            scaler.step(self.optimizer)
            self.scheduler.step()
            scaler.update()
            total_loss += loss.item()
            total_reg_loss += reg_loss.item()
            total_flow_loss += flow_loss.item()
            total_epe += metrics['epe']
            iters += 1
            istr = 'iters{:02d}  {:05d}/{:05d} Training Loss:{:2.6f}(Reg loss{:2.3f}, Flow loss{:2.3f}) AEE: {:2.6f}'.\
                format(iters+start_epoch*val_iters, iters, val_iters, total_loss/iters, total_reg_loss/iters, total_flow_loss/iters, metrics['epe'])
            if iters % 10 == 0:
                self.logger.write_line(istr, True)
            if self.visualize_map:
                if iters % 100 ==0:
                        # idx = batch['idx'].cpu().data.numpy()[0]
                        idx = iters + start_epoch*val_iters
                        flow_est = f_est[-1].cpu().data.numpy()[0].transpose(1,2,0)
                        self.visualize_optical_flow(flow_est, str(idx)+'_flow_est.jpg')
                        flow_gt = f_gt.cpu().data.numpy()[0].transpose(1,2,0)
                        self.visualize_optical_flow(flow_gt, str(idx)+'_flow_gt.jpg')

                        map0, map1 = self.get_key_map(batch)
                        self.vis_map_RGB(map0, str(idx)+'_events_map0.jpg')
                        self.vis_map_RGB(map1, str(idx)+'_events_map1.jpg')

                        events0, events1 = self.get_events(batch)
                        d_events0, d_events1 = self.get_dense_events(batch)
                        self.vis_map_RGB(events0, str(idx)+'_events0.jpg')
                        self.vis_map_RGB(events1, str(idx)+'_events1.jpg')
                        self.vis_map_RGB(d_events0, str(idx)+'_events_dense0.jpg')
                        self.vis_map_RGB(d_events1, str(idx)+'_events_dense1.jpg')

            if(iters >= val_iters):
                break
        mloss = total_loss / iters
        mepe = total_epe / iters
        istr = '{:d}th {:d}iters:  Mean Loss:{:2.6f}  Mean AEE: {:2.6f}'.format(start_epoch+1, val_iters, mloss, mepe)
        self.logger.write_line(istr, True)

        if self.return_epe:
            return model, mepe
        return model
    
    def train_iters_without_reg(self, model, start_epoch=0, val_iters=500):
        model.module.change_imagesize(self.image_size)
        model.cuda()
        model.train()

        model.module.freeze_bn()

        scaler = GradScaler(enabled=self.mixed_precision)
        iters = 0
        total_loss = 0.0
        total_epe = 0.0
        for batch_idx, batch in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            # Move Data to GPU
            batch = self.move_batch_to_cuda(batch)
            # Network Forward Pass
            self.run_network(model,batch)
            f_est, f_flow_mask = self.get_estimation_and_target(batch)
            f_gt = f_flow_mask[0]
            f_mask = f_flow_mask[1]
            loss, metrics = self.sequence_loss(f_est, f_gt, f_mask, self.gamma)

            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
            scaler.step(self.optimizer)
            self.scheduler.step()
            scaler.update()
            total_loss += loss.item()
            total_epe += metrics['epe']
            iters += 1
            istr = 'iters{:02d}  {:05d} / {:05d}  Training Loss:{:2.6f}  AEE: {:2.6f}'.format(iters+start_epoch*val_iters, iters, val_iters, loss.item(), metrics['epe'])
            if iters % 10 == 0:
                self.logger.write_line(istr, True)
            if self.visualize_map:
                if iters % 100 ==0:
                        # idx = batch['idx'].cpu().data.numpy()[0]
                        idx = iters + start_epoch*val_iters
                        flow_est = f_est[-1].cpu().data.numpy()[0].transpose(1,2,0)
                        self.visualize_optical_flow(flow_est, str(idx)+'_flow_est.jpg')
                        flow_gt = f_gt.cpu().data.numpy()[0].transpose(1,2,0)
                        self.visualize_optical_flow(flow_gt, str(idx)+'_flow_gt.jpg')

                        map1, map2 = self.get_key_map(batch)
                        self.vis_map_RGB(map1, str(idx)+'_map1.jpg')
                        self.vis_map_RGB(map2, str(idx)+'_map2.jpg') 
            if(iters >= val_iters):
                break
        mloss = total_loss / iters
        mepe = total_epe / iters
        istr = '{:d}th {:d}iters:  Mean Loss:{:2.6f}  Mean AEE: {:2.6f}'.format(start_epoch+1, val_iters, mloss, mepe)
        self.logger.write_line(istr, True)

        if self.return_epe:
            return model, mepe
        return model
    
    def train_unet_iters(self, model, start_epoch=0, val_iters=500):
        model.module.change_imagesize(self.image_size)
        model.cuda()
        model.train()

        model.module.freeze_bn()

        scaler = GradScaler(enabled=self.mixed_precision)
        iters = 0

        total_reg_loss = 0.0

        for batch_idx, batch in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            # Move Data to GPU
            batch = self.move_batch_to_cuda(batch)
            # Network Forward Pass
            self.run_unet(model,batch)

            d_event = self.get_dense_events_batch(batch)
            d_event_pre = self.get_unet_out_batch(batch)

            reg_loss = self.dice_reg(d_event, d_event_pre)

            scaler.scale(reg_loss).backward()
            scaler.unscale_(self.optimizer)                
            torch.nn.utils.clip_grad_norm_(model.module.unet.parameters(), self.clip)
            scaler.step(self.optimizer)
            self.scheduler.step()
            scaler.update()

            total_reg_loss += reg_loss.item()
            iters += 1
            istr = 'iters{:02d}  {:05d} / {:05d}  Reg Loss:{:2.6f}'.format(iters+start_epoch*val_iters, iters, val_iters, total_reg_loss/iters)
            if iters % 10 == 0:
                self.logger.write_line(istr, True)
                
            if(iters >= val_iters):
                break
            
        mloss = total_reg_loss / iters
        istr = '{:d}th {:d}iters:  Mean reg Loss:{:2.6f}'.format(start_epoch+1, val_iters, mloss)
        self.logger.write_line(istr, True)
        
        return model
    
    def get_mimounet_out(self, batch):
        maps_list = batch['map_list'][-1]
        if not self.downsample:
            map1 = maps_list[0].cpu().data
            map2 = maps_list[1].cpu().data
        else:
            map1 = torch.nn.functional.interpolate(maps_list[0].cpu().data, scale_factor=0.5)
            map2 = torch.nn.functional.interpolate(maps_list[1].cpu().data, scale_factor=0.5)
        return map1, map2 

    def train_mimounet_iters(self, model, start_epoch=0, val_iters=500, without_reg=False, compute_density=False):
        model.module.change_imagesize(self.image_size)
        model.cuda()
        model.train()

        # model.module.freeze_bn()
  
        scaler = GradScaler(enabled=self.mixed_precision)
        iters = 0
        total_loss = 0.0
        total_epe = 0.0
        total_reg_loss = 0.0
        total_flow_loss = 0.0
        for batch_idx, batch in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            # Move Data to GPU
            batch = self.move_batch_to_cuda(batch)
            # Network Forward Pass
            self.run_network(model,batch)
 
            if(without_reg):
                f_est, f_flow_mask = self.get_estimation_and_target(batch)
                f_gt = f_flow_mask[0]
                f_mask = f_flow_mask[1]
                loss, metrics = self.sequence_loss(f_est, f_gt, f_mask, self.gamma)
                flow_loss = loss
                reg_loss = loss - flow_loss
            elif(compute_density):
                loss, metrics, reg_loss, flow_loss = self.compute_mimounet_loss_with_density(batch)
            else:
                loss, metrics, reg_loss, flow_loss = self.compute_mimounet_loss(batch)

            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)                
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), self.clip)
            scaler.step(self.optimizer)
            self.scheduler.step()
            scaler.update()
            total_loss += loss.item()
            total_reg_loss += reg_loss.item()
            total_flow_loss += flow_loss.item()
            total_epe += metrics['epe']
            iters += 1
            istr = 'iters{:02d}  {:05d}/{:05d} Training Loss:{:2.6f}(Reg loss{:2.3f}, Flow loss{:2.3f}) AEE: {:2.6f}'.\
                format(iters+start_epoch*val_iters, iters, val_iters, total_loss/iters, total_reg_loss/iters, total_flow_loss/iters, metrics['epe'])
            
            if iters % 10 == 0:
                self.logger.write_line(istr, True)
            if self.visualize_map:
                if iters % 100 ==0:
                        # idx = batch['idx'].cpu().data.numpy()[0]

                        f_est, f_flow_mask = self.get_estimation_and_target(batch)
                        f_gt = f_flow_mask[0]

                        idx = iters + start_epoch*val_iters
                        flow_est = f_est[-1].cpu().data.numpy()[0].transpose(1,2,0)
                        self.visualize_optical_flow(flow_est, str(idx)+'_flow_est.jpg')
                        flow_gt = f_gt.cpu().data.numpy()[0].transpose(1,2,0)
                        self.visualize_optical_flow(flow_gt, str(idx)+'_flow_gt.jpg')

                        map0, map1 = self.get_mimounet_out(batch)
                        self.vis_map_RGB(map0, str(idx)+'_events_map0.jpg')
                        self.vis_map_RGB(map1, str(idx)+'_events_map1.jpg')

                        events0, events1 = self.get_events(batch)
                        d_events0, d_events1 = self.get_dense_events(batch)
                        self.vis_map_RGB(events0, str(idx)+'_events0.jpg')
                        self.vis_map_RGB(events1, str(idx)+'_events1.jpg')
                        self.vis_map_RGB(d_events0, str(idx)+'_events_dense0.jpg')
                        self.vis_map_RGB(d_events1, str(idx)+'_events_dense1.jpg')

            if(iters >= val_iters):
                break
        mloss = total_loss / iters
        mepe = total_epe / iters
        istr = '{:d}th {:d}iters:  Mean Loss:{:2.6f}  Mean AEE: {:2.6f}'.format(start_epoch+1, val_iters, mloss, mepe)
        self.logger.write_line(istr, True)

        if self.return_epe:
            return model, mepe
        return model
    

    def train_mimounet_iters_test_loss(self, model, start_epoch=0, val_iters=500, just_density=False):
        model.module.change_imagesize(self.image_size)
        model.cuda()
        model.train()

        model.module.freeze_bn()
  

        scaler = GradScaler(enabled=self.mixed_precision)
        iters = 0
        total_loss = 0.0
        total_epe = 0.0
        total_reg_loss = 0.0
        total_flow_loss = 0.0
        for batch_idx, batch in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            # Move Data to GPU
            batch = self.move_batch_to_cuda(batch)
            # Network Forward Pass
            self.run_network(model,batch)
 

            if(just_density):
                loss, metrics, reg_loss, flow_loss = self.compute_mimounet_loss_just_density(batch)
            else:
                loss, metrics, reg_loss, flow_loss = self.compute_mimounet_fix_density(batch)

            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)                
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), self.clip)
            scaler.step(self.optimizer)
            self.scheduler.step()
            scaler.update()
            total_loss += loss.item()
            total_reg_loss += reg_loss.item()
            total_flow_loss += flow_loss.item()
            total_epe += metrics['epe']
            iters += 1
            istr = 'iters{:02d}  {:05d}/{:05d} Training Loss:{:2.6f}(Reg loss{:2.3f}, Flow loss{:2.3f}) AEE: {:2.6f}'.\
                format(iters+start_epoch*val_iters, iters, val_iters, total_loss/iters, total_reg_loss/iters, total_flow_loss/iters, metrics['epe'])
            
            if iters % 10 == 0:
                self.logger.write_line(istr, True)
            if self.visualize_map:
                if iters % 100 ==0:
                        # idx = batch['idx'].cpu().data.numpy()[0]

                        f_est, f_flow_mask = self.get_estimation_and_target(batch)
                        f_gt = f_flow_mask[0]

                        idx = iters + start_epoch*val_iters
                        flow_est = f_est[-1].cpu().data.numpy()[0].transpose(1,2,0)
                        self.visualize_optical_flow(flow_est, str(idx)+'_flow_est.jpg')
                        flow_gt = f_gt.cpu().data.numpy()[0].transpose(1,2,0)
                        self.visualize_optical_flow(flow_gt, str(idx)+'_flow_gt.jpg')

                        map0, map1 = self.get_mimounet_out(batch)
                        self.vis_map_RGB(map0, str(idx)+'_events_map0.jpg')
                        self.vis_map_RGB(map1, str(idx)+'_events_map1.jpg')

                        events0, events1 = self.get_events(batch)
                        d_events0, d_events1 = self.get_dense_events(batch)
                        self.vis_map_RGB(events0, str(idx)+'_events0.jpg')
                        self.vis_map_RGB(events1, str(idx)+'_events1.jpg')
                        self.vis_map_RGB(d_events0, str(idx)+'_events_dense0.jpg')
                        self.vis_map_RGB(d_events1, str(idx)+'_events_dense1.jpg')

            if(iters >= val_iters):
                break
        mloss = total_loss / iters
        mepe = total_epe / iters
        istr = '{:d}th {:d}iters:  Mean Loss:{:2.6f}  Mean AEE: {:2.6f}'.format(start_epoch+1, val_iters, mloss, mepe)
        self.logger.write_line(istr, True)

        if self.return_epe:
            return model, mepe
        return model

class TrainGMflowEvents(train):
    def move_batch_to_cuda(self, batch):
        return move_dict_to_cuda(batch, self.gpu)    
    
    def get_estimation_and_target(self, batch):
        if not self.downsample:
            if 'valid' in batch.keys():
                return batch['flow_est'], (batch['flow'], batch['valid'])
            return batch['flow_est'], batch['flow']
        else:
            f_est = batch['flow_est']
            f_gt = torch.nn.functional.interpolate(batch['flow'], scale_factor=0.5)
            if 'valid' in batch.keys():
                f_mask = torch.nn.functional.interpolate(batch['valid'], scale_factor=0.5)
                return f_est, (f_gt, f_mask)
            return f_est, f_gt

    def run_network(self, model, batch):
        # RAFT just expects two images as input. cleanest. code. ever.
        if not self.downsample:
                events1 = batch['event_volume_old']
                events2 = batch['event_volume_new']
        else:
            events1 = torch.nn.functional.interpolate(batch['event_volume_old'], scale_factor=0.5)
            events2 = torch.nn.functional.interpolate(batch['event_volume_new'], scale_factor=0.5)
        batch['map_list'], batch['flow_list'] = model(events1=events1, events2=events2,
                                    attn_splits_list=self.config['attn_splits_list'],
                                    corr_radius_list=self.config['corr_radius_list'],
                                    prop_radius_list=self.config['prop_radius_list'])
        batch['flow_est'] = batch['flow_list']

class TrainGMflowEventsDense(TrainDenseSparse):
    def move_batch_to_cuda(self, batch):
        return move_dict_to_cuda(batch, self.gpu)    
    
    def get_estimation_and_target(self, batch):
        if not self.downsample:
            if 'valid' in batch.keys():
                return batch['flow_est'], (batch['flow'], batch['valid'])
            return batch['flow_est'], batch['flow']
        else:
            f_est = batch['flow_est']
            f_gt = torch.nn.functional.interpolate(batch['flow'], scale_factor=0.5)
            if 'valid' in batch.keys():
                f_mask = torch.nn.functional.interpolate(batch['valid'], scale_factor=0.5)
                return f_est, (f_gt, f_mask)
            return f_est, f_gt

    def run_network(self, model, batch):
        # RAFT just expects two images as input. cleanest. code. ever.
        if not self.downsample:
                events1 = batch['event_volume_old']
                events2 = batch['event_volume_new']
        else:
            events1 = torch.nn.functional.interpolate(batch['event_volume_old'], scale_factor=0.5)
            events2 = torch.nn.functional.interpolate(batch['event_volume_new'], scale_factor=0.5)
        batch['map_list'], batch['flow_list'] = model(events1=events1, events2=events2,
                                    attn_splits_list=self.config['attn_splits_list'],
                                    corr_radius_list=self.config['corr_radius_list'],
                                    prop_radius_list=self.config['prop_radius_list'])
        batch['flow_est'] = batch['flow_list']

class TrainGMflownet(train):

    def run_network(self, model, batch):
        # RAFT just expects two images as input. cleanest. code. ever.
        if not self.downsample:
                im1 = batch['event_volume_old']
                im2 = batch['event_volume_new']
        else:
            im1 = torch.nn.functional.interpolate(batch['event_volume_old'], scale_factor=0.5)
            im2 = torch.nn.functional.interpolate(batch['event_volume_new'], scale_factor=0.5)
        batch['map_list'], batch['flow_list'] = model(im1,im2)
        batch['flow_est'] = batch['flow_list']

    def get_events(self, batch):
        if not self.downsample:
            events0 = batch['event_volume_old'].cpu().data
            events1 = batch['event_volume_new'].cpu().data
        else:
            events0 = torch.nn.functional.interpolate(batch['event_volume_old'], scale_factor=0.5)
            events1 = torch.nn.functional.interpolate(batch['event_volume_new'], scale_factor=0.5)
        return events0, events1
    
    def get_dense_events(self, batch):
        if not self.downsample:
            events0 = batch['d_event_volume_old'].cpu().data
            events1 = batch['d_event_volume_new'].cpu().data
        else:
            events0 = torch.nn.functional.interpolate(batch['d_event_volume_old'].cpu().data, scale_factor=0.5)
            events1 = torch.nn.functional.interpolate(batch['d_event_volume_old'].cpu().data, scale_factor=0.5)
        return events0, events1

    def get_events_batch(self, batch):
        if not self.downsample:
            events0 = batch['event_volume_old']
            events1 = batch['event_volume_new']
        else:
            events0 = torch.nn.functional.interpolate(batch['event_volume_old'], scale_factor=0.5)
            events1 = torch.nn.functional.interpolate(batch['event_volume_new'], scale_factor=0.5)
        return events0, events1

    def get_dense_events_batch(self, batch):
        if not self.downsample:
            d_event0 = batch['d_event_volume_old']
            d_event1 = batch['d_event_volume_new']
        else:
            d_event0 = torch.nn.functional.interpolate(batch['d_event_volume_old'], scale_factor=0.5)
            d_event1 = torch.nn.functional.interpolate(batch['d_event_volume_new'], scale_factor=0.5)
        return torch.cat((d_event0, d_event1), dim=0)
    
    def get_unet_out_batch(self, batch):
        if not self.downsample:
            map1 = batch['map_list'][0]
            map2 = batch['map_list'][1]
        else:
            map1 = torch.nn.functional.interpolate(batch['map_list'], scale_factor=0.5)
            map2 = torch.nn.functional.interpolate(batch['map_list'], scale_factor=0.5)
        return torch.cat((map1, map2), dim=0)
        # return map1, map2
    
    def get_mimo_unet_out_batch(self, batch):
        map_list = []
        for maps in batch['map_list']:
            if not self.downsample:
                map1 = maps[0]
                map2 = maps[1]
            else:
                map1 = torch.nn.functional.interpolate(maps[0], scale_factor=0.5)
                map2 = torch.nn.functional.interpolate(maps[1], scale_factor=0.5)
            map_list.append(torch.cat((map1, map2), dim=0))
        return map_list
    
    def get_mimounet_out(self, batch):
        maps_list = batch['map_list'][-1]
        if not self.downsample:
            map1 = maps_list[0].cpu().data
            map2 = maps_list[1].cpu().data
        else:
            map1 = torch.nn.functional.interpolate(maps_list[0].cpu().data, scale_factor=0.5)
            map2 = torch.nn.functional.interpolate(maps_list[1].cpu().data, scale_factor=0.5)
        return map1, map2 

    def sequence_loss(self, train_outputs, image1, image2, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW, use_matching_loss=True):
        import configparser
        """ Loss function defined over sequence of flow predictions """
        flow_preds, softCorrMap = train_outputs

        # original RAFT loss
        n_predictions = len(flow_preds)
        flow_loss = 0.0

        # exclude invalid pixels and extremely large displacements
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        valid = (valid >= 0.5) & (mag < max_flow)

        for i in range(n_predictions):
            i_weight = gamma**(n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None].float()  * i_loss).mean()

        epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }

        if use_matching_loss:
            # enable global matching loss. Try to use it in late stages of the trianing
            img_2back1 = backwarp(image2, flow_gt)
            occlusionMap = (image1 - img_2back1).mean(1, keepdims=True) #(N, H, W)
            occlusionMap = torch.abs(occlusionMap) > 20
            occlusionMap = occlusionMap.float()

            conf_matrix_gt = compute_supervision_coarse(flow_gt, occlusionMap, 8) # 8 from RAFT downsample

            matchLossCfg = configparser.ConfigParser()
            matchLossCfg.POS_WEIGHT = 1
            matchLossCfg.NEG_WEIGHT = 1
            matchLossCfg.FOCAL_ALPHA = 0.25
            matchLossCfg.FOCAL_GAMMA = 2.0
            matchLossCfg.COARSE_TYPE = 'cross_entropy'
            match_loss = compute_coarse_loss(softCorrMap, conf_matrix_gt, matchLossCfg)

            flow_loss = flow_loss + 0.01*match_loss

        return flow_loss, metrics
    
    def train_iters(self, model, start_epoch=0, val_iters=500):
        model.module.change_imagesize(self.image_size)
        model.cuda()
        model.train()       

        model.module.freeze_bn()

        scaler = GradScaler(enabled=self.mixed_precision)
        iters = 0
        total_loss = 0.0
        total_epe = 0.0
        for batch_idx, batch in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            # Move Data to GPU
            batch = self.move_batch_to_cuda(batch)
            # Network Forward Pass
            self.run_network(model,batch)
            train_outputs, f_flow_mask = self.get_estimation_and_target(batch)
            f_est = train_outputs[0]
            f_gt = f_flow_mask[0]
            f_mask = f_flow_mask[1]
            events1, events2 = self.get_events_batch(batch)
            loss, metrics = self.sequence_loss(train_outputs, events1, events2, f_gt, f_mask, self.gamma)

            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
            scaler.step(self.optimizer)
            self.scheduler.step()
            scaler.update()
            total_loss += loss.item()
            total_epe += metrics['epe']
            iters += 1
            istr = 'iters{:02d}  {:05d} / {:05d}  Training Loss:{:2.6f}  AEE: {:2.6f}'.format(iters+start_epoch*val_iters, iters, val_iters, total_loss/iters, total_epe/iters)
            if iters % 10 == 0:
                self.logger.write_line(istr, True)
            if self.visualize_map:
                if iters % 100 ==0:
                        # idx = batch['idx'].cpu().data.numpy()[0]
                        idx = iters + start_epoch*val_iters
                        flow_est = f_est[-1].cpu().data.numpy()[0].transpose(1,2,0)
                        self.visualize_optical_flow(flow_est, str(idx)+'_flow_est.jpg')
                        flow_gt = f_gt.cpu().data.numpy()[0].transpose(1,2,0)
                        self.visualize_optical_flow(flow_gt, str(idx)+'_flow_gt.jpg')

                        map1, map2 = self.get_key_map(batch)
                        self.vis_map_RGB(map1, str(idx)+'_map1.jpg')
                        self.vis_map_RGB(map2, str(idx)+'_map2.jpg') 
            if(iters >= val_iters):
                break
        mloss = total_loss / iters
        mepe = total_epe / iters
        istr = '{:d}th {:d}iters:  Mean Loss:{:2.6f}  Mean AEE: {:2.6f}'.format(start_epoch+1, val_iters, mloss, mepe)
        self.logger.write_line(istr, True)

        if self.return_epe:
            return model, mepe
        return model
    
    def compute_density_loss(self, d_event_pre, d_event):
        def compute_density(event):
            c,h,w = event.shape
            value_th = 0
            event_sum = torch.sum(torch.abs(event), dim=0)
            # event_map = torch.where(event_sum>value_th)
            density = torch.sum(event_sum>value_th)/ (h*w)
            return density
        criterion = torch.nn.L1Loss()
        loss = 0.
        batch_size = d_event.shape[0]
        for i in range(batch_size):    
            x_density = compute_density(d_event_pre[i])
            t_density = compute_density(d_event[i])
            
            loss += criterion(x_density, t_density)

        return loss

    def dice_reg(self, x, target):
        # Average of Dice coefficient for all batches, or for a single mask
        regress = 0.
        batch_size = x.shape[0]
        for i in range(batch_size):
            x_i = x[i].reshape(-1)
            t_i = target[i].reshape(-1)
            regress += torch.sum((x_i - t_i) ** 2, dim=0).sqrt()
        
        return regress / batch_size

    def compute_mimounet_loss_with_density(self, batch):
        # criterion = torch.nn.L1Loss()

        d_event_pre = self.get_mimo_unet_out_batch(batch)

        d_event = self.get_dense_events_batch(batch)

        # print(d_event.shape)
        # print(d_event_pre[-1].shape)
        d_event2 = torch.nn.functional.interpolate(d_event, scale_factor=0.5, mode='bilinear')
        d_event4 = torch.nn.functional.interpolate(d_event, scale_factor=0.25, mode='bilinear')

        l1 = self.dice_reg(d_event_pre[0], d_event4)
        l2 = self.dice_reg(d_event_pre[1], d_event2)
        l3 = self.dice_reg(d_event_pre[2], d_event)
        reg_loss = l1+l2+l3

        density_loss = self.compute_density_loss(d_event_pre[2], d_event)

        reg_loss += density_loss * 50

        train_outputs, f_flow_mask = self.get_estimation_and_target(batch)
        f_est = train_outputs[0]
        f_gt = f_flow_mask[0]
        f_mask = f_flow_mask[1]
        events1, events2 = self.get_events_batch(batch)
        flow_loss, metrics = self.sequence_loss(train_outputs, events1, events2, f_gt, f_mask, self.gamma)

        den = 200.0
        loss = reg_loss  + flow_loss * den 

        return loss, metrics, reg_loss, flow_loss

    def train_mimounet_iters(self, model, start_epoch=0, val_iters=500, without_reg=False, compute_density=False):
        model.module.change_imagesize(self.image_size)
        model.cuda()
        model.train()

        model.module.freeze_bn()
  

        scaler = GradScaler(enabled=self.mixed_precision)
        iters = 0
        total_loss = 0.0
        total_epe = 0.0
        total_reg_loss = 0.0
        total_flow_loss = 0.0
        for batch_idx, batch in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            # Move Data to GPU
            batch = self.move_batch_to_cuda(batch)
            # Network Forward Pass
            self.run_network(model,batch)
 

            if(compute_density):
                loss, metrics, reg_loss, flow_loss = self.compute_mimounet_loss_with_density(batch)

            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)                
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), self.clip)
            scaler.step(self.optimizer)
            self.scheduler.step()
            scaler.update()
            total_loss += loss.item()
            total_reg_loss += reg_loss.item()
            total_flow_loss += flow_loss.item()
            total_epe += metrics['epe']
            iters += 1
            istr = 'iters{:02d}  {:05d}/{:05d} Training Loss:{:2.6f}(Reg loss{:2.3f}, Flow loss{:2.3f}) AEE: {:2.6f}'.\
                format(iters+start_epoch*val_iters, iters, val_iters, total_loss/iters, total_reg_loss/iters, total_flow_loss/iters, metrics['epe'])
            
            if iters % 10 == 0:
                self.logger.write_line(istr, True)
            if self.visualize_map:
                if iters % 100 ==0:
                        # idx = batch['idx'].cpu().data.numpy()[0]

                        train_outputs, f_flow_mask = self.get_estimation_and_target(batch)
                        f_est = train_outputs[0]
                        f_gt = f_flow_mask[0]

                        idx = iters + start_epoch*val_iters
                        flow_est = f_est[-1].cpu().data.numpy()[0].transpose(1,2,0)
                        self.visualize_optical_flow(flow_est, str(idx)+'_flow_est.jpg')
                        flow_gt = f_gt.cpu().data.numpy()[0].transpose(1,2,0)
                        self.visualize_optical_flow(flow_gt, str(idx)+'_flow_gt.jpg')

                        map0, map1 = self.get_mimounet_out(batch)
                        self.vis_map_RGB(map0, str(idx)+'_events_map0.jpg')
                        self.vis_map_RGB(map1, str(idx)+'_events_map1.jpg')

                        events0, events1 = self.get_events(batch)
                        d_events0, d_events1 = self.get_dense_events(batch)
                        self.vis_map_RGB(events0, str(idx)+'_events0.jpg')
                        self.vis_map_RGB(events1, str(idx)+'_events1.jpg')
                        self.vis_map_RGB(d_events0, str(idx)+'_events_dense0.jpg')
                        self.vis_map_RGB(d_events1, str(idx)+'_events_dense1.jpg')

            if(iters >= val_iters):
                break
        mloss = total_loss / iters
        mepe = total_epe / iters
        istr = '{:d}th {:d}iters:  Mean Loss:{:2.6f}  Mean AEE: {:2.6f}'.format(start_epoch+1, val_iters, mloss, mepe)
        self.logger.write_line(istr, True)

        if self.return_epe:
            return model, mepe
        return model
            





