import os,sys
current_path = os.path.dirname(os.path.abspath(__file__))
proc_path = current_path.rsplit("/",1)[0]
sys.path.append(current_path)
sys.path.append(proc_path)

import h5py
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import os
import numpy
from utils import filename_templates as TEMPLATES
from loader.loader_utils import *
from utils.transformers import *
from utils.augumentor import DenseSparseAugmentor, FlowAugmentor
from utils_luo.tools import file_tools, tensor_tools
from utils_luo.event_utils import vis_map_RGB
import cv2
import glob
from matplotlib import colors

from matplotlib.colors import hsv_to_rgb

import imageio

import pdb

def check_out_bounds(point_i,point_j,height,width):
    if(point_i>=height):
        point_i=height-1
    elif(point_i<0):
        point_i=0
    if(point_j>=width):
        point_j=width-1
    elif(point_j<0):
        point_j=0
    return point_i,point_j

def motion_propagate(fflow, height, width, mesh_size=16, radius=3):
    from scipy.signal import medfilt2d

    if(fflow.shape[0]==2):
        fflow.transpose(1,2,0)
    u = fflow[...,0]
    v = fflow[...,1]

    # spreads motion over the mesh for the old_frame    
    mesh_cols, mesh_rows = width//mesh_size, height//mesh_size
    x_motion = {}
    y_motion = {}
    for i in range(mesh_size):
        for j in range(mesh_size):
            x_motion.update({(i,j):[]})
            y_motion.update({(i,j):[]})

            for r in range(radius):
                offect_x = r*mesh_rows//2
                offect_y = r*mesh_cols//2
                point_i = mesh_rows*i+offect_x
                point_j = mesh_cols*j+offect_y
                point_i,point_j=check_out_bounds(point_i,point_j,height,width)
                x_motion[i, j].append(u[point_i,point_j])
                y_motion[i, j].append(v[point_i,point_j])
                point_i = mesh_rows*i+offect_x
                point_j = mesh_cols*j-offect_y
                point_i,point_j=check_out_bounds(point_i,point_j,height,width)
                x_motion[i, j].append(u[point_i,point_j])
                y_motion[i, j].append(v[point_i,point_j])
                point_i = mesh_rows*i-offect_x
                point_j = mesh_cols*j+offect_y
                point_i,point_j=check_out_bounds(point_i,point_j,height,width)
                x_motion[i, j].append(u[point_i,point_j])
                y_motion[i, j].append(v[point_i,point_j])
                point_i = mesh_rows*i-offect_x
                point_j = mesh_cols*j-offect_y
                point_i,point_j=check_out_bounds(point_i,point_j,height,width)
                x_motion[i, j].append(u[point_i,point_j])
                y_motion[i, j].append(v[point_i,point_j])

    # apply median filter (f-1) on obtained motion for each vertex
    x_motion_mesh = np.zeros((mesh_size, mesh_size), dtype=float)
    y_motion_mesh = np.zeros((mesh_size, mesh_size), dtype=float)
    for key in x_motion.keys():
        if(len(x_motion[key])>0):
            x_motion[key].sort()
            x_motion_mesh[key] = x_motion[key][len(x_motion[key])//2]
        if(len(y_motion[key])>0):
            y_motion[key].sort()
            y_motion_mesh[key] = y_motion[key][len(y_motion[key])//2]

    # apply second median filter (f-2) over the motion mesh for outliers
    filter_size = 5
    pad_size = (filter_size - 1) // 2
    x_motion_mesh_ = cv2.copyMakeBorder(x_motion_mesh,pad_size,pad_size,pad_size,pad_size,cv2.BORDER_REPLICATE)
    y_motion_mesh_ = cv2.copyMakeBorder(y_motion_mesh,pad_size,pad_size,pad_size,pad_size,cv2.BORDER_REPLICATE)
    x_motion_mesh_ = medfilt2d(x_motion_mesh_, [filter_size, filter_size])
    y_motion_mesh_ = medfilt2d(y_motion_mesh_, [filter_size, filter_size])

    return x_motion_mesh_[pad_size:(pad_size+mesh_size),pad_size:(pad_size+mesh_size)], y_motion_mesh_[pad_size:(pad_size+mesh_size),pad_size:(pad_size+mesh_size)]


def visualize_optical_flow(flow, save_path, name=None):
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
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(os.path.join(save_path, name+".jpg"), out)
    return 

class HREMEventFlow(Dataset):
    def __init__(self, args, train = True):
        super(HREMEventFlow, self).__init__()
        self.input_type = 'events'
        self.type = 'train' if train else 'val'
        self.evaluation_type = args['eval_type']
        self.dt = args['event_interval']
         
        self.image_width = 1280
        self.image_height = 720

        self.num_bins = args['num_voxel_bins']
        self.voxel = EventSequenceToVoxelGrid_Pytorch(
            num_bins=self.num_bins, 
            normalize=True, 
            gpu=False
        )
        
        if 'aug_params' in args.keys():
            self.aug_params = args['aug_params']
            self.augmentor = FlowAugmentor(**self.aug_params)
        else:
            self.augmentor = None
        
        self.get_data_ls()   
    
    def get_data_ls(self):
        if(self.type == 'train'):

            self.dataset_dir = os.path.join(proc_path, 'dataset/HREM/train/{:s}'.format(self.dt))

            self.data_ls = []
            for names in sorted(os.listdir(self.dataset_dir)):

                event1_file_path = os.path.join(self.dataset_dir, names, "events1.npz")
                event2_file_path = os.path.join(self.dataset_dir, names, "events2.npz")
                flow_file_path = os.path.join(self.dataset_dir, names, "flow.flo")

                if(os.path.exists(event1_file_path) and os.path.exists(event2_file_path)):

                    self.data_ls.append({"names":names, "event0":event1_file_path, "event1":event2_file_path, "fflow":flow_file_path})


        else:
            self.dataset_dir = os.path.join(proc_path, 'dataset/HREM/test/{:s}'.format(self.dt))
            sequence_list = sorted(os.listdir(self.dataset_dir))

            self.nori_list = {}
            for sequence in sequence_list:
                self.dataset_sequence_dir = os.path.join(self.dataset_dir, sequence)
                data_ls = []
                for names in sorted(os.listdir(self.dataset_sequence_dir)):

                    event1_file_path = os.path.join(self.dataset_sequence_dir, names, "events1.npz")
                    event2_file_path = os.path.join(self.dataset_sequence_dir, names, "events2.npz")
                    flow_file_path = os.path.join(self.dataset_sequence_dir, names, "flow.flo")

                    if(os.path.exists(event1_file_path) and os.path.exists(event2_file_path)):

                        data_ls.append({"names":names, "event0":event1_file_path, "event1":event2_file_path, "fflow":flow_file_path})
                
                self.nori_list.update({sequence:data_ls})

    
    def change_test_sequence(self, sequence):
        self.data_ls = self.nori_list[sequence]

    def __len__(self):

        return len(self.data_ls)
    
    def get_sample(self, idx):
        
        sample = self.data_ls[idx]
        names = sample["names"]

        # Load Flow
        fflow = read_flo(sample['fflow'])
        fflow2 = fflow.copy()

        if(fflow.ndim == 4):
            fflow = fflow[0]
        if(fflow.shape[-1]==2):
            fflow = fflow.transpose(2,0,1)
        
        height, width = fflow2.shape[0], fflow2.shape[1]
        x_mesh, y_mesh = motion_propagate(fflow2, height, width)
        meshflow = np.stack([x_mesh, y_mesh], axis=-1)
        if(meshflow.shape[-1]==2):
            meshflow = meshflow.transpose(2,0,1)

        return_dict = {'names': names,
                    'flow': torch.from_numpy(meshflow),
                    "fflow": torch.from_numpy(fflow),
                    "valid": None,
                    }

        # Load Events 
        params = {'height': self.image_height, 'width': self.image_width}
        events_old = get_compressed_events(sample['event0'])
        events_new = get_compressed_events(sample['event1'])
        ev_seq_old = EventSequence(None, params, features=events_old, timestamp_multiplier=1e6, convert_to_relative=True)
        ev_seq_new = EventSequence(None, params, features=events_new, timestamp_multiplier=1e6, convert_to_relative=True)
        event_volume_old = self.voxel(ev_seq_old).cpu()
        event_volume_new = self.voxel(ev_seq_new).cpu()

        return_dict['event_volume_new'] = event_volume_new
        return_dict['event_volume_old'] = event_volume_old
        

        event_valid = np.sum(event_volume_old.data.numpy(), axis=0)
        return_dict['event_valid'] = torch.from_numpy(event_valid).unsqueeze(dim=0)

        return return_dict
    
    def __getitem__(self, idx):
        
        if self.type == 'train':
            sample = self.get_sample(idx % len(self))

            img1 = sample['event_volume_old'].permute(1,2,0).numpy()
            img2 = sample['event_volume_new'].permute(1,2,0).numpy()

            meshflow = sample['flow'].permute(1,2,0).numpy()
            img1, img2, flow = self.augmentor(img1, img2, meshflow, without_resize=True)

            sample['flow'] = torch.from_numpy(meshflow).permute(2, 0, 1).float()
            sample['valid'] = torch.ones(size=meshflow.shape[0:2])


            sample['event_volume_old'] = torch.from_numpy(img1).permute(2, 0, 1).float()
            sample['event_volume_new'] = torch.from_numpy(img2).permute(2, 0, 1).float()

        elif self.type == 'val':
            sample = self.get_sample(idx % len(self))

            flow = sample['flow']
            print("Meshflow size:{},{}".format(flow.shape[-2],flow.shape[-1]))
            flow_t = flow.unsqueeze(dim=0)
            flow_t = F.interpolate(flow_t, size=(self.image_height, self.image_width), mode='bilinear', align_corners=False)
            sample['flow'] = flow_t.squeeze()

            flow = flow_t.squeeze().permute(1,2,0).numpy()
            valid = np.logical_and(np.logical_and(~np.isinf(flow[:, :, 0]), ~np.isinf(flow[:, :, 1])), np.linalg.norm(flow, axis=2) > 0)
            sample['valid'] = torch.from_numpy(valid).float()
           
        return sample

if __name__ == '__main__':
    config_path = '/home/luoxinglong/meshflow/config/a_meshflow.json'
    config = json.load(open(config_path))
    config["data_loader"]["test"]["args"].update({"event_interval":'dt4'})

    save_root = '/home/luoxinglong/meshflow/vis_loader/align_image'

    test_set = HREMEventFlow(
        args = config["data_loader"]["test"]["args"],
        train=False
    )

    sequence_list = ["indoor_fast", "indoor_slow", "outdoor_fast", "outdoor_slow"]

    for sequence in sequence_list:
        print(sequence)
        test_set.change_test_sequence(sequence)
        save_path = os.path.join(save_root, sequence)

        os.makedirs(save_path, exist_ok=True)

        for i in range(len(test_set)):

            data = test_set[i]
            
            names = data['names']
            print(data['event_volume_old'].shape)
            event1 = data['event_volume_old'].permute(1,2,0).numpy()
            event2 = data['event_volume_new'].permute(1,2,0).numpy()
            flow = data['flow'].permute(1,2,0).numpy()

            vis_map_RGB(event1, save_path, '{:s}_event1'.format(names))
            vis_map_RGB(event2, save_path, '{:s}_event2'.format(names))
            visualize_optical_flow(flow, save_path, "{:s}_flow".format(names))

            print(event1.shape)
            print(event2.shape)
            print(flow.shape)