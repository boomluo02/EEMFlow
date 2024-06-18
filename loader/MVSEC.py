import os,sys
current_path = os.path.dirname(os.path.abspath(__file__))
proc_path = current_path.rsplit("/",1)[0]
sys.path.append(current_path)
sys.path.append(proc_path)
import numpy as np
import torch
import cv2
import json
import h5py
import pandas

from matplotlib import pyplot as plt
from matplotlib import colors
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from loader_utils import get_events
from loader_utils import FlowAugmentor, DenseSparseAugmentor, EventSequence, EventSequenceToVoxelGrid_Pytorch

import pdb

Valid_Time_Index = {
    'indoor_flying1': [(314, 2199)],
    'indoor_flying2': [(314, 2199)],
    'indoor_flying3': [(314, 2199)],
    'indoor_flying4': [(196, 570)],
    'outdoor_day1': [(245, 3000)],
    'outdoor_day2': [(4375, 7002)]
}

class MvsecEventFlow(Dataset):
    def __init__(self, args, train = True):
        super(MvsecEventFlow, self)

        self.input_type = 'events'
        self.type = 'train' if train else 'val'
        self.evaluation_type = args['eval_type']

        self.change_test_sequence(args['sequence'])

        self.image_width = 346
        self.image_height = 260
        self.num_bins = args['num_voxel_bins']
        self.voxel = EventSequenceToVoxelGrid_Pytorch(
            num_bins=self.num_bins, 
            normalize=True, 
            gpu=True
        )
        self.cropper = transforms.CenterCrop((256,256))
        
        if 'aug_params' in args.keys():
            self.aug_params = args['aug_params']
            self.augmentor = FlowAugmentor(**self.aug_params)
            self.dense_augmentor = DenseSparseAugmentor(**self.aug_params)
        else:
            self.augmentor = None

    def change_test_sequence(self, sequence):

        self.names = []
        for s in Valid_Time_Index[sequence]:
            for ind in range(s[0], s[1]):
                self.names.append(ind) 

        self.sequence = sequence
        if('outdoor_day1' in sequence):
            self.sequence = 'outdoor_day1'

        if(self.sequence == 'indoor_flying1'):
            self.flowgt_path = os.path.join(proc_path, 'dataset/MVSEC_test/{:s}/flowgt_dt1'.format(self.sequence))
            self.event_path = os.path.join(proc_path, 'dataset/MVSEC_test/{:s}/event'.format(self.sequence))
            self.sequence = 'indoor_flying1_new'
        else:
            self.flowgt_path = os.path.join(proc_path, 'dataset/MVSEC/{:s}/flowgt_dt1'.format(self.sequence))
            self.event_path = os.path.join(proc_path, 'dataset/MVSEC/{:s}/event'.format(self.sequence))
        
        event_list = os.listdir(self.event_path)
        event_list.sort(key= lambda x : int(x[:-3]))
        self.flow_list = [os.path.join(self.flowgt_path, '{:d}.npy'.format(i)) for i in self.names]

        self.event_list = []
        self.flow_list = []

        for i in self.names:
            self.flow_list.append(os.path.join(self.flowgt_path, '{:d}.npy'.format(i)))
            self.event_list.append(os.path.join(self.event_path, '{:06d}.h5'.format(i+1)))
        flow_idx_end = i
        self.event_list.append(os.path.join(self.event_path, '{:06d}.h5'.format(flow_idx_end+2)))


    def summary(self, logger):
        logger.write_line("================================== Dataloader Summary ====================================", True)
        logger.write_line("Loader Type:\t\t" + self.__class__.__name__ + " for {}".format(self.type), True)

    def get_sample(self, idx):

        names = self.names[idx]

        # Load Flow
        flow = np.load(self.flow_list[idx])

        if(flow.shape[-1]==2):
            flow = flow.transpose(2,0,1)

        return_dict = {'idx': names,
                    'flow': torch.from_numpy(flow),
                    "valid": None
                    }

        # Load Events 
        params = {'height': self.image_height, 'width': self.image_width}
        
        event_path_old = self.event_list[idx]
        event_path_new = self.event_list[(idx+1) % len(self.event_list)]
        
        events_old = get_events(event_path_old)
        events_new = get_events(event_path_new)
        
        ev_seq_old = EventSequence(events_old, params, timestamp_multiplier=1e6, convert_to_relative=True)
        ev_seq_new = EventSequence(events_new, params, timestamp_multiplier=1e6, convert_to_relative=True)
        event_volume_new = self.voxel(ev_seq_new).cpu()
        event_volume_old = self.voxel(ev_seq_old).cpu()

        return_dict['event_volume_new'] = event_volume_new
        return_dict['event_volume_old'] = event_volume_old 

        return_dict['d_event_volume_new'] = event_volume_new
        return_dict['d_event_volume_old'] = event_volume_old 

        if(self.type == 'val'):

            seq = ev_seq_old.get_sequence_only()
            h = self.image_height
            w = self.image_width
            hist, _, _ = np.histogram2d(x=seq[:,1], y=seq[:,2],
                                    bins=(w,h),
                                    range=[[0,w], [0,h]])
            hist = hist.transpose()
            ev_mask = hist > 0
            return_dict['event_valid'] = torch.from_numpy(ev_mask).unsqueeze(dim=0)

        return return_dict


    def __len__(self):
        
        return len(self.names)

    def __getitem__(self, idx):
        
        sample = self.get_sample(idx % len(self))

        if self.type == 'train':
            
            event1 = sample['event_volume_old'].permute(1,2,0).numpy()
            event2 = sample['event_volume_new'].permute(1,2,0).numpy()
            flow = sample['flow'].permute(1,2,0).numpy()

            if("d_event_volume_old" in sample.keys()):
            
                d_event1 = sample['d_event_volume_old'].permute(1,2,0).numpy()
                d_event2 = sample['d_event_volume_new'].permute(1,2,0).numpy()
                
                event1, event2, d_event1, d_event2, flow_crop = self.dense_augmentor(event1, event2, d_event1, d_event2, flow)

                valid = np.logical_and(np.logical_and(~np.isinf(flow_crop[:, :, 0]), ~np.isinf(flow_crop[:, :, 1])), np.linalg.norm(flow_crop, axis=2) > 0)

                sample['event_volume_old'] = torch.from_numpy(event1).permute(2, 0, 1).float()
                sample['event_volume_new'] = torch.from_numpy(event2).permute(2, 0, 1).float()
                sample['d_event_volume_old'] = torch.from_numpy(d_event1).permute(2, 0, 1).float()
                sample['d_event_volume_new'] = torch.from_numpy(d_event2).permute(2, 0, 1).float()
                sample['flow'] = torch.from_numpy(flow_crop).permute(2, 0, 1).float()
                sample['valid'] = torch.from_numpy(valid).float()

            else:
                
                event1, event2, flow = self.augmentor(event1, event2, flow)
                valid = np.logical_and(np.logical_and(~np.isinf(flow[:, :, 0]), ~np.isinf(flow[:, :, 1])), np.linalg.norm(flow, axis=2) > 0)

                sample['event_volume_old'] = torch.from_numpy(event1).permute(2, 0, 1).float()
                sample['event_volume_new'] = torch.from_numpy(event2).permute(2, 0, 1).float()
                sample['flow'] = torch.from_numpy(flow).permute(2, 0, 1).float()
                sample['valid'] = torch.from_numpy(valid).float()

        elif self.type == 'val':

            sample['flow'] = self.cropper(sample['flow'])
            sample['valid'] = (sample['flow'][0].abs() < 1000) & (sample['flow'][1].abs() < 1000)
            sample['event_volume_old'] = self.cropper(sample['event_volume_old'])
            sample['event_volume_new'] = self.cropper(sample['event_volume_new'])
            sample['event_valid'] = self.cropper(sample['event_valid'])

        return sample

class MvsecEventFlow_dt4(MvsecEventFlow):

    def change_test_sequence(self, sequence):

        self.names = []
        for s in Valid_Time_Index[sequence]:
            for ind in range(s[0], s[1]):
                self.names.append(ind) 

        self.sequence = sequence
        if('outdoor_day1' in sequence):
            self.sequence = 'outdoor_day1'

        self.flowgt_path = os.path.join(proc_path, 'dataset/MVSEC/{:s}/flowgt_dt4'.format(self.sequence))
        self.event_path = os.path.join(proc_path, 'dataset/MVSEC/{:s}/event'.format(self.sequence))
        
        event_list = os.listdir(self.event_path)
        event_list.sort(key= lambda x : int(x[:-3]))

        self.event_list = []
        self.flow_list = []

        for i in self.names:
            self.flow_list.append(os.path.join(self.flowgt_path, '{:d}.npy'.format(i)))
            self.event_list.append(os.path.join(self.event_path, '{:06d}.h5'.format(i+1)))
        flow_idx_end = i

        for j in range(5):
            self.event_list.append(os.path.join(self.event_path, '{:06d}.h5'.format(flow_idx_end+2+j)))
        
    
    def get_sample(self, idx):
        names = self.names[idx]

        # Load Flow
        flow = np.load(self.flow_list[idx])
        if(flow.shape[2]==2):
            flow = flow.transpose(2,0,1)


        return_dict = {'idx': names,
                    'flow': torch.from_numpy(flow),
                    "valid": None,
                    }

        # Load Events 
        params = {'height': self.image_height, 'width': self.image_width}
        
        events_old_list = []
        events_new_list = []
        for i in range(4):
            event_path_old = self.event_list[idx+i]
            events_old = get_events(event_path_old)
            events_old_list.append(events_old)

            event_path_new = self.event_list[(idx+i+1) % len(self.event_list)]
            events_new = get_events(event_path_new)
            events_new_list.append(events_new)
        
        events0 = pandas.concat(events_old_list)
        events0.sort_values(by = ['ts'])
        events1 = pandas.concat(events_new_list) 
        events1.sort_values(by = ['ts'])

        ev_seq_old = EventSequence(events0, params, timestamp_multiplier=1e6, convert_to_relative=True)
        ev_seq_new = EventSequence(events1, params, timestamp_multiplier=1e6, convert_to_relative=True)
        
        event_volume_new = self.voxel(ev_seq_new).cpu()
        event_volume_old = self.voxel(ev_seq_old).cpu()

        return_dict['event_volume_new'] = event_volume_new
        return_dict['event_volume_old'] = event_volume_old

        return_dict['d_event_volume_new'] = event_volume_new
        return_dict['d_event_volume_old'] = event_volume_old 

        if(self.type == 'val'):

            seq = ev_seq_old.get_sequence_only()
            h = self.image_height
            w = self.image_width
            hist, _, _ = np.histogram2d(x=seq[:,1], y=seq[:,2],
                                    bins=(w,h),
                                    range=[[0,w], [0,h]])
            hist = hist.transpose()
            ev_mask = hist > 0
            return_dict['event_valid'] = torch.from_numpy(ev_mask).unsqueeze(dim=0)

        return return_dict


if __name__ == '__main__':
    config_path = os.path.join(proc_path, 'config/mvsec.json')
    config = json.load(open(config_path))

    config["data_loader"]["test"]["args"].update({"sequence": "outdoor_day1"})

    test_set = MvsecEventFlow(
        args = config["data_loader"]["test"]["args"],
        train=False
    )

    test_set_loader = DataLoader(test_set,
                            batch_size=2,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=True)

    for batch_idx, data in enumerate(test_set_loader):

        idx = data['idx']
        event_volume_old = data['event_volume_old']
        flow = data['event_volume_new']
        print(event_volume_old.max())
        print(event_volume_old.max())