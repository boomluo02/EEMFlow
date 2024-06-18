# Bottow from https://github.com/ruizhao26/STE-FlowNet
import os,sys
current_path = os.path.dirname(os.path.abspath(__file__))
proc_path = current_path.rsplit("/",1)[0]
sys.path.append(current_path)
sys.path.append(proc_path)

import os.path as osp
import h5py
import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from loader_utils import estimate_corresponding_gt_flow

import pdb

class Events(object):
    def __init__(self, num_events, width=346, height=260):
        self.data = np.rec.array(None, dtype=[('x', np.uint16), ('y', np.uint16), ('p', np.bool_), ('ts', np.float64)],
                                 shape=(num_events))
        self.width = width
        self.height = height

    def h5_to_torch(self, e):
        return torch.stack((
        torch.from_numpy(e[:,0]),
        torch.from_numpy(e[:,1]),
        torch.from_numpy(e[:,2]),
        torch.from_numpy(e[:,3]), 
        ), dim=-1)

    def split_events(self, events):
        return events[:, 0].long(), events[:, 1].long(), events[:, 2].float(), events[:, 3].float()


    def generate_fimage(self, input_event=0,  image_raw_event_inds_temp=0, image_raw_ts_temp=0, dt_time_temp=0):

        split_interval = image_raw_ts_temp.shape[0]
        
        t_index = 0
        encoding_length = split_interval - (dt_time_temp - 1)

        for i in range(split_interval - (dt_time_temp - 1)):
            
            if(osp.exists(osp.join(event_dir, "{:06d}.h5".format(i)))):
                print('{:s} event {:05d} already exists'.format(args.save_env, i))
                continue

            if image_raw_event_inds_temp[i - 1] < 0:
                frame_data = input_event[0:image_raw_event_inds_temp[i + (dt_time_temp - 1)], :]
            else:
                frame_data = input_event[image_raw_event_inds_temp[i - 1]:image_raw_event_inds_temp[i + (dt_time_temp - 1)], :]
            
            st = time.time()
            if frame_data.size > 0:
                x = frame_data[:,0]
                y = frame_data[:,1]
                ts = frame_data[:,2]
                p = frame_data[:,3]

                new_frame = np.stack((ts,x,y,p), axis=1)
                df_data = pd.DataFrame(new_frame, columns=['ts', 'x', 'y', 'p'])
                df_data.to_hdf(os.path.join(event_dir, "{:06d}.h5".format(i)),"myDataset")

            t_index = t_index + 1

            if args.sparse_print:
                if i % 1000 == 0:
                    print('Dataset {:s} sp={:02d} Finish Encoding {:05d} / {:05d} Time: {:.2f} !'.format(args.save_env, args.data_split, i, encoding_length - 1, time.time()-st))
            else:
                print('Dataset {:s} sp={:02d} Finish Encoding {:05d} / {:05d} Time: {:.2f} !'.format(args.save_env, args.data_split, i, encoding_length - 1, time.time()-st))


class Test_loading_dt1(Dataset):
    # Initialize your data, download, etc.
    def __init__(self):
        self.dt = 1
        self.xoff = 45
        self.yoff = 2
        self.split = 5

        self.image_raw_ts = image_raw_ts
        self.length = gray_image.shape[0]

    def __getitem__(self, index):
        if (index + 8 < self.length) and (index > 20):
            aa = np.ones((256, 256, self.split), dtype=np.uint8)
            bb = np.ones((256, 256, self.split), dtype=np.uint8)
            
            return aa, bb, self.image_raw_ts[index], self.image_raw_ts[index + self.dt]
        else:
            pp = np.zeros((image_resize, image_resize, self.split))
            return pp, pp, np.zeros((self.image_raw_ts[index].shape)), np.zeros(
                (self.image_raw_ts[index].shape))

    def __len__(self):
        return self.length


class Test_loading_dt4(Dataset):
    # Initialize your data, download, etc.
    def __init__(self):
        self.dt = 4
        self.xoff = 45
        self.yoff = 2
        self.split = args.data_split

        self.image_raw_ts = image_raw_ts
        self.length = gray_image.shape[0]

    def __getitem__(self, index):
        if (index + 8 < self.length) and (index > 20):
            aa = np.ones((256, 256, int(self.dt*self.split)), dtype=np.uint8)
            bb = np.ones((256, 256, int(self.dt*self.split)), dtype=np.uint8)

            return aa, bb, self.image_raw_ts[index], self.image_raw_ts[index+self.dt]
        else:
            pp = np.zeros((image_resize,image_resize,int(self.split*self.dt/2)))
            return pp, pp, np.zeros((self.image_raw_ts[index].shape)), np.zeros((self.image_raw_ts[index].shape))

    def __len__(self):
        return self.length


def generate_flowgt(test_loader, flowgt_path):
    global args, image_resize, sp_threshold
    d_label = h5py.File(gt_file, 'r')
    gt_temp = np.float32(d_label['davis']['left']['flow_dist'])
    gt_ts_temp = np.float64(d_label['davis']['left']['flow_dist_ts'])
    
    # pdb.set_trace()
    for i, data in enumerate(test_loader, 0):
        
        if osp.exists(osp.join(flowgt_path, str(i)+'.npy')):
            print('flowgt_dt{:d} of {:s} frame {:05d} already exists'.format(args.dt, args.save_env, i))
            continue
               
        inputs_on, inputs_off, st_time, ed_time = data

        if torch.sum(inputs_on + inputs_off) > 0:
            time_start = time.time()

            U_gt_all = np.array(gt_temp[:, 0, :, :])
            V_gt_all = np.array(gt_temp[:, 1, :, :])

            U_gt, V_gt = estimate_corresponding_gt_flow(U_gt_all, V_gt_all, gt_ts_temp, np.array(st_time), np.array(ed_time))
            gt_flow = np.stack((U_gt, V_gt), axis=2)

            curr_path = osp.join(flowgt_path, str(i))
            np.save(curr_path, gt_flow)
            print('Finish saving flowgt_dt{:d} of {:s} frame {:05d} time={:.2f} '.format(args.dt, args.save_env, i, time.time() - time_start))

    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mvsec dataset Encoding')
    parser.add_argument('--data-split', '-sp', type=int, default=1)
    parser.add_argument('--save-dir', '-sd', type=str, default='/home/luoxinglong/ADMFlow/dataset/MVSEC', metavar='PARAMS',
                        help='Main Directory to save all encoding results')
    parser.add_argument('--out-dir', '-od', type=str, default='dataset/MVSEC_test', metavar='PARAMS')
    parser.add_argument('--save-env', '-se', type=str, default='indoor_flying1', metavar='PARAMS',
                        help='Sub-Directory name to save Environment specific encoding results')
    parser.add_argument('--sparse_print', '-s', action='store_true', help='saprse print log')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--dt', '-dt', type=int, default=1, help='time interval')
    parser.add_argument('--image_resize', type=int, default=256)
    parser.add_argument('--only_event', action='store_true')
    args = parser.parse_args()
    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

    #### Data Path
    save_path = osp.join(proc_path, args.out_dir, args.save_env)

    event_dir = osp.join(save_path, 'event')
    if not os.path.exists(event_dir):
        print('making', event_dir)
        os.makedirs(event_dir)

    #### Read Files
    image_resize = args.image_resize

    args.data_path = args.save_dir + '/' + args.save_env + '/' + args.save_env + '_data.hdf5'

    d_set = h5py.File(args.data_path, 'r')
    raw_data = d_set['davis']['left']['events']
    image_raw_event_inds = d_set['davis']['left']['image_raw_event_inds']
    image_raw_ts = np.float64(d_set['davis']['left']['image_raw_ts'])
    gray_image = d_set['davis']['left']['image_raw']
    d_set = None

    # Events
    dt_time = 1
    td = Events(raw_data.shape[0])
    td.generate_fimage(input_event=raw_data, image_raw_event_inds_temp=image_raw_event_inds,
                    image_raw_ts_temp=image_raw_ts, dt_time_temp=dt_time)
    
    print('\nEvents has been saved!\n')

    if(not args.only_event):
        
        # Flow
        image_resize = 256
        gt_file = args.save_dir + '/' + args.save_env + '/' + args.save_env + '_gt.hdf5'

        flowgt_path = osp.join(save_path, 'flowgt_dt{:d}'.format(args.dt))
        os.makedirs(flowgt_path, exist_ok = True)

        if args.dt == 1:
            Test_dataset = Test_loading_dt1()
        elif args.dt == 4:
            Test_dataset = Test_loading_dt4()

        test_loader = DataLoader(dataset=Test_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.workers)

        generate_flowgt(test_loader, flowgt_path)

        raw_data = None

        print('\nFlow_gt encoding complete!\n')