import os,sys
current_path = os.path.dirname(os.path.abspath(__file__))
proc_path = current_path.rsplit("/",1)[0]
sys.path.append(current_path)
sys.path.append(proc_path)
from loader.HREM import HREMEventFlow
from test_mvsec import *
from train_mvsec import *

from utils.logger import *
import utils.helper_functions as helper
import json
from torch.utils.data.dataloader import DataLoader
from utils import visualization as visualization
import argparse

import git
import torch.nn

def get_visualizer(args):
        return visualization.FlowVisualizerEvents
        
def train(args):

    config_path = 'config/a_meshflow.json'

    config = json.load(open(config_path))
    
    # Load Model
    if(args.model_name == "eraft"):
        from model.eraft import ERAFT as RAFT
        model = RAFT(config=config)
    elif(args.model_name == "kpaflow"):
        from model.KPAflow.KPAFlow import KPAFlow
        model = KPAFlow(config=config)
    elif(args.model_name == "GMA"):
        from model.GMA.network import RAFTGMA
        model = RAFTGMA(
            config=config,
            n_first_channels=config['data_loader']['train']['args']['num_voxel_bins']
        )
    elif(args.model_name == "flowformer"):
        from model.flowformer.FlowFormer import build_flowformer
        from model.flowformer.config import get_cfg
        cfg = get_cfg()
        model = build_flowformer(cfg)
    elif(args.model_name == "skflow"):
        from model.SKflow.models.sk_decoder import SK_Decoder
        model = SK_Decoder(config=config)
    elif(args.model_name == "irrpwc"):
        from model.IRRPWC.pwcnet_irr import PWCNet
        model = PWCNet(config=config)
    elif(args.model_name == "EEMFlow"):
        from model.EEMFlow.EEMFlow import EEMFlow
        model = EEMFlow(config=config, n_first_channels=5)

    config = json.load(open(config_path))
    config["train"]["lr"] = args.lr
    config["train"]["wdecay"] = args.wd
    config["train"]["num_steps"] =  args.train_iters
    config['data_loader']['train']['args']['batch_size'] = args.batch_size

    # # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))

    config['name'] = "lr{:5f}_we{:5f}".format(args.lr, args.wd)
    save_path = os.path.join(proc_path, "exp_HREM_meshflow/{}_{}".format(args.model_name, args.input_type), config['name'].lower())
    os.makedirs(save_path, exist_ok=True)

    config["data_loader"]["train"]["args"].update({'type':'train', 'event_interval':args.input_type})

    print('Storing output in folder {}'.format(save_path))
    # Copy config file to save dir
    json.dump(config, open(os.path.join(save_path, 'config.json'), 'w'),
              indent=4, sort_keys=False)

    if args.start_epoch:
        states = torch.load(os.path.join(save_path, 'lasted_ckpt.pth.tar'))
        state_dict = {}
        for key, param in states['state_dict'].items():
            state_dict.update({key.replace('module.',''): param})
        model.load_state_dict(state_dict)

        start_epoch = states['epoch']
    else:
        start_epoch = 0

    # Instantiate Dataloader
    train_logger = Logger(save_path, custom_name='train.log')
    train_logger.initialize_file("train")
    train_set = HREMEventFlow(
        args = config["data_loader"]["train"]["args"],
        train = True
    )

    
    train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.num_workers,pin_memory=True)

    # Get Visualizer
    visualizer = get_visualizer(args)

    train = TrainRaftEvents(
        model=model,
        config=config,
        args=args,
        data_loader=train_set_loader,
        train_logger=train_logger,
        save_path=save_path,
        visualizer=visualizer,
        visualizer_map=True
    )


    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    train.fetch_optimizer(model)

    epochs = args.train_iters // args.val_iters

    for epoch in range(start_epoch, epochs):

        train.summary()
        model = train.train_iters(model, start_epoch=epoch, val_iters=args.val_iters)

        save_dict = {'epoch': epoch,
        'state_dict': model.module.state_dict()}

        torch.save(save_dict, os.path.join(save_path, 'lasted_ckpt.pth.tar'))
        
    return

            
if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--visualize', action='store_true', help='Provide this argument s.t. DSEC results are visualized. MVSEC experiments are always visualized.')
    parser.add_argument('-n', '--num_workers', default=0, type=int, help='How many sub-processes to use for data loading')
    parser.add_argument('--train_iters', default=6000000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-se','--start-epoch', action='store_true', help='restart')
    parser.add_argument('-be','--best_epe', default=1e5, type=float)
    parser.add_argument('--val_iters', default=10000, type=int, metavar='N',help='Evaluate every \'evaluate interval')
    parser.add_argument('--lr', default=1e-5, type=float, help='learnning rate')
    parser.add_argument('--wd', default=0, type=float, help='weight decay')

    parser.add_argument('--batch_size', '-bs', default=6, type=int, help='batch size in trainning')

    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--test_sequence', '-sq', default='indoor_flying2', type=str)
    parser.add_argument('--dense', action='store_true')
    parser.add_argument('--density', '-d', default='ct0.05', type=str)
    parser.add_argument('--model_name', '-model', default='EEMFlow', type=str)
    parser.add_argument('--input_type', '-int', default='dt1', type=str)
    args = parser.parse_args()

    # Run Test Script
    train(args)
