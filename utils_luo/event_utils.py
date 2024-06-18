import torch
import numpy
import pandas
import torch.nn.functional as F
from matplotlib import colors
import os
import cv2

def warp_events_flow_torch(xt, yt, tt, pt, flow_field, t0=None,
        batched=False, batch_indices=None):
    """
    Given events and a flow field, warp the events by the flow
    Parameters
    ----------
    xs : list of event x coordinates 
    ys : list of event y coordinates 
    ts : list of event timestamps 
    ps : list of event polarities 
    flow_field : 2D tensor containing the flow at each x,y position
    t0 : the reference time to warp events to. If empty, will use the
        timestamp of the last event
    Returns
    -------
    warped_xt: x coords of warped events
    warped_yt: y coords of warped events
    """
    if len(xt.shape) > 1:
        xt, yt, tt, pt = xt.squeeze(), yt.squeeze(), tt.squeeze(), pt.squeeze()
    if t0 is None:
        t0 = tt[-1]
    while len(flow_field.size()) < 4:
        flow_field = flow_field.unsqueeze(0)
    if len(xt.size()) == 1:
        event_indices = torch.transpose(torch.stack((xt, yt), dim=0), 0, 1)
    else:
        event_indices = torch.transpose(torch.cat((xt, yt), dim=1), 0, 1)
    #event_indices.requires_grad_ = False
    event_indices = torch.reshape(event_indices, [1, 1, len(xt), 2])

    # Event indices need to be between -1 and 1 for F.gridsample
    event_indices[:,:,:,0] = event_indices[:,:,:,0]/(flow_field.shape[-1]-1)*2.0-1.0
    event_indices[:,:,:,1] = event_indices[:,:,:,1]/(flow_field.shape[-2]-1)*2.0-1.0

    flow_at_event = F.grid_sample(flow_field, event_indices, align_corners=True) 
    
    dt = (tt-t0).squeeze()

    warped_xt = xt+flow_at_event[:,0,:,:].squeeze()*dt
    warped_yt = yt+flow_at_event[:,1,:,:].squeeze()*dt

    return warped_xt, warped_yt

""" Method 1: visualize event as image """
def events_to_event_image(event_sequence, resolution):
    height, width = resolution[0], resolution[1]

    polarity = event_sequence[:, 3] == -1.0
    x_negative = event_sequence[~polarity, 0].astype(numpy.uint16)
    y_negative = event_sequence[~polarity, 1].astype(numpy.uint16)
    x_positive = event_sequence[polarity, 0].astype(numpy.uint16)
    y_positive = event_sequence[polarity, 1].astype(numpy.uint16)

    positive_histogram, _, _ = numpy.histogram2d(
        x_positive,
        y_positive,
        bins=(width, height),
        range=[[0, width], [0, height]])
    negative_histogram, _, _ = numpy.histogram2d(
        x_negative,
        y_negative,
        bins=(width, height),
        range=[[0, width], [0, height]])

    # Red -> Negative Events
    red = numpy.transpose((negative_histogram >= positive_histogram) & (negative_histogram != 0))
    # Blue -> Positive Events
    blue = numpy.transpose(positive_histogram > negative_histogram)

    height, width = red.shape
    background = torch.full((3, height, width), 255).byte()

    points_on_background = plot_points_on_background(
        torch.nonzero(torch.from_numpy(red.astype(numpy.uint8))), background,
        [255, 0, 0])
    points_on_background = plot_points_on_background(
        torch.nonzero(torch.from_numpy(blue.astype(numpy.uint8))),
        points_on_background, [0, 0, 255])
    return points_on_background.permute(1,2,0).numpy()

def plot_points_on_background(points_coordinates,
                              background,
                              points_color=[0, 0, 255]):
    """
    Args:
        points_coordinates: array of (y, x) points coordinates
                            of size (number_of_points x 2).
        background: (3 x height x width)
                    gray or color image uint8.
        color: color of points [red, green, blue] uint8.
    """
    if not (len(background.size()) == 3 and background.size(0) == 3):
        raise ValueError('background should be (color x height x width).')
    _, height, width = background.size()
    background_with_points = background.clone()
    y, x = points_coordinates.transpose(0, 1)
    if len(x) > 0 and len(y) > 0: # There can be empty arrays!
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        if not (x_min >= 0 and y_min >= 0 and x_max < width and y_max < height):
            raise ValueError('points coordinates are outsize of "background" '
                             'boundaries.')
        background_with_points[:, y, x] = torch.Tensor(points_color).type_as(
            background).unsqueeze(-1)
    return background_with_points 

""" Method 1: visualize optical flow """
def visualize_optical_flow(flow,save_flow_path,name):
    # flow -> np array 2 x height x width
    # 2,h,w -> h,w,2
    if(flow.shape[0]==2):
        flow = flow.transpose(1,2,0)
    flow[numpy.isinf(flow)]=0
    assert flow.shape[-1]==2
    # Use Hue, Saturation, Value colour model
    hsv = numpy.zeros((flow.shape[0], flow.shape[1], 3), dtype=float)

    # The additional **0.5 is a scaling factor
    mag = numpy.sqrt(flow[...,0]**2+flow[...,1]**2)**0.5

    ang = numpy.arctan2(flow[...,1], flow[...,0])
    ang[ang<0]+=numpy.pi*2
    hsv[..., 0] = ang/numpy.pi/2.0 # Scale from 0..1
    hsv[..., 1] = 1
    hsv[..., 2] = (mag-mag.min())/(mag-mag.min()).max() # Scale from 0..1
    rgb = colors.hsv_to_rgb(hsv)
    # This all seems like an overkill, but it's just to exactly match the cv2 implementation
    bgr = numpy.stack([rgb[...,2],rgb[...,1],rgb[...,0]], axis=2)
    out = bgr*255
    if not os.path.exists(save_flow_path):
        os.makedirs(save_flow_path)
    cv2.imwrite(os.path.join(save_flow_path, name), out)
    return


class EventSequenceToVoxelGrid_Pytorch(object):
    # Source: https://github.com/uzh-rpg/rpg_e2vid/blob/master/utils/inference_utils.py#L480
    def __init__(self, num_bins, gpu=False, gpu_nr=0, normalize=True, forkserver=True):
        if forkserver:
            try:
                torch.multiprocessing.set_start_method('forkserver')
            except RuntimeError:
                pass
        self.num_bins = num_bins
        self.normalize = normalize
        if gpu:
            if not torch.cuda.is_available():
                print('Warning: There\'s no CUDA support on this machine!')
            else:
                self.device = torch.device('cuda:' + str(gpu_nr))
        else:
            self.device = torch.device('cpu')

    def __call__(self, event_sequence):
        """
        Build a voxel grid with bilinear interpolation in the time domain from a set of events.
        :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
        :param num_bins: number of bins in the temporal axis of the voxel grid
        :param width, height: dimensions of the voxel grid
        :param device: device to use to perform computations
        :return voxel_grid: PyTorch event tensor (on the device specified)
        """

        events = event_sequence.features.astype('float')

        width = event_sequence.image_width
        height = event_sequence.image_height

        assert (events.shape[1] == 4)
        assert (self.num_bins > 0)
        assert (width > 0)
        assert (height > 0)

        with torch.no_grad():

            events_torch = torch.from_numpy(events)
            # with DeviceTimer('Events -> Device (voxel grid)'):
            events_torch = events_torch.to(self.device)

            # with DeviceTimer('Voxel grid voting'):
            voxel_grid = torch.zeros(self.num_bins, height, width, dtype=torch.float32, device=self.device).flatten()

            # normalize the event timestamps so that they lie between 0 and num_bins
            last_stamp = events_torch[-1, 0]
            first_stamp = events_torch[0, 0]

            assert last_stamp.dtype == torch.float64, 'Timestamps must be float64!'
            # assert last_stamp.item()%1 == 0, 'Timestamps should not have decimals'

            deltaT = last_stamp - first_stamp

            if deltaT == 0:
                deltaT = 1.0

            events_torch[:, 0] = (self.num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
            ts = events_torch[:, 0]
            xs = events_torch[:, 1].long()
            ys = events_torch[:, 2].long()
            pols = events_torch[:, 3].float()
            pols[pols == 0] = -1  # polarity should be +1 / -1


            tis = torch.floor(ts)
            tis_long = tis.long()
            dts = ts - tis
            vals_left = pols * (1.0 - dts.float())
            vals_right = pols * dts.float()

            valid_indices = tis < self.num_bins
            valid_indices &= tis >= 0

            if events_torch.is_cuda:
                datatype = torch.cuda.LongTensor
            else:
                datatype = torch.LongTensor

            voxel_grid.index_add_(dim=0,
                                  index=(xs[valid_indices] + ys[valid_indices]
                                         * width + tis_long[valid_indices] * width * height).type(
                                      datatype),
                                  source=vals_left[valid_indices])


            valid_indices = (tis + 1) < self.num_bins
            valid_indices &= tis >= 0

            voxel_grid.index_add_(dim=0,
                                  index=(xs[valid_indices] + ys[valid_indices] * width
                                         + (tis_long[valid_indices] + 1) * width * height).type(datatype),
                                  source=vals_right[valid_indices])

            voxel_grid = voxel_grid.view(self.num_bins, height, width)

        if self.normalize:
            mask = torch.nonzero(voxel_grid, as_tuple=True)
            if mask[0].size()[0] > 0:
                mean = voxel_grid[mask].mean()
                std = voxel_grid[mask].std()
                if std > 0:
                    voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                else:
                    voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid

class EventSequence(object):
    def __init__(self, dataframe, params, features=None, timestamp_multiplier=None, convert_to_relative=False):
        if isinstance(dataframe, pandas.DataFrame):
            self.feature_names = dataframe.columns.values
            self.features = dataframe.to_numpy()
        else:
            self.feature_names = numpy.array(['ts', 'x', 'y', 'p'], dtype=object)
            if features is None:
                self.features = numpy.zeros([1, 4])
            else:
                self.features = features
        self.image_height = params['height']
        self.image_width = params['width']
        if not self.is_sorted():
            self.sort_by_timestamp()
        if timestamp_multiplier is not None:
            self.features[:,0] *= timestamp_multiplier
        if convert_to_relative:
            self.absolute_time_to_relative()

    def get_sequence_only(self):
        return self.features

    def __len__(self):
        return len(self.features)

    def __add__(self, sequence):
        event_sequence = EventSequence(dataframe=None,
                                       features=numpy.concatenate([self.features, sequence.features]),
                                       params={'height': self.image_height,
                                               'width': self.image_width})
        return event_sequence

    def is_sorted(self):
        return numpy.all(self.features[:-1, 0] <= self.features[1:, 0])

    def sort_by_timestamp(self):
        if len(self.features[:, 0]) > 0:
            sort_indices = numpy.argsort(self.features[:, 0])
            self.features = self.features[sort_indices]

    def absolute_time_to_relative(self):
        """Transforms absolute time to time relative to the first event."""
        start_ts = self.features[:,0].min()
        assert(start_ts == self.features[0,0])
        self.features[:,0] -= start_ts

def vis_map_RGB(map, save_map_path, name):
    channel,h,w = map.shape
    if(channel==5): # events
        map = numpy.concatenate([map, numpy.zeros((1,h,w))], axis=0)
        map_img1 = map[:3, ...]
        map_img2 = map[3:, ...]
        for c in range(3):
            if(map_img1[c].mean() != 0):
                map_img1[c] = (map_img1[c] - map_img1[c].min()) / (map_img1[c].max() - map_img1[c].min()) * 255
            if(map_img2[c].mean() != 0):
                map_img2[c] = (map_img2[c] - map_img2[c].min()) / (map_img2[c].max() - map_img2[c].min()) * 255

        map_img_sum = numpy.concatenate([map_img1, map_img2], axis=2) # 叠加在
        map_img_sum = numpy.asarray(map_img_sum.transpose(1,2,0), dtype=numpy.uint8)
    elif(channel==3):
        map_img = (map - numpy.min(map)) / (numpy.max(map) - numpy.min(map)) * 255
        map_img_sum = numpy.asarray(map_img.transpose(1,2,0), dtype=numpy.uint8)

    elif(channel>5):
        map_img = numpy.squeeze(numpy.sum(map, axis=0))
        if(map_img.mean() != 0):
            map_img = (map_img - map_img.min()) / (map_img.max() - map_img.min())
        map_img_sum = numpy.asarray(map_img * 255, dtype=numpy.uint8)

    if not os.path.exists(save_map_path):
        os.makedirs(save_map_path)
    cv2.imwrite(os.path.join(save_map_path, name), map_img_sum)
    return 

def vis_map_RGB(map, save_path, name):
    import numpy as np
    def plot_points_on_background(points_coordinates,
                            background,
                            points_color=[0, 0, 255]):
        """
        Args:
            points_coordinates: array of (y, x) points coordinates
                                of size (number_of_points x 2).
            background: (3 x height x width)
                        gray or color image uint8.
            color: color of points [red, green, blue] uint8.
        """
        if not (len(background.size()) == 3 and background.size(0) == 3):
            raise ValueError('background should be (color x height x width).')
        _, height, width = background.size()
        background_with_points = background.clone()
        y, x = points_coordinates.transpose(0, 1)
        if len(x) > 0 and len(y) > 0: # There can be empty arrays!
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            if not (x_min >= 0 and y_min >= 0 and x_max < width and y_max < height):
                raise ValueError('points coordinates are outsize of "background" '
                                'boundaries.')
            background_with_points[:, y, x] = torch.Tensor(points_color).type_as(
                background).unsqueeze(-1)
        return background_with_points 
    
    if(map.shape[0]<10):
        map = map.transpose(1,2,0)

    height,width,_ = map.shape

    map_img = np.squeeze(np.sum(map, axis=-1))
    # if(map_img.mean() != 0):
    #     map_img = (map_img - map_img.min()) / (map_img.max() - map_img.min())
    density = np.sum(map_img>0.5) / (height * width)

    mean = map_img.mean()

    # Red -> Negative Events
    red = (map_img <= (mean-0.2))
    # Blue -> Positive Events
    blue = (map_img >= (mean+0.2))
    
    background = torch.full((3, height, width), 255).byte()
    points_on_background = plot_points_on_background(
    torch.nonzero(torch.from_numpy(red.astype(np.uint8))), background,
    [255, 0, 0])
    points_on_background = plot_points_on_background(
    torch.nonzero(torch.from_numpy(blue.astype(np.uint8))),
    points_on_background, [0, 0, 255])

    map_img_sum = points_on_background.permute(1,2,0).numpy()

    os.makedirs(save_path, exist_ok=True)
    name = name.replace(".jpg","")
    cv2.imwrite(os.path.join(save_path, name+"_{:.3f}.jpg".format(density)), map_img_sum)
    return 
