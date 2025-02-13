from model import SimVP_adv
import torch
import os
from datetime import datetime, timedelta
import numpy as np
import gzip
from pyproj import Proj
from tqdm import tqdm

import argparse

# load station coordinates
def load_stations(args):
    def location(lat, long):
        p = Proj("+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38.0 +lon_0=126.0 +x_0=450000 +y_0=450000 +no_defs +ellps=WGS84 +units=km", preserve_units=True)
        x0, y0 = p(126.0, 38.0)
        x, y = p(long, lat)
        return (int(round(450 + (y - y0)/2)), int(round(450 + (x - x0)/2)))
    lats = []
    lons = []
    korea_stn_regions = []
    korea_stn_num = 0
    china_stn_num = 0
    xys = []
    with open(f"{args.data_path}/station_infos/korea.txt", 'r') as f:
        for line in f:
            data = line.strip().split(",")
            lats.append(float(data[2]))
            lons.append(float(data[3]))
            korea_stn_regions.append(data[-1])
            korea_stn_num += 1
            xys.append(location(float(data[2]), float(data[3])))
    with open(f"{args.data_path}/station_infos/china.txt", 'r') as f:
        for line in f:
            data = line.strip().split(",")
            lats.append(float(data[2]))
            lons.append(float(data[3]))
            china_stn_num += 1
            xys.append(location(float(data[2]), float(data[3])))
    lats = torch.tensor(lats)
    lons = torch.tensor(lons)
    xys = torch.LongTensor(np.array(xys))
    return lats, lons, korea_stn_regions, korea_stn_num, china_stn_num, xys

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run SimVP model')
    parser.add_argument("--model_name", type=str, default='',
                        help="SimVP model checkpoint")
    parser.add_argument("--input_dim", type=int, default=13,
                        help="input window size")
    parser.add_argument("--output_dim", type=int, default=6,
                        help="number of predictions")
    parser.add_argument("--patience", type=int, default=5,
                        help="patience for early stopping")
    parser.add_argument("--max_time", type=int, default=6,
                        help="current max time for training")
    parser.add_argument("--data_path", type=str, default='data',
                        help="path of data")
    parser.add_argument("--sat_data_path", type=str, default='data/satellite_image',
                        help="path of data")
    parser.add_argument("--feat_dim", type=int, default=12,
                        help="feature dimension")

    parser.add_argument("--gpus", type=str, default='0',
                        help="gpu id for execution")

    args = parser.parse_args()

    if torch.cuda.is_available():
        if args.gpus == 'cpu':
            args.all_devices = None
            args.default_device = 'cpu:0'
        elif args.gpus is not None:
            args.all_devices = list(map(int, args.gpus.split(',')))
            args.default_device = args.all_devices[0]
        else:
            args.all_devices = [i for i in range(torch.cuda.device_count())]
            args.default_device = torch.cuda.current_device()
    else:
        args.all_devices = None
        args.default_device = 'cpu:0'
    torch.cuda.set_device(args.default_device)
    print(f'enabled gpu_devices: {args.all_devices}, default device: {args.default_device}')
    print(args)

    lats, lons, korea_stn_regions, korea_stn_num, china_stn_num, coors_xy = load_stations(args)
    img_model = SimVP_adv(shape_in=[args.input_dim,1,900,900], hid_S=4, hid_T=64, groups=4, C_out=6).to(args.default_device)
    img_model = torch.nn.DataParallel(img_model, device_ids=args.all_devices, output_device=args.default_device)

    load_model = torch.load(f'check_points/{args.model_name}.pkt')
    img_model.load_state_dict(load_model)

    start_t = datetime(2021, 1, 1, 0, 0)
    end_t = datetime(2021, 3, 31, 23, 0)

    os.makedirs(f'{args.sat_data_path}/observation', exist_ok=True)
    os.makedirs(f'{args.sat_data_path}/prediction', exist_ok=True)

    # List for storing total time
    total_t_list = []

    # Add all the timestamp between start and end time
    current_t = start_t
    while current_t <= end_t:
        total_t_list.append(current_t)
        current_t += timedelta(hours=1)

    for current_t in tqdm(total_t_list):

        input_img_list = []

        # Start time for input data
        input_start_t = current_t - timedelta(hours= (args.input_dim - 1))
        current_t_str = current_t.strftime('%Y%m%d%H%M')

        date_list = []

        # Add all the timestamp needed for satellite data
        cur_t = input_start_t
        for i in range(args.input_dim + args.output_dim):
            date_list.append(cur_t.strftime('%Y%m%d%H%M'))
            cur_t += timedelta(hours=1)

        # Load satellite data
        for date in date_list:
            # first verify if the file exists
            try:
                with gzip.open(f"{args.sat_data_path}/APPS_kr/apps_{date}.bin.gz", "r") as f:
                    target = f.read()
                    result = np.frombuffer(target, 'f4').reshape(900, 900)
                    img = np.maximum(result, 0)
                    img[img>100] = -1
                    input_img_list.append(img)
            except:
                input_img_list.append(-np.ones((900,900)))

        satellite_for_simvp = torch.FloatTensor(np.stack(input_img_list, axis=0)[:args.input_dim])
        satellite_inputs = satellite_for_simvp.clone()

        satellite_for_simvp = satellite_for_simvp.unsqueeze(1).to(args.default_device)
        satellite_for_simvp = satellite_for_simvp.unsqueeze(0)
        output = img_model(satellite_for_simvp)
        output = output.squeeze(0)
        output = output.squeeze(1)
        output = output[:args.output_dim,coors_xy[:,1], coors_xy[:,0]].detach().cpu().permute(1,0).numpy()

        satellite_inputs = satellite_inputs[:, coors_xy[:,1], coors_xy[:,0]].cpu().permute(1,0).numpy()

        np.save(f'{args.sat_data_path}/observation/{current_t_str}.npy', satellite_inputs)
        np.save(f'{args.sat_data_path}/prediction/{current_t_str}.npy', output)
