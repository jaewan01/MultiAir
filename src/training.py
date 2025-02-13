from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import time

from dataset import Air_with_fixed_Sat_Dataset
from model import MultiAir

import random
from pyproj import Proj
import os
import argparse

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=.1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = args.gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if args.use_focal_L1:
            pt = torch.abs(input-target) # MAE
        else:
            pt = torch.abs(input-target)**2 # MSE

        logpt = torch.sigmoid((0.25*pt)**self.gamma)
        F_loss = (2*(logpt)-1)*pt
        
        return F_loss.mean()

# assign PM2.5 class accorrding to the range
def assign_class(arr, range, classes):
    return np.select([np.logical_and(arr > r[0], arr <= r[1]) for r in range], classes, default=0)

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

def train(args):
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    feat_dim = args.feat_dim
    input_dim = args.input_dim
    output_dim = args.output_dim

    lats, lons, korea_stn_regions, korea_stn_num, china_stn_num, coors_xy = load_stations(args)

    time_index_train = {}
    with open(f"{args.data_path}/train_time.txt", "r") as f:
        train_times = f.read().strip().split(' ')
        for idx, key in enumerate(train_times):
            time_index_train[int(key)] = idx
    
    with open(f"{args.data_path}/validation_days.txt", "r") as f:
        validation_days = f.read().strip().split(' ')
    
    del train_times

    validation_days = list(map(int, validation_days))
    validation_days = set(validation_days)

    train_times = sorted(time_index_train.keys())

    train_time_len = len(time_index_train)

    feat_train = np.zeros((train_time_len, korea_stn_num + china_stn_num, feat_dim))
    mask_train = np.zeros((train_time_len, korea_stn_num + china_stn_num))
    sat_output_train = np.zeros((train_time_len, korea_stn_num + china_stn_num, output_dim))
    sat_input_train = np.zeros((train_time_len, korea_stn_num + china_stn_num, 13))


    for t in tqdm(train_times, desc='loading train data'):
        cur_data = np.load(f"{args.data_path}/ground_obs/{t}.npy")
        feat_train[time_index_train[t]] = cur_data[:, :feat_dim]
        mask_train[time_index_train[t]] = cur_data[:, -1]
        sat_output_data = np.load(f"{args.sat_data_path}/prediction/{t}00.npy")
        sat_output_train[time_index_train[t]] = sat_output_data
        sat_input_data = np.load(f"{args.sat_data_path}/observation/{t}00.npy")
        sat_input_train[time_index_train[t]] = sat_input_data

    with open(f"{args.data_path}/feat_infos.txt", "r") as f:
        for line in f.readlines():
            feat_name, mean, std = line.strip().split(',')
            if feat_name == 'PM2.5':
                pm25_mean, pm25_std = float(mean), float(std)
                break
    
    train_dataset = Air_with_fixed_Sat_Dataset(train_times, time_index_train, sat_output_train, sat_input_train, feat_train, mask_train, input_dim, output_dim, args.prev_len, korea_stn_num, china_stn_num)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=5, collate_fn = train_dataset.collate_fn, drop_last=True)
    valid_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=5, collate_fn = train_dataset.collate_fn, drop_last=True)
    
    # load model
    model = MultiAir(input_dim = input_dim, lats = lats, lons = lons, feat_dim=feat_dim, hidden_dim=args.hidden_dim, pm25_mean=pm25_mean, pm25_std=pm25_std, output_dim = output_dim, prev_len=args.prev_len, korea_stn_num=korea_stn_num, china_stn_num=china_stn_num).to(args.default_device)
    model = torch.nn.DataParallel(model, device_ids=args.all_devices, output_device=args.default_device)

    os.makedirs('check_points', exist_ok=True)
    
    if args.use_pretrained: 
        model.load_state_dict(torch.load(f'check_points/{args.pretrained_model}.pkt'))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.use_focal_loss:
        criterion = FocalLoss()
    else:
        criterion = torch.nn.MSELoss()

    max_time = args.max_time

    epoch_cnt = 0
    train_loss_sum, train_acc = 0., 0
    TP1, TN1, FP1, FN1 = 0, 0, 0, 0
    TP2, TN2, FP2, FN2 = 0, 0, 0, 0
    TP3, TN3, FP3, FN3 = 0, 0, 0, 0
    TP = np.array([0 for _ in range(3 * max_time)])
    TN = np.array([0 for _ in range(3 * max_time)])
    FP = np.array([0 for _ in range(3 * max_time)])
    FN = np.array([0 for _ in range(3 * max_time)])

    p_TP = np.array([0 for _ in range(3 * max_time)])
    p_TN = np.array([0 for _ in range(3 * max_time)])
    p_FP = np.array([0 for _ in range(3 * max_time)])
    p_FN = np.array([0 for _ in range(3 * max_time)])

    range_4class = [(-1,15),(15,35),(35,75), (75,np.Inf)]
    class_four = [0,1,2,3]

    os.makedirs('logs', exist_ok=True)

    model.train()
    f_log = open("logs/training_" + args.model_name + ".log", "a")
    f_log.write(str(args))
    f_log.write('\n')
    f_log.flush()

    best_loss = np.Inf
    patience_check = 0

    # model training start
    while epoch_cnt < args.epoch:

        start = time.time()
        step_cnt = 0
        for feats, masks, sat_outputs, sat_inputs, pred_classes, pred_vals, pred_mask, raw_times, prev_vals in tqdm(iter(train_loader), desc='training epoch'+str(epoch_cnt + 1)):

            optimizer.zero_grad()

            valid_batches = torch.ones(feats.shape[0], dtype=bool).to(args.default_device)
            for i in range(feats.shape[0]):
                cur_date = int(raw_times[i, input_dim-1, 0] * 10000 + raw_times[i, input_dim-1, 1] * 100 + raw_times[i, input_dim-1, 2])
                if cur_date in validation_days:
                    valid_batches[i] = 0
                    continue
                for j in range(max_time):
                    if torch.sum(pred_mask[i, j]) == 0:
                        valid_batches[i] = 0
            
            if torch.sum(valid_batches) == 0:
                continue
            
            feats = feats.to(args.default_device)
            masks = masks.to(args.default_device)
            pred_classes = pred_classes.to(args.default_device)
            pred_vals = pred_vals.to(args.default_device)
            pred_mask = pred_mask.to(args.default_device)
            raw_times = raw_times.to(args.default_device)
            prev_vals = prev_vals.to(args.default_device)
            sat_outputs = sat_outputs.to(args.default_device)
            sat_inputs = sat_inputs.to(args.default_device)

            feats = feats[valid_batches]
            masks = masks[valid_batches]
            pred_classes = pred_classes[valid_batches]
            pred_vals = pred_vals[valid_batches]
            pred_mask = pred_mask[valid_batches]
            raw_times = raw_times[valid_batches]
            prev_vals = prev_vals[valid_batches]
            sat_outputs = sat_outputs[valid_batches]
            sat_inputs = sat_inputs[valid_batches]

            last_PM = feats[:,-1,:korea_stn_num,0].clone()
            last_PM = last_PM.repeat(1,output_dim).view(last_PM.shape[0], output_dim, korea_stn_num)
            last_PM_np = np.array(last_PM.cpu().numpy())
            persistent_PM = assign_class(last_PM_np, range_4class, class_four) # predicted PM2.5 class by persistence model
            persistent_PM = torch.from_numpy(persistent_PM).to(args.default_device)

            preds = model(feats, masks, raw_times[:,:,1:], prev_vals, sat_outputs, sat_inputs)
            
            preds = preds.permute(0, 2, 1)[:, :max_time, :]
            predict_PM_np = np.array(preds.detach().cpu().numpy())
            predict_PM_class = assign_class(predict_PM_np, range_4class, class_four)
            labels = torch.from_numpy(predict_PM_class)[:, :max_time, :].to(args.default_device)
            pred_classes = pred_classes[:, :max_time, :].to(args.default_device)

            pred_vals = pred_vals[:, :max_time, :].to(args.default_device)

            korea_mask = pred_mask[:,:max_time,:]

            loss = criterion(preds[korea_mask] , pred_vals[korea_mask])

            loss.backward()

            optimizer.step()

            step_cnt += 1
            
            train_loss_sum += loss.item()

            cur_labels = labels[korea_mask]
            cur_preds = pred_classes[korea_mask]

            train_acc += (cur_labels == cur_preds).float().sum().item()

            TP1 += ((cur_labels > 0) & (cur_preds > 0)).float().sum().item()
            TN1 += ((cur_labels == 0) & (cur_preds == 0)).float().sum().item()
            FP1 += ((cur_labels > 0) & (cur_preds == 0)).float().sum().item()
            FN1 += ((cur_labels == 0) & (cur_preds > 0)).float().sum().item()
            
            TP2 += ((cur_labels > 1) & (cur_preds > 1)).float().sum().item()
            TN2 += ((cur_labels < 2) & (cur_preds < 2)).float().sum().item()
            FP2 += ((cur_labels > 1) & (cur_preds < 2)).float().sum().item()
            FN2 += ((cur_labels < 2) & (cur_preds > 1)).float().sum().item()

            TP3 += ((cur_labels > 2) & (cur_preds > 2)).float().sum().item()
            TN3 += ((cur_labels < 3) & (cur_preds < 3)).float().sum().item()
            FP3 += ((cur_labels > 2) & (cur_preds < 3)).float().sum().item()
            FN3 += ((cur_labels < 3) & (cur_preds > 2)).float().sum().item()

            for i in range(1, 3+1):
                for j in range(max_time):
                    cur_labels = labels[:,j][korea_mask[:, j]]
                    cur_preds = pred_classes[:,j][korea_mask[:, j]]
                    cur_perst = persistent_PM[:,j][korea_mask[:, j]]
                    TP[(i-1) * max_time + j] += ((cur_labels > i-1) & (cur_preds > i-1)).sum()
                    TN[(i-1) * max_time + j] += ((cur_labels < i) & (cur_preds < i)).sum()
                    FP[(i-1) * max_time + j] += ((cur_labels > i-1) & (cur_preds < i)).sum()
                    FN[(i-1) * max_time + j] += ((cur_labels < i) & (cur_preds > i-1)).sum()

                    p_TP[(i-1) * max_time + j] += ((cur_perst > i-1) & (cur_preds > i-1)).sum()
                    p_TN[(i-1) * max_time + j] += ((cur_perst < i) & (cur_preds < i)).sum()
                    p_FP[(i-1) * max_time + j] += ((cur_perst > i-1) & (cur_preds < i)).sum()
                    p_FN[(i-1) * max_time + j] += ((cur_perst < i) & (cur_preds > i-1)).sum()

            del preds, loss, labels, feats, masks, pred_classes, pred_vals, raw_times, prev_vals, korea_mask, cur_labels, cur_preds, cur_perst

        # calculate CSI and F1 score for persistence model and ASOC model
        CSI = ' '.join(map(str, (TP / (TP + FN + FP)).tolist()))
        F1 = ' '.join(map(str, (2 * TP / (2 * TP + FN + FP)).tolist()))

        p_CSI = ' '.join(map(str, (p_TP / (p_TP + p_FN + p_FP)).tolist()))
        p_F1 = ' '.join(map(str, (2 * p_TP / (2 * p_TP + p_FN + p_FP)).tolist()))

        epoch_cnt += 1
        f_log.write(f'Epoch #{epoch_cnt}: train_loss is {train_loss_sum / step_cnt} train_acc: {train_acc / (TP1 + FN1 + FP1 + TN1)} ')
        f_log.write(f'F1_15: {(2 * TP1) / (2 * TP1 + FN1 + FP1)} F1_35: {(2 * TP2) / (2 * TP2 + FN2 + FP2)} F1_75: {(2 * TP3) / (2 * TP3 + FN3 + FP3)}\t')
        f_log.write(f'CSI_15: {(TP1) / (TP1 + FN1 + FP1)} CSI_35: {(TP2) / (TP2 + FN2 + FP2)} CSI_75: {(TP3) / (TP3 + FN3 + FP3)}\n')
        f_log.write(F1 + '\n')
        f_log.write(CSI + '\n')
        f_log.write('persistence model \n')
        f_log.write(p_F1 + '\n')
        f_log.write(p_CSI + '\n')
        print('training loss:', train_loss_sum / step_cnt)

        f_log.flush()
        train_loss_sum, train_acc = 0., 0
        TP1, TN1, FP1, FN1 = 0, 0, 0, 0
        TP2, TN2, FP2, FN2 = 0, 0, 0, 0
        TP3, TN3, FP3, FN3 = 0, 0, 0, 0
        TP = np.array([0 for _ in range(3 * max_time)])
        TN = np.array([0 for _ in range(3 * max_time)])
        FP = np.array([0 for _ in range(3 * max_time)])
        FN = np.array([0 for _ in range(3 * max_time)])
        p_TP = np.array([0 for _ in range(3 * max_time)])
        p_TN = np.array([0 for _ in range(3 * max_time)])
        p_FP = np.array([0 for _ in range(3 * max_time)])
        p_FN = np.array([0 for _ in range(3 * max_time)])


        end = time.time()
        f_log.write(f'Duration time is {end - start}\n')
        f_log.flush()
        model.eval()
        torch.save(model.state_dict(), 'check_points/{}.pkt'.format(epoch_cnt))
        val_loss_sum, val_acc = 0., 0
        _TP1, _TN1, _FP1, _FN1 = 0, 0, 0, 0
        _TP2, _TN2, _FP2, _FN2 = 0, 0, 0, 0
        _TP3, _TN3, _FP3, _FN3 = 0, 0, 0, 0
        _TP = np.array([0 for _ in range(3 * output_dim)])
        _TN = np.array([0 for _ in range(3 * output_dim)])
        _FP = np.array([0 for _ in range(3 * output_dim)])
        _FN = np.array([0 for _ in range(3 * output_dim)])
        p_TP = np.array([0 for _ in range(3 * output_dim)])
        p_TN = np.array([0 for _ in range(3 * output_dim)])
        p_FP = np.array([0 for _ in range(3 * output_dim)])
        p_FN = np.array([0 for _ in range(3 * output_dim)])

        # model validation start for model selection
        with torch.no_grad():
            val_step_cnt = 0
            valid_diff_sum = 0
            valid_diff_squares_sum = 0
            valid_entry_cnt = 0
            for feats, masks, sat_outputs, sat_inputs, pred_classes, pred_vals, pred_mask, raw_times, prev_vals in tqdm(iter(valid_loader), desc='validation epoch'+str(epoch_cnt)):
                
                valid_batches = torch.zeros(feats.shape[0], dtype=bool)
                for i in range(feats.shape[0]):
                    cur_date = int(raw_times[i, input_dim-1, 0] * 10000 + raw_times[i, input_dim-1, 1] * 100 + raw_times[i, input_dim-1, 2])
                    if cur_date in validation_days:
                        valid_batches[i] = 1
                        continue
                
                if torch.sum(valid_batches) == 0:
                    continue

                feats = feats[valid_batches]
                masks = masks[valid_batches]
                pred_classes = pred_classes[valid_batches]
                pred_vals = pred_vals[valid_batches]
                pred_mask = pred_mask[valid_batches]
                raw_times = raw_times[valid_batches]
                prev_vals = prev_vals[valid_batches]
                sat_outputs = sat_outputs[valid_batches]
                sat_inputs = sat_inputs[valid_batches]

                last_PM = feats[:,-1,:korea_stn_num,0].clone()
                last_PM = last_PM.repeat(1,output_dim).view(last_PM.shape[0], output_dim, korea_stn_num)
                last_PM_np = np.array(last_PM.cpu().numpy())
                persistent_PM = assign_class(last_PM_np, range_4class, class_four) # predicted PM2.5 class by persistence model
                persistent_PM = torch.from_numpy(persistent_PM).to(args.default_device)
                preds = model(feats, masks, raw_times[:,:,1:], prev_vals, sat_outputs, sat_inputs)
                
                preds = preds.permute(0, 2, 1)
                predict_PM_np = np.array(preds.detach().cpu().numpy())
                predict_PM_class = assign_class(predict_PM_np, range_4class, class_four)
                labels = torch.from_numpy(predict_PM_class).to(args.default_device)
                pred_classes = pred_classes.to(args.default_device)

                pred_vals = pred_vals.to(args.default_device)

                loss_mask = pred_mask[:,:max_time,:]
                
                val_loss_sum += criterion(preds[:, :max_time,:][loss_mask] , pred_vals[:, :max_time,:][loss_mask]).item()                        
                
                valid_diffs = preds[:, :max_time,:][loss_mask] - pred_vals[:, :max_time,:][loss_mask]
                valid_diff_sum += torch.abs(valid_diffs).sum().item()
                valid_diff_squares_sum += (valid_diffs ** 2).sum().item()
                valid_entry_cnt += loss_mask.sum().item()

                val_step_cnt += 1

                cur_labels = labels[pred_mask]
                cur_preds = pred_classes[pred_mask]

                val_acc += (cur_labels == cur_preds).float().sum().item()
                
                _TP1 += ((cur_labels > 0) & (cur_preds > 0)).float().sum().item()
                _TN1 += ((cur_labels == 0) & (cur_preds == 0)).float().sum().item()
                _FP1 += ((cur_labels > 0) & (cur_preds == 0)).float().sum().item()
                _FN1 += ((cur_labels == 0) & (cur_preds > 0)).float().sum().item()
                _TP2 += ((cur_labels > 1) & (cur_preds > 1)).float().sum().item()
                _TN2 += ((cur_labels < 2) & (cur_preds < 2)).float().sum().item()
                _FP2 += ((cur_labels > 1) & (cur_preds < 2)).float().sum().item()
                _FN2 += ((cur_labels < 2) & (cur_preds > 1)).float().sum().item()
                _TP3 += ((cur_labels > 2) & (cur_preds > 2)).float().sum().item()
                _TN3 += ((cur_labels < 3) & (cur_preds < 3)).float().sum().item()
                _FP3 += ((cur_labels > 2) & (cur_preds < 3)).float().sum().item()
                _FN3 += ((cur_labels < 3) & (cur_preds > 2)).float().sum().item()
                
                for i in range(1, 3+1):
                    for j in range(output_dim):
                        cur_labels = labels[:,j][pred_mask[:, j]]
                        cur_preds = pred_classes[:,j][pred_mask[:, j]]
                        cur_perst = persistent_PM[:,j][pred_mask[:, j]]
                        _TP[(i-1) * output_dim + j] += ((cur_labels > i-1) & (cur_preds > i-1)).sum()
                        _TN[(i-1) * output_dim + j] += ((cur_labels < i) & (cur_preds < i)).sum()
                        _FP[(i-1) * output_dim + j] += ((cur_labels > i-1) & (cur_preds < i)).sum()
                        _FN[(i-1) * output_dim + j] += ((cur_labels < i) & (cur_preds > i-1)).sum()

                        p_TP[(i-1) * output_dim + j] += ((cur_perst > i-1) & (cur_preds > i-1)).sum()
                        p_TN[(i-1) * output_dim + j] += ((cur_perst < i) & (cur_preds < i)).sum()
                        p_FP[(i-1) * output_dim + j] += ((cur_perst > i-1) & (cur_preds < i)).sum()
                        p_FN[(i-1) * output_dim + j] += ((cur_perst < i) & (cur_preds > i-1)).sum()
                del preds, labels, feats, masks, pred_classes, pred_vals, raw_times, prev_vals, pred_mask, cur_labels, cur_preds, cur_perst
                
                
        _CSI = ' '.join(map(str, (_TP / (_TP + _FN + _FP)).tolist()))
        _F1 = ' '.join(map(str, (2 * _TP / (2 * _TP + _FN + _FP)).tolist()))

        p_CSI = ' '.join(map(str, (p_TP / (p_TP + p_FN + p_FP)).tolist()))
        p_F1 = ' '.join(map(str, (2 * p_TP / (2 * p_TP + p_FN + p_FP)).tolist()))

        val_loss = val_loss_sum / val_step_cnt

        f_log.write(f'val_loss is {val_loss} val_acc: {val_acc / (_TP1 + _TN1 + _FP1 + _FN1)}\n')
        f_log.write(_F1 + '\n')
        f_log.write(_CSI + '\n')
        f_log.write('persistence model \n')
        f_log.write(p_F1 + '\n')
        f_log.write(p_CSI + '\n')
        f_log.flush()
        print('validation loss:', val_loss)
        print('presist', p_CSI)
        print('ASOC:', _CSI)
        model.train()

        # model selection
        if val_loss > best_loss:
            patience_check += 1
            if patience_check == args.patience:
                best_epoch = epoch_cnt - args.patience
                model.load_state_dict(torch.load('check_points/{}.pkt'.format(best_epoch)))
                torch.save(model.state_dict(), 'check_points/'+args.model_name +'.pkt')
                break
        else:
            best_loss = val_loss
            patience_check = 0

            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='MultiAir PM2.5 training model')
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay for optimizer")
    parser.add_argument("--epoch", type=int, default=200,
                        help="number of epoch")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed")
    parser.add_argument("--model_name", type=str, default='',
                        help="name of saved model")
    parser.add_argument("--batch_size", type=int, default=25,
                        help="number of batch size")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="hidden dimension for LSTM")
    parser.add_argument("--input_dim", type=int, default=7,
                        help="input window size")
    parser.add_argument("--output_dim", type=int, default=6,
                        help="number of predictions")
    parser.add_argument("--patience", type=int, default=5,
                        help="patience for early stopping")
    parser.add_argument("--max_time", type=int, default=6,
                        help="current max time for training")
    parser.add_argument("--prev_len", type=int, default=7,
                        help="previous length for statistics of data")
    parser.add_argument("--data_path", type=str, default='data',
                        help="path of data")
    parser.add_argument("--sat_data_path", type=str, default='data/satellite_image',
                        help="path of satellite data")
    parser.add_argument("--feat_dim", type=int, default=12,
                        help="feature dimension")

    parser.add_argument("--gpus", type=str, default='0',
                        help="gpu id for execution")
    parser.add_argument('--use_focal_loss', default=False, action='store_true', help='focal loss')
    parser.add_argument('--use_focal_L1', default=False, action='store_true', help='focal L1 loss')
    parser.add_argument('--use_pretrained', default=False, action='store_true', help='Utilizing pretrained model')
    parser.add_argument("--pretrained_model", type=str, default='1hr',
                        help="name of pretrained model")
    parser.add_argument('--dish_init', type=str, default='standard') # standard, 'avg' or 'uniform'
    parser.add_argument("--gamma", type=int, default=2, help="gamma of focal loss")
    
    args = parser.parse_args()

    # set gpu devices
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
    train(args)
