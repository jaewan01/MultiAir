from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import Air_with_fixed_Sat_Dataset
from model import MultiAir

from pyproj import Proj
import pandas as pd

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

# Total evaluation code
def evaluation(args):

    feat_dim = args.feat_dim
    input_dim = args.input_dim
    output_dim = args.output_dim

    lats, lons, korea_stn_regions, korea_stn_num, china_stn_num, coors_xy = load_stations(args)
    
    time_index_test = {}
    with open(f"{args.data_path}/test_time.txt", "r") as f:
        test_times = f.read().strip().split(' ')
        for idx, key in enumerate(test_times):
            time_index_test[int(key)] = idx

    del test_times

    test_times = sorted(time_index_test.keys())

    test_time_len = len(test_times)

    feat_test = np.zeros((test_time_len, korea_stn_num + china_stn_num, feat_dim))
    mask_test = np.zeros((test_time_len, korea_stn_num + china_stn_num))    
    sat_output_test = np.zeros((test_time_len, korea_stn_num + china_stn_num, output_dim))
    sat_input_test = np.zeros((test_time_len, korea_stn_num + china_stn_num, 13))

    for t in tqdm(test_times, desc='loading test data'):
        cur_data = np.load(f"{args.data_path}/ground_obs/{t}.npy")
        feat_test[time_index_test[t]] = cur_data[:, :feat_dim]
        mask_test[time_index_test[t]] = cur_data[:, -1]
        sat_output_data = np.load(f"{args.sat_data_path}/prediction/{t}00.npy")
        sat_output_test[time_index_test[t]] = sat_output_data
        sat_input_data = np.load(f"{args.sat_data_path}/observation/{t}00.npy")
        sat_input_test[time_index_test[t]] = sat_input_data
        
    with open(f"{args.data_path}/feat_infos.txt", "r") as f:
        for line in f.readlines():
            feat_name, mean, std = line.strip().split(',')
            if feat_name == 'PM2.5':
                pm25_mean, pm25_std = float(mean), float(std)
                break
    
    test_dataset = Air_with_fixed_Sat_Dataset(test_times, time_index_test, sat_output_test,sat_input_test, feat_test, mask_test, input_dim, output_dim, args.prev_len, korea_stn_num, china_stn_num)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=5, collate_fn = test_dataset.collate_fn, drop_last=False)
    

    criterion = torch.nn.MSELoss()
    model = MultiAir(input_dim = input_dim, lats = lats, lons = lons, feat_dim=feat_dim, hidden_dim=args.hidden_dim, pm25_mean=pm25_mean, pm25_std=pm25_std, output_dim = output_dim, prev_len=args.prev_len, korea_stn_num=korea_stn_num, china_stn_num=china_stn_num).to(args.default_device)
    model = torch.nn.DataParallel(model, device_ids=args.all_devices, output_device=args.default_device)

    model.load_state_dict(torch.load(f'check_points/{args.model_name}.pkt'))
        
    val_loss_sum, val_acc = 0., 0
    _TP1, _TN1, _FP1, _FN1 = 0, 0, 0, 0
    _TP2, _TN2, _FP2, _FN2 = 0, 0, 0, 0
    _TP3, _TN3, _FP3, _FN3 = 0, 0, 0, 0
    total_a1, total_a2, total_a3, total_a4 = 0, 0, 0, 0
    total_b1, total_b2, total_b3, total_b4 = 0, 0, 0, 0
    total_c1, total_c2, total_c3, total_c4 = 0, 0, 0, 0
    total_d1, total_d2, total_d3, total_d4 = 0, 0, 0, 0

    per_total_a1, per_total_a2, per_total_a3, per_total_a4 = 0, 0, 0, 0
    per_total_b1, per_total_b2, per_total_b3, per_total_b4 = 0, 0, 0, 0
    per_total_c1, per_total_c2, per_total_c3, per_total_c4 = 0, 0, 0, 0
    per_total_d1, per_total_d2, per_total_d3, per_total_d4 = 0, 0, 0, 0
    
    _TP = np.array([0 for _ in range(3 * output_dim)])
    _TN = np.array([0 for _ in range(3 * output_dim)])
    _FP = np.array([0 for _ in range(3 * output_dim)])
    _FN = np.array([0 for _ in range(3 * output_dim)])
    p_TP = np.array([0 for _ in range(3 * output_dim)])
    p_TN = np.array([0 for _ in range(3 * output_dim)])
    p_FP = np.array([0 for _ in range(3 * output_dim)])
    p_FN = np.array([0 for _ in range(3 * output_dim)])
    
    _RMSE_np = np.array([0. for _ in range(output_dim)])
    _MAE_np = np.array([0. for _ in range(output_dim)])
    p_RMSE_np = np.array([0. for _ in range(output_dim)])
    p_MAE_np = np.array([0. for _ in range(output_dim)])
    valid_count = np.array([0 for _ in range(output_dim)])

    range_4class = [(-1,15),(15,35),(35,75), (75,np.Inf)]
    class_four = [0,1,2,3]

    gt_vals = []
    gt_masks = []
    model_vals = []
    times = []

    f_log = open("logs/test_"+args.model_name+".log", "a")
    f_log.write(str(args))
    f_log.write('\n')
    f_log.flush()

    # model evaluation start
    model.eval()
    with torch.no_grad():
        val_step_cnt = 0
        valid_diff_sum = 0
        valid_diff_sum_p = 0
        valid_diff_squares_sum = 0
        valid_diff_squares_sum_p = 0
        valid_entry_cnt = 0
        valid_entry_diff = 0

        valid_norm_diff_sum = 0
        valid_norm_diff_sum_p = 0
        valid_norm_diff_abs_sum = 0
        valid_norm_diff_abs_sum_p = 0

        valid_vals_gt = []
        valid_vals_model = []
        valid_vals_p = []
        valid_nonzero_entry_cnt = 0

        for feats, masks, sat_outputs, sat_inputs, pred_classes, pred_vals, pred_mask, raw_times, prev_vals in tqdm(iter(test_loader), desc='evaluation'):

            last_PM = feats[:,-1,:korea_stn_num,0].clone()
            last_PM = last_PM.repeat(1,output_dim).view(last_PM.shape[0], output_dim, korea_stn_num).to(args.default_device)
            last_PM_np = np.array(last_PM.cpu().numpy())
            persistent_PM = assign_class(last_PM_np, range_4class, class_four) # predicted PM2.5 class by persistence model
            persistent_PM = torch.from_numpy(persistent_PM).to(args.default_device)

            preds = model(feats, masks, raw_times[:,:,1:], prev_vals, sat_outputs, sat_inputs)
            
            preds = preds.permute(0, 2, 1)
            preds[preds < 0.] = 0.
            
            predict_PM_np = np.array(preds.detach().cpu().numpy())
            predict_PM_class = assign_class(predict_PM_np, range_4class, class_four)
            labels = torch.from_numpy(predict_PM_class).to(args.default_device)
            pred_classes = pred_classes.to(args.default_device)

            pred_vals = pred_vals.to(args.default_device)

            for i in range(preds.shape[0]):
                gt_vals.append(pred_vals[i].detach().cpu().numpy())
                gt_masks.append(pred_mask[i].detach().cpu().numpy())
                model_vals.append(preds[i].detach().cpu().numpy())
                cur_time = raw_times[i,input_dim-1,0].item() * 1000000 + raw_times[i,input_dim-1,1].item() * 10000 + raw_times[i,input_dim-1,2].item() * 100 + raw_times[i,input_dim-1,3].item()
                times.append(int(cur_time))

            mask_diff = pred_mask[:, 1:] & pred_mask[:, :-1]

            valid_entry_diff += mask_diff.sum().item()

            pred_mask = pred_mask.to(args.default_device)

            val_loss_sum += criterion(preds[pred_mask], pred_vals[pred_mask]).item()

            valid_diffs = preds[pred_mask] - pred_vals[pred_mask]
            valid_diff_sum += torch.abs(valid_diffs).sum().item()
            valid_diff_squares_sum += (valid_diffs ** 2).sum().item()
            
            valid_diffs_p = last_PM[pred_mask] - pred_vals[pred_mask]
            valid_diff_sum_p += torch.abs(valid_diffs_p).sum().item()
            valid_diff_squares_sum_p += (valid_diffs_p ** 2).sum().item()

            valid_entry_cnt += pred_mask.sum().item()

            nonzero_mask = pred_mask & (pred_vals > 0)
            valid_nonzero_entry_cnt += nonzero_mask.sum().item()

            valid_nonzero_diffs = preds[nonzero_mask] - pred_vals[nonzero_mask]
            valid_norm_diff_sum += ((valid_nonzero_diffs) / pred_vals[nonzero_mask]).sum().item()
            valid_norm_diff_abs_sum += torch.abs((valid_nonzero_diffs) / pred_vals[nonzero_mask]).sum().item()

            valid_nonzero_diffs_p = last_PM[nonzero_mask] - pred_vals[nonzero_mask]
            valid_norm_diff_sum_p += ((valid_nonzero_diffs_p) / pred_vals[nonzero_mask]).sum().item()
            valid_norm_diff_abs_sum_p += torch.abs((valid_nonzero_diffs_p) / pred_vals[nonzero_mask]).sum().item()

            valid_vals_gt += pred_vals[pred_mask].tolist()
            valid_vals_model += preds[pred_mask].tolist()
            valid_vals_p += last_PM[pred_mask].tolist()
            
            cur_labels = labels[pred_mask]
            cur_preds = pred_classes[pred_mask]
            cur_perst = persistent_PM[pred_mask]

            val_acc += ((cur_labels == cur_preds)).float().sum().item()
            
            val_step_cnt += 1

            # count of occurrences for all combinations of ground and predicted classes
            total_a1 += ((cur_labels == 0) & (cur_preds == 0)).float().sum().item()
            total_a2 += ((cur_labels == 0) & (cur_preds == 1)).float().sum().item()
            total_a3 += ((cur_labels == 0) & (cur_preds == 2)).float().sum().item()
            total_a4 += ((cur_labels == 0) & (cur_preds == 3)).float().sum().item()
            total_b1 += ((cur_labels == 1) & (cur_preds == 0)).float().sum().item()
            total_b2 += ((cur_labels == 1) & (cur_preds == 1)).float().sum().item()
            total_b3 += ((cur_labels == 1) & (cur_preds == 2)).float().sum().item()
            total_b4 += ((cur_labels == 1) & (cur_preds == 3)).float().sum().item()
            total_c1 += ((cur_labels == 2) & (cur_preds == 0)).float().sum().item()
            total_c2 += ((cur_labels == 2) & (cur_preds == 1)).float().sum().item()
            total_c3 += ((cur_labels == 2) & (cur_preds == 2)).float().sum().item()
            total_c4 += ((cur_labels == 2) & (cur_preds == 3)).float().sum().item()
            total_d1 += ((cur_labels == 3) & (cur_preds == 0)).float().sum().item()
            total_d2 += ((cur_labels == 3) & (cur_preds == 1)).float().sum().item()
            total_d3 += ((cur_labels == 3) & (cur_preds == 2)).float().sum().item()
            total_d4 += ((cur_labels == 3) & (cur_preds == 3)).float().sum().item()


            per_total_a1 += ((cur_perst == 0) & (cur_preds == 0)).float().sum().item()
            per_total_a2 += ((cur_perst == 0) & (cur_preds == 1)).float().sum().item()
            per_total_a3 += ((cur_perst == 0) & (cur_preds == 2)).float().sum().item()
            per_total_a4 += ((cur_perst == 0) & (cur_preds == 3)).float().sum().item()
            per_total_b1 += ((cur_perst == 1) & (cur_preds == 0)).float().sum().item()
            per_total_b2 += ((cur_perst == 1) & (cur_preds == 1)).float().sum().item()
            per_total_b3 += ((cur_perst == 1) & (cur_preds == 2)).float().sum().item()
            per_total_b4 += ((cur_perst == 1) & (cur_preds == 3)).float().sum().item()
            per_total_c1 += ((cur_perst == 2) & (cur_preds == 0)).float().sum().item()
            per_total_c2 += ((cur_perst == 2) & (cur_preds == 1)).float().sum().item()
            per_total_c3 += ((cur_perst == 2) & (cur_preds == 2)).float().sum().item()
            per_total_c4 += ((cur_perst == 2) & (cur_preds == 3)).float().sum().item()
            per_total_d1 += ((cur_perst == 3) & (cur_preds == 0)).float().sum().item()
            per_total_d2 += ((cur_perst == 3) & (cur_preds == 1)).float().sum().item()
            per_total_d3 += ((cur_perst == 3) & (cur_preds == 2)).float().sum().item()
            per_total_d4 += ((cur_perst == 3) & (cur_preds == 3)).float().sum().item()     
            
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
                    cur_labels = labels[:,j][pred_mask[:,j]]
                    cur_preds = pred_classes[:,j][pred_mask[:,j]]
                    cur_perst = persistent_PM[:,j][pred_mask[:,j]]

                    _TP[(i-1) * output_dim + j] += ((cur_labels > i-1) & (cur_preds > i-1)).sum()
                    _TN[(i-1) * output_dim + j] += ((cur_labels < i) & (cur_preds < i) & (cur_preds > -1)).sum()
                    _FP[(i-1) * output_dim + j] += ((cur_labels > i-1) & (cur_preds < i) & (cur_preds > -1)).sum()
                    _FN[(i-1) * output_dim + j] += ((cur_labels < i) & (cur_preds > i-1)).sum()

                    p_TP[(i-1) * output_dim + j] += ((cur_perst > i-1) & (cur_preds > i-1)).sum()
                    p_TN[(i-1) * output_dim + j] += ((cur_perst < i) & (cur_preds < i) & (cur_preds > -1)).sum()
                    p_FP[(i-1) * output_dim + j] += ((cur_perst > i-1) & (cur_preds < i) & (cur_preds > -1)).sum()
                    p_FN[(i-1) * output_dim + j] += ((cur_perst < i) & (cur_preds > i-1)).sum()

            for i in range(output_dim):
                _RMSE_np[i] += ((preds[:,i][pred_mask[:,i] & (pred_classes[:,j] >= 0)] - pred_vals[:,i][pred_mask[:,i] & (pred_classes[:,j] >= 0)])**2).sum().item()
                _MAE_np[i] += (torch.abs(preds[:,i][pred_mask[:,i] & (pred_classes[:,j] >= 0)] - pred_vals[:,i][pred_mask[:,i] & (pred_classes[:,j] >= 0)])).sum().item()
                p_RMSE_np[i] += ((last_PM[:,i][pred_mask[:,i] & (pred_classes[:,j] >= 0)] - pred_vals[:,i][pred_mask[:,i] & (pred_classes[:,j] >= 0)])**2).sum().item()
                p_MAE_np[i] += torch.abs(last_PM[:,i][pred_mask[:,i] & (pred_classes[:,j] >= 0)] - pred_vals[:,i][pred_mask[:,i] & (pred_classes[:,j] >= 0)]).sum().item()
                valid_count[i] += (pred_mask[:,i] & (pred_classes[:,j] >= 0)).sum().item()

            del preds, labels, feats, masks, pred_classes, pred_vals, raw_times, prev_vals, pred_mask, cur_labels, cur_preds, cur_perst

        valid_vals_gt_np = np.array(valid_vals_gt)
        valid_vals_model_np = np.array(valid_vals_model)
        valid_vals_p_np = np.array(valid_vals_p)

        nmb_p = np.sum((valid_vals_p_np - valid_vals_gt_np)) / np.sum(valid_vals_gt_np) * 100
        nme_p = np.sum(np.abs(valid_vals_p_np - valid_vals_gt_np)) / np.sum(valid_vals_gt_np) * 100

        nmb = np.sum((valid_vals_model_np - valid_vals_gt_np)) / np.sum(valid_vals_gt_np) * 100
        nme = np.sum(np.abs(valid_vals_model_np - valid_vals_gt_np)) / np.sum(valid_vals_gt_np) * 100

        # print results of evaluation
        _RMSE = np.sqrt(_RMSE_np / valid_count)
        _MAE = _MAE_np / valid_count
        
        p_RMSE = np.sqrt(p_RMSE_np / valid_count)
        p_MAE = p_MAE_np / valid_count

        _CSI_real = (_TP / (_TP + _FN + _FP))
        _p_CSI_real = (p_TP / (p_TP + p_FN + p_FP))

        _F1_real = (2 * _TP / (2 * _TP + _FN + _FP))
        _p_F1_real = (2 * p_TP / (2 * p_TP + p_FN + p_FP))

        model_CSI_dict = {'normal':_CSI_real[:output_dim], 'high':_CSI_real[output_dim:2 * output_dim], 'very high':_CSI_real[2 * output_dim:]}
        persist_CSI_dict = {'normal':_p_CSI_real[:output_dim], 'high':_p_CSI_real[output_dim:2 * output_dim], 'very high':_p_CSI_real[2 * output_dim:]}
        model_F1_dict = {'normal':_F1_real[:output_dim], 'high':_F1_real[output_dim:2 * output_dim], 'very high':_F1_real[2 * output_dim:]}
        persist_F1_dict = {'normal':_p_F1_real[:output_dim], 'high':_p_F1_real[output_dim:2 * output_dim], 'very high':_p_F1_real[2 * output_dim:]}

        model_RMSE_dict = {'total':_RMSE}
        model_MAE_dict = {'total':_MAE}
        persist_RMSE_dict = {'total':p_RMSE}
        persist_MAE_dict = {'total':p_MAE}
        
        per_ACC = (per_total_a1 + per_total_b2 + per_total_c3 + per_total_d4) / (per_total_a1+per_total_a2+per_total_a3+per_total_a4+per_total_b1+per_total_b2+per_total_b3+per_total_b4+per_total_c1+per_total_c2+per_total_c3+per_total_c4+per_total_d1+per_total_d2+per_total_d3+per_total_d4)
        per_POD = (per_total_c3 + per_total_c4 + per_total_d3 + per_total_d4) / (per_total_a3+per_total_a4+per_total_b3+per_total_b4+per_total_c3+per_total_c4+per_total_d3+per_total_d4)
        per_FAR = (per_total_c1 + per_total_c2 + per_total_d1 + per_total_d2) / (per_total_c1+per_total_c2+per_total_c3+per_total_c4+per_total_d1+per_total_d2+per_total_d3+per_total_d4)
        
        valid_vals_gt_np = np.array(valid_vals_gt)
        valid_vals_model_np = np.array(valid_vals_model)
        valid_vals_p_np = np.array(valid_vals_p)

        valid_vals_gt_np = valid_vals_gt_np - np.mean(valid_vals_gt_np)
        valid_vals_model_np = valid_vals_model_np - np.mean(valid_vals_model_np)
        valid_vals_p_np = valid_vals_p_np - np.mean(valid_vals_p_np)

        ACC = (total_a1 + total_b2 + total_c3 + total_d4) / (total_a1 + total_a2 + total_a3 + total_a4 + total_b1 + total_b2 + total_b3 + total_b4 + total_c1 + total_c2 + total_c3 + total_c4 + total_d1 + total_d2 + total_d3 + total_d4)
        POD = (total_c3 + total_c4 + total_d3 + total_d4) / (total_a3+total_a4+total_b3+total_b4+total_c3+total_c4+total_d3+total_d4)
        FAR = (total_c1 + total_c2 + total_d1 + total_d2) / (total_c1+total_c2+total_c3+total_c4+total_d1+total_d2+total_d3+total_d4)
        r = np.sum(valid_vals_model_np * valid_vals_gt_np) / (np.sqrt(np.sum(valid_vals_model_np ** 2)) * np.sqrt(np.sum(valid_vals_gt_np ** 2)))
        r_p = np.sum(valid_vals_p_np * valid_vals_gt_np) / (np.sqrt(np.sum(valid_vals_p_np ** 2)) * np.sqrt(np.sum(valid_vals_gt_np ** 2)))

        model_CSI_pd = pd.DataFrame(model_CSI_dict)
        persist_CSI_pd = pd.DataFrame(persist_CSI_dict)
        model_F1_pd = pd.DataFrame(model_F1_dict)
        persist_F1_pd = pd.DataFrame(persist_F1_dict)

        model_RMSE_pd = pd.DataFrame(model_RMSE_dict)
        model_MAE_pd = pd.DataFrame(model_MAE_dict)
        persist_RMSE_pd = pd.DataFrame(persist_RMSE_dict)
        persist_MAE_pd = pd.DataFrame(persist_MAE_dict)

        f_log.write('persist total ACC: ' + str(per_ACC) + '\n')
        f_log.write('persist total POD: ' + str(per_POD) + '\n')
        f_log.write('persist total FAR: ' + str(per_FAR) + '\n')
        f_log.write('persist total F1 score: ' + str(2 * (per_POD * (1 - per_FAR)) / (per_POD + (1 - per_FAR))) + '\n')
        f_log.write('persist MAE: ' + str(valid_diff_sum_p / valid_entry_cnt) + '\n')
        f_log.write('persist RMSE: ' + str((valid_diff_squares_sum_p / valid_entry_cnt) ** 0.5) + '\n')
        f_log.write('persist NMB: ' + str(nmb_p) + '\n')
        f_log.write('persist NME: ' + str(nme_p) + '\n')
        f_log.write('persist R: ' + str(r_p) + '\n')

        f_log.write('model total ACC: ' + str(ACC) + '\n')
        f_log.write('model total POD: ' + str(POD) + '\n')
        f_log.write('model total FAR: ' + str(FAR) + '\n')
        f_log.write('model total F1 score: ' + str(2 * (POD * (1 - FAR)) / (POD + (1 - FAR))) + '\n')
        f_log.write('model MAE: ' + str(valid_diff_sum / valid_entry_cnt) + '\n')
        f_log.write('model RMSE: ' + str((valid_diff_squares_sum / valid_entry_cnt) ** 0.5) + '\n')
        f_log.write('model NMB: ' + str(nmb) + '\n')
        f_log.write('model NME: ' + str(nme) + '\n')
        f_log.write('model R: ' + str(r) + '\n')

        f_log.write('persistance model CSI:\n' + persist_CSI_pd.to_string() + '\n')
        f_log.write('persistance model F1:\n' + persist_F1_pd.to_string() + '\n')
        f_log.write('persistance model RMSE:\n' + persist_RMSE_pd.to_string() + '\n')
        f_log.write('persistance model MAE:\n' + persist_MAE_pd.to_string() + '\n')
        f_log.write('MultiAir CSI:\n' + model_CSI_pd.to_string() + '\n')
        f_log.write('MultiAir F1:\n' + model_F1_pd.to_string() + '\n')
        f_log.write('MultiAir RMSE:\n' + model_RMSE_pd.to_string() + '\n')
        f_log.write('MultiAir MAE:\n' + model_MAE_pd.to_string() + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='evaluation MultiAir')
    parser.add_argument("--batch_size", type=int, default=24,
                        help="number of batch size")
    parser.add_argument("--data_path", type=str, default='data',
                        help="path of data")
    parser.add_argument("--sat_data_path", type=str, default='data/satellite_image',
                        help="path of satellite data")
    parser.add_argument("--model_name", type=str, default='',
                        help="name of model to evaluate")
    parser.add_argument("--gpus", type=str, default='0',
                        help="gpu id for execution")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="hidden dimension for LSTM")
    parser.add_argument("--output_dim", type=int, default=6,
                        help="number of predictions")
    parser.add_argument("--input_dim", type=int, default=7,
                        help="input window size")
    parser.add_argument("--prev_len", type=int, default=7,
                        help="previous length for statistics of data")
    parser.add_argument("--feat_dim", type=int, default=12,
                        help="feature dimension")
    
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
    evaluation(args)
