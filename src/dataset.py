import numpy as np
import torch

# assign PM2.5 class accorrding to the range
def assign_class(arr, range, classes):
  return np.select([np.logical_and(arr > r[0], arr <= r[1]) for r in range], classes, default=-1)

def assign_class2(pm_arr, pm_mask, range, classes):
    cur_class = np.select([np.logical_and(pm_arr > r[0], pm_arr <= r[1]) for r in range], classes, default=-1)
    cur_class[~pm_mask] = -1
    return cur_class

class Air_with_fixed_Sat_Dataset(torch.utils.data.Dataset):
    """
    :param times: list of time in the dataset
    :param time_index: index of time in the dataset
    :param sat_outputs: satellite predictions
    :param sat_inputs: satellite observations
    :param feats,masks: features and corresponding masks
    :param input_dim: length of time series input
    :param output_dim: length of time series output
    :param prev_len: length of previous values for normalization
    :param korea_stn_num,china_stn_num: number of stations in Korea and China
    """
    def __init__(self, times, time_index, sat_outputs, sat_inputs, feats, masks, input_dim, output_dim, prev_len, korea_stn_num, china_stn_num):
        
        self.times = times
        self.time_index = time_index
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prev_len = prev_len
        self.time_index = time_index
        self.korea_stn_num = korea_stn_num
        self.china_stn_num = china_stn_num
        self.total_stn_num = korea_stn_num + china_stn_num
        
        self.feat_torch = torch.FloatTensor(feats)
        self.mask_torch = torch.from_numpy(masks)
        self.sat_outputs_torch = torch.FloatTensor(sat_outputs)
        self.sat_inputs_torch = torch.FloatTensor(sat_inputs)

    
    def load_feats(self, idx):
        feats = self.feat_torch[idx - self.input_dim + 1: idx + 1]
        if (idx - self.input_dim + 1 < 0):
            feats = torch.zeros((self.input_dim, self.total_stn_num, self.feat_torch.shape[-1]))
        return feats
    
    def load_masks(self, idx):
        masks = self.mask_torch[idx - self.input_dim + 1: idx + 1]
        if (idx - self.input_dim + 1 < 0):
            masks = torch.zeros((self.input_dim, self.total_stn_num), dtype=bool)
        return masks
    
    def __len__(self) -> int:
        return len(self.times)
    
    def __getitem__(self, idx):
        feats = self.load_feats(idx)
        masks = self.load_masks(idx)
        sat_outputs = self.sat_outputs_torch[idx]
        sat_inputs = self.sat_inputs_torch[idx]

        cur_mask = masks[0].clone()

        for i in range(1, self.input_dim):
            cur_mask = torch.logical_and(cur_mask, masks[i])

        pred_pm25_vals = self.feat_torch[idx+1:idx+1+self.output_dim, :self.korea_stn_num, 0].numpy() 

        pred_pm25_mask = self.feat_torch[idx+1:idx+1+self.output_dim, :self.korea_stn_num, 6].numpy().astype(bool)

        for i in range(pred_pm25_mask.shape[0]):
            pred_pm25_mask[i] = np.logical_and(~pred_pm25_mask[i], cur_mask[:self.korea_stn_num])

        if idx < self.input_dim:
            pred_pm25_vals.fill(0.)
            pred_pm25_mask.fill(False)

        if len(pred_pm25_vals) < self.output_dim:
            if len(pred_pm25_vals) == 0:
                pred_pm25_vals = np.full((self.output_dim, self.korea_stn_num), 0.)
                pred_pm25_mask = np.full((self.output_dim, self.korea_stn_num), False)
            else:
                for i in range(self.output_dim - len(pred_pm25_vals)):
                    pred_pm25_vals = np.vstack([pred_pm25_vals, np.full_like(pred_pm25_vals[0], 0.)])
                    pred_pm25_mask = np.vstack([pred_pm25_mask, np.full_like(pred_pm25_mask[0], False)])

        prev_pm25_vals = self.feat_torch[idx-self.prev_len + 1:idx + 1, :, 0].numpy()

        if len(prev_pm25_vals) < self.prev_len:
            if len(prev_pm25_vals) == 0:
                prev_pm25_vals = np.full((self.prev_len, self.total_stn_num), np.nan)
            else:
                for i in range(self.prev_len - len(prev_pm25_vals)):
                    prev_pm25_vals = np.vstack([np.full_like(prev_pm25_vals[0], np.nan), prev_pm25_vals])      

        raw_times = []

        for t_idx in range(self.input_dim + self.output_dim):
            cur_idx = idx - self.input_dim + 1 + t_idx
            if cur_idx < 0 or cur_idx >= len(self.times):
                raw_times.append([0,0,0,0])
            else:
                _year = self.times[cur_idx] // 1000000
                _month = (self.times[cur_idx] // 10000) % 100
                _day = (self.times[cur_idx] // 100) % 100
                _time = self.times[cur_idx] % 100
                raw_times.append([_year, _month, _day, _time])

        range_4class = [(-1,15),(15,35),(35,75), (75,np.Inf)]
        class_four = [0,1,2,3]
        pred_pm25_class = assign_class2(pred_pm25_vals, pred_pm25_mask, range_4class, class_four)
        
        return feats, cur_mask, sat_outputs, sat_inputs, torch.IntTensor(pred_pm25_class), torch.FloatTensor(pred_pm25_vals), torch.BoolTensor(pred_pm25_mask), torch.FloatTensor(raw_times), torch.FloatTensor(prev_pm25_vals)
    
    def collate_fn(self, samples):
        feats, masks, sat_outputs, sat_inputs, pred_classes, pred_vals, pred_mask, raw_times, prev_vals = zip(*samples)
        feats = torch.stack(feats, dim=0)
        masks = torch.stack(masks, dim=0)
        sat_outputs = torch.stack(sat_outputs, dim=0)
        sat_inputs = torch.stack(sat_inputs, dim=0)
        pred_classes = torch.stack(pred_classes, dim=0)
        pred_vals = torch.stack(pred_vals, dim=0)
        pred_mask = torch.stack(pred_mask, dim=0)
        raw_times = torch.stack(raw_times)
        prev_vals = torch.stack(prev_vals, dim=0)
        return feats, masks, sat_outputs, sat_inputs, pred_classes, pred_vals, pred_mask, raw_times, prev_vals
