In this file, we explain the format of the data for MultiAir
*Note1: all of the data we provide are randomly generated to help the understanding of the format. For details on the data please refer to our paper.
*Note2: please unpack ground_obs.tar.gz and satellite_image.tar.gz to use

train_time.txt, test_time.txt:
List of training time and test time in order. 
*Note1: order of the time should be consecutive.
*Note2: days for validations should be also included in train_time.txt

validation_days.txt:
List of dates used for validation

feat_infos.txt:
Information of each features (name, mean, standard deviation).

station_infos:
Information on ground stations. More explanation in readme.txt in the directory.

ground_obs:
Hourly observation from ground stations.
Each numpy file is of shape (number of stations, 13), where 13 is consisted of
PM2.5 (raw), PM10, SO2, NO2 ,O3, CO (z-score normalized), PM2.5 mask, SO2 mask, NO2 mask, O3 mask, CO mask, valid mask
where mask for feature is 0 if ground-truth exists and 1 if it was filled from other station, and valid mask means the validity of this station, whether to use it or not for attention.

satellite_image:
This contains satellite image of target region (In APPS_kr), along with observation & prediction generated through sat_model_img_extract.py.
