import pandas as pd
import numpy as np
from sklearn import preprocessing
from Calc_Element import Calc_Sunrad_for_Temp, Calc_RH11

def process(sta, lon, lat):
    Afile = f'temp/{sta}.temp.csv'

    data = pd.read_csv(Afile)

    Nhours = 0
    for hr in range(0,24,1):
        # hr_count
        hr_count    = data['HOUR'] == hr
        if hr_count.sum() > 0 :
            Nhours = Nhours+1


    Nparam = 24 / Nhours
    print ( "Dataset:{:f}, Nparam:{:f}".format(len(data.index),Nparam ) )

    #--- Make Target Data
    (y, errflg) = Temp_Target(data)
    if errflg == 1: return

    Y_train = y.loc[:,'ObsTemp'].values

    #--- NN -> add "MON"
    (x_with_hour, errflg) = Temp_Input(data, lat, lon)
    if errflg == 1: return

    #--- remove sunrad
    zero_sunrad = x_with_hour['sunrad'] == 0.0
    if zero_sunrad.sum() > len(x_with_hour) * 0.8:
        print("{:s}: [Check] {:s}-{:s} remove Sunrad for feature." .format(myname,contxt,lclid))
        if str('sunrad')   in list(x_with_hour.columns):
            x_with_hour = x_with_hour.drop(['sunrad'], axis=1)
        if str('se_delay') in list(x_with_hour.columns):
            x_with_hour = x_with_hour.drop(['se_delay'], axis=1)

    x = x_with_hour.drop('HOUR', axis=1)
    X_train = x.values

    return X_train, Y_train

def Temp_Input(data, lat, lon):
    
    Data_Elems = list(data.columns)

    #--- Surface
    tmp_sfc = data[['SFC-AIRTMP']]
    uv_sfc  = data[['SFC-UWIND','SFC-VWIND']]

    #--  MM & HH
    mon  = data[['MON']]
    hour = data[['HOUR']]

    tmp_low = data[['850-AIRTMP','925-AIRTMP','950-AIRTMP','975-AIRTMP']]
    uv_low  = data[['900-UWIND','900-VWIND','925-UWIND','925-VWIND','950-UWIND','950-VWIND']]

    #####
    YYYY = data.loc[:,'YEAR'].values
    MM   = data.loc[:,'MON'].values
    DD   = data.loc[:,'DAY'].values
    HH   = data.loc[:,'HOUR'].values
    date    = pd.DataFrame({'year':YYYY,'month':MM,'day':DD,'hour':HH})
    UTC_dt  = pd.to_datetime(date)
    UTC_dti = pd.DatetimeIndex(UTC_dt).tz_localize('UTC')
    #####

    rh = data[['300-RHUM','400-RHUM','500-RHUM','600-RHUM','700-RHUM','800-RHUM','850-RHUM','900-RHUM','925-RHUM','950-RHUM','975-RHUM']]
    rh_norm = pd.DataFrame(rh.values*0.01,columns=['300-rh','400-rh','500-rh','600-rh','700-rh','800-rh','850-rh','900-rh','925-rh','950-rh','975-rh'])

    rh11 = Calc_RH11('JMA_MSM',data)
    RH11 = rh11.values

    (SUNRAD,A,SE_delay) = Calc_Sunrad_for_Temp('JMA_MSM',lat,lon,UTC_dti,RH11)

    sunrad   = pd.DataFrame({'sunrad':SUNRAD})
    a        = pd.DataFrame({'a':A})
    se_delay = pd.DataFrame({'se_delay':SE_delay})

    lb = preprocessing.LabelBinarizer()
    lb.fit([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
    HH_LB   = lb.transform(hour)
    hour_lb = pd.DataFrame(HH_LB,\
            columns=['0hr','1hr','2hr','3hr','4hr','5hr','6hr','7hr','8hr','9hr','10hr','11hr','12hr','13hr','14hr','15hr','16hr','17hr','18hr','19hr','20hr','21hr','22hr','23hr'])

    lb = preprocessing.LabelBinarizer()
    lb.fit([1,2,3,4,5,6,7,8,9,10,11,12])
    MM_LB   = lb.transform(mon)
    mon_lb  = pd.DataFrame(MM_LB,columns=['1mon','2mon','3mon','4mon','5mon','6mon','7mon','8mon','9mon','10mon','11mon','12mon'])

    x = pd.concat([hour,mon_lb,hour_lb,se_delay,tmp_low,uv_low,uv_sfc,rh_norm,sunrad,tmp_sfc],axis=1).round(4)
    return x, 0

def Temp_Target(data):

    #--  Check columns
    Need_Elems = ['ObsTemp']
    Data_Elems = list(data.columns)

    for column in Need_Elems:
        if not column in Data_Elems:
            print('Warning: No value')

            return -9999., 1

    y = data[['ObsTemp']]

    return y, 0

if __name__ == '__main__':
    sta = 'AMEDAS-44132'
    lat = 35.69166666666667
    lon = 139.75
    process(sta, lon, lat)
