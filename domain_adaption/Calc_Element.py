#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import math
from time import time
from datetime import datetime
from datetime import timedelta
from pytz import timezone
#from timezonefinder import TimezoneFinder



#--- Define Parameter for TAU
# input parameter  : model
# output parameter : upper_param1, middle_param1, lower_param1, upper_param2, middle_param2, lower_param2
def Define_Param_for_TAU(model):

        #Default Parameter
	cc1 = 78
	cc2 = 75
	cc3 = 40
	cor1 = 5.6
	cor2 = 6.9
	cor3 = 16.0

	#Customized Parameter
	if model == "ECMWF_HIRES_0p1":

		cc1 = 70
		cc2 = 70
		cc3 = 40
		cor1 = 9.0
		cor2 = 7.4
		cor3 = 16.0

	elif model == "ECMWF_ENS_0p2":

		cc1 = 70
		cc2 = 70
		cc3 = 40
		cor1 = 7.8
		cor2 = 7.4
		cor3 = 16.0

	elif  model == "NCEP_ENS_0p5":

		cc1 = 70
		cc2 = 70
		cc3 = 40
		cor1 = 8.6
		cor2 = 7.0
		cor3 = 14.0

	elif model == "NCEP_GFS_0p25":

		cc1 = 70
		cc2 = 70
		cc3 = 40
		cor1 = 10.6
		cor2 = 7.8
		cor3 = 18.0

	return (cc1,cc2,cc3,cor1,cor2,cor3)



#--- Calc RH(11 layers) by Interpolation
# input parameter : model, rh(layer is depends on models)
# outpu parameter : RH(11 layers)
def Calc_RH11(model, data):


	if model == "ECMWF_HIRES_0p1":
		rh11 = pd.DataFrame({'300-RHUM':data['300-RHUM'],
			'400-RHUM': data['300-RHUM'].values*1/2 + data['500-RHUM'].values*1/2, \
			'500-RHUM': data['500-RHUM'],
			'600-RHUM': data['500-RHUM'].values*1/2 + data['700-RHUM'].values*1/2, \
			'700-RHUM': data['700-RHUM'],
			'800-RHUM': data['800-RHUM'],
			'850-RHUM': data['850-RHUM'],
			'900-RHUM': data['900-RHUM'],
			'925-RHUM': data['925-RHUM'],
			'950-RHUM': data['950-RHUM'],
			'975-RHUM': data['950-RHUM'].values*1/2 + data['1000-RHUM'].values*1/2})

	elif model == "ECMWF_ENS_0p2":
		rh11 = pd.DataFrame({'300-RHUM':data['300-RHUM'],
			'400-RHUM': data['300-RHUM'].values*1/2 + data['500-RHUM'].values*1/2, \
			'500-RHUM': data['500-RHUM'],
			'600-RHUM': data['500-RHUM'].values*1/2 + data['700-RHUM'].values*1/2,  \
			'700-RHUM': data['700-RHUM'],
			'800-RHUM': data['700-RHUM'].values*1/3 + data['850-RHUM'].values*2/3,  \
			'850-RHUM': data['850-RHUM'],
			'900-RHUM': data['850-RHUM'].values*1/3 + data['925-RHUM'].values*2/3,  \
			'925-RHUM': data['925-RHUM'],
			'950-RHUM': data['925-RHUM'].values*2/3 + data['1000-RHUM'].values*1/3, \
			'975-RHUM': data['925-RHUM'].values*1/3 + data['1000-RHUM'].values*2/3})

	elif model == "NCEP_ENS_0p5":
		rh11 = pd.DataFrame({'300-RHUM':data['250-RHUM'],
			'400-RHUM': data['250-RHUM'].values*1/2 + data['500-RHUM'].values*1/2, \
			'500-RHUM': data['500-RHUM'],
			'600-RHUM': data['500-RHUM'].values*1/2 + data['700-RHUM'].values*1/2,  \
			'700-RHUM': data['700-RHUM'],
			'800-RHUM': data['700-RHUM'].values*1/3 + data['850-RHUM'].values*2/3,  \
			'850-RHUM': data['850-RHUM'],
			'900-RHUM': data['850-RHUM'].values*1/3 + data['925-RHUM'].values*2/3,  \
			'925-RHUM': data['925-RHUM'],
			'950-RHUM': data['925-RHUM'].values*2/3 + data['1000-RHUM'].values*1/3, \
			'975-RHUM': data['925-RHUM'].values*1/3 + data['1000-RHUM'].values*2/3})

	else:
		rh11 = data[['300-RHUM','400-RHUM','500-RHUM','600-RHUM','700-RHUM','800-RHUM',\
			'850-RHUM','900-RHUM','925-RHUM','950-RHUM','975-RHUM']]


	return rh11




#--- Calc TAU from RH(11 layers)
# input parameter : RH(11 layers), upper_param1, middle_param1, lower_param1, upper_param2, middle_param2, lower_param2
# output paramter : TAU
def Calc_TAU11(RH11,cc1,cc2,cc3,cor1,cor2,cor3):

	TAU = []
	for ii in range(0,11,1):
			
		if   ii == 0:
			TAU  = 1.0*((100.0-np.exp((RH11[:,ii]-cc3)/cor3))/100.0)
		elif ii <= 4:  # 300~700hPa
			#TAU  = TAU*((100.0-np.exp((RH11[:,ii]-cc3)/cor3))/100.0)
			TAU  = np.array([TAU[jj] if TAU[jj] < 0.1 else TAU[jj]*((100.0-math.exp((RH11[jj,ii]-cc3)/cor3))/100.0) for jj in range(0,len(TAU),1)])
		elif ii <= 6:  # 800~850hPa
			#TAU  = TAU*((100.0-np.exp((RH11[:,ii]-cc2)/cor2))/100.0)
			TAU  = np.array([TAU[jj] if TAU[jj] < 0.1 else TAU[jj]*((100.0-math.exp((RH11[jj,ii]-cc2)/cor2))/100.0) for jj in range(0,len(TAU),1)])
		elif ii <= 11: # 900~975hPa
			#TAU  = TAU*((100.0-np.exp((RH11[:,ii]-cc1)/cor1))/100.0)
			TAU  = np.array([TAU[jj] if TAU[jj] < 0.1 else TAU[jj]*((100.0-math.exp((RH11[jj,ii]-cc1)/cor1))/100.0) for jj in range(0,len(TAU),1)])

	return TAU


#--- Calc TOA for Radiation
# input parameter : lat, lon, UTC(datetime index)
# output paramter : TOA_Radiation
def Calc_TOA(lat,lon,UTC_dti):

	a1 = 0.000075
	a2 = 0.001868
	a3 = 0.032077
	a4 = 0.014615
	a5 = 0.040849

	b1 = 0.006918
	b2 = 0.399912
	b3 = 0.070257
	b4 = 0.006758
	b5 = 0.000907
	b6 = 0.002697
	b7 = 0.001480

	c1 = 1.000110
	c2 = 0.034221
	c3 = 0.001280
	c4 = 0.000719
	c5 = 0.000077

	sc = 1366 #Solar Constant
	alvedo = 0.15 #alvedo
	p = 0.84

	pi = 3.141592653589


	#--- Calculation DOY
	dlon = lon
	dlon=np.where(dlon < 0, dlon+360, dlon)
	tz = (dlon+7.5)/15
	tz = tz.astype(np.int64)
	tz=np.where(tz > 12, tz-24, tz)
	tz=np.where((tz == 12)&(dlon>180), -12, tz)

	ydays = []        
         
	if tz.size == 1:
		tz=np.full(len(UTC_dti), tz)

	for ii in range(0,len(UTC_dti),1):
		if tz[ii] > 0 :
			first_dt = datetime(UTC_dti[ii].year-1,12,31,24-tz[ii])
		else :
			first_dt = datetime(UTC_dti[ii].year,1,1,0-tz[ii])
		yday = ( UTC_dti[ii] - timezone('UTC').localize(first_dt)).days
		ydays.append(yday)
	DOY = np.array(ydays)

	#--- Calculation Each Parameters
	A  = 2.0*pi*(DOY/365)

	SD = b1 - b2*np.cos(A) + b3*np.sin(A) - b4*np.cos(2*A) + b5*np.sin(2*A) - b6*np.cos(3*A) + b7*np.sin(3*A)
	SE_Distance = 1.0/np.sqrt(c1 + c2*np.cos(A) + c3*np.sin(A) + c4*np.cos(2*A) + c5*np.sin(2*A))
	ET = (a1 + a2*np.cos(A) - a3*np.sin(A) - a4*np.cos(2*A) - a5*np.sin(2*A)) * 12/pi
	GMT = UTC_dti.hour
	MST = GMT + lon / 15.0
	TST = MST + ET
	T_RAD = pi/12*(TST-12)
	lat_rad = lat * pi / 180

	SE = np.arcsin(np.sin(lat_rad)*np.sin(SD)+np.cos(lat_rad)*np.cos(SD)*np.cos(T_RAD))
	SE_deg = (180/pi)*SE

	#--- Calculation Radiation
	DirectRadiationAir = np.array([0 if np.sin(SE[ii])<=0  else p**(1/np.sin(SE[ii])) for ii in range(0,len(SE),1)])
	ScatRadiationAir=(1.0/2.0)*(1-DirectRadiationAir)/(1.0-1.4*np.log(p))

	RawRadiation = sc * np.sin(SE) * (DirectRadiationAir+ScatRadiationAir)
	TOA_Radiation = np.array([0 if RawRadiation[ii]<=0 else RawRadiation[ii] for ii in range(0,len(RawRadiation),1)])

	#--- return
	return TOA_Radiation


#--- Calc TOA for Temp
# input parameter : lat, lont, UTC(datetime index)
# output paramter : TOA_Radiation, A(Radian of DOY), Sun Elevation(2hr delay)
def Calc_TOA_for_Temp(lat,lon,UTC_dti):

	a1 = 0.000075
	a2 = 0.001868
	a3 = 0.032077
	a4 = 0.014615
	a5 = 0.040849

	b1 = 0.006918
	b2 = 0.399912
	b3 = 0.070257
	b4 = 0.006758
	b5 = 0.000907
	b6 = 0.002697
	b7 = 0.001480

	c1 = 1.000110
	c2 = 0.034221
	c3 = 0.001280
	c4 = 0.000719
	c5 = 0.000077

	sc = 1366 #Solar Constant
	alvedo = 0.15 #alvedo
	p = 0.84

	pi = 3.141592653589


	#--- Calculation timezone automatically(from lat/lon)
	#tf = TimezoneFinder()
	#tz_str = tf.timezone_at(lat=lat, lng=lon)

	#--- Calculation timezone manually
	dlon = lon
	dlon=np.where(dlon < 0, dlon+360, dlon)
	tz = (dlon+7.5)/15
	tz = tz.astype(np.int64)
	tz=np.where(tz > 12, tz-24, tz)
	tz=np.where((tz == 12)&(dlon>180), -12, tz)
	#dlon = lon
	#if dlon < 0:
	#	dlon = dlon + 360.0
	#tz = int((dlon+7.5)/15)
	#if tz > 12:
	#	tz = tz - 12
	#elif tz == 12 and dlon > 180.0:
	#	tz = -12

	#--- Calculation LST
	#if tz_str != None: # Successful to Calc. timezone from lat/lon
	#	LST_dti = UTC_dti.tz_convert(tz_str)
	#else:             # Failed to Calc.timezone from lat/lon -> Using Manually Calc. timezone
	#	LST_dti = UTC_dti + timedelta(hours=tz)
	#LST_dti = UTC_dti + timedelta(hours=tz)

	#--- Calculation DOY
	ydays = []
	#for ii in range(0,len(LST_dti),1):
	#	first_dt = datetime(LST_dti[ii].year,1,1,0)

		#if tz_str != None: # Successful to Calc. timezone from lat/lon
		#	yday = ( LST_dti[ii] - timezone(tz_str).localize(first_dt)).days
		#else:             # Failed to Calc.timezone from lat/lon -> Using Manually Calc. timezone
		#	yday = ( LST_dti[ii] - timezone('UTC').localize(first_dt)).days
	#	yday = ( LST_dti[ii] - timezone('UTC').localize(first_dt)).days
	#	ydays.append(yday)

	if tz.size == 1:
		tz=np.full(len(UTC_dti), tz)

	for ii in range(0,len(UTC_dti),1):
		if tz[ii] > 0 :
			first_dt = datetime(UTC_dti[ii].year-1,12,31,24-tz[ii])
		else:
			first_dt = datetime(UTC_dti[ii].year,1,1,0-tz[ii])
		yday = ( UTC_dti[ii] - timezone('UTC').localize(first_dt)).days
		ydays.append(yday)

	DOY = np.array(ydays)

	#--- Calculation Each Parameters
	A  = 2.0*pi*(DOY/365)

	SD = b1 - b2*np.cos(A) + b3*np.sin(A) - b4*np.cos(2*A) + b5*np.sin(2*A) - b6*np.cos(3*A) + b7*np.sin(3*A)
	SE_Distance = 1.0/np.sqrt(c1 + c2*np.cos(A) + c3*np.sin(A) + c4*np.cos(2*A) + c5*np.sin(2*A))
	ET = (a1 + a2*np.cos(A) - a3*np.sin(A) - a4*np.cos(2*A) - a5*np.sin(2*A)) * 12/pi
	GMT = UTC_dti.hour
	MST = GMT + lon / 15.0
	TST = MST + ET
	T_RAD = pi/12*(TST-12)
	lat_rad = lat * pi / 180

	SE = np.arcsin(np.sin(lat_rad)*np.sin(SD)+np.cos(lat_rad)*np.cos(SD)*np.cos(T_RAD))
	SE_deg = (180/pi)*SE

	#-- SE 2hr delay
	MST_delay = GMT + lon / 15.0  - 2
	MST_delay = np.array([24+MST_delay[ii] if MST_delay[ii]<0 else MST_delay[ii] for ii in range(0,len(MST_delay),1)])
	TST_delay = MST_delay + ET
	T_RAD_delay = pi/12*(TST_delay-12)
	SE_delay = np.arcsin(math.sin(lat_rad)*np.sin(SD)+math.cos(lat_rad)*np.cos(SD)*np.cos(T_RAD_delay))

	#--- Calculation Radiation
	DirectRadiationAir = np.array([0 if np.sin(SE[ii])<=0  else p**(1/np.sin(SE[ii])) for ii in range(0,len(SE),1)])
	ScatRadiationAir=(1.0/2.0)*(1-DirectRadiationAir)/(1.0-1.4*np.log(p))

	RawRadiation = sc * np.sin(SE) * (DirectRadiationAir+ScatRadiationAir)
	TOA_Radiation = np.array([0 if RawRadiation[ii]<=0 else RawRadiation[ii] for ii in range(0,len(RawRadiation),1)])

	#--- return
	return (TOA_Radiation,A,SE_delay)



#--- Calc Sunrad
# input parameter  : model, lon, lat, UTC(datetime index), RH(11 layers)
# output parameter : SolarRadiation(normalized)
def Calc_Sunrad(model,lat,lon,UTC_dti,RH11):

	sc = 1366 #Solar Constant

	#--- Calculation TAU
	(cc1,cc2,cc3,cor1,cor2,cor3) = Define_Param_for_TAU(model)

	TAU = Calc_TAU11(RH11,cc1,cc2,cc3,cor1,cor2,cor3)

	#--- Calculation Top Of Atomosphere Radiation	
	TOA_Radiation = Calc_TOA(lat,lon,UTC_dti)

	#--- Calculation Radiation
	Radiation = TOA_Radiation * TAU

	#--- Normalize
	return Radiation/sc



#--- Calc Sunrad for Temp( Add some elements related to Solar)
# input parameter  : model, lat, lon, UTC(datetime index), RH(11 layers)
# output parameter : SolarRadiation(normalized), A(Radian of DOY), Sun Elevation(2hr delay)
def Calc_Sunrad_for_Temp(model,lat,lon,UTC_dti,RH11):

	sc = 1366 #Solar Constant

	#--- Calculation TAU
	(cc1,cc2,cc3,cor1,cor2,cor3) = Define_Param_for_TAU(model)

	TAU = Calc_TAU11(RH11,cc1,cc2,cc3,cor1,cor2,cor3)

	#--- Calculation Top Of Atomosphere Radiation	
	(TOA_Radiation, A, SE_delay) = Calc_TOA_for_Temp(lat,lon,UTC_dti)

	#--- Calculation Radiation
	Radiation = TOA_Radiation * TAU

	#--- Normalize
	return (Radiation/sc), A, SE_delay



#--- Calc TAU from Srad
# input parameter  : lat, lon , UTC(datetime index), Sunrad
# output parameter : TAU
def Calc_Srad2Tau(lat,lon,UTC_dti,srad):

	sc = 1366 #Solar Constant

	#--- Calculation Top of Atomosphere Radiation
	TOA_Radiation = Calc_TOA(lat,lon,UTC_dti)

	TAU = np.array([-9999 if TOA_Radiation[ii]==0 else srad[ii]/TOA_Radiation[ii] for ii in range(0,len(TOA_Radiation),1)])

	#--- return
	return TAU



#--- Calc Srad from TAU
# input parameter  : lat, lon, UTC(datetime index), TAU
# output parameter : SolarRadiation(NOT normalized)
def Calc_Tau2Srad(lat,lon,UTC_dti,TAU):

	sc = 1366 #Solar Constant

	#--- Calculation Top of Atomosphere Radiation
	TOA_Radiation = Calc_TOA(lat,lon,UTC_dti)
	
	#--- Calculation Radiation
	Radiation = TOA_Radiation * TAU

	#Radiation = np.array([0 if RawRadiation[ii]<=0 or TAU[ii]<=0 else RawRadiation[ii] for ii in range(0,len(RawRadiation),1)])
	Radiation = np.array([-9999 if TAU[ii]>1 else Radiation[ii] for ii in range(0,len(Radiation),1)])

	#--- return
	return Radiation



