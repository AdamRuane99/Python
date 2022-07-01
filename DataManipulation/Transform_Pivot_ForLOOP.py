# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:21:58 2022

@author:AR
"""
from datetime import datetime
from dateutil.relativedelta import relativedelta
date = "01/06/2022"


date_inp = datetime.strptime(date, '%d/%m/%Y')

date_inp + relativedelta(months=-1)
Year =  date_inp.strftime("%Y")
Month = date_inp.strftime("%m")
Day =   date_inp.strftime("%d")
Month_data = date_inp + relativedelta(months=-1)
Month2 = Month_data.strftime("%B")

import os

UserName =  os.environ.get('USERNAME')

dest = "C:/Users/" + UserName + "/Documents/Adam's File/"

Qloc ='Q:/AR/01 - DroppedData'




from datetime import datetime
time = datetime.now()
zip_file_name = Year  +  Month+ Day +"_MonthlyClaimsAnalysis_"+  Month2

os.mkdir(dest+ zip_file_name)


import zipfile
with zipfile.ZipFile(Qloc + zip_file_name+".zip", 'r') as zip_ref:
    zip_ref.extractall(dest+ zip_file_name)


import os
loc = (dest+ zip_file_name+"/")
files = os.listdir(loc)

import pandas as pd
start =  Year  +  Month+ Day +"_CLAIMS_ANALYSIS_"+  Month2
Gdf = pd.DataFrame()
for filename in files:    
    if filename.startswith(start):
        
        df = pd.read_excel(loc+filename)
        df = df[(df['Amounts/Volumes']=='Amounts')] 
        columns = list(df)
        headers = []
        months = []
        for col in columns:
            if col.startswith('Month'):
                months.append(col)
            else:
                headers.append(col)
        df2 = pd.melt(df,
                  id_vars=headers,
                  value_vars=months,
                  var_name='Date',
                  value_name='Val')

        df2['Date'] = pd.to_datetime(Year + '-0' + df2['Date'].str[6:7] + '-01')
        df2['Date'] = df2['Date'].dt.date
        i = -1
        while( filename[i] != "_"):
            i = i - 1
        i = i +1
        df2['Country'] = filename[i:-5]
        res = [Gdf, df2]
        Gdf = pd.concat(res, axis=0)
        df2.to_excel(loc+filename, index = False)

Gdf.to_csv(dest+"ClaimsDataSet/"+"DataSet.csv",mode = 'w', index = False,header = True)