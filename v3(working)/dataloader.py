import akshare as ak
import numpy as np
import pandas as pd
import datetime as dt
import time
import os 
from tqdm import tqdm
import alphalens
from scipy.stats import rankdata

class dataloader():
    def __init__(self):

        # 待选股列表
        self.slist = None
        self.slist_path ='slist.csv'

        #  历史成分股（不全）
        self.slist_hist = None

        # 股票信息
        self.sinfo = None
        self.sinfo_path = 'sinfo.csv'
        
        # 不考虑的股票
        self.filtered_slist = None

        # 股票数据 
        self.sdata = None
        self.sdata_path = 'sdata.csv'

        self.slist_used = None

        # 已加载因子和产生的交易
        self.factor = []
        self.trade = {}

    def get_slist(self,symbol='000300'):
        # 最新中正成分股（包含沪深300）
        if self.slist is not None:
            self.slist_used = self.slist.code.values
            return self.slist
        elif os.path.exists(self.slist_path):
            # local file exist
            self.slist = pd.read_csv(self.slist_path,encoding='gbk',dtype={'code':str})
            self.slist_used = self.slist.code.values
        else:
            # extract info
            self.slist = ak.index_stock_cons_csindex(symbol=symbol).iloc[:,[0,4,5]]
            self.slist.columns = ['date','code','name']
            self.slist.date = pd.to_datetime(self.slist.date)
            self.slist.to_csv(self.slist_path,index=False,encoding='gbk')
            self.slist_used = self.slist.code.values
    
    def get_slist_hist(self,symbol):
        # index_stock_hist is bad function, data is incomplete.
        try:
            if symbol[:3]=='000':
                self.slist_hist = ak.index_stock_hist(symbol='sz'+symbol)
            elif symbol[:3]=='399':
                self.slist_hist = ak.index_stock_hist(symbol='sh'+symbol)
            else:
                return 'Code not accepted'
        except:
            return 'Code not found'

    def get_slist_info(self):
        # 获取股票数据信息
        if self.sinfo is not None:
            return self.sinfo
        elif os.path.exists(self.sinfo_path):
            self.sinfo = pd.read_csv(self.sinfo_path,encoding='gbk',dtype={'股票代码':str})
            self.sinfo['上市时间'] = pd.to_datetime(self.sinfo['上市时间'],format='%Y%m%d')
        else:
            self.sinfo = pd.DataFrame(columns=['总市值', '流通市值', '行业', '上市时间', '股票代码', '股票简称', '总股本', '流通股'])

            # count for sleep
            count = 0
            for i in self.slist.code.unique():
                self.sinfo.loc[len(self.sinfo)] = ak.stock_individual_info_em(symbol=i).value.values
                if count%20 ==0:
                    time.sleep(0.7)
            self.sinfo['上市时间'] = pd.to_datetime(self.sinfo['上市时间'],format='%Y%m%d')
            self.sinfo.to_csv(self.sinfo_path,index=False,encoding='gbk')

            return self.sinfo

    def filter_slist(self,date):
        # 过滤上市在date之后的股票
        if self.sinfo is None:
           _ = self.get_slist_info()

        self.filtered_slist = self.sinfo[self.sinfo['上市时间']>date].股票代码.sort_values().values
        self.slist_used = self.slist_used[~np.isin(self.slist_used,self.filtered_slist)]
        return self.filtered_slist

    def get_sdata(self,start="2021/01/01",end='2021/01/231',adjust='qfq'):
        # 获取股票数据
        start_date=pd.to_datetime(start).strftime('%Y%m%d')
        end_date=pd.to_datetime(end).strftime('%Y%m%d')
        if self.sdata:
            return self.sdata
        elif os.path.exists(self.sdata_path):
            self.sdata = pd.read_csv(self.sdata_path,dtype={'code':str})
            self.sdata.date = pd.to_datetime(self.sdata.date)
            self.sdata.set_index(['date','code'],inplace=True)
            return self.sdata
        else:
            self.sdata=pd.DataFrame(columns=['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率','code'])

            # count for sleep
            count = 0            
            for i in tqdm(self.slist_used):
                temp=ak.stock_zh_a_hist(symbol=i, period="daily", start_date=start_date, end_date=end_date, adjust=adjust)
                temp['code']=i
                self.sdata=self.sdata.append(temp,ignore_index=True)
                if count%20 ==0:
                    time.sleep(0.7)

            self.sdata.drop(['振幅', '涨跌幅', '涨跌额', '换手率'], axis=1 , inplace=True)
            self.sdata.columns = ['date','open','close','high','low','volume','amount','code']
            self.sdata.volume = self.sdata.volume*100
            self.sdata.date = pd.to_datetime(self.sdata.date)
            self.sdata.to_csv(self.sdata_path,index=False,encoding='gbk')
            
            self.sdata.set_index(['date','code'],inplace=True)
            
            return self.sdata

    def get_sdata_tdx(self,file_path,start="2021/01/01",end='2021/01/31'):
        # 通达信数据
        false = []
        if self.sdata is not None:
            return self.sdata
        else:
            self.sdata=pd.DataFrame()
            for i in tqdm(self.slist_used):
                path = file_path+i+'.csv'
                if not os.path.exists(path):
                    false.append(i)
                else:
                    temp = pd.read_csv(file_path+i+'.csv',header=None)
                    temp['code'] = i
                    self.sdata = self.sdata.append(temp)
            self.sdata.columns = ['date','open','high','low','close','volume','amount','code']
            self.sdata.date = pd.to_datetime(self.sdata.date)
            self.sdata = self.sdata[(self.sdata.date>pd.to_datetime(start))&(self.sdata.date<pd.to_datetime(end))].reset_index(drop=True)

            self.sdata.set_index(['date','code'],inplace=True)
            print('No data:',false)
            return self.sdata
            

    def clean_tdx(self,path):
        # 删除数据最后一行
        for i in os.listdir(path):
            temp = pd.read_csv(path+i,encoding='gbk',header=None)
            temp = temp.iloc[:-1,:]
            temp.to_csv(path+i,index=False,header=None)


    def add_factor(self,x,name):
        if name in self.sdata.columns:
            print('Name exist')
        else:
            self.sdata[name] = x
            self.factor.append(name)

    def first_n(self,x,n):
        return x[x.groupby(by='date').rank()<(n+1)]

    def first_n_todict(self,name,n):
        if name in self.factor:        
            self.trade[name] = self.first_n(self.sdata[name],n).reset_index(level=0).groupby('date').groups
            #return self.trade[name]

    def date_trade(self,name,date):
        return self.trade[name][pd.to_datetime(date)]
        

def sma(x,n):
    return x.groupby(by='code').rolling(n).mean()

def ts_sum(x,n):
    return x.groupby(by='code').apply(lambda x: x.rolling(n).sum())

def rank(x):
    return x.groupby(by='date').rank(pct=True)

def ts_rank(x,n):
    # keep multiindex
    return x.groupby('code').rolling(n).apply(lambda x: rankdata(x)[-1]).set_axis(x.index)

def stddev(x,n):
    return x.groupby(by='code').apply(lambda x: x.rolling(n).std())

def covariance(x,y,n):
    return pd.concat([x,y],axis=1).reset_index(level=1).groupby(by='code').rolling(n).cov().unstack().iloc[:,-2].reorder_levels(['date','code'])

def correlation(x,y,n):
    return pd.concat([x,y],axis=1).reset_index(level=1).groupby(by='code').rolling(n).corr().unstack().iloc[:,-2].reorder_levels(['date','code'])

def delta(x,n):
    return x-x.groupby('code').shift(n)

def adv(x,n):
    return x.volume.groupby('code').apply(lambda x: x.rolling(n).mean())

def shift_ndate(x,n):
    return x.groupby(by='code').shift(n)

#==============================================
# Formula
def alpha_6(data):
    return shift_ndate(-1*correlation(data.open,data.volume,10),1)

def alpha_17(data):
    return shift_ndate(-1*rank(ts_rank(data.close,10))*rank(delta(delta(data.close,1),1))*rank(ts_rank(data.volume/adv(data,20),5)),1)

def momentum_nd(data,n):
    return shift_ndate(delta(data.close,n),1)

#==============================================
def alphalens_fullsheet(x):
    if 'factor' in x.columns:
        # The required format for alphalens.
        x.reset_index(inplace=True)
        x.index=pd.to_datetime(x.date)
        x.set_index([x.index,x.code],inplace=True)

        alpha_pri = x.pivot(index='date',columns='code',values='open')
        alpha_pri.index = pd.to_datetime(alpha_pri.index)

        ret = alphalens.utils.get_clean_factor_and_forward_returns(factor=x.factor,prices=alpha_pri)
        alphalens.tears.create_full_tear_sheet(ret)

    return ret