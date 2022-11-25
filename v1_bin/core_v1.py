import numpy as np
import pandas as pd
import warnings
import dataloader as dal

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rcParams.update(mpl.rcParamsDefault)

# https://zhuanlan.zhihu.com/p/480348215
# 一些默认配置，使得图表更美观
large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
# style
plt.style.use('seaborn-whitegrid')
#sns.set_style("white")
# 设置matplotlib正常显示中文
plt.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus']=False 

warnings.filterwarnings('ignore')

tdt = pd.to_datetime

# Backtracking core
class PDcore():
    def __init__(self,slist,dlist,cash=100000):
        self.slist = slist
        self.dlist = dlist

        self.today = self.dlist.min()
        self.position = {}
        self.strat_position = {}
        self.profit = pd.DataFrame(0,index=self.dlist,columns=self.slist)
        self.init_cash = cash
        self.cash = cash
        self.trade = {}
        self.balance = pd.DataFrame(0,index=self.dlist,columns=self.slist)
        self.balance['cash'] = 0

        self.x = 'open'
        
        for s in self.slist:
            self.trade[s]={}
        
    def _run(self,row,amount,verbose = True):
        s = row.code
        d = row.date
        # 记录每日结束现金余量
        if self.today<d:
            self.balance.loc[self.today,'cash'] = self.cash*1.
            self.today = d    

        if amount > 0:
            self.buy(row,amount,verbose=verbose)
            
        elif amount < 0:
            self.sell(row,-amount,verbose=verbose)

    def buy(self,row,amount,verbose = True):
        s = row.code
        d = row.date
        price = getattr(row,self.x)
        if price*amount<=self.cash:
            if s in self.position:
                # 现金变动
                self.cash -= price*amount
                # 记录交易
                self.trade[s][d] = [price,amount]
                # 调整仓位
                value = (self.position[s][2]+price*amount)
                self.position[s] = [
                    value/(amount + self.position[s][1]),
                    amount + self.position[s][1],
                    value
                ]
                if verbose:
                    print('Date %s | Code %s | Amount %s | Add' % (d,s,amount))
            else:
                self.cash -= price*amount
                self.position[s] = [price, amount,price*amount]
                self.trade[s][d] = [price, amount]
                if verbose:
                    print('Date %s | Code %s | Amount %s | New' % (d,s,amount))

    
    def sell(self,row,amount,verbose=True):
        s = row.code
        d = row.date
        price = getattr(row,self.x)
        # 是否有仓位
        if s in self.position:
            #是否有足够库存
            if self.position[s][1]>amount:
                #现金变动
                self.cash += price*amount
                # 记录交易
                self.trade[s][d]= [price,-amount]
                # 调整仓位
                self.position[s] = [
                    self.position[s][0],
                    self.position[s][1]-amount,
                    self.position[s][2]-self.position[s][0]*amount
                ]
                # 计算利润
                self.profit.loc[d,s] = (price-self.position[s][0])*amount
                if verbose:
                    print('Date %s | Code %s | Amount %s | Sell' % (d,s,amount))
            else:
                # 清仓
                self.clear_position(row)
                if verbose:
                    print('Date %s | Code %s | Amount %s | Clear' % (d,s,amount))
                
    
    def clear_position(self, row):
        s = row.code
        d = row.date
        price = getattr(row,self.x)
        #现金变动
        self.cash += price*self.position[s][1]
        # 记录交易
        self.trade[s][d]= [price,-self.position[s][1]]
        # 计算利润
        self.profit.loc[d,s] = (price-self.position[s][0])*self.position[s][1]
        # 清仓
        del self.position[s]

    def end(self,d):
        # cash
        self.balance.loc[d,'cash'] = self.cash*1.
        # profit
        self.stock_profit = self.profit.sum()
        self.date_profit = self.profit.sum(axis=1)
        self.value = self.balance.sum(axis=1)
        self.drt=((self.value-self.value.shift(1).fillna(self.init_cash))/self.value).values
        _ = self.sharpe()

    def plot_value(self):
        self.value.plot()
        self.balance.cash.plot()
        plt.title('Value and cash')
        plt.ylim(bottom = 0)
        plt.show()

    def plot_rt(self):
        cumsum = self.drt.cumsum()
        sup = self.drt[0]
        drawback = []
        for i in cumsum:
            if i>=sup:
                sup = i
                drawback.append(0)
            else:
                drawback.append(sup-i)
        
        fig,ax=plt.subplots(3,1,sharex=True)
        ax[0].plot(self.balance.index,cumsum)
        ax[0].set_ylabel('Cum return')
        ax[1].plot(self.balance.index,-np.array(drawback))
        ax[1].set_ylabel('Drawback')
        ax[2].scatter(self.balance.index,self.drt)
        ax[2].set_ylabel('Daily return')
        ax[2].set_xlabel('Date')
        fig.subplots_adjust(hspace=0.3)
        fig.suptitle(f'Snapshot\n Sharpe ratio:{self.sharpe_ratio}')
        plt.show()

    def date_balance(self,date:str):
        temp = self.balance.loc[pd.to_datetime(date)]
        return temp[temp!=0]

    def sharpe(self,rf=0,period=250):
        riskfree = 0
        if rf!=0:
            print('RF')
            riskfree = np.pow(1+rf,1/period)
        _drt = self.drt-riskfree
        self.sharpe_ratio =  _drt.mean()/_drt.std(ddof=1)*np.sqrt(period)
        return self.sharpe_ratio


# Backtracking cerebro
class PDcerebro(PDcore):
    def __init__(self,data,cash):
        self.data = data.dropna().reset_index().sort_values(by=['date','code'])
        self.slist = self.data.code.unique()
        self.dlist = self.data.date.unique()
        super().__init__(self.slist,self.dlist,cash)
        
        self.strategy = {}
        self.pseduo_position = {}
        
    def run(self,verbose = True):
        if self.strategy:
            # 有策略才运行
            for row in self.data.itertuples():
                s = row.code
                d = row.date

                for name , strat in self.strategy.items():
                    amount = strat.execute(row,self.strat_position[name])
                    self.adjust_strat_position(row,self.strat_position[name],amount)
                    self._run(row,amount,verbose=verbose)

                    #if amount != 0:
                        #print('Date %s | Code %s | Name %s' % (d,s,name))
                        #print(self.position)
                        #print(self.cash)
                        #print(self.balance)

                # 记录仓位价值
                if s in self.position:
                    self.balance.loc[d,s] = row.close*self.position[s][1]
                    

             # 最后一天总结
            self.end(d)
            for _ , strat in self.strategy.items():
                strat.end(d)
                
        else:
            print('No strategy')

    def add_strats(self,strats,name = None):
        strats.init(self.slist,self.dlist,self.cash)

        if name:
            pass
        elif self.strategy:
            name = len(self.strategy)
        else:
            name = 0
            
        self.strategy[name] = strats
    
        self.strat_position[name] = {}

    def adjust_strat_position(self,row,position,amount):
        # control the position of strategies in the cerebro
        s = row.code
        d = row.date
        if s in position:
            if position[s] + amount <=0:
                del position[s]
            else:
                position[s]+= amount
        else:
            if amount >0:
                position[s] = amount

    def date_data(self,code,date):
        # 代码日期截取股票数据
        return self.data[(self.data.date==pd.to_datetime(date))&(self.data.code == code)]

# Backtracking strategy
class PDstrategy(PDcore):
    def __init__(self,verbose = True):
        self.active = True
        self.verbose =verbose

    def init(self, slist, dlist, cash=100000):
        super().__init__(slist, dlist, cash)
        self.active = True

    def cal(self,row):
        pass
    
    def execute(self,row,position):
        s = row.code
        d = row.date
        if self.active:
            # Strategy run
            _amount = self.cal(row,self.position)
            self._run(row,_amount,self.verbose)
            # Record balance at the end of the day
            if s in self.position:
                self.balance.loc[d,s] = row.close*self.position[s][1]
                
        # Cerebro amount
        return self.cal(row,position)

# Sample strategy

class alpha6(PDstrategy):
    def __init__(self,date_trade):
        super().__init__()
        # date-trade dict
        self.date_trade = date_trade
        
    def cal(self,row,position):
        s = row.code
        d = row.date
        if s in position:
            if s in self.date_trade[d]:
                return 0
            else:
                return -100
        else:
            if s in self.date_trade[d]:
                return 100
            else:
                return 0
