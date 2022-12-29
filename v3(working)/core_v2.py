import numpy as np
import pandas as pd
import warnings
import dataloader as dal

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

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
plt.style.use('seaborn')
#sns.set_style("white")
# 设置matplotlib正常显示中文
plt.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus']=False 
sns.set_palette('coolwarm')

warnings.filterwarnings('ignore')

tdt = pd.to_datetime


_FLATUI_COLORS = {'yellow':"#fedd78",'blue':"#348dc1" ,'red':"#af4b64",'green': "#4fa487",'purple': "#9b59b6",'grey':"#808080"}

# Backtracking core
class PDcore():
    def __init__(self,slist,dlist,cash = 100000):
        self.slist = slist
        self.dlist = dlist

        self.today = self.dlist.min()
        self.yesterday = self.dlist.min()
        self.position = {}

        self.profit = pd.DataFrame(0,index=self.dlist,columns=self.slist)
        self.init_cash = cash
        self.cash = cash
        self.trade = {}
        self.balance = pd.DataFrame(0,index=self.dlist,columns=self.slist)
        self.balance['cash'] = 0
        self.sell_commission = 0.002

        self.future_row = pd.DataFrame(columns=['date','open','close','high','low','volume','amount','code'])

        self.suspension = {}

        self.verbose = True
        
        self.x = 'open'
        
        self.temp= None

        for s in self.slist:
            self.trade[s]={}
        
    def _run(self,row,amount):
        s = row.code
        d = row.date
        # 记录每日结束现金余量
        if self.today<d:
            # if self.today in [tdt('2019-03-20'),tdt('2019-03-21')]:
            #     print(self.today,'=============================')
            #     print(self.position)
            #     for i in self.position:
            #         print(i,'|',self.balance.loc[self.today,i])
         
            for i in self.position:
                #如果股票x不交易则记录
                if self.balance.loc[self.today,i]==0:
                    self.balance.loc[self.today,i] = self.balance.loc[self.yesterday,i]
                    if i not in self.suspension:
                        count = 0
                        # 找到下一日期
                        nextrow = self.get_next_row(i,self.yesterday)
                        self.temp = nextrow
                        while nextrow.iloc[0,:].date not in self.dlist:
                            nextrow = self.get_next_row(i,nextrow.iloc[0,:].date)
                            print(f'Row not found,{nextrow.iloc[0,:].date} {nextrow.iloc[0,:].code}')
                            if count>10:
                                break
                                
                                
                        # 记录日期
                        self.suspension[i] = [self.yesterday,nextrow.date.iloc[0],getattr(nextrow,self.x).iloc[0]]
                        self.future_row = self.future_row.append(nextrow,ignore_index=True)

            for _row in self.future_row[self.future_row.date==self.today].itertuples():
                if _row.code in self.position:
                    print(f'Deal | {_row.code} |{ _row.date}')    
                    del self.suspension[_row.code]
                    self.clear_position(_row)
                    self.balance.loc[self.today,_row.code]=0
                    print(self.position)
                    # 异常：strategy没strat_position,无法调用。
                    try:
                        for name in self.strat_position:
                            if _row.code in self.strat_position[name]:
                                del self.strat_position[name][_row.code]
                    except:
                        pass
                else:
                    del self.suspension[_row.code]

            self.balance.loc[self.today,'cash'] = self.cash*1.
            self.yesterday = self.today
            self.today = d
            
        if amount > 0:
            self.buy(row,amount)
            
        elif amount < 0:
            self.sell(row,-amount)

    def buy(self,row,amount):
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
                if self.verbose:
                    print('Date %s | Code %s | Amount %s | Add' % (d,s,amount))
            else:
                self.cash -= price*amount
                self.position[s] = [price, amount,price*amount]
                self.trade[s][d] = [price, amount]
                # print
                if self.verbose:
                    print('Date %s | Code %s | Amount %s | New' % (d,s,amount))
        else:
            print('Not enough money|',d,s)

    
    def sell(self,row,amount):
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
                # print
                if self.verbose:
                    print('Date %s | Code %s | Amount %s | Sell' % (d,s,amount))
            else:
                # 清仓
                self.clear_position(row)
                if self.verbose:
                    print('Date %s | Code %s | Amount %s | Clear' % (d,s,amount))
                
    
    def clear_position(self, row):
        s = row.code
        d = row.date
        price = getattr(row,self.x)
        # 现金变动
        self.cash += price*self.position[s][1]
        # 记录交易
        self.trade[s][d]= [price,-self.position[s][1]]
        # 计算利润
        self.profit.loc[d,s] = (price-self.position[s][0])*self.position[s][1]
        # print
        if self.verbose:
            print('Date %s | Code %s | Amount %s | Clear' % (d,s,self.position[s][1]))
        # 清仓
        del self.position[s]

    def end(self,d):
        # cash
        self.balance.loc[d,'cash'] = self.cash*1.
        # profit
        self.stock_profit = self.profit.sum()
        self.date_profit = self.profit.sum(axis=1)
        self.value = self.balance.sum(axis=1)
        self.drt=(self.value-self.value.shift(1).fillna(self.init_cash))/self.value
        _ = self.sharpe()

    def plot_value(self):

        # Adjust ticker
        def formatter(x,pos=None):
            if x<0 or x>len(x_str)-1:
                return ''
            return x_str[int(x)]

        x_int = list(range(len(self.dlist)))
        x_str = pd.Series(self.dlist).apply(lambda x :x.strftime('%Y-%m-%d'))

        ax=plt.gca()
        ax.xaxis.set_major_locator(plt.MultipleLocator(72))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(formatter))
        plt.xticks(rotation=15)

        # plot
        plt.plot(x_int,self.value.values)
        plt.plot(x_int,self.balance.cash.values)
        plt.title('Value and cash')
        plt.ylim(bottom = 0)
        plt.show()

    def plot_rt(self):

        # Adjust ticker
        def formatter(x,pos=None):
            if x<0 or x>len(x_str)-1:
                return ''
            return x_str[int(x)]

        x_int = list(range(len(self.dlist)))
        x_str = pd.Series(self.dlist).apply(lambda x :x.strftime('%Y-%m-%d'))
        # plot

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
        ax[0].plot(x_int,cumsum.values)
        ax[0].set_ylabel('Cum return')
        ax[1].plot(x_int,-np.array(drawback))
        ax[1].set_ylabel('Drawback')
        ax[2].scatter(x_int,self.drt)
        ax[2].set_ylabel('Daily return')
        ax[2].set_xlabel('Date')

        for i in range(3):
            ax[i].xaxis.set_major_locator(plt.MultipleLocator(72))
            ax[i].xaxis.set_major_formatter(ticker.FuncFormatter(formatter))
        plt.xticks(rotation=15)
        fig.subplots_adjust(hspace=0.3)
        fig.suptitle(f'Snapshot\n Sharpe ratio:{self.sharpe_ratio}')
        plt.show()

    def date_balance(self,date:str):
        # Get non zero balance by date
        temp = self.balance.loc[pd.to_datetime(date)]
        return temp[temp!=0]

    def _sharpe(self,rf=0,period=250):
        # From quanstats
        riskfree = 0
        if rf!=0:
            print('RF')
            riskfree = np.power(1+rf,1/period)
        _drt = self.drt-riskfree
        self.sharpe_ratio =  _drt.mean()/_drt.std(ddof=1)*np.sqrt(period)
        return self.sharpe_ratio

    def sharpe(self,rf=0.04,period=250):
        # From jointquant
        annualized_return=np.power(1+(self.value[-1]-self.value[0])/self.value[0] , period/len(self.value))-1
        annualized_volatility = np.sqrt(250/(len(self.value)-1)*(np.sum((self.drt-self.drt.mean())**2)))
        self.sharpe_ratio = annualized_return/annualized_volatility
        return self.sharpe_ratio

    def get_next_row(self,code,date):
        # Get next row by code and date
        temp=pd.read_csv(f'../../tdx/{code}.csv',header=None)
        temp['code'] = code
        temp.columns = ['date','open','high','low','close','volume','amount','code']
        temp.date = pd.to_datetime(temp.date)

        result = temp.loc[temp.date.shift().eq(date)]
        if len(result)==0:
            return temp[temp.date == date]
        elif len(result) == 1:
            return temp.loc[temp.date.shift().eq(date)]
        else:
            raise ValueError('Repeated row!!')
        
# Backtracking cerebro
class PDcerebro(PDcore):
    def __init__(self,data,cash,verbose = True):
        self.data = data.dropna().reset_index().sort_values(by=['date','code'])
        # All stocks
        self.slist = self.data.code.unique()
        # All trading date
        self.dlist = self.data.date.unique()
        self.verbose = verbose
        super().__init__(self.slist,self.dlist,cash)
        
        # The contaniner for strategies
        self.strategy = {}
        self.strat_position = {}

    def run(self):
        if self.strategy:
            # 有策略才运行
            for row in self.data.itertuples():
                s = row.code
                d = row.date
                price = getattr(row,self.x)
                for name , strat in self.strategy.items():
                    amount = strat.execute(row,self.strat_position[name])
                    self.adjust_strategy_position(row,self.strat_position[name],amount)
                    self._run(row,amount)

                # 记录仓位价值
                if s in self.position:
                    self.balance.loc[d,s] = row.open*self.position[s][1]
                    
             # 最后一天总结
            self.end(d)
            for _ , strat in self.strategy.items():
                strat.end(d)
                
        else:
            print('No strategy')

    def add_strategy(self,strats,name = None):
        strats.init(self.slist,self.dlist,self.cash)

        if name:
            pass
        elif self.strategy:
            name = len(self.strategy)
        else:
            name = 0
            
        self.strategy[name] = strats
    
        self.strat_position[name] = {}

        self.strategy[name].code_data = self.code_data

    def adjust_strategy_position(self,row,position,amount):
        # control the position of strategies in the cerebro
        s = row.code
        d = row.date
        if s in position:
            if position[s][1] + amount <=0:
                del position[s]
            else:
                position[s][1]+= amount
        else:
            if amount >0:
                position[s] =[0,amount,0]

    def code_date_data(self,code,date):
        # 代码日期截取股票数据
        return self.data[(self.data.date==pd.to_datetime(date))&(self.data.code == code)]

    def plot_trade(self,code):
        plt.plot(self.data[self.data.code==code].date,self.data[self.data.code==code].open,alpha=0.6)
        for i in self.trade[code]:
            if self.trade[code][i][1]<0:
                plt.scatter(i,self.trade[code][i][0],color = _FLATUI_COLORS['green'],marker='v',alpha=1)
            else:
                plt.scatter(i,self.trade[code][i][0],color = _FLATUI_COLORS['red'],marker='^',alpha=1)
        plt.show()

    def code_data(self,code):
        return self.data[self.data.code==code]

# Backtracking strategy
class PDstrategy(PDcore):
    def __init__(self,verbose = True):
        self.active = True
        self.verbose =verbose
        self.risk = None

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
            amount = self.cal(row,self.position)
            if self.risk:
                # 风控模块(风控下单覆盖策略下单)
                _amount = self.risk.run(row,self.position)
                if _amount < 0:
                    amount = _amount
            self._run(row,amount)
            # Record balance at the end of the day
            if s in self.position:
                self.balance.loc[d,s] = row.close*self.position[s][1]
                
        # Cerebro amount
        return self.cal(row,position)

    def add_risk(self,risk):
        risk.init(slist=self.slist,x=self.x)
        self.risk = risk

    def plot_trade(self,code):
        temp = self.code_data(code)
        plt.plot(temp.date,temp.open,alpha=0.6)
        for i in self.trade[code]:
            if self.trade[code][i][1]<0:
                plt.scatter(i,self.trade[code][i][0],color = _FLATUI_COLORS['green'],marker='v',alpha=1)
            else:
                plt.scatter(i,self.trade[code][i][0],color = _FLATUI_COLORS['red'],marker='^',alpha=1)
        plt.show()
# Sample strategy

class buy_n_stock(PDstrategy):
    def __init__(self,date_trade,n=100):
        super().__init__()
        # date-trade dict
        self.date_trade = date_trade
        self.n = n
        
    def cal(self,row,position):
        s = row.code
        d = row.date
        if s in position:
            if s in self.date_trade[d]:
                return 0
            else:
                return -self.n
        else:
            if s in self.date_trade[d]:
                return self.n
            else:
                return 0

class buy_n_value(PDstrategy):
    def __init__(self,date_trade,buyvalue = 10000):
        super().__init__()
        # date-trade dict
        self.date_trade = date_trade
        self.buyvalue = buyvalue
        
    def cal(self,row,position):
        s = row.code
        d = row.date
        price = getattr(row,self.x)
        if s in position:
            if s in self.date_trade[d]:
                return 0
            else:
                return -position[s][1]
        else:
            if s in self.date_trade[d]:
                return 100*self.buyvalue//(price*100)
            else:
                return 0

class PDrisk():
    def __init__(self):
        self.record = {}
        self.blacklist = {}

        self.max = {}
        self.day = {}
        
    def init(self,slist,x):
        for s in slist:
            self.record[s] = {}
        self.x = x

    def run(self,row, position):
        # keep track of indicators
        s = row.code
        d = row.date
        price = getattr(row,self.x)
        
        # 风控黑名单（持续n天）
        if s in self.blacklist:
            self.blacklist[s] -= 1
            if self.blacklist[s]<0:
                del self.blacklist[s]
            else:
                return -999

        # 记录最大值和持续天数
        if s in position:
            # 如果持仓，则运行self.cal。如果返回0，则正常；小于0斩仓。
            if s in self.day:
                self.max[s] = max(self.max[s],price)
                self.day[s] += 1
            else:
                self.max[s] = max(position[s][0],price)
                self.day[s] = 1
            return self.cal(row,position)
        else:
            # 如果不持仓，清空指标
            if s in self.day:
                del self.max[s]
                del self.day[s]
            return 0

    def cal(self,row,position):
        # Risk control strategy
        return 0

    def execute(self,s,d,price,amount,profit):
        self.record[s][d] = [price,amount,profit]

class Cut(PDrisk):
    def __init__(self,percent=0.10,n=10):
        super().__init__()
        self.percent = percent
        self.n = n
    def cal(self,row,position):
        s = row.code
        d = row.date
        price = getattr(row,self.x)
        if (self.max[s]-price)/self.max[s] >self.percent:
            self.blacklist[s] = self.n
            return -position[s][1]
        else:
            return 0