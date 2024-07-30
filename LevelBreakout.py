import logging
import sys
import os

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

pd.options.mode.copy_on_write = True

LOGGING_DEFAULT_LEVEL = logging.INFO


def create_dir_if_not_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)



def get_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("ticker")
    p.add_argument("begin")
    p.add_argument("end")
    
    p.add_argument("-g", "--gap", type=int, help="gap window")
    p.add_argument("-b", "--backcandles", type=int, help="number of bars in zone (backcandles)")
    p.add_argument("-p", "--pwindow", type=int, help="number of bars in pivot window")
    p.add_argument("-n", "--nbars", type=int, help="max no of bars from start")
    p.add_argument("-z", "--zoneheight", type=float, help="the height of the zone")
    p.add_argument("-d", "--distance", type=float, help="distance of stop (as a fraction of close)")
    p.add_argument("-l", "--leverage", type=float, help="the height of the zone")
    p.add_argument("-r", "--pwratio", type=float, help="limit/stop ratio")
    p.add_argument("-s", "--size", type=float, help="allocation to trade [0.0, 1.0]")
    #p.add_argument("-r", "--run", action="store_true", help="normal mode of operation")
    #p.add_argument("-o", "--optimize", action="store_true", help="testing parameter combinations")
    p.add_argument("-save", "--save", action="store_true", help="saving input to csv file")
    p.add_argument("-plot", "--plot", action="store_true", help="showing plot of pivots")
    p.add_argument("-log", "--loglevel", help="loglevel (default=INFO)")

    return p.parse_args() 
    


def get_loglevel(loglevel):
    if not loglevel:
        return LOGGING_DEFAULT_LEVEL
    
    return logging.getLevelName(loglevel.upper())




class Application :
    CONFIG_DIR = 'conf'
    OUTPUT_DIR = 'out'
    LOGGING_DIR = 'log'
    NOW = pd.Timestamp.today().replace(microsecond=0)    

    logger = logging.getLogger()
    ticker = None
   
    
    @staticmethod
    def initialize():
        
        # create directories    
        create_dir_if_not_exists(Application.OUTPUT_DIR)
        create_dir_if_not_exists(Application.LOGGING_DIR)
        
        # create log file
        logging.basicConfig(
            filename=f'{Application.LOGGING_DIR}/{__name__}{Application.NOW}.log',
            format='%(levelname)s %(name)s: %(asctime)s %(message)s',
            filemode='w')       

        Application.logger.setLevel('INFO')
               
    
    @staticmethod
    def load_data(ticker, b, e):  
        
        if ticker in ('H', 'D', 'h', 'd'): # test data, ignoring b, e
            data = pd.read_csv(f'data/eurusd_{ticker.lower()}.csv')
            data = data.rename(columns={"Gmt time":"Time"})
            data.Time = pd.to_datetime(data.Time, format="%d.%m.%Y %H:%M:%S.%f")
            if ticker in ('H', 'h'):
                data = data[data['Volume'] != 0]
                data.reset_index(inplace=True, drop=True)
         
        else:                   
            data = yf.download(ticker, start=b, end=e)
            
            # create numeric index, keep Date column             
            data.reset_index(inplace=True, drop=False)  
            if "Date" in data.columns:  
                data = data.rename(columns={"Date":"Time"})
               
        if len(data) == 0:
            raise Exception(f'No data for ticker: {ticker}')       
        return data
 
    
    @staticmethod
    def save_snapshot(data):
        data.to_csv(f'{Application.OUTPUT_DIR}/{Application.NOW.strftime("%Y%m%d-%H:%M:%S")}_{Application.ticker}_snapshot.csv', sep='\t')
    
    
    @staticmethod
    def get_filename(name):
        return f'{Application.OUTPUT_DIR}/{Application.NOW.date().strftime("%Y%m%d")}_{Application.ticker}_{name}_results.csv'
    

 
RESISTANCE  = 1
SUPPORT     = 2


def _is_pivot_candle(candle_idx, window):
    """
    function that detects if a candle is a pivot/fractal point
    args: candle index, window before and after candle to test if pivot
    returns: 1 if pivot high, 2 if pivot low, 3 if both and 0 default
    """
    if candle_idx < window or (candle_idx + window) >= len(df):
        return 0
    
    r = range(candle_idx-window, candle_idx+window+1)
    
    pivot_low = SUPPORT 
    for i in r:
        if df.iloc[candle_idx].Low > df.iloc[i].Low:
            pivot_low = 0
            break
            
    pivot_high = RESISTANCE
    for i in r:           
        if df.iloc[candle_idx].High < df.iloc[i].High:
            pivot_high = 0
            break
            
    return pivot_high | pivot_low        
 
def is_pivot(df, window):
    return df.apply(lambda x: _is_pivot_candle(x.name, window), axis=1)



def is_trend(df, backcandles=10):    
    ema_signal = [0]*len(df)

    for row in range(backcandles, len(df)):
        upt = 2 # 0x0010
        dnt = 1 # 0x0001
        
        r = range(row-backcandles, row+1)
        for i in r:
            if max(df.Open[i], df.Close[i]) > df.EMA[i]:
                dnt = 0
                break
            
        for i in r:               
            if min(df.Open[i], df.Close[i]) < df.EMA[i]:
                upt = 0
                break
                
        ema_signal[row] = upt | dnt
        
    return ema_signal




SELL = 1
BUY  = 2


#
# used in determine_signal
#
def _check_breakout(f_breakout_test, values, nbounces, zone_height, cclose):
    
    # helper functions
    def _is_zone(values, mean, height):
        for values in values:
            if abs(values-mean) > height:
                return False   
        return True
  

    # main function body    
    if len(values) == nbounces:  
        mean = values.mean()
        if _is_zone(values, mean, zone_height):
            return f_breakout_test(mean, cclose, zone_height)
       
    return 0    # not a break



#
#   determine buy/sell signal 
#
#   0 - no signal
#   1 - sell
#   2 - buy
#
def determine_signal(candle_idx, backcandles, gap_window, zone_height, nbounces=3): 
     
    begin = candle_idx - backcandles - gap_window
    end   = candle_idx - gap_window   
    
    if begin < 0 or candle_idx + gap_window >= len(df):
        return 0
    
    # window must be greater than pivot window to avoid look ahead bias 
    localdf = df.iloc[begin:end] 
    # current close
    cclose = df.loc[candle_idx].Close
      
    # check if current close breaks above resistance level                       
    signal = _check_breakout(
                lambda mean, cclose, height: (BUY if (cclose - mean) > height * 2 else 0),
                localdf[localdf['pivot'] == RESISTANCE].High.tail(nbounces).values, 
                nbounces, 
                zone_height, 
                cclose
            )
    
    if signal != 0:  
        return signal
    
    # check if current close breaks below support level
    signal = _check_breakout(                          
                lambda mean, cclose, height: (SELL if (mean - cclose) > height * 2 else 0),                      
                localdf[localdf['pivot'] == SUPPORT].Low.tail(nbounces).values, 
                nbounces, 
                zone_height, 
                cclose
            )    

    return signal




from backtesting import Strategy
from backtesting import Backtest

 
def SIGNAL():
    return data.signal


class BreakoutStrategy(Strategy):
   
    pos_sz = .7         # position size
    tp_sl_ratio = 2     # profit/loss ratio
    distance = 0.03     # distance 3% below or above current price depending on direction
    
    
    def init(self):        
        super().init()
                
        if self.data == None:
            raise Exception('Set BreakoutStrategy.data to the appropriate DataFrame')
        
        self.signal = self.I(SIGNAL)
        
 
    def get_signal(self):
        return self.signal[-1]
 
    def next(self):
        super().next()        

        #if len(self.trades) == 1:
        #    for trade in self.trades:
        #        trade.sl = trade.entry_price

        if len(self.trades) == 0:
            close = self.data.Close[-1]
            s_fac = self.distance
            r     = self.tp_sl_ratio
            
            match self.get_signal():
                
                case 1: # SELL
                    stop1  = close * (1.0 + s_fac)
                    limit1 = close * (1.0 - r * s_fac) 
                    self.sell(sl=stop1, tp=limit1, size=self.pos_sz)
                    
                case 2: # BUY
                    stop1  = close * (1.0 - s_fac)
                    limit1 = close * (1.0 + r * s_fac)
                                        
                    self.buy(sl=stop1, tp=limit1, size=self.pos_sz)
                
                case 0:
                    pass
                case _:
                    raise ValueError(f'Signal: {self.get_signal()} is invalid. Should be either 0, 1, or 2')
                          
 


def show_pivot_plot(dfpl):
    
    def _ypos(x):
        if x.pivot == SUPPORT:
            return x.Low - 1e-3
        elif x.pivot == RESISTANCE:
            return x.High + 1e-3
        else:
            return np.nan



    dfpl['delta-y'] = dfpl.apply(lambda row: _ypos(row), axis=1)

    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                open=dfpl['Open'],
                high=dfpl['High'],
                low=dfpl['Low'],
                close=dfpl['Close'])])

    fig.add_scatter(
        x=dfpl.index, 
        y=dfpl['delta-y'], 
        mode="markers",
        marker=dict(size=5, color="MediumPurple"),
        name="pivot")

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()

 

def usage():
    print("USAGE: ticker start_date end_date [config.json]")
                

#
# MAIN FUNDTION
#
import time

if __name__ == '__main__':
    #
    # initialize the application
    #
    args  = get_args()  
    ticker = args.ticker.upper()  
    
   
    #
    # cmd line args: ticker from_date to_date [config.json]     
    # 

          
    try:
        if ticker in ('H', 'D'): 
            Application.ticker = 'USDEUR'
            b = 0
            e = 0      
            # ignore begin and end with test data
        else:             
            # mandatory 'begin' and 'end' for downloads                                      
            b = datetime.strptime(args.begin, '%Y%m%d').date()
            e = datetime.strptime(args.end, '%Y%m%d').date()
            Application.ticker = ticker        
        
        Application.initialize()        
        Application.logger.info("START")    
               
        Application.logger.info("\n")
        Application.logger.info(f'{ticker} : [{b}, {e}]\n')
        
        # load data into pandas DataFrame
        print("Loading data...")
        df = Application.load_data(ticker, b, e) 

        #
        # parameters
        #
        pv_window   = args.pwindow if args.pwindow else 6         
        backcandles = args.backcandles if args.backcandles else 40      
        gap_window  = args.gap if args.gap else pv_window+1
        zone_height = 0.001 if ticker == 'H' else 0.01
        zone_height = args.zoneheight if args.zoneheight else zone_height
        nbars       = args.nbars       
      
        Application.logger.info(
            f'(backcandles {backcandles}), (pv window {pv_window}), (gap window {gap_window}), (zone height {zone_height}), (number of bars {nbars}), (leverage {args.leverage}), (size {args.size})'            
        )
                 
        df = df[0:nbars] if nbars else df[0:]         
     
        if df.empty:
            Application.logger("Dataset is empty")
            print("Dataset is empty")
            sys.exit(0)
            
        # save snapshot
        if args.save:
            Application.save_snapshot(df)    
    
        # generating pivots
        print("Calculating pivots...")
        start = time.time()
        df['pivot'] = is_pivot(df, window=pv_window)            
        df[df['pivot']!=0].to_csv(f"out/{Application.ticker.lower()}-{ticker.lower()}-pivots.csv")
        print (f'time in s.: {(time.time() - start):6.1f}')
        
        # generating signals
        print("Calculating signals...")      
        start = time.time()        
        df['signal'] = df.apply(lambda row: 
                                    determine_signal(
                                        candle_idx=row.name, 
                                        backcandles=backcandles, 
                                        gap_window=gap_window, 
                                        zone_height=zone_height
                                    ), 
                                axis=1
                                )
        print (f'time in s.: {(time.time() - start):6.1f}')
       

        # storing signals
        print("Storing signals...")
        df[df['signal'] !=0].to_csv(f"out/{Application.ticker.lower()}-{ticker.lower()}-signals.csv")
        
        # other indicators   
        #df['atr'] = ta.atr(df.High, df.Low, df.Close)
        #df['rsi'] = ta.rsi(df.Close)    
        #df['ema'] = ta.ema(df.Close, length=100)
        
        #df['ema_signal'] = is_trend(df, backcandles=10)    
                            
        #show_pivot_plot(df[4300:4600])        
    
        # input for run
        data = df               
        cash = 10_000
        margin = 1.0/args.leverage if args.leverage else 1.0
        p_size = args.size if args.size else 0.7
        tp_sl_ratio = args.pwratio if args.pwratio else 2.0
        distance = args.distance if args.distance else 0.03
    
        
        Application.logger.info(
            f'(cash {cash}), (margin {margin}), (position size {p_size})'            
        )
       
        
        # run
        data.set_index('Time', inplace=True)
        bt = Backtest(data, BreakoutStrategy, cash=cash, margin=margin)
        stat = bt.run(pos_sz = p_size, tp_sl_ratio = tp_sl_ratio, distance = distance)
        
        print(stat)

        #bt.plot()
            
        
     
    except Exception as e:
        print(f'That\'s an error: {e}')
        Application.logger.error(e)
    finally:
        Application.logger.info("DONE")
        
        