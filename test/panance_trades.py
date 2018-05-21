# -*- coding:utf-8 -*-
from pprint import pprint
import pandas as pd
import time as tm
import term as trm

from finta import TA
from panance import Panance

pd.options.display.precision = 8

api = Panance()

wLn = trm.writeLine
symbol = 'TRX/BTC'

while True:
    raw = api.get_trades(symbol, limit=100)

    trades = raw.query('amount != 1.0')
    sells = trades.loc[trades.side == "sell"]
    buys = trades.loc[trades.side == "buy"]
    num_sells = len(sells)
    num_buys = len(buys)
    if num_sells > num_buys:
        sells = sells[-num_buys:]
    else:
        buys = buys[-num_sells:]
    trm.clear(), trm.pos(1, 1)

    # print(buys.tail(3))
    sells_index = trades.side == "sell"
    datos_precios = (trades['price'] * sells_index.apply(lambda x: -1.0 if x else 1.0))  # type: pd.Series
    datos_volumen = (trades['cost'] * sells_index.apply(lambda x: -1.0 if x else 1.0))  # type: pd.Series

    buy_cost = pd.DataFrame(buys['cost'].values, columns=['close'], index=buys.index)
    buy_price = pd.DataFrame(sells['price'].values, columns=['close'], index=buys.index)
    sell_cost = pd.DataFrame(sells['cost'].values, columns=['close'], index=sells.index)
    sell_price = pd.DataFrame(sells['price'].values, columns=['close'], index=buys.index)
    # buy_cost.index = buys.index
    # sell_cost.index = sells.index
    # sell_cost['close'] = sells['cost']
    # buy_cost['close'] = buys['cost']
    # wLn(datos_volumen.describe())
    wLn(trm.center(' === {} === ').format(symbol))
    grouped = trades.set_index('datetime').cost.groupby(pd.Grouper(freq='10s'))
    print(grouped.ohlc().tail())
    # print(grouped.sort_index(ascending=False).head(10))
    grp_cumsum = grouped.cumsum()[-6:].diff().dropna()
    print(str(grouped.mean().index[-1]), grouped.mean().diff().sum())
    print(str(grouped.mean().index[-2]), grouped.mean().diff().sum())
    wLn(trm.center('{}').format(str(trades.datetime[-1]).split(' ')[1]))
    wLn(' - Sells   : {:d} [{:9.8f}]'.format(num_sells, sells['cost'].sum()))
    wLn(' - Buy    : {:d} [{:9.8f}]'.format(num_sells, buys['cost'].sum()))
    wLn(' - Balance : {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}'.format(*grp_cumsum.sort_index()))
    wLn(' - Primero : {:.8f} [{:>8}]'.format(trades['price'][0], str(trades.datetime[0]).split(' ')[1]))
    wLn(' - Ultimo  : {:.8f} [{:>8}]'.format(trades['price'][-1], str(trades.datetime[-1]).split(' ')[1]))
    wLn(' - Media   : {:.8f}'.format(trades['cost'][-10:].mean()))
    wLn(' - Std Dev : {:.8f}'.format(trades['cost'][-10:].std()))
    wLn(' - Maximo  : {:.8f}'.format(trades['cost'].max()))
    # wLn(' - Minimo  : {:.8f}'.format(trades['cost'].min()))
    # wLn(' - Mediana : {:.8f}'.format(trades['cost'][-10:].median()))
    wLn(' - Moda    : {:.8f}'.format(trades['cost'][-10:].mode().values[0]))
    # wLn(' - Varianza: {:.12f}'.format(trades['price'].var()))
    tm.sleep(3)
# print('Moda    : {:.8f}'.format(balance.))
# trades['cost'] =
# pprint(trades)
# buys = api.get_trades('ZIL/BTC', limit=10, side='buy')
# pprint(buys)
# sells = api.get_trades('ZIL/BTC', limit=10, side='sell')
# pprint(sells)
