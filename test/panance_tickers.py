# -*- coding:utf-8 -*-
from pprint import pprint

import pandas as pd

from panance import Panance

pd.options.display.precision = 8

api = Panance()

pprint(api.get_tickers(symbols=['BTC/USDT', 'ETH/USDT']))
# pprint(api.get_tickers(market='BTC'))
# pprint(api.get_tickers(market='USDT'))
# pprint(api.get_tickers(market='USDT')['BTC/USDT'])
