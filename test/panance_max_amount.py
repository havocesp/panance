# -*- coding:utf-8 -*-
import os

import pandas as pd

from panance import Panance

pd.options.display.precision = 8

key = os.getenv('BINANCE_KEY')
secret = os.getenv('BINANCE_SECRET')
api = Panance(key=key, secret=secret)
print(api._get_amount('BTC', 'max'))
print(api._get_amount('BTC', '50%'))
print(api._get_amount('BTC', .01))
print(api._get_price('BTC/USDT', 'bid'))
print(api._get_price('BTC/USDT', 'ask'))
print(api._get_price('BTC/USDT', .01))
