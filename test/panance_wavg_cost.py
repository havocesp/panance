# -#-*- coding:utf-8 -*-
import os

import pandas as pd

from panance import Panance

pd.options.display.precision = 8

key = os.getenv('BINANCE_KEY')
secret = os.getenv('BINANCE_SECRET')
api = Panance(key=key, secret=secret)

print(api.get_weighted_average_cost('ZIL/BTC'))
