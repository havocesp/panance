# -*- coding:utf-8 -*-
import os
from pprint import pprint

import pandas as pd

from panance import Panance

pd.options.display.precision = 8

key = os.getenv('BINANCE_KEY')
secret = os.getenv('BINANCE_SECRET')
api = Panance(key=key, secret=secret)

trades = api.get_user_trades('ZIL/BTC', limit=10)
pprint(trades)
buys = api.get_user_trades('ZIL/BTC', limit=10, side='buy')
pprint(buys)
sells = api.get_user_trades('ZIL/BTC', limit=10, side='sell')
pprint(sells)
