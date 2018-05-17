# -*- coding:utf-8 -*-
from pprint import pprint

import pandas as pd

from panance import Panance

pd.options.display.precision = 8

api = Panance()

trades = api.get_trades('ZIL/BTC', limit=10)
pprint(trades)
buys = api.get_trades('ZIL/BTC', limit=10, side='buy')
pprint(buys)
sells = api.get_trades('ZIL/BTC', limit=10, side='sell')
pprint(sells)
