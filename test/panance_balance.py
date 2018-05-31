# -*- coding:utf-8 -*-
import os
from pprint import pprint

import pandas as pd

from panance import Panance

pd.options.display.precision = 8

key = os.getenv('BINANCE_KEY')
secret = os.getenv('BINANCE_SECRET')
api = Panance(key=key, secret=secret)

balance = api.get_balances(detailed='USDT')

pprint(balance)
# pprint(balance['BTC'])
#
# balance = api.get_balances(detailed=True)
# pprint(balance['BTC'])
#
# balance = api.get_balances(coin='BTC')
# pprint(balance)
#
# balance = api.get_balances(coin='BTC', detailed=True)
# pprint(balance)
