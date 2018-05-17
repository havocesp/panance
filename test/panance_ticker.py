# -*- coding:utf-8 -*-
from pprint import pprint

import pandas as pd

from panance import Panance

pd.options.display.precision = 8

api = Panance()
ticker = api.get_ticker('ETH/BTC')

pprint(ticker)
