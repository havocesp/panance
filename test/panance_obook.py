# -*- coding:utf-8 -*-
from pprint import pprint

import pandas as pd

from panance import Panance

pd.options.display.precision = 8

api = Panance()

book = api.get_depth('ZIL/BTC', limit=10)
pprint(book)
book = api.get_asks('ZIL/BTC', limit=5)
pprint(book.ask[0])
book = api.get_bids('ZIL/BTC', limit=5)
pprint(book.bid[0])
