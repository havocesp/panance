# -*- coding:utf-8 -*-
from pprint import pprint

import pandas as pd

from panance import Panance

pd.options.display.precision = 8

api = Panance()

book = api.get_depth('ZIL/BTC', limit=10)
pprint(book)
