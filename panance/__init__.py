# -*- coding:utf-8 -*-
"""
Binance API wrapper over Pandas lib.
"""
import inspect
import os
import sys
import time as tm
import warnings
from collections import Iterable
from functools import partial

import ccxt
import numpy as np
import pandas as pd
import requests as req
from ccxt.base import errors as apierr
from decorator import decorator

from panance.utils import cnum, is_empty

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

pd.options.display.precision = 8
warnings.filterwarnings(action='ignore', category=FutureWarning)

__version__ = '0.1.6'
__author__ = 'Daniel J. Umpierrez'
__license__ = 'UNLICENSE'
__package__ = 'panance'
__description__ = 'Python 3 Binance API wrapper built over Pandas Library'
__site__ = 'https://github.com/havocesp/panance'
__email__ = 'umpierrez@pm.me'
__requirements__ = ['ccxt', 'pandas', 'numpy', 'requests', 'decorator']
__all__ = ['Panance', '__package__', '__version__', '__author__', '__site__',
           '__description__', '__email__', '__requirements__', '__license__']

_LIMITS = [5, 10, 20, 50, 100, 500, 1000]


@decorator
def checker(fn, *args, **kwargs):
    """
    Param validator decorator.

    :param fn: reference to caller class instance
    :param args: method call args
    :param kwargs: method class kwargs
    :return:
    """
    args = [v for v in args]
    self = args.pop(0)  # type: ccxt.binance

    try:
        sig = inspect.signature(fn)
    except Exception as err:
        print(str(err))
        return None

    param_names = [p for p in sig.parameters.keys()]
    detected_params = [f for f in ['currency', 'limit', 'coin', 'symbol', 'symbols'] if f in param_names]
    if len(detected_params):

        def get_value(_v):
            value = kwargs.get(_v)
            if _v in param_names and value is None:
                arg_position = param_names.index(_v)
                value = args[arg_position - 1]
            return value

        for dp in detected_params:
            param_value = get_value(dp)
            if param_value is None:
                continue
            if 'limit' in dp and not str(param_value) in [l for l in map(str, _LIMITS)]:
                str_limits = ','.join([l for l in map(str, _LIMITS)])
                raise ValueError('Invalid limit: {}\nAccepted values: {}'.format(str(param_value), str_limits))
            elif dp in ['currency', 'coin', 'symbol', 'symbols']:
                if 'symbols' not in dp and not isinstance(dp, Iterable):
                    param_value = [param_value]
                symbol_list = [str(s).upper() for s in param_value]

                if self.symbols is None or not len(self.symbols):
                    self.load_markets(True)

                if not all([any((s not in self.currencies, s not in self.symbols)) for s in symbol_list]):
                    raise ValueError(
                            'There is a not a valid currency or symbol in function params: {}'.format(symbol_list))

    return fn(self, *args, **kwargs)


class Panance(ccxt.binance):
    """
    Binance API wrapper over Pandas lib.
    """
    usd = 'USDT'

    def __init__(self, key=None, secret=None, config=None):
        """
        Constructor.

        :param str key: user account Binance api key
        :param str secret: user account Binance secret key
        :param dict config: ccxt.binance configuration dict
        """

        if config is None or not isinstance(config, dict):
            config = dict(verbose=False, enableRateLimit=True, timeout=15000)

        if 'apiKey' not in config or 'secret' not in config:
            if [k for k in os.environ if 'BINANCE_KEY' in k and 'BINANCE_SECRET' in 'k']:
                config.update(apiKey=os.getenv('BINANCE_KEY'), secret=os.getenv('BINANCE_SECRET'))
            elif not is_empty(key) and not is_empty(secret):
                config.update(apiKey=key, secret=secret)

        super(Panance, self).__init__(config=config)

        self.load_time_difference()
        self.markets = self.load_markets()
        self.symbols = [k for k in self.markets if k[-5:] in str('/' + self.usd) or k[-4:] in '/BTC']
        self.currencies = [s for s in {k.split('/')[0] for k in self.symbols}]
        self.currencies.append(self.usd)

        self.usd_symbols = [k for k in self.symbols if k[-5:] in str('/' + self.usd)]
        self.usd_currencies = [k.split('/')[0] for k in self.usd_symbols]

    @checker
    def _get_amount(self, coin, amount):
        """
        Get coin amount.

        Amount should be a float / int or an string value like "max" or a percentage like "10%",

        :param coin: the coin where amount will be returned.
        :param amount: a float or int with price, "max" word or a percentage like "10%"
        :type amount: str pr float or int
        :return float: amount as countable item, this is as a float instance
        """

        if amount and isinstance(amount, str):
            amount = str(amount).lower()

            balance = self.get_balances(coin=coin)
            if amount in 'max':
                percent = 1.0
            elif len(amount) > 1 and amount[-1] in '%':
                percent = float(amount[:-1])
                percent /= 100.0
            else:
                raise ValueError('Invalid amount.')

            if all((balance is not None, not balance.empty)):
                amount = balance['total'] * percent
            else:
                raise ValueError('Not enough balance for {} currency.'.format(coin))

        if amount and isinstance(amount, float):
            amount = round(amount, 8)
        else:
            raise ValueError('Invalid amount.')
        return amount

    @checker
    def _get_price(self, symbol, price):
        """
        Get price for a symbol.

        If price contains "ask" or "bid", it's value will be retrieve from order book ask or bid entries.

        :param symbol: slash sep formatted pair (example: BTC/USDT)
        :param price: a float or int with price, "ask" or "bid"
        :type price: str pr float or int
        :return:
        """
        if price is not None:
            if str(price).lower() in ['ask', 'bid']:
                field = str(price).lower()
                return self.get_depth(symbol, limit=5)[field][0]
            elif isinstance(price, float):
                return round(price, 8)
            else:
                raise ValueError('Invalid price')
        else:
            raise ValueError('Invalid price')

    @checker
    def _get_since(self, timeframe='15m', limit=100):
        """
        Return number of seconds resulting by doing:
        >>> self.parse_timeframe(timeframe) * limit

        :param str timeframe: accepted values: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 12h, 1d
        :param int limit: limit of timeframes
        :return int: number of seconds for limit and timeframe
        """

        timeframe_mills = self.parse_timeframe(timeframe) * 1000.0
        return int(ccxt.Exchange.milliseconds() - timeframe_mills * limit)

    @checker
    def get_tickers(self, symbols=None, market=None):
        """
        Get all tickers (use market param to filter result by market).


        :param list symbols: list of trade pairs
        :param str market: accepted values: BTC, USDT
        :return pd.DataFrame: ticker data filtered by market (if set)
        """

        market = str(market).upper() if market and market in ['BTC', self.usd] else None
        if market is None and symbols is not None:
            symbols = [str(s).upper() for s in symbols if s in self.symbols]
        elif market is not None and symbols is None:
            symbols = [s for s in self.symbols if s.split('/')[1] in market]
        else:
            symbols = None

        try:
            if symbols:
                raw = self.fetch_tickers(symbols)
            else:
                raw = self.fetch_tickers()
        except (apierr.RequestTimeout, apierr.DDoSProtection, apierr.InvalidNonce) as err:
            print(str(err))
            return None

        columns = [k for k in [k for k in raw.values()][0].keys()]
        transposed = zip(k for k in [v.values() for v in raw.values()])
        dict_data = dict(zip(columns, transposed))
        del dict_data['info'], dict_data['average'], dict_data['timestamp'], dict_data['datetime']
        df = pd.DataFrame(dict_data).dropna(axis=1)
        df = df.round(8).set_index('symbol')

        if (df.ask < 10.0).all():
            df = df.round(dict(bidVolume=3, askVolume=3, baseVolume=0, percentage=2, quoteVolume=2))
        return df.sort_values('quoteVolume', ascending=False)

    @checker
    def get_ticker(self, symbol):
        """
            Get ticker for symbol.

            Ticker fields:
                ask                              0.084969
                askVolume                           7.997
                baseVolume                      89046.924
                bid                               0.08493
                bidVolume                           2.301
                change                           0.000385
                close                            0.084969
                datetime         2018-05-17T16:07:50.610Z
                high                               0.0854
                last                             0.084969
                low                               0.08371
                open                             0.084584
                percentage                          0.455
                previousClose                    0.084585
                quoteVolume                     7538.2366
                timestamp                   1526573270061
                vwap                           0.08465466

        :param str symbol: slash sep formatted pair (example: BTC/USDT)
        :return pd.Series: ticker data for symbol.
        """

        try:
            raw = self.fetch_ticker(symbol)
        except (apierr.RequestTimeout,) as err:
            print(str(err))
            return None

        del raw['info'], raw['symbol'], raw['average']
        return pd.DataFrame({symbol: raw})[symbol]

    @checker
    def get_ohlc(self, symbol, timeframe='5m', limit=100):
        """
        Get OHLC data for specific symbol and timeframe.

        :param str symbol: a valid slash separated trade pair
        :param str timeframe: accepted values: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 12h, 1d
        :param int limit: result rows limit
        :return pd.DataFrame: OHLC data for specific symbol and timeframe.
        """

        cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        since = self._get_since(timeframe=timeframe, limit=limit)
        try:
            data = self.fetch_ohlcv(symbol, timeframe=timeframe, since=since)
        except (apierr.RequestTimeout, apierr.InvalidNonce) as err:
            print(str(err))
            tm.sleep(3)
            return None
        except (apierr.DDoSProtection,) as err:
            print(str(err))
            tm.sleep(15)
            return None

        seconds2datetime = partial(pd.to_datetime, unit='ms')
        date = [seconds2datetime(v.pop(0)).round('1s') for v in data]
        dt_index = pd.DatetimeIndex(date, name='date', tz='Europe/Madrid')

        df = pd.DataFrame(data, columns=cols[1:], index=dt_index)

        return df

    @checker
    def get_balances(self, coin=None, detailed=False):
        """
        Get balance data.

        :param str coin: if set only data for currency "coin" will be returned
        :param detailed: if True detailed data will be added to result
        :type detailed: bool
        :return pd.DataFrame: balance data
        """

        try:
            raw = self.fetch_balance()
        except (apierr.RequestTimeout, apierr.InvalidNonce, apierr.RequestTimeout) as err:
            print(str(err))
            return None

        [raw.pop(f) for f in ['total', 'used', 'free', 'info'] if f in raw]
        df = pd.DataFrame(raw).T.query('total > 0.0').T

        result = pd.DataFrame()

        if detailed:
            symbols = ['BTC/USDT']

            if all((coin is not None, str(coin).upper() in self.currencies, str(coin).upper() not in ['BTC', 'USDT'])):
                symbols.append('{}/BTC'.format(coin))
            else:
                for c in df.keys():
                    if c not in ['BTC', 'USDT']:
                        symbols.append('{}/BTC'.format(c))

            tickers = self.get_tickers(symbols=symbols)
            if tickers is not None:
                tickers = tickers.T
            else:
                print(' - [ERROR] Server return None for ticker data.')
                sys.exit(1)
            btc_usdt_last = tickers['BTC/USDT']['last']

            for s in symbols:
                c, b = s.split('/')
                c_balance = df[c]
                coin_total = c_balance['total']

                if c in ['USDT', 'BTC']:
                    c_balance['total_{}'.format(c.lower())] = coin_total

                    if 'USDT' in c:
                        c_balance['total_btc'] = coin_total / btc_usdt_last
                    else:
                        c_balance['total_usdt'] = btc_usdt_last * c_balance['total_btc']
                else:
                    ticker = tickers['{}/BTC'.format(c)]
                    c_balance['total_btc'] = coin_total * ticker['last']
                    c_balance['total_usdt'] = c_balance['total_btc'] * btc_usdt_last
                result = result.append(c_balance)
        else:
            result = df

        if all((coin is not None, str(coin).upper() in self.currencies, str(coin).upper() in result.T)):
            result = result.T[str(coin).upper()]

        return result.fillna(0.0)

    @checker
    def get_aggregated_trades(self, symbol, from_id=None, start=None, end=None, limit=500):
        """
        Get aggregated trades for a symbol.

        :param str symbol: trade pair
        :param int from_id:	get trades from specific id
        :param int start: unix datetime starting date
        :param int end:	unix datetime  ending date
        :param int limit: row limits, max. 500 (default 500)
        :return pd.DataFrame: aggregated trades as a Pandas DataFrame
        """

        url = 'https://api.binance.com/api/v1/aggTrades?symbol={}'.format(symbol.replace('/', '').upper())
        if from_id and isinstance(from_id, int):
            url += '&fromId={:d}'.format(from_id)
        else:
            if start and isinstance(start, (int, float)):
                start = int(start)
                url += '&startTime={:d}'.format(start)

            if end and isinstance(end, (int, float)):
                end = int(end)
                url += '&startTime={:d}'.format(end)

        if limit != 500:
            url += '&limit={:d}'.format(limit)
        try:
            response = req.get(url)
        except (req.RequestException,) as err:
            print(str(err))
            return None

        if response.ok:
            raw = response.json()
            cols = ['price', 'amount', 'first_id', 'last_id', 'timestamp']
            df = pd.DataFrame([[r['p'], r['q'], r['f'], r['l'], r['T']] for r in raw], columns=cols).dropna(axis=1,
                                                                                                            how='any')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['price'] = df['price'].apply(float)
            df['amount'] = df['amount'].apply(float)
            df['first_id'] = df['first_id'].apply(int)
            df['last_id'] = df['last_id'].apply(int)
            df = df['first_id'][0]
            df.set_index('timestamp', inplace=True)
            grouped = pd.DataFrame()
            grouped['price'] = df.price.groupby(pd.Grouper(freq='1s')).mean().apply(round, args=(8,))
            grouped['amount'] = df.amount.groupby(pd.Grouper(freq='1s')).mean().apply(round, args=(3,))
            grouped['total'] = (grouped['price'] * grouped['amount']).apply(round, args=(8,))
            return grouped.dropna(axis=1, how='all').bfill()
        else:
            response.raise_for_status()

    @checker
    def get_trades(self, symbol, limit=100, side=None, from_id=None):
        """
        Get last symbol trades.

        :param str symbol: a valid trade pair.
        :param int limit: result rows limit.
        :param str side: accepted values: "buy", "sell", None.
        :param int from_id: id where to start data retrieval.
        :return pd.DataFrame: last symbol trades.
        """

        params = dict()
        if from_id and isinstance(from_id, int):
            params = dict(fromId=from_id)
        if len(params):
            raw = self.fetch_trades(symbol, limit=limit, params=params)
        else:
            raw = self.fetch_trades(symbol, limit=limit)
        result = self._parse_trades(raw, side)
        return result

    @staticmethod
    def _parse_trades(raw, side=None):
        """
        Parse trades data.

        :param list raw: raw data from a trades like query to server.
        :param str side: accepted values: "buy", "sell", None.
        :return pd.DataFrame: parsed trades data.
        """
        side = str(side).lower() if side and str(side).lower() in ['buy', 'sell'] else None

        data = [{k: v for k, v in r.items() if k not in ['info', 'type']} for r in raw]
        trades = pd.DataFrame(data)
        ts = trades.pop('timestamp') / 1000
        trades.drop(['symbol', 'datetime'], axis=1, inplace=True)
        trades['datetime'] = pd.to_datetime(ts.apply(int), unit='s')

        fee = trades.pop('fee')

        if fee.any():
            fee_currency = pd.Series(fee.apply(lambda v: v['currency']), index=trades.index.values, name='fee_currency')
            trades['fee_currency'] = fee_currency
            trades['fee_percent'] = trades.T.apply(lambda v: 0.05 if 'BNB' in v['fee_currency'] else 0.1).T
            trades['fee_base'] = trades['fee_percent'] / 100. * trades['cost']
            trades['total'] = trades.T.apply(
                    lambda v: v['cost'] - v['fee_base'] if v['side'] in 'sell' else v['cost'] + v['fee_base']).T

        else:
            trades = trades.drop(['takerOrMaker', 'order'], axis=1)
        if side and side.lower() in ['buy', 'sell']:
            trades = trades.query('side == "{}"'.format(side.lower()))

        return trades.set_index('id')

    @checker
    def get_user_trades(self, symbol, limit=100, side=None):
        """
        Get last user trades for a symbol.

        :param str symbol: a valid trade pair
        :param int limit: result rows limit
        :param str side: accepted values: "buy", "sell", None
        :return pd.DataFrame: last user trades for a symbol
        """

        try:
            raw = self.fetch_my_trades(symbol, limit=limit)
        except (apierr.RequestTimeout,) as err:
            print(str(err))
            return None
        return self._parse_trades(raw=raw, side=side) if raw else pd.DataFrame()

    @checker
    def get_profit(self, coin):
        """
        Returns current profit for a currency and its weighted average buy cost
        :param str coin: a valid currency to use at profit and cost calc
        :return: current profit and weighted average buy cost as a tuple
        """

        if str(coin).upper() not in self.currencies:
            print('[ERROR] {} is not a valid currency.'.format(str(coin).upper()))
            sys.exit(1)
        else:
            coin = str(coin).upper()
        btc_symbol = '{}/BTC'.format(coin)
        balance = self.get_balances(coin=coin, detailed=True)

        if all((balance is not None, not balance.empty, balance['total_btc'] > 0.0)):

            real_cost = self.get_weighted_average_cost(symbol=btc_symbol)
            coin_ticker = self.get_ticker(btc_symbol)['last']
            return cnum((coin_ticker * balance.total) - (real_cost * balance['total']), 8), cnum(real_cost, 8)
        else:
            return 0.0, 0.0

    @checker
    def get_weighted_average_cost(self, symbol):
        """
        Get weighted average buy cost for a symbol.

        :param str symbol: a valid slash separated trade pair
        :return float: weighted average cost (0.0 if currency not in balance)
        """

        quote, base = str(symbol).upper().split('/')

        balances = self.get_balances(coin=quote, detailed=True)

        if all((balances is not None, not balances.empty)):

            if balances['total_btc'] >= 0.001:

                last_symbol_user_trades = self.get_user_trades(symbol, side='buy')
                last_symbol_user_trades.sort_values(by='datetime', ascending=False, inplace=True)

                if not is_empty(last_symbol_user_trades):
                    amounts = list()
                    balance = balances['total']

                    for amount in last_symbol_user_trades.query('side == "buy"').amount:
                        if balance - amount <= 0.0:
                            amounts.append(balance)
                            break
                        else:
                            balance -= amount
                            amounts.append(amount)

                    prices = last_symbol_user_trades.price.values[:len(amounts)]

                    return cnum(np.average(prices, weights=amounts), 8)
                else:
                    print(' - [ERROR] Balance returned by server is not valid.')
            else:
                print(' - [ERROR] Not enough balance returned by server is not valid.')
        else:
            print(' - [ERROR] Balance returned by server is not valid.')
        return -1.0

    @checker
    def get_depth(self, symbol, limit=5, split=False):
        """
        Get order book data for a symbol.

        :param split:
        :type split:
        :param str symbol: a valid slash separated trade pair
        :param int limit: result rows limit
        :return pd.DataFrame: data frame with depth row for a symbol.
        """

        try:
            raw = self.fetch_order_book(symbol, limit=limit)
        except (apierr.RequestTimeout,) as err:
            print(str(err))
            return None

        data = pd.DataFrame(raw)

        if split:
            return data['asks'], data['bids']
        else:
            rows = pd.DataFrame(data['asks'] + data['bids'])
            return pd.DataFrame(sum(rows.values.tolist(), []), columns=['ask', 'ask_amount', 'bid', 'bid_amount'])

    @checker
    def get_asks(self, symbol, limit=10):
        """
        Return asks data from order book for a symbol.

        :param str symbol: a valid slash separated trade pair
        :param int limit: result rows limit
        :return pd.Series: asks data from order book for a symbol
        """
        try:
            raw = self.fetch_order_book(symbol, limit=int(limit))
        except (apierr.RequestTimeout,) as err:
            print(str(err))
            return None

        return pd.DataFrame(raw['asks'], columns=['ask', 'amount'])

    @checker
    def get_bids(self, symbol, limit=10):
        """
        Return bids data from order book for a symbol.

        :param str symbol: a valid slash separated trade pair
        :param int limit: result rows limit
        :return pd.Series: bids data from order book for a symbol
        """

        try:
            raw = self.fetch_order_book(symbol, limit=limit)
        except (apierr.RequestTimeout, apierr.InvalidNonce) as err:
            print(str(err))
            return None

        return pd.DataFrame(raw['bids'], columns=['bid', 'amount'])

    @checker
    def market_buy(self, symbol, amount='max'):
        """
        Place a market buy order.

        :param str symbol: a valid trade pair symbol
        :param str, float amount: quote amount to buy or sell or 'max' get the max amount from balance
        :return dict: order info
        """

        try:
            order_data = self.create_market_buy_order(symbol, amount=amount)
        except (apierr.RequestTimeout, apierr.InvalidNonce, apierr.InsufficientFunds) as err:
            print(str(err))
            return None
        return order_data

    @checker
    def market_sell(self, symbol, amount='max'):
        """
        Place a market sell order.

        :param str symbol: a valid trade pair symbol
        :param str, float amount: quote amount to buy or sell or 'max' get the max amount from balance
        :return dict: order info
        """

        try:
            order_data = self.create_market_buy_order(symbol, amount=amount)
        except (apierr.RequestTimeout, apierr.InsufficientFunds, apierr.InvalidNonce) as err:
            print(str(err))
            return None
        return order_data

    @checker
    def limit_buy(self, symbol, amount='max', price='ask'):
        """
        Place a limit buy order.

        :param str symbol: a valid trade pair symbol
        :param str, float amount: quote amount to buy or sell or 'max' get the max amount from balance
        :param str, float price: valid price or None for a market order
        :return dict: order info
        """
        base, quote = symbol.split('/')
        amount = self._get_amount(quote, amount)
        price = self._get_price(symbol, price)

        try:
            order_data = self.create_limit_buy_order(symbol, amount, price)
        except (apierr.RequestTimeout,) as err:
            print(str(err))
            return None
        return order_data

    @checker
    def limit_sell(self, symbol, amount='max', price='bid'):
        """
        Place a limit sell order.

        :param str symbol: a valid trade pair symbol
        :param str, float amount: quote amount to buy or sell or 'max' get the max amount from balance
        :param str, float price: valid price or None for a market order
        :return dict: order info
        """
        base, quote = symbol.split('/')
        amount = self._get_amount(base, amount)
        price = self._get_price(symbol, price)

        try:
            order_data = self.create_limit_sell_order(symbol, amount, price)
        except (apierr.RequestTimeout,) as err:
            print(str(err))
            return None
        return order_data

    @checker
    def download_trade_history(self, symbol, limit=500, start=None, end=None, from_id=None):
        """
        FIXIT not full implemented

        :param symbol:
        :param limit:
        :param start:
        :param end:
        :param from_id:
        """

        if from_id:
            from_id = from_id
        start = int(start) if start else int(tm.time() * 1000.0)
        end = int(end) if end else int(tm.time() * 1000.0)

        trades = self.get_aggregated_trades(symbol, from_id, start, end, limit)  # type: pd.DataFrame
        if not trades.empty:
            filename = '{}_trades.csv'.format(symbol.lower())
            df = pd.read_csv(filename)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            d = pd.concat([df, trades]).drop_duplicates()
            if not d.empty:
                d.to_csv(filename, index_label='date', mode='w', header=True)

    get_orderbook = get_depth
    get_book = get_depth
    get_obook = get_depth


if __name__ == '__main__':
    api = Panance()
