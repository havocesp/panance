# -*- coding:utf-8 -*-
"""
Binance API wrapper over Pandas lib.
"""
import os
import sys
import warnings

import ccxt
import numpy as np
import pandas as pd

from panance.utils import cnum, is_empty

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

pd.options.display.precision = 8
warnings.filterwarnings(action='ignore', category=FutureWarning)

__all__ = ['Panance']


class Panance(ccxt.binance):
    """
    Binance API wrapper over Pandas lib.
    """
    usd = 'USDT'

    def __init__(self, key=None, secret=None, config=None):
        """
        Constructor.

        :param str key: api key
        :param str secret: api secret key
        :param dict config: ccxt.binance configuration dict
        """
        self._LIMITS = [5, 10, 20, 50, 100, 500, 1000]
        if config is None or not isinstance(config, dict):
            config = dict(verbose=False, enableRateLimit=True, timeout=15000)

        if not 'apiKey' in config or not 'secret' in config:
            if [k for k in os.environ if 'BINANCE_KEY' in k and 'BINANCE_SECRET' in 'k']:
                config.update(apiKey=os.getenv('BINANCE_KEY'), secret=os.getenv('BINANCE_SECRET'))
            elif not is_empty(key) and not is_empty(secret):
                config.update(apiKey=key, secret=secret)

        super().__init__(config=config)

        self.load_time_difference()
        self.markets = self.load_markets()
        self.symbols = [k for k in self.markets if k[-5:] in str('/' + self.usd) or k[-4:] in '/BTC']
        self.currencies = [*{k.split('/')[0] for k in self.symbols}, self.usd]

        self.usd_symbols = [k for k in self.symbols if k[-5:] in str('/' + self.usd)]
        self.usd_currencies = [k.split('/')[0] for k in self.usd_symbols]

    def _check_limit(self, limit):
        if limit and int(limit) in self._LIMITS:
            return int(limit)
        str_limits = ', '.join([*map(str, self._LIMITS)])
        raise ValueError('Invalid limit: {}\nAccepted values: {}'.format(str(limit), str_limits))

    def _get_amount(self, coin, amount):
        coin = self._check(coin)
        if amount and isinstance(amount, str):
            amount = str(amount).lower()

            balance = self.get_balances(coin=coin)
            if amount in 'max':
                percent = 1.0
            elif len(amount) > 1 and amount[-1] in '%':
                percent = float(amount[:-1]) / 100.
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

    def _check(self, v):
        """
        Currency / Symbol validator.

        :param str v: currency or symbol to check
        :return str: currency or symbol
        """
        v = str(v).upper()
        if v in self.symbols or v in self.currencies:
            return v
        else:
            raise ValueError('{} is not a valid currency or symbol'.format(v))

    def _get_price(self, symbol, price):
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

    def _get_since(self, timeframe='15m', limit=100):
        """
        Return number of seconds resulting from:
            timeframe2seconds * limit

        :param str timeframe: accepted values: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 12h, 1d
        :param int limit: limit of timeframes
        :return int: number of seconds for limit and timeframe
        """
        limit = self._check_limit(limit)
        timeframe_mills = self.parse_timeframe(timeframe) * 1000.0
        return int(ccxt.Exchange.milliseconds() - timeframe_mills * limit)

    def get_tickers(self, symbols=None, market=None):
        """
            Get all tickers (use market param to filter result by market).
            Ticker fields:
             timestamp
             datetime
             high
             low
             bid
             bidVolume
             ask
             askVolume
             vwap
             open
             close
             last
             previousClose
             change
             percentage
             average
             baseVolume
             quoteVolume
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

        if symbols:
            raw = self.fetch_tickers(symbols)
        else:
            raw = self.fetch_tickers()

        return pd.DataFrame(raw).drop(['info', 'average', 'symbol'])

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
        :return pd.Series: ticker data for symbol
        """
        symbol = self._check(symbol)
        raw = self.fetch_ticker(symbol)
        del raw['info'], raw['symbol'], raw['average']
        return pd.DataFrame({symbol: raw})[symbol]

    def get_ohlc(self, symbol, timeframe='5m', limit=100):
        """
        Get OHLC data for specific symbol and timeframe.

        :param str symbol: a valid slash separated trade pair
        :param str timeframe: accepted values: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 12h, 1d
        :param int limit: result rows limit
        :return pd.DataFrame: OHLC data for specific symbol and timeframe
        """
        limit = self._check_limit(limit)
        data = self.fetch_ohlcv(self._check(symbol), timeframe=timeframe,
                                since=self._get_since(timeframe=timeframe, limit=limit))
        df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = df['date'] + 3600 * 3 * 1000
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        return df.set_index('date')

    def get_balances(self, coin=None, detailed=False):
        """
        Get balance data

        :param str coin: if set only data for currency "coin" will be returned
        :param bool detailed: if True detailed data will be added to result
        :return pd.DataFrame: balance data
        """
        coin = self._check(coin) if coin else None
        raw = self.fetch_balance()
        [raw.pop(f) for f in ['total', 'used', 'free', 'info'] if f in raw]
        df = pd.DataFrame(raw).T.query('total > 0.0').T

        if detailed:

            tickers = self.get_tickers(symbols=['{}/BTC'.format(c) for c in [*df.keys()]])
            for s in tickers.keys():
                quote, base = s.split('/')
                last = tickers[s]['last']
                df[quote]['total_btc'] = df[quote]['total'] * last
        if coin:
            return df[coin].fillna(0.0) if coin in df else pd.DataFrame()
        else:
            return df.fillna(0.0)

    def get_trades(self, symbol, limit=100, side=None):
        """
        Get last symbol trades.

        :param str symbol: a valid trade pair
        :param int limit: result rows limit
        :param str side: accepted values: "buy", "sell", None
        :return pd.DataFrame: last symbol trades
        """
        limit = self._check_limit(limit)
        raw = self.fetch_trades(self._check(symbol), limit=limit)
        return self._parse_trades(raw, side) if raw else pd.DataFrame()

    def _parse_trades(self, raw, side=None):
        """
        Parse trades data.

        :param list raw: raw data from a trades like query to server
        :param str side: accepted values: "buy", "sell", None
        :return pd.DataFrame: parsed trades data
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

    def get_user_trades(self, symbol, limit=100, side=None):
        """
        Get last user trades for a symbol.

        :param str symbol: a valid trade pair
        :param int limit: result rows limit
        :param str side: accepted values: "buy", "sell", None
        :return pd.DataFrame: last user trades for a symbol
        """
        limit = self._check_limit(limit)
        raw = self.fetch_my_trades(self._check(symbol), limit=limit)
        return self._parse_trades(raw, side) if raw else pd.DataFrame()

    def get_profit(self, coin):
        """
        Returns current profit for a currency and its weighted average buy cost
        :param str coin: a valid currency to use at profit and cost calc
        :return: current profit and weighted average buy cost as a tuple
        """
        coin = self._check(coin)
        btc_symbol = '{}/BTC'.format(coin)
        balance = self.get_balances(coin, detailed=True)
        print(balance)
        if all((balance is not None, not balance.empty)) and balance['btc_total'] > 0.0:

            trades = self.get_user_trades(btc_symbol, 24)  # type: pd.DataFrame
            trades = trades.query('side == "buy"')

            if all((trades is not None, not trades.empty)):
                coin_total = balance.total
                prices, amounts = list(), list()
                for idx, trd in trades.iterrows():
                    if (coin_total - trd['amount']) <= 0.0:
                        prices.append(trd['price'])
                        amounts.append(coin_total)
                        break
                    else:
                        prices.append(trd['price'])
                        amounts.append(trd['amount'])
                        coin_total -= trd['amount']
                real_cost = cnum(np.average(prices, weights=amounts), 8)
                coin_ticker = self.get_ticker(btc_symbol).last
                return cnum((coin_ticker * balance.total) - (real_cost * balance.total), 8), cnum(real_cost, 8)
        else:
            return 0.0, 0.0

    def get_weighted_average_cost(self, symbol):
        """
        Get weighted average buy cost for a symbol.

        :param str symbol: a valid slash separated trade pair
        :return float: weighted average cost (0.0 if currency not in balance)
        """
        symbol = self._check(symbol)
        coin, exchange = symbol.split('/')
        balances = self.get_balances(coin=coin, detailed=True)
        if all((balances is not None, not balances.empty)) and balances['total_btc'] > 0.0:
            last_symbol_user_trades = self.get_user_trades(symbol)
            last_symbol_user_trades.dropna(axis=1, inplace=True)
            if not is_empty(last_symbol_user_trades):
                amounts = list()
                balance = balances['total']

                last_symbol_user_trades = last_symbol_user_trades.sort_values(by='id', ascending=False)
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
            return 0.0

    def get_depth(self, symbol, limit=5):
        """
        Get order book data for a symbol.
        :param str symbol: a valid slash separated trade pair
        :param limit: result rows limit
        :return pd.DataFrame:
        """
        limit = self._check_limit(limit)
        raw = self.fetch_order_book(self._check(symbol), limit=limit)
        data = pd.DataFrame(raw)
        rows = pd.DataFrame(data['asks'] + data['bids'])
        return pd.DataFrame(sum(rows.values.tolist(), []), columns=['ask', 'ask_amount', 'bid', 'bid_amount'])

    def get_asks(self, symbol, limit=10):
        """
        Return asks data from order book for a symbol.

        :param str symbol: a valid slash separated trade pair
        :param int limit: result rows limit
        :return pd.Series: asks data from order book for a symbol
        """
        raw = self.fetch_order_book(self._check(symbol), limit=limit)
        return pd.DataFrame(raw['asks'], columns=['ask', 'amount'])

    def get_bids(self, symbol, limit=10):
        """
        Return bids data from order book for a symbol.

        :param str symbol: a valid slash separated trade pair
        :param int limit: result rows limit
        :return pd.Series: bids data from order book for a symbol
        """
        limit = self._check_limit(limit)
        raw = self.fetch_order_book(self._check(symbol), limit=limit)
        return pd.DataFrame(raw['bids'], columns=['bid', 'amount'])

    def market_buy(self, symbol, amount='max'):
        """
        Place a market buy order.

        :param str symbol: a valid trade pair symbol
        :param str, float amount: quote amount to buy or sell or 'max' get the max amount from balance
        :return dict: order info
        """
        symbol = self._check(symbol)
        return self.create_market_buy_order(symbol, amount=amount)

    def market_sell(self, symbol, amount='max'):
        """
        Place a market sell order.

        :param str symbol: a valid trade pair symbol
        :param str, float amount: quote amount to buy or sell or 'max' get the max amount from balance
        :return dict: order info
        """
        symbol = self._check(symbol)
        return self.create_market_buy_order(symbol, amount=amount)

    def limit_buy(self, symbol, amount='max', price='ask'):
        """
        Place a limit buy order.

        :param str symbol: a valid trade pair symbol
        :param str, float amount: quote amount to buy or sell or 'max' get the max amount from balance
        :param str, float price: valid price or None for a market order
        :return dict: order info
        """
        symbol = self._check(symbol)

        amount = self._get_amount(symbol, amount)
        assert amount is not None and isinstance(amount, float)
        price = self._get_price(symbol, price)
        assert amount is not None and isinstance(amount, float)

        return self.create_limit_buy_order(symbol, amount, price)

    def limit_sell(self, symbol, amount='max', price='bid'):
        """
        Place a limit sell order.

        :param str symbol: a valid trade pair symbol
        :param str, float amount: quote amount to buy or sell or 'max' get the max amount from balance
        :param str, float price: valid price or None for a market order
        :return dict: order info
        """
        symbol = self._check(symbol)
        amount = self._get_amount(symbol, amount)
        assert amount is not None and isinstance(amount, float)
        price = self._get_price(symbol, price)
        assert price is not None and isinstance(price, float)
        return self.create_limit_sell_order(symbol, amount, price)

    get_orderbook = get_depth
    get_book = get_depth
    get_obook = get_depth
