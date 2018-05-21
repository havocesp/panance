# Panance
Python 3 Binance API wrapper built over Pandas and CCXT Libraries

* **Author**: Daniel J. Umpierrez
* **Version**: 0.1.6

# Requirements
 * python >= 3.6
 * pandas >= 23.0
 * ccxt

# Changelog

## Version 0.1.6
 * New method _get_aggregated_trades_ (not available in _ccxt_)
 * Added _ccxt.base.errors.RequestTimeout_ exception catch.
 * Decorator implemented for check some method args values.
 * Improved _get_trades_ method by adding a _fromId_ param
 * Fixed _"invalid amount"_ error at _buy_limit_ method
## Version 0.1.5
 * Deleted repeated code for _get_profit_ method
 * Minor fixes
## Version 0.1.4
 * Minor fix
## Version 0.1.3
* Minor fix
## Version 0.1.2
* Minor fix
## Version 0.1.1
 * _market_buy_ method to place market buy orders
 * _market_sell_ method to place market sell orders
 * _limit_buy_ method limit to place limit buy orders
 * _limit_sell_ method to place limit buy orders
## Version 0.1.0
 * Initial commit


