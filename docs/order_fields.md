# Order Fields

```json
{
    "id":                "12345-67890:09876/54321", // string
    "datetime":          "2017-08-17 12:42:48.000", // ISO8601 datetime of "timestamp" with milliseconds
    "timestamp":          1502962946216, // order placing/opening Unix timestamp in milliseconds
    "lastTradeTimestamp": 1502962956216, // Unix timestamp of the most recent trade on this order
    "status":     "open",         // "open", "closed", "canceled"
    "symbol":     "ETH/BTC",      // symbol
    "type":       "limit",        // "market", "limit"
    "side":       "buy",          // "buy", "sell"
    "price":       0.06917684,    // float price in quote currency
    "amount":      1.5,           // ordered amount of base currency
    "filled":      1.1,           // filled amount of base currency
    "remaining":   0.4,           // remaining amount to fill
    "cost":        0.076094524,   // "filled" * "price"
    "trades":    [  ],            // a list of order trades/executions
    "fee": {                      // fee info, if available
        "currency": "BTC",        // which currency the fee is (usually quote)
        "cost": 0.0009,           // the fee amount in that currency
        "rate": 0.002,            // the fee rate (if available)
    },
    "info": { ... },              // the original unparsed order structure as is
}
```
