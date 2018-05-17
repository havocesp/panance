# -*- coding:utf-8 -*-
from collections import Mapping, Iterable
from decimal import Decimal


def _num_parser(value, precision=8):
    """
    Detect value num type (int, float, Decimal) from str or any num type.

    If detection success return it's parsed value (rounded to precision if type in float or Decimal), if not,
    a str(value) will be returned.

    :param float, Decimal, int value: value to parse
    :param int precision:  precision desired (default 8)
    :return float: parsed value (rounded) or str(value) if it is not a parserable num type
    """
    if isinstance(value, Decimal):
        value = float(value)
    strval = str(value)
    try:
        if '.' in strval or 'e' in strval.lower():
            r = round(float(strval), precision)
        else:
            try:
                r = int(strval)
            except ValueError:
                r = round(float(strval), precision)
    except ValueError:
        r = value
    return r


def cnum(data, ndecims=8):
    """
    Data type infer and parser

    Accept any Iterable (dict, list, tuple, set, ...) or built-in data types int, float, str, ... and try  to
    convert it a number data type (int, float)
    """
    if isinstance(data, (str, int, float, Decimal)):
        r = _num_parser(str(data), ndecims)
    elif isinstance(data, Mapping):
        r = {k: cnum(v, ndecims) if isinstance(v, Iterable) else v for k, v in data.items()}
    elif isinstance(data, Iterable):
        r = [cnum(n, ndecims) if isinstance(n, Iterable) else n for n in data]
    else:
        r = _num_parser(str(data), ndecims)
    return r


def is_empty(v):
    """
    Empty variable checker

    If variable is None a False value will be returned also.

    :param dict, list, tuple, set v: variable to check
    :return: returns True if v is empty or None and False if not
    """
    return v is None or not len(v)
