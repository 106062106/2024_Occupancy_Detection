"""
    This module contains project directory prefixes,
    output formatting and initialization methods.
"""

import os

from typing import IO
from collections.abc import Mapping, Iterable

# log path prefix
LOG = os.path.join("log", "")

# output path prefix
OUT = os.path.join("out", "")

# csv resource path prefix
CSV = os.path.join("res", "csv", "")
# json resource path prefix
JSON = os.path.join("res", "json", "")
# mp4 resource path prefix
MP4 = os.path.join("res", "mp4", "")
# model resource path prefix
MDL = os.path.join("res", "mdl", "")

def tab(msg: str):
    """
        Prepends the given message string with 4 spaces.

        Keyword Arguments
        :msg (str) -- the message string to be prepended
    """
    return "    {msg}".format(msg=msg)

def out(msg: str):
    """
        Tabs and shows the given message string,
        then gives an empty line.

        Keyword Arguments
        :msg (str) -- the message string to be outputted
    """
    print(tab(msg))
    print()

def outs(msgLst: list):
    """
        Tabs and shows the message strings within the given list,
        then gives an empty line.

        Keyword Arguments
        :msgLst (list[str]) -- the message strings to be outputted
    """
    for msg in msgLst:
        print(tab(msg))
    print()

def log(file: IO, msg: str):
    """
        Tabs and writes the given message string to log
        with an appended endline character.

        Keyword Arguments
        :file (IO) -- the file for output log
        :msg (str) -- the message string to be outputted
    """
    file.write("{msg}\n".format(msg=msg))

def show(obj: object):
    """
        Shows the object type and instance value.

        Keyword Arguments
        :obj (object) -- the object to be shown
    """
    msgLst = []
    msgLst.append(type(obj))

    try:
        if (isinstance(obj, str)):
            msgLst.append(obj)
        elif (isinstance(obj, Mapping)):
            for key in obj.keys():
                tup = "{key}: {val}".format(key=key, val=obj[key])
                msgLst.append(tup)
        elif (isinstance(obj, Iterable)):
            for itm in obj:
                msgLst.append(itm)
        else:
            msgLst.append(obj)
    except Exception:
        msgLst.append(obj)

    outs(msgLst)

def listlist(cnt: int):
    """
        Initialize a list of empty lists.

        Keyword Arguments
        :cnt (int) -- number of empty lists
    """
    lstLst = []
    for _ in range(cnt):
        lstLst.append([])
    return lstLst
