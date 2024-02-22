"""
    This module contains methods to manipulate json files.
"""

import json
import traceback

def read_as_object(pth: str):
    """
        Reads the objects from the json file.

        Keyword Arguments
        :pth (str) -- the source json file path
    """
    try:
        with open(pth) as file:
            obj = json.load(file)
            return obj
    except Exception as exc:
        traceback.print_exception(exc)
        return None

def write_by_object(pth: str, obj: object):
    """
        Writes given object into the json file.

        Keyword Arguments
        :pth (str) -- the destination json file path
        :obj (object) -- object to be serialized
    """
    try:
        with open(pth, mode = "w", newline = "") as file:
            file.write(json.dumps(obj))
    except Exception as exc:
        traceback.print_exception(exc)
