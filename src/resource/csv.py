"""
    This module contains methods to manipulate csv files.
"""

import csv
import traceback

def read_as_list(pth: str):
    """
        Reads the csv file records into a list.

        Keyword Arguments
        :pth (str) -- the source csv file path
    """
    try:
        with open(pth, newline='') as file:
            rdr = csv.reader(file)

            recLst = []
            for row in rdr:
                recLst.append(row)
            return recLst
    except Exception as exc:
        traceback.print_exception(exc)
        return None
    
def read_as_dict(pth: str):
    """
        Reads the csv file records into a dictionary of lists with header columns as keys.

        Note: Assumes the first row being the header.

        Keyword Arguments
        :pth (str) -- the source csv file path
    """
    try:
        recLst = read_as_list(pth)

        hdrLst = recLst[0]
        hdrCnt = len(hdrLst)
        recLst = recLst[1:]

        lstDct = {}
        for c in range(hdrCnt):
            clmLst = []
            for rec in recLst:
                clmLst.append(rec[c])

            lstDct[hdrLst[c]] = clmLst

        return lstDct
    except Exception as exc:
        traceback.print_exception(exc)
        return None

def write_by_list(pth: str, recLst: list):
    """
        Writes given record list into the csv file.

        Note: Record list should be a 2D-list

        Keyword Arguments
        :pth (str) -- the destination csv file path
        :recLst (list[list[object]]) -- record list to be written
    """
    try:
        with open(pth, mode = "w", newline = "") as file:
            wtr = csv.writer(file)
            wtr.writerows(recLst)
    except Exception as exc:
        traceback.print_exception(exc)
