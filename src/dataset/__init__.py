"""
    This package contains defined datasets and manipulating methods.
"""

import traceback

import torch.utils.data as data

import src.core as core

from . import json_listdict_naive
from . import json_listdict_lookback

__all__ = [
    "json_listdict_naive",
    "json_listdict_lookback"
]

def try_float(obj: object):
    """
        Converts object into float or retains original value.

        Keyword Argument
        :obj (object) -- converting object
    """
    try:
        obj = float(obj)
    except Exception:
        pass
    return obj

def try_int(obj: object):
    """ 
        Converts object into integer or retains original value.

        Keyword Argument
        :obj (object) -- converting object
    """
    try:
        obj = int(obj)
    except Exception:
        pass
    return obj

def try_float_listlist(lstLst: list):
    """
        Converts all objects in the list of list into float or retains original value.

        Keyword Arguments
        :lstLst (list[list[object]]) -- converting list
    """
    try:
        outLst = []
        for lst in lstLst:
            out = []
            for elm in lst:
                out.append(try_float(elm))
            outLst.append(out)
        return outLst
    except Exception as exc:
        traceback.print_exception(exc)
        return None
    
def unpack(lstLst: list):
    """
        Concatenates the top level list elements.

        Keyword Arguments
        :lstLst (list[list[object]]) -- flattening list
    """
    try:
        outLst = []
        for lst in lstLst:
            outLst += lst
        return outLst
    except Exception as exc:
        traceback.print_exception(exc)
        return None
    
def pack(lstLst: list):
    """
        Turns each elements in the list being a single element list.

        Keyword Arguments
        :lstLst (list[list[object]]) -- unflattening list
    """
    try:
        outLst = []
        for lst in lstLst:
            outLst.append([lst])
        return outLst
    except Exception as exc:
        traceback.print_exception(exc)
        return None

def data_from_listdict(
    lstDct: dict,
    attClm: list,
    lblClm: list,
    attCnv: object,
    lblCnv: object
):
    """
        Extracts data from a dictionary of lists,
        the attributes and labels are defined by the provided lists of dictionary keys,
        then the elements of same index are concatenated into a list in the requested order.

        Note: Each attribute and label column values are converted by the provided the convertion method.

        Note: Expecting each list in the dictionary has the same length.

        Note: Expecting column names to be valid.

        Keyword Arguments
        :lstDct (dict{str:list}) -- dictionary where keys are column names and values are column data
        :attClm (list[str]) -- attribute column names
        :lblClm (list[str]) -- label column names
        :attCnv (object) -- attribute column convertion method
        :lblCnv (object) -- label column convertion method
    """
    try:
        recCnt = len(lstDct[attClm[0]])

        attLst = core.listlist(recCnt)
        lblLst = core.listlist(recCnt)

        # extract attribute data
        for clm in attClm:
            clmLst = lstDct[clm]
            for r in range(recCnt):
                elm = clmLst[r]
                attLst[r].append(attCnv(elm))

        # extract label data
        for clm in lblClm:
            clmLst = lstDct[clm]
            for r in range(recCnt):
                elm = clmLst[r]
                lblLst[r].append(lblCnv(elm))

        return attLst, lblLst
    except Exception as exc:
        traceback.print_exception(exc)
        return None, None

def filter_data(
    fltLst: list,
    asoLst: list,
    fltIdx: int,
    low: float,
    upp: float
):
    """
        Filters and adjusts associated data lists with a numeric range (inclusive).

        Note: Expecting both lists having the same length.

        Keyword Arguments
        :fltLst (list[list[object]]) -- filtering list
        :asoLst (list[list[object]]) -- associated list
        :fltIdx (int) -- filtering column index
        :low (float) -- filtering lower bound (inclusive)
        :upp (float) -- filtering upper bound (inclusive)
    """
    try:        
        # remaining list (of filtering list)
        rmnLst = []
        # adjusting list (of associated list)
        adjLst = []

        datCnt = len(fltLst)
        for d in range(datCnt):
            fltDat = fltLst[d]
            asoDat = asoLst[d]

            val = fltDat[fltIdx]
            if (val < low or val > upp):
                continue
            
            rmnLst.append(fltDat)
            adjLst.append(asoDat)
        
        return rmnLst, adjLst
    except Exception as exc:
        traceback.print_exception(exc)
        return None, None

def split_train_eval(
    datSet: data.Dataset,
    trnRat: float,
    evlRat: float,
    bchSiz: float,
    sflDat: bool,
    drpLst: bool
):
    """
        Splits dataset into training, evaluating datasets and dataloaders.

        Keyword Arguments
        :datSet (Dataset) -- dataset
        :trnRat (float; [0.0, 1.0]) -- dataset training subset ratio
        :evlRat (float; [0.0, 1.0]) -- dataset evaluating subset ratio
        :bchSiz (int) -- batch size
        :sflDat (bool) -- dataloader flag to shuffle data entries
        :drpLst (bool) -- dataloader flag to drop the last batch if incompleted
    """
    try:
        TRAIN_IDX = 0
        TEST_IDX = 1

        # dataset shuffle split
        setLst = data.random_split(datSet, [trnRat, evlRat])
        lodLst = []
        for subset in setLst:
            subLod = data.DataLoader(
                subset,
                batch_size = bchSiz,
                shuffle = sflDat,
                drop_last = drpLst
            )
            lodLst.append(subLod)

        # training, evaluating datasets
        trnSet = setLst[TRAIN_IDX]
        evlSet = setLst[TEST_IDX]
        # training, evaluating dataloaders
        trnLod = lodLst[TRAIN_IDX]
        evlLod = lodLst[TEST_IDX]

        return trnSet, evlSet, trnLod, evlLod
    except Exception as exc:
        traceback.print_exception(exc)
        return None, None, None, None

def dataloader(
    datSet: data.Dataset,
    bchSiz = 64,
    datSfl = True,
    drpLst = True
):
    """
        A dataloader configuration template.

        Keyword Arguments
        :datSet (Dataset) -- loading dataset
        :bchSiz (int) -- batch size
        :datSfl (bool) -- data shuffling flag
        :drpLst (bool) -- drop incomplete batch flag
    """
    try:
        datLod = data.DataLoader(
            datSet,
            batch_size=bchSiz,
            shuffle=datSfl,
            drop_last=drpLst)
        
        return datLod
    except Exception as exc:
        traceback.print_exception(exc)
        return None, None, None, None
