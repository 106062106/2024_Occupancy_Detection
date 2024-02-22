"""
    This module contains a defined dataset class initialized by JSON files
    and each data could consists of multiple records.
"""

import os
import traceback

import torch

import src.core as core
import src.dataset as dataset
import src.resource.json as json

def set_scores(
    dirPth = core.JSON,
    attClm = ["motion_score", "micro_motion_score"],
    lblClm = ["occupancy"],
    bckCnt = 4
):
    """
        A JSON_ListDict_Lookback dataset configuration template.

        Keyword Arguments
        :dirPth (str) -- the lookup directory path
        :attClm (list[str]) -- attribute column names
        :lblClm (list[str]) -- label column names
        :bckCnt (int) -- lookback record count
    """
    return dataset.json_listdict_lookback.JSON_ListDict_Lookback(
        dirPth,
        attClm,
        lblClm,
        bckCnt
    )

class JSON_ListDict_Lookback(torch.utils.data.Dataset):
    """
        This class is a dataset initialized by JSON files found in the given directory,
        which expected to contain a dictionary of list.

        The attributes and labels are defined by providing lists of dictionary keys,
        then the elements of same index are concatenated into a list in the requested order.

        For each attribute list, a number of its preceding entries are copy and prepended,
        those has not enough precedents are discarded along with their corresponding label list.
        Then each of the remaining pair of attribute and label lists then becomes a data entry of the dataset.

        Note: Precedence isn't carried over between files.

        Class Attributes
        :attLst (list[list[object]]) -- attribute entries
        :attClm (list[str]) -- attribute column names
        :lblLst (list[list[object]]) -- label entries
        :lblClm (list[str]) -- label column names
    """
    attLst = []
    attClm = []

    lblLst = []
    lblClm = []

    def __init__(
        self,
        dir: str,
        attClm: list,
        lblClm: list,
        bckCnt: int
    ):
        """
            Initialize the dataset by all JSON found
            with the provided lookup directory, data column list, label column list and segment size.

            Note: The json content must be convertible into a dictionary of lists.

            Note: Any numeric string is converted into float.

            Keyword Arguments
            :dir (str) -- string of the json lookup directory
            :attClm (list[str]) -- list of attribute column names
            :lblClm (list[str]) -- list of label column names
            :bckCnt (int) -- attribute entry lookback count
        """
        self.attLst = []
        self.attClm = attClm*(bckCnt + 1)

        self.lblLst = []
        self.lblClm = lblClm

        try:
            elmClm = attClm
            for curDir, _, filLst in os.walk(dir):
                for file in filLst:
                    if (file.endswith(".json")):
                        pth = os.path.join(curDir, file)
                        lstDct = json.read_as_object(pth)

                        elmLst, lblLst = dataset.data_from_listdict(
                            lstDct,
                            elmClm,
                            lblClm,
                            dataset.try_float,
                            dataset.try_int
                        )

                        elmLen = len(elmClm)
                        elmCnt = len(elmLst)
                        if (elmCnt < bckCnt):
                            continue

                        bckBuf = []
                        # filling lookback buffer
                        for b in range(bckCnt):
                            elm = elmLst[b]
                            for e in range(elmLen):
                                bckBuf.append(elm[e])
                        
                        insIdx = bckCnt
                        bufSiz = bckCnt*elmLen
                        # sliding window for data
                        for b in range(bckCnt, elmCnt):
                            elm = elmLst[b]
                            att = bckBuf[insIdx:] + bckBuf[:insIdx] + elm
                            lbl = lblLst[b]
                        
                            self.attLst.append(att)
                            self.lblLst.append(lbl)

                            # update window buffer
                            for e in range(elmLen):
                                if (insIdx == bufSiz):
                                    insIdx = 0

                                bckBuf[insIdx] = elm[e]
                                insIdx += 1
                            
        except Exception as exc:
            traceback.print_exception(exc)

    def __getitem__(self, idx: int):
        """
            Get data at the provided index.

            Keyword Arguments
            :idx (int) -- index
        """
        tup = (
            torch.tensor(self.attLst[idx]),
            torch.tensor(self.lblLst[idx])
        )
        return tup
    
    def __len__(self):
        """
            Get number of data in the dataset.
        """
        return len(self.attLst)
