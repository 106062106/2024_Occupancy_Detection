"""
    This module contains a defined dataset class initialized by JSON files
    and each data consists of single record.
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
    lblClm = ["occupancy"]
):
    """
        A JSON_ListDict_Naive dataset configuration template.

        Keyword Arguments
        :dirPth (str) -- the lookup directory path
        :attClm (list[str]) -- attribute column names
        :lblClm (list[str]) -- label column names
    """
    return dataset.json_listdict_naive.JSON_ListDict_Naive(
        dirPth,
        attClm,
        lblClm
    )

class JSON_ListDict_Naive(torch.utils.data.Dataset):
    """
        This class is a dataset initialized by JSON files found in the given directory,
        which expected to contain a dictionary of list.

        The attributes and labels are defined by providing lists of dictionary keys,
        then the elements of same index are concatenated into a list in the requested order.

        Each pair of attribute and label lists then becomes a data entry of the dataset.

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
        dirPth: str,
        attClm: list,
        lblClm: list
    ):
        """
            Initialize the dataset by the JSON files found in the provided directory path,
            which expected to contain a dictionary of list.

            The attributes and labels are defined by providing lists of dictionary keys,
            then the elements of same index are concatenated into a list in the requested order.

            Each pair of attribute and label lists then becomes a data of the dataset.

            Note: Any numeric attribute string is converted into float.

            Note: Any numeric label string is converted into integer.

            Keyword Arguments
            :dirPth (str) -- the lookup directory path
            :attClm (list[str]) -- attribute column names
            :lblClm (list[str]) -- label column names
        """
        self.attLst = []
        self.attClm = attClm

        self.lblLst = []
        self.lblClm = lblClm

        try:
            for curDir, _, filLst in os.walk(dirPth):
                for file in filLst:
                    if (file.endswith(".json")):
                        pth = os.path.join(curDir, file)
                        lstDct = json.read_as_object(pth)

                        attLst, lblLst = dataset.data_from_listdict(
                            lstDct,
                            attClm,
                            lblClm,
                            dataset.try_float,
                            dataset.try_int
                        )

                        self.attLst += attLst
                        self.lblLst += lblLst
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
