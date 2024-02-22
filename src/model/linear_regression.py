"""
    This module contains a linear regression model nad its training/testing methods.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.utils.data as data

import src.model as model

import src.core as core
import src.dataset as dataset

class LinearRegression(nn.Module):
    """
        A simple multi-layer linear regression pytorch model.
    """
    def __init__(self, dimIn: int, dimOut: int):
        """
            Initialize model parameters.

            Keyword Arguments
            :dimIn (int) -- integer of input dimension
            :dimOut (int) -- integer of output dimension
        """
        super(LinearRegression, self).__init__()
        self.modLst = nn.ModuleList()

        dimLst = dimIn
        # input layer
        dimMlt = 32
        dimNxt = dimLst*dimMlt

        self.modLst.append(nn.Linear(dimLst, dimNxt))
        dimLst = dimNxt
        # hidden layers
        modLst, dimLst = model.layer_linear_batchnorm1d_relu(dimLst, 8, 1)
        self.modLst += modLst
        # output layer
        self.modLst.append(nn.Linear(dimLst, dimOut))    

    def forward(self, x):
        """
            Forward propagation of the model.

            Keyword Arguments
            :x -- input tensor
        """
        for mod in self.modLst:
            x = mod(x)
        return x
        
def train_scores(
        datSet = dataset.json_listdict_naive.set_scores(),
        trnRat = 0.6,
        evlRat = 0.4,
        mdlPth = os.path.join(core.MDL, "LinearRegression.pt"),
        maxEpc = 256,
        lrnRat = 1e-4,
        bchSiz = 64,
        sflDat = True,
        drpLst = True,
        pltPth = os.path.join(core.LOG, "LinearRegression_loss.png"),
        logPth = os.path.join(core.LOG, "LinearRegression_log.txt")
    ):
    """
        A LinearRegression model training configuration template.

        Keyword Arguments
        :datSet (Dataset) -- assigned dataset
        :trnRat (float; [0.0, 1.0]) -- dataset training subset ratio
        :evlRat (float; [0.0, 1.0]) -- dataset evaluating subset ratio
        :mdlPth (str) -- model file path
        :maxEpc (int) -- max training epoch
        :lrnRat (float) -- learning rate
        :bchSiz (int) -- batch size
        :sflDat (bool) -- dataloader flag to shuffle data entries
        :drpLst (bool) -- dataloader flag to drop the last batch if incompleted
        :pltPth (str) -- loss plot path
        :logPth (str) -- log file path
    """
    try:
        # cast label as float
        datSet.lblLst = dataset.try_float_listlist(datSet.lblLst)

        # split training & evaluating data
        _, _, trnLod, evlLod = dataset.split_train_eval(datSet, trnRat, evlRat, bchSiz, sflDat, drpLst)
        
        # model
        mdl = LinearRegression(len(datSet.attClm), len(datSet.lblClm))
        mdl = model.model_load(mdl, mdlPth)

        # optimizer
        opt = torch.optim.SGD(mdl.parameters(), lr=lrnRat)
        # loss function
        losFnc = func.mse_loss

        # log file
        logFil = open(logPth, "w")

        # training model
        losLst = model.train(trnLod, bchSiz, maxEpc, mdl, losFnc, opt, logFil)
        
        def round_float(tsr: torch.Tensor):
            """
                Cast as float then rounding for evaluation.

                Keyword Argument
                :tsr (Tensor) -- target tensor
            """
            elm = round(float(tsr))
            return elm
        # evaluating model
        model.eval(evlLod, bchSiz, mdl, logFil, round_float, round_float)
    except KeyboardInterrupt:
        pass
    finally:
        model.plot_save_y(pltPth, losLst, "epoch", "avg. loss")
        model.model_save(mdl, mdlPth)

        logFil.close()
