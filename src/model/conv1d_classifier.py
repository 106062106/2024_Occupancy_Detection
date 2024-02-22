"""
    This module contains a basic linear classifier model.
"""

import os

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as func

import src.model as model

import src.core as core
import src.dataset as dataset

class Conv1dClassifier(nn.Module):
    """
        A simple multi-layer linear classifier pytorch model.
    """
    def __init__(self, dimIn: int, dimOut: int):
        """
            Initialize model parameters.

            Keyword Arguments
            :dimIn (int) -- integer of input dimension
            :dimOut (int) -- integer of output dimension
        """
        super(Conv1dClassifier, self).__init__()
        self.modLst = nn.ModuleList()

        dimLst = dimIn
        # hidden layers
        modLst, chnLst, dimRes = model.layer_conv1d_batchnorm1d_relu(dimLst, 1, [2, 4, 8, 16], [3, 3, 3, 3])
        self.modLst += modLst
        self.modLst.append(nn.Flatten())
        dimLst = chnLst*dimRes
        # output layer (no softmax; when using cross entropy as loss func.)
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
    datSet = dataset.json_listdict_lookback.set_scores(),
    clsCnt = 4,
    trnRat = 0.6,
    evlRat = 0.4,
    mdlPth = os.path.join(core.MDL, "Conv1dClassifier.pt"),
    maxEpc = 256,
    lrnRat = 1e-4,
    bchSiz = 64,
    sflDat = True,
    drpLst = True,
    pltPth = os.path.join(core.LOG, "Conv1dClassifier_loss.png"),
    logPth = os.path.join(core.LOG, "Conv1dClassifier_log.txt")
):
    """
        A LinearClassifier model training configuration template.

        Keyword Arguments
        :datSet (Dataset) -- assigned dataset
        :clsCnt (int) -- class count
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
        # pack dataset attribute
        datSet.attLst = dataset.pack(datSet.attLst)
        # unpack dataset labels
        datSet.lblLst = dataset.unpack(datSet.lblLst)

        # split training & evaluating data
        _, _, trnLod, evlLod = dataset.split_train_eval(datSet, trnRat, evlRat, bchSiz, sflDat, drpLst)
        
        # model
        mdl = Conv1dClassifier(len(datSet.attClm), clsCnt)
        mdl = model.model_load(mdl, mdlPth)

        # optimizer
        opt = torch.optim.SGD(mdl.parameters(), lr=lrnRat)
        # loss function
        losFnc = func.cross_entropy

        # log file
        logFil = open(logPth, "w")

        # training model
        losLst = model.train(trnLod, bchSiz, maxEpc, mdl, losFnc, opt, logFil)
        
        # evaluating model
        model.eval(evlLod, bchSiz, mdl, logFil, torch.argmax)

    except KeyboardInterrupt:
        pass
    finally:
        model.plot_save_y(pltPth, losLst, "epoch", "avg. loss")
        model.model_save(mdl, mdlPth)

        logFil.close()
    