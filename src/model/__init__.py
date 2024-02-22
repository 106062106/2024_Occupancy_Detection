"""
    This package contains pytroch models.
"""

import traceback
from typing import IO

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import matplotlib.pyplot as plot

import src.core as core

from . import conv1d_classifier
from . import linear_classifier
from . import linear_regression

__all__ = [
    "conv1d_classifier",
    "linear_classifier",
    "linear_regression"
]

def plot_show_y(y: list, xLbl = "x", yLbl = "y"):
    """
        Plots and shows given list by matplotlib.

        Keyword Arguments
        :y (list) -- list to be plotted
        :xLbl (str) -- x axis name
        :yLbl (str) -- y axis name
    """
    x = list(range(len(y)))
    plot.plot(x, y, 'bx:', label='y')
    plot.xlabel(xLbl)
    plot.ylabel(yLbl)
    plot.legend()
    plot.show()

def plot_save_y(pth: str, y: list, xLbl = "x", yLbl = "y"):
    """
        Plots and saves given list by matplotlib.

        Keyword Arguments
        :pth (str) -- output file path
        :y (list) -- list to be plotted
        :xLbl (str) -- x axis name
        :yLbl (str) -- y axis name
    """
    try:
        x = list(range(len(y)))
        plot.plot(x, y, 'bx:', label='y')
        plot.xlabel(xLbl)
        plot.ylabel(yLbl)
        plot.legend()
        plot.savefig(pth)
        plot.close()
    except Exception as exc:
        traceback.print_exception(exc)

def model_save(mdl: nn.Module, pth: str):
    """
        Saves torch model by given path.

        Keyboard Arguments
        :mdl (Module) -- torch module to be saved
        :pth (str) -- saving file path
    """
    try:
        torch.save(mdl.state_dict(), pth)
    except Exception as exc:
        traceback.print_exception(exc)

def model_load(mdl: nn.Module, pth: str):
    """
        Loads torch model by given path.

        Keyword Arguments
        :mdl (Module) -- torch module to be loaded
        :pth (str) -- loading file path
    """
    try:
        mdl.load_state_dict(torch.load(pth))
    except Exception:
        pass
    return mdl

def train(
    trnLod: data.DataLoader,
    bchSiz: int,
    maxEpc: int,
    mdl: nn.Module,
    losFnc: object,
    opt: optim.Optimizer,
    logFil: IO
):
    """
        Trains the model.

        Keyword Arguments
        :trnLod (DataLoader) -- training data loader
        :bchSiz (int) -- batch size
        :maxEpc (int) -- max training epoch
        :mdl (Module) -- training pytorch model
        :losFnc (object) -- loss function object
        :opt (Optimizer) -- pytorch optimizer
        :logFil (IO) -- log file
    """
    try:
        mdl.train()

        losLst = []
        trnCnt = len(trnLod)*bchSiz
        for epc in range(maxEpc):
            epcLos = 0.0
            for attBch, lblBch in trnLod:
                opt.zero_grad()
                outBch = mdl(attBch)

                loss = losFnc(outBch, lblBch)
                loss.backward()

                epcLos += loss.item()
                opt.step()
            epcLos /= trnCnt
            losLst.append(epcLos)

            msg = "training epoch: {epc}, avg. loss: {loss}".format(epc=epc, loss=epcLos)
            core.log(logFil, msg)
            core.out(msg)
    
        return losLst
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        traceback.print_exception(exc)
        return None
    
def eval(
    evlLod: data.DataLoader,
    bchSiz: int,
    mdl: nn.Module,
    logFil: IO,
    cvtOut = None,
    cvtLbl = None
):
    """
        Evaluates the regression model.

        Keyword Arguments
        :evlLod (DataLoader) -- evaluating data loader
        :bchSiz (int) -- batch size
        :mdl (Module) -- training pytorch model
        :logFil (IO) -- log file
        :cvtOut (object) -- method to convert model output elements
        :cvtLbl (object) -- method to convert label elements
    """
    try:
        mdl.eval()

        errRat = 0.0
        evlCnt = len(evlLod)*bchSiz
        for attBch, lblBch in evlLod:
            outBch = mdl(attBch)

            for b in range(bchSiz):
                out = outBch[b]
                lbl = lblBch[b]

                if (cvtOut != None):
                    out = cvtOut(out)
                if (cvtLbl != None):
                    lbl = cvtLbl(lbl)

                if (out != lbl):
                    errRat += 1.0
        errRat /= evlCnt

        msg = "evaluation error rate: {errRat}, over {evlCnt} data".format(errRat=errRat, evlCnt=evlCnt)
        core.log(logFil, msg)
        core.out(msg)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        traceback.print_exception(exc)

def layer_linear_batchnorm1d_relu(dimIn: int, lyrCnt: int, lyrMlt: int):
    """
        Gives a number of sequences of linear, batchnorm1d and relu layers
        with linear layer output dimension being a geometric sequence.

        Keyword Arguments
        :dimIn (int) -- input dimension
        :lyrCnt (int) -- number of linear layers
        :mltRat (int) -- dimension multiply ratio for each layer
    """
    try:
        modLst = nn.ModuleList()

        dimLst = dimIn
        dimNxt = dimIn*lyrMlt
        for _ in range(lyrCnt):
            modLst.append(
                nn.Sequential(
                    nn.Linear(dimLst, dimNxt),
                    nn.BatchNorm1d(dimNxt),
                    nn.ReLU()
                )
            )
            dimLst = dimNxt
            dimNxt *= lyrMlt
        return modLst, dimLst
    except Exception as exc:
        traceback.print_exception(exc)
        return None, None
    
def layer_conv1d_batchnorm1d_relu(
    dimIn: int,
    chnIn: int,
    chnOut: list,
    knlLst: list
):
    """
        Gives a sequential of conv1d, batchnorm1d and relu layers
        with the provided channel list, stride list and padding list.

        Note: Uses the length of the channel list as layer count.

        Keyword Arguments
        :dimIn (int) -- input dimension
        :chnLst (list) -- layer channel count list
        :knlLst (list) -- kernel size list
    """
    try:
        modLst = nn.ModuleList()

        dimRes = dimIn
        chnLst = chnIn
        lyrCnt = len(chnOut)
        for l in range(lyrCnt):
            chnNxt = chnOut[l]
            dimRes -= knlLst[l] - 1

            modLst.append(nn.Conv1d(chnLst, chnNxt, knlLst[l]))
            modLst.append(nn.BatchNorm1d(chnNxt))
            modLst.append(nn.ReLU())

            chnLst = chnNxt
        return modLst, chnLst, dimRes
    except Exception as exc:
        traceback.print_exception(exc)
        return None, None, None
