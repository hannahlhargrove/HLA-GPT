#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/11/23
@author yrh

"""
#EDITEDITEDIT: HLH, added pandas and os
import pandas as pd
import os
import click
import numpy as np
from functools import partial
from pathlib import Path
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
from logzero import logger


from deepmhcii.data_utils import *
from deepmhcii.datasets import MHCIIDataset
from deepmhcii.models import Model
from deepmhcii.networks import DeepMHCII
from deepmhcii.evaluation import output_res, CUTOFF

def test(model, model_cnf, test_data):
    data_loader = DataLoader(MHCIIDataset(test_data, **model_cnf['padding']),
                             batch_size=model_cnf['test']['batch_size'])
    return model.predict(data_loader)


def get_binding_core(data_list, model_cnf, model_path, start_id, num_models, core_len=9):
    scores_list = []
    for model_id in range(start_id, start_id + num_models):
        model = Model(DeepMHCII, model_path=model_path.with_stem(f'{model_path.stem}-{model_id}'), pooling=False,
                      **model_cnf['model'])
        scores_list.append(test(model, model_cnf, data_list))
    return (scores:=np.mean(scores_list, axis=0)).argmax(-1), scores

#HLH:Added progress bar function
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)

def main(randYN = False, mode='binding',start_id=0,num_models=20):
    yaml = YAML(typ='safe')
    if not randYN:
        data_cnf, model_cnf = yaml.load(Path("configure/GPT_peps.yaml")), yaml.load(Path("configure/deepmhcii.yaml"))
    else:
        data_cnf, model_cnf = yaml.load(Path("configure/rand_peps.yaml")), yaml.load(Path("configure/deepmhcii.yaml"))
    model_name = model_cnf['name']
    logger.info(f'Model Name: {model_name}')
    model_path = Path(model_cnf['path'])/f'{model_name}.pt'
    res_path = Path(data_cnf['results'])/f'{model_name}'
    model_cnf.setdefault('ensemble', 20)
    mhc_name_seq = get_mhc_name_seq(data_cnf['mhc_seq'])
    get_data_fn = partial(get_data, mhc_name_seq=mhc_name_seq)
    if mode == 'binding':
        blank = pd.DataFrame()
        blank.to_csv(f"{os.getcwd()}/binding_results.csv")
        model_cnf['padding'] = model_cnf['binding']
        data_list = get_binding_data(data_cnf['binding'], mhc_name_seq, model_cnf['model']['peptide_pad'])
        (core_pos, scores), correct = get_binding_core(data_list, model_cnf, model_path, start_id, num_models), 0
        count=0
        for d, core_pos_, scores_ in zip(data_list, core_pos, scores):
            (pdb, mhc_name, core), peptide_seq = d[0], d[1]
            core_ = peptide_seq[core_pos_: core_pos_ + 9]
            printProgressBar(count,len(data_list),prefix = "Processing...", suffix = '', length = 50)
            count+=1
            #EDITEDITEDIT: HLH, added line to write directly to a .csv file when printing. 
            HLHrow = pd.DataFrame(data = [pdb, mhc_name, peptide_seq, core, core_, core == core_])
            HLHrow.to_csv(f"{os.getcwd()}/binding_results.csv",header=False,mode="a")
            if core != core_ or core == core_: #EDITEDITEDIT:HLH, added condition "or core == core_" to print scores no matter what.
                for i, s in enumerate(scores_[:len(peptide_seq) - len(core) + 1]):
                    #EDITEDITEDIT: HLH, added line to write directly to a .csv file when printing. 
                    HLHrow = pd.DataFrame(data = [peptide_seq[i: i + len(core)], s])
                    HLHrow.to_csv(f"{os.getcwd()}/binding_results.csv",header=False,mode="a")
            correct += core_ == core
        
    