from EL_model import CapHLA_EL
from BA_model import CapHLA_BA
from utils import Logger, load_data, predict_ms, predict_ba
import os
import pickle
import pandas as pd
import argparse
import sys
import torch
from tqdm import tqdm

# script help
description = """peptide HLA-I and HLA-II binding prediction
input file must be .csv format with no header.
The first column must be peptide sequences, peptide length should range from 7-25 and is composed of normal amino acid.
The second column must contain HLA allele names within HLA library"""

#Edit HLH: Changed input/output path values from parser-based system to being hard-coded in, changed gpu and BA values to be called in a module. 

# parser = argparse.ArgumentParser(description=description)
# parser.add_argument('--input', type=str, help='the path of the .csv file contains peptide and HLA allele name',default='CapHLA-2_inputs.csv')
# parser.add_argument('--output', type=str, help='the path of the output file',default = 'CapHLA-2_outputs.csv')
# parser.add_argument('--gpu', type=str, default='False', help='whether use gpu to predict')
# parser.add_argument('--BA', type=str, default='False', help='whether to predict binding affinity by BA data')
# args = parser.parse_args()
def main(gpu = False, BA= False,randYN=False):
    if not randYN:
        cap_input = 'CapHLA-2_inputs.csv'
        cap_output = 'CapHLA-2_outputs.csv'
    else:
        cap_input = 'CapHLA-2_random_inputs.csv'
        cap_output = 'CapHLA-2_random_outputs.csv'
    pwd = os.getcwd()
    logpath = os.path.join(pwd, 'error.log')
    # if not cap_input:
    #     log = Logger(logpath)
    #     log.logger.critical('your input file path is empty')
    #     sys.exit()
    # if not cap_output:
    #     log = Logger(logpath)
    #     log.logger.critical('your output file path is empty')
    #     sys.exit()
    
    # reading file and check it quality
    upscaleAA = {'A', 'R', 'N', 'D', 'V', 'Q', 'E', 'G', 'H', 'I',
                 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'C'}
    
    main_dir = os.path.dirname(__file__)
    hla_df = pd.read_csv(os.path.join(main_dir, 'HLA_library.csv'))
    hla_lib = dict(zip(hla_df['Allele Name'], hla_df['MHC pseudo-seq']))
    
    input_df = pd.read_csv(cap_input, header=None)
    #HLH: Added this to ensure that HLA subtypes not found in HLA_lib are *not* carried over into processing.
    cleaned_df = pd.DataFrame()
    for i in range(input_df.shape[0]):
        if input_df.iloc[i,1] in hla_lib.keys():
            row = pd.DataFrame(data=input_df.iloc[i,:]).T
            cleaned_df = pd.concat([cleaned_df,row],axis=0,ignore_index=True)
    input_df = cleaned_df
    
    if input_df.shape[1] == 3:
        three = True
        input_df.columns = ['peptide', 'Allele Name', 'Annotation']
    else:
        three = False
        input_df.columns = ['peptide', 'Allele Name']
    
    try:
        input_df['MHC pseudo-seq'] = input_df['Allele Name'].apply(lambda x: hla_lib[x])
    except KeyError:
        log = Logger(logpath)
        log.logger.critical('The HLA allele name is invalid, please check whether your HLA allele names are contained in the HLA allele library.')
        try:
            log.logger.critical(f"mhc-{input_df['MHC pseudo-seq']}")
        except KeyError:
            pass
        try:
            log.logger.critical(f"name-{input_df['Allele Name']}")
        except KeyError:
            pass
        #sys.exit()
            
    for pep in input_df['peptide']:
        if len(pep) > 25 or len(pep) < 7:
            log = Logger(logpath)
            log.logger.critical('The peptide is invalid, please check whether their length ranges from 7-25.')
            #sys.exit(0)
        if not set(list(pep)).issubset(upscaleAA):
            log = Logger(logpath)
            log.logger.critical('The peptide is invalid, please check whether they contain abnormal amino acid.')
            #sys.exit(0)
    input_iter = load_data(input_df)
    print('file QC achieved!')
    
    allele_dict_path = os.path.join(main_dir, 'allele_dict.pickle')
    allele_dict = pickle.load(open(allele_dict_path, 'rb'))
    allele_list = input_df['Allele Name'].value_counts().index
    
    
    # ms model load and predict input peptides
    device = torch.device('cuda' if gpu == False else 'cpu')
    params_dir = os.path.join(main_dir, 'params')
    result_el = pd.DataFrame()
    print('5-fold model prediction start!')
    for fold in tqdm(range(5)):
        net = CapHLA_EL().to(device)
        params_path = os.path.join(params_dir, f'el_fold{fold}.params')
        net.load_state_dict(torch.load(params_path, map_location=device,weights_only=False))
        net.eval()
        score = predict_ms(net, input_iter, device)
        result_el[f'fold{fold}'] = score
    input_df['presentation_score'] = result_el.mean(axis=1)
    # ba model load and predict input peptides
    if three:
        output_df = input_df.loc[:, ['peptide', 'Allele Name', 'Annotation', 'presentation_score']]
    else:
        output_df = input_df.loc[:, ['peptide', 'Allele Name', 'presentation_score']]
                                     
    
    if BA == True:
        result_ba = pd.DataFrame()
        for fold in tqdm(range(5)):
            net = CapHLA_BA().to(device)
            params_path = os.path.join(params_dir, f'ba_fold{fold}.params')
            net.load_state_dict(torch.load(params_path, map_location=device,weights_only=False))
            net.eval()
            score = predict_ba(net, input_iter, device)
            result_ba[f'fold{fold}'] = score
        output_df['affinity_score'] = result_ba.mean(axis=1)
    
    # output
    output_path = os.path.join(pwd, cap_output)
    output_df.to_csv(output_path, index=False)
    print('Successful finished')
