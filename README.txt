================================================================================================================================================================
Original code sources:
    GPT template sourced from Jay Mody's "GPT in 60 Lines of Numpy", template available on GitHub: https://github.com/jaymody/picoGPT
    DeepMHCII source code by You et. al (https://doi.org/10.1093/bioinformatics/btac225), source code available on GitHub: https://github.com/yourh/DeepMHCII
    CapHLA-2.0 source code by Chang et. al (https://doi.org/10.1093/bib/bbae595), source code available on GitHub: https://github.com/changyunjian/CapHLA
================================================================================================================================================================
Modifications to code sources: 
    Modifications to GPT template:
        - Novel encode/decode modules to accommodate peptide sequences rather than English text.
        - Novel "get_acids" module to convert a peptide sequence to a list of amino acids, with a "<start>" token at the beginning or the list and an "<end>" token at the end of the list. 
        - Novel "tokenize" module to convert the "get_acids" list into a list of N-gram tokens, N = [1,2,3].
        - Altered "train" module to include an early stop which triggers when validation loss is observed to be increasing for two iterations of loss calculations. 
        - Altered module "generate_text" to be "generate_peps", adding limitations to constrain the desired output peptide length, and to limit generation to a single peptide at a time. 
        - Novel module "mass_generate_peps" to dictate the number of novel peptides to be generated, writing the output peptides to a .txt format which can be read by DeepMHCII. Output peptides have "XXXX" set as a dummy PDBID, and "LLLLLLLLL" as a dummy predicted binding core (this is not used for anything, it is purely to satisfy the input 
          requirements of DeepMHCII).
        - Encased the GPT structure within a "main" module to allow for the GPT to be called with variable input
        
    Modified template available as ./submodels/core_GPT.py
    
    Modifications to DeepMHCII source code:
        - Removed Bovine Leukemia (BoLa) alleles and non-DR, -DQ, or -DP alleles from the list of possible pseudosequences, leaving only human allele types. 
        - DeepMHCII model trained as original suggestion from You et. al, using data.yaml as the training input file and deepmhcii.yaml as the model file.
        - ./DeepMHCII_mod/configure/data.yaml copied to a new file, ./DeepMHCII_mod/configure/GPT_peps.yaml 
        - Altered variable "binding" in GPT_peps.yaml to match the filepath for peptides to be tested for DeepMHCII binding. 
        - Added "import pandas" to DeepMHCII's main.py
        - Added lines 141 and 142 of DeepMHCII_mod main.py to write the information for the input datapoint under consideration directly to the output file [pdbid, mhc_name, peptide_seq, core, core_, core == core_]
        - Altered line 143 of DeepMHCII_mod main.py to report binding score regardless of whether the core being measured matches the predicted binding core (this is why the input predicted binding core can be set to a dummy value of "LLLLLLLLL"). 
        - Altered lines 147 and 148 of DeepMHCII_mod main.py to write al binding core and score pairs directly to the output file
        - For use in HLA-GPT, altered DeepMHCII-specific import paths to work within the current folder configuration.
    Modified DeepMHCII configuration available as ./DeepMHCII_mod
        - Altered specific HLA allele naming convention to match up with CapHLA-2.0 notation
        - Removed any alleles not present in the CapHLA-2.0 dataset, to prevent the model from attempting to predict MHC pseudosequence data it doesn't posess. 
    CapHLA-2.0 was not modified from the source code, as the model already included a full data report. CapHLA-2.0 provided as-is for use in replicating experiments as ./CapHLA_2.0

================================================================================================================================================================

Original code:
    - ensemble.py -- runs the GPT ensemble where the training is separated by subtype. Takes in a single subtype and the number of peptides to be generated, as well as the opportunity to manually dictate the inclusion threshold. If the inclusion threshold is not manually dictated, the threshold will be set to the optimized threshold for the 
                     associated subtype, per the publication. 
    - single_model.py -- runs the generalized GPT ensemble where training is not separated by subtype. Takes in the number of peptides to be generated, as well as the opportunity to manually dictate the inclusion threshold. If the inclusion threshold is not manually dictated, the threshold will be set to the optimized threshold for the 
                         generalized model (10nM), per the publication. 

All source code modifications and original code were produced by Hannah Hargrove

Contact: X. Frank Zhang (frank.zhang@umass.edu)