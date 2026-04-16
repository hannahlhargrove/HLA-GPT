import core_GPT
import pandas as pd
import numpy as np
import logo
import sys
import random
import os
import datetime

sys.path.append('../')
#functions to normalize and de-normalize values according to the manuscript
def normalize(val):
    return 1-(np.log10(val)/np.log10(50000))
def denormalize(val):
    d_val = 10**(-(val-1)*np.log10(50000))
    #make sure denormalized value isn't hyperspecific -- limit to 2 sig figs, since that is what's typically reported for DeepMHCII and CapHLA normalized scores:
    return float('%.2g' % d_val)
    
def get_rands(peplen = 15,numpeps=100): #get random sequences for the experiment control
    print(f"Peptides to be generated: {numpeps}")
    blank = pd.DataFrame(columns=["Peptide_sequence","Timestamp"])
    blank.to_csv("outputs/random_peptides.csv",header=True,mode="w",index=False)
    for pep in range(numpeps):
        seq = ""
        for acid in range(peplen):
            seq += random.choice(alphabet)
        seq = pd.DataFrame(data=[seq],columns=["Peptide_sequence"])
        mark = pd.DataFrame(data=[str(datetime.datetime.now())],columns=["Timestamp"])
        seq = pd.concat([seq,mark],axis=1)
        seq.to_csv("outputs/random_peptides.csv",header=False,mode="a",index=False)

#Get specific pseudosequences for the relevant subtype
def get_pseudos(subtype,specific_alleles = []): 
    #Get pseudosequences for HLA-II alleles:
    pseudos = pd.read_csv("data/pseudosequence.2016.all.X.dat",sep="\t",header=None)
    #Establish the pseudosequences as a dict() with alleles as keys and sequences as values
    p_dict = dict(zip(list(pseudos.iloc[:,0]),list(pseudos.iloc[:,1])))
    relevant = dict()
    if len(specific_alleles)==0: #if the specific alleles weren't specified and the user wants to test against all alleles in the subtype:
        for key in p_dict.keys():
            if subtype in key:
                relevant[key] = p_dict[key]
    else:
        for key in specific_alleles:
            if subtype in key:
                relevant[key] = p_dict[key]
    return relevant
#From the generated peptides list, make the results interpretable by DeepMHCII:
def get_DeepMHC_input(filepath, pseudos,randYN = False):
    data = pd.read_csv(filepath)
    #Get a list of the novel generated peptide sequences:
    peps = list(data["Peptide_sequence"])
    #Set PDB ID to XXXX, set predicted binding core to LLLLLLLLL, make a point for each relevant pseudosequence for each peptide generated
    hold = pd.DataFrame()
    for pep in peps:
        for pseud in pseudos.keys():
            row = pd.DataFrame(data=["XXXX",pseud,pseudos[pseud], pep, "LLLLLLLLL"]).T
            hold = pd.concat([hold,row],axis=0,ignore_index=True)
    if not randYN:
        hold.to_csv("data/peps.txt",sep="\t",header=None,index=False)
    else:
        hold.to_csv("data/random_peps.txt",sep="\t",header=None,index=False)

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
flag=""
answers = ["Y","N"] #Basic Y N answer list, makes notation easier to read. 

while flag not in answers:
    flag = input("WARNING: ALL OLD DATA WILL BE ERASED AT START OF PROGRAM. PROCEED? Y/N ").upper()
    if flag=="N":
        sys.exit()
pertinent_files = ["best_DeepMHCII_results.csv","best_random_DeepMHCII_results.csv","binding_results.csv","CapHLA-2_inputs.csv","CapHLA-2_outputs.csv","CapHLA-2_random_inputs.csv","CapHLA-2_random_outputs.csv","compound_scores.csv","final_ensemble_output.csv","polished_DeepMHCII_outputs.csv","rand_compound_scores.csv","rand_ensemble_output.csv","random_DeepMHCII_outputs.csv","outputs/random_peptides.csv","outputs/generated_peptides.csv","data/peps.txt","data/random_peps.txt"]
#Remove all old data to prevent leakage
for file in pertinent_files:
    try:
        os.remove(file)
    except FileNotFoundError:
        pass

logo.main()

alphabet = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
subtype = ""
sts = ["DP","DQ","DR"]


while subtype not in sts:
    subtype = input("Enter the HLA subtype (DP, DQ, or DR): ").upper()
    if subtype not in sts:
        print("Please enter 'DP','DQ', or 'DR' without spaces or special characters: ")
print(f"Subtype:{subtype}")

thresh = -1
indic = ""

while indic not in answers:
    indic = input("Would you like to manually set the inclusion threshold? (y/n): ").upper()
    if indic not in answers:
        print("Please only enter y or n: y to manually set the inclusion threshold, or n to defer to the optimized inclusion threshold.")

if indic == "Y":
    print("The inclusion threshold dictates the kD/EC50/IC50 value below which inclusion is acceptable. For example, an inclusion threshold of 10 indicates all points where kD/EC50/IC50<= 10 should be included in the dataset.")
    while thresh <=-1:
        try:
            thresh = float(input("Inclusion threshold: "))
        except ValueError:
            print("Please enter the inclusion threshold in units of nM, using only digits. E.g. '10' ")

num_peps = -1
while num_peps <= -1:
    try:
        num_peps = int(input("Number of novel peptides to be generated: "))
        if num_peps<1:
            num_peps=-1
            print("Please enter an integer value greater than 0.")
        elif num_peps>1000:
            checkpoint = ""
            while checkpoint not in answers:
                checkpoint = input("Peptide count in excess of 1000 may lead to long generation times, proceed? (y/n): ").upper()
                if checkpoint == "Y":
                    pass
                elif checkpoint == "N":
                    num_peps=-1
                else:
                    print("Please select y or n.")
            
    except ValueError:
        print("Please enter an integer value.")
core_GPT.main(cutoff=thresh,tag=subtype,final_pep_count = num_peps)
print()

start_DeepMHC = ""
while start_DeepMHC not in answers:
    start_DeepMHC = input(f"Proceed to DeepMHCII processing? (y/n): ").upper()
    if start_DeepMHC not in answers:
        print("Please only enter y or n: y to proceed to DeepMHCII processing, or n to exit the ensemble program.")
print()
#Note: Below this point all code should be contingent on DeepMHCII=="Y", otherwise nothing should happen. 
if start_DeepMHC == "Y": 
    spec = ""
    while spec not in answers:
        spec=input("Would you like to specify target allele(s)? (This will significantly improve program speed) (y/n): ").upper()
        if spec not in answers:
            print(f"Please only enter y or n: y to specify target allele(s), n to test every allele in the {subtype} subset.")

    if spec == "Y":
        allele = ""
        a_data = pd.read_csv("HLA_library.csv")
        a_dict = dict()
        a_list = []
        for i in range(a_data.shape[0]):
            name = a_data.iloc[i,0]
            if subtype in name:
                mhc = a_data.iloc[i,1]
                a_dict[name]=mhc
                
        print("Please enter target alleles one at a time, pressing enter between each allele.")
        if subtype == "DR":
            print("Use the format: HLA-DRB1*01:01")
        elif subtype == "DP":
            print("Use the format: HLA-DPA1*01:03/DPB1*01:01")
        else:
            print("Use the format: HLA-DQA1*01:01/DQB1*02:01")
        print("When you are finished, please enter 'done'.")
        while allele != "done":
            allele = input("Target allele: ")
            if allele not in a_dict.keys() and allele != 'done':
                print("Please select an existing allele in the specified format.")
            elif allele == 'done':
                pass
            else:
                print(f"{allele} added!")
                a_list.append(allele)
        print("Priming data...")
        get_DeepMHC_input("outputs/generated_peptides.csv",get_pseudos(subtype,a_list))
    else:
        allele = ""
        a_data = pd.read_csv("HLA_library.csv")
        a_dict = dict()
        a_list = []
        for i in range(a_data.shape[0]):
            name = a_data.iloc[i,0]
            if subtype in name:
                mhc = a_data.iloc[i,1]
                a_dict[name]=mhc
        for key in list(a_dict.keys()):
            a_list.append(key)
            
        print("Priming data...")
        get_DeepMHC_input("outputs/generated_peptides.csv",get_pseudos(subtype))
    print("Loading DeepMHCII...")
    import deepmhcii_mod_main as deepmhcii
    print("Initializing...")
    deepmhcii.main()

    path = "binding_results.csv"
    text = ""
    with open(path,"r") as f:
        text += f.read()
    words = text.split(",")
    data = []
    for y in range(len(words)):
        w = words[y]
        if w[0] in alphabet or (y>=1 and words[y-1][0] in alphabet):
            if "\n" in w:
                x = w.split("\n")
                data.append(x[0])
    pol_df = pd.DataFrame()
    newline=pd.DataFrame()
    for piece in data:
        if "HLA" in piece and newline.shape[0] > 0:
            pol_df = pd.concat([pol_df,newline],axis=0,ignore_index=True)
            newline=pd.DataFrame()
        newline = pd.concat([newline,pd.DataFrame([piece])],axis=1,ignore_index=True)
    pol_df = pd.concat([pol_df,newline],axis=0,ignore_index=True)
    #if deepmhcii supplies more than 6 potential binding cores, limit to 6:
    pol_df = pol_df.iloc[:,:17]
    for x in range(2,5): pol_df.pop(x)
    names = ["mhc_name","pep_seq"]
    for n in range(1,7): names.extend((f"binding_core_{n}",f"binding_core_score_{n}"))
    pol_df.columns=names
    pol_df.to_csv("polished_DeepMHCII_outputs.csv",index=False,header=True)
    #Find best score for each peptide:
    n_pol = pd.DataFrame()
    for i in range(pol_df.shape[0]):
        printProgressBar(i,pol_df.shape[0],prefix="Polishing data...",suffix="",length=50)
        row = pd.DataFrame(pol_df.iloc[i,:]).T
        mhc=row["mhc_name"].iloc[0]
        pep = row["pep_seq"].iloc[0]
        m=0
        m_name=""
        for key in row.columns:
            if "score" in key:
                v = float(row[key].iloc[0])
                s = key.split("score_")
                tag = s[0] + s[1]
                name = row[tag].iloc[0]
                if v>m:
                    m=v
                    m_name=name
        n_row = pd.DataFrame(data=[mhc,pep,m_name,m]).T
        n_row.columns = ["HLA Allele", "Peptide Sequence", "Predicted Binding Core", "Maximum DeepMHCII Binding Score"]
        n_pol = pd.concat([n_pol,n_row],axis=0,ignore_index=True)
    
    n_pol=n_pol.sort_values("Maximum DeepMHCII Binding Score",ascending=False)
    n_pol.reset_index(drop=True,inplace = True)
    n_pol.to_csv("best_DeepMHCII_results.csv")
    print()
    print("Top ten results:")
    print(n_pol.head(10))

    print()
    #CapHLA-2.0 analysis should only be done when DeepMHCII processing has already happened, or the program will break. 
    start_Cap = ""
    while start_Cap not in answers:
        start_Cap = input(f"Proceed to CapHLA-2.0 processing? (y/n): ").upper()
        if start_Cap not in answers:
            print("Please only enter y or n: y to proceed to CapHLA-2.0 processing, or n to exit the ensemble program.")
    print()
    
    import CapHLA
    if start_Cap == "Y":
        minimize = ""
        thresh = -1
        print(f"Would you like to specify a cutoff Maximum DeepMHCII Binding Score for inclusion in CapHLA processing?\nThis reduces processing to only instances above this threshold.\nThe range of Maximum DeepMHCII Binding Scores for your data is {round(min(n_pol['Maximum DeepMHCII Binding Score']),3)} to {round(max(n_pol['Maximum DeepMHCII Binding Score']),3)}, with a median of {round(np.median(n_pol['Maximum DeepMHCII Binding Score']),3)}.")
        while minimize not in answers:
            minimize = input("Please enter Y to select a cutoff threshold or N to process all datapoints: ").upper()
            if minimize == "Y":
                while thresh <= -1:
                    thresh = input("Please enter the cutoff threshold (instances less than or equal to this value will be dropped)")
                    try:
                        thresh = float(thresh)
                        if thresh > round(max(n_pol['Maximum DeepMHCII Binding Score']),2) or thresh < round(min(n_pol['Maximum DeepMHCII Binding Score']),2):
                            thresh = -1
                            print("Please only enter numbers within the specified maximum and minimum of your data range.")
                    except ValueError:
                        thresh=-1
                        print("Please only enter a float value.")
    
        # #Make CapHLA-readable outputs:
        df = pd.DataFrame()
        for i in range(n_pol.shape[0]):
            row = pd.DataFrame(n_pol.iloc[i,:]).T
            if round(row["Maximum DeepMHCII Binding Score"].iloc[0],3) >= thresh: #if thresh never specified, values always > -1, so all are included. 
                df = pd.concat([df,row],axis=0,ignore_index=True)
    
        names = df["HLA Allele"]
        seq = df["Peptide Sequence"]
        
        n_df = pd.concat([seq,names],axis=1)
        n_df.to_csv("CapHLA-2_inputs.csv",index=False, header=False)
        print("Beginning Cap-HLA-2.0 processing...")
        CapHLA.main(gpu=False,BA=True)
#Finalize outputs:
print("Finalizing outputs...")
deep=pd.read_csv("best_DeepMHCII_results.csv",index_col=0)
cap = pd.read_csv("CapHLA-2_outputs.csv")
cap = cap.sort_values(["peptide","Allele Name"])
cap.reset_index(inplace = True, drop = True)
deep = deep.sort_values(["Peptide Sequence","HLA Allele"])
deep.reset_index(inplace = True, drop = True)
final = pd.concat([deep,cap],axis=1)
final.pop("peptide")
pep_score = (final["Maximum DeepMHCII Binding Score"] + final["presentation_score"] + final["affinity_score"])/3
final = pd.concat([final,pd.DataFrame(pep_score,columns=["PepScore"])],axis=1)
BA = pd.DataFrame()
for i in final["affinity_score"]:
    BA = pd.concat([BA,pd.DataFrame(data=[denormalize(i)],columns = ["EC50"])],axis=0,ignore_index=True)
final.pop("affinity_score")
final = pd.concat([final,BA],axis=1)

dmhc = pd.DataFrame()
for i in final["Maximum DeepMHCII Binding Score"]:
    dmhc = pd.concat([dmhc,pd.DataFrame(data=[denormalize(i)],columns = ["IC50"])],axis=0,ignore_index=True)
final.pop("Maximum DeepMHCII Binding Score")
final = pd.concat([final,dmhc],axis=1)
PB = final.pop("presentation_score")
PB = pd.DataFrame(PB)
PB.columns = ["PB"]
final = pd.concat([final,PB],axis=1)
final.pop("Allele Name")


#Prepare documentation of CompoundScore:
final_groups = final.groupby("Peptide Sequence")
compounds = pd.DataFrame()
cdf = dict()
for f in final_groups:
    d = pd.DataFrame([f[0],(np.average(f[1]["PepScore"]))]).T
    cdf[f[0]]=np.average(f[1]["PepScore"])
    d.columns = ["Peptide Sequence","CompoundScore"]
    compounds = pd.concat([compounds,d],axis=0,ignore_index=True)
    
compounds = compounds.sort_values(by="CompoundScore",ascending=False)
compounds.to_csv("compound_scores.csv",index=False)

fdf = pd.DataFrame()
for i in range(final.shape[0]):
    row = pd.DataFrame(final.iloc[i,:]).T
    row.reset_index(inplace=True,drop=True)
    seq = row["Peptide Sequence"].iloc[0]
    nrow = pd.DataFrame(data=[cdf[seq]],columns=["CompoundScore"])
    nrow = pd.concat([row,nrow],axis=1)
    fdf = pd.concat([fdf,nrow],axis=0,ignore_index=True)
final = fdf.sort_values(by=["CompoundScore","PepScore"],ascending=False)
final.to_csv("final_ensemble_output.csv",index=False)
#ADD RANDOMIZED CONTROL
print("Adding randomized control...")
get_rands(peplen=15,numpeps=num_peps)

if start_DeepMHC == "Y": 
    if spec == "Y":
        print("Priming data...")
        get_DeepMHC_input("outputs/random_peptides.csv",get_pseudos(subtype,a_list),randYN=True)
    else:
        print("Priming data...")
        get_DeepMHC_input("outputs/random_peptides.csv",get_pseudos(subtype),randYN=True)
    import deepmhcii_mod_main as deepmhcii
    print("Initializing...")
    deepmhcii.main(randYN=True)

    path = "binding_results.csv"
    text = ""
    with open(path,"r") as f:
        text += f.read()
    words = text.split(",")
    data = []
    for y in range(len(words)):
        w = words[y]
        if w[0] in alphabet or (y>=1 and words[y-1][0] in alphabet):
            if "\n" in w:
                x = w.split("\n")
                data.append(x[0])
    rand_pol_df = pd.DataFrame()
    newline=pd.DataFrame()
    for piece in data:
        if "HLA" in piece and newline.shape[0] > 0:
            rand_pol_df = pd.concat([rand_pol_df,newline],axis=0,ignore_index=True)
            newline=pd.DataFrame()
        newline = pd.concat([newline,pd.DataFrame([piece])],axis=1,ignore_index=True)
    rand_pol_df = pd.concat([rand_pol_df,newline],axis=0,ignore_index=True)
    #if deepmhcii supplies more than 6 potential binding cores, limit to 6:
    rand_pol_df = rand_pol_df.iloc[:,:17]
    for x in range(2,5): rand_pol_df.pop(x)
    names = ["mhc_name","pep_seq"]
    for n in range(1,7): names.extend((f"binding_core_{n}",f"binding_core_score_{n}"))
    rand_pol_df.columns=names
    rand_pol_df.to_csv("random_DeepMHCII_outputs.csv",index=False,header=True)
    #Find best score for each peptide:
    rand_pol = pd.DataFrame()
    for i in range(rand_pol_df.shape[0]):
        printProgressBar(i,rand_pol_df.shape[0],prefix="Polishing data...",suffix="",length=50)
        row = pd.DataFrame(rand_pol_df.iloc[i,:]).T
        mhc=row["mhc_name"].iloc[0]
        pep = row["pep_seq"].iloc[0]
        m=0
        m_name=""
        for key in row.columns:
            if "score" in key:
                v = float(row[key].iloc[0])
                s = key.split("score_")
                tag = s[0] + s[1]
                name = row[tag].iloc[0]
                if v>m:
                    m=v
                    m_name=name
        n_row = pd.DataFrame(data=[mhc,pep,m_name,m]).T
        n_row.columns = ["HLA Allele", "Peptide Sequence", "Predicted Binding Core", "Maximum DeepMHCII Binding Score"]
        rand_pol = pd.concat([rand_pol,n_row],axis=0,ignore_index=True)
    
    rand_pol=rand_pol.sort_values("Maximum DeepMHCII Binding Score",ascending=False)
    rand_pol.reset_index(drop=True,inplace = True)
    rand_pol.to_csv("best_random_DeepMHCII_results.csv")
    print()
    if start_Cap == "Y":
        # #Make CapHLA-readable outputs:
        df = pd.DataFrame()
        for i in range(rand_pol.shape[0]):
            row = pd.DataFrame(rand_pol.iloc[i,:]).T
            if round(row["Maximum DeepMHCII Binding Score"].iloc[0],3) >= thresh: #if thresh never specified, values always > -1, so all are included. 
                df = pd.concat([df,row],axis=0,ignore_index=True)
    
        names = df["HLA Allele"]
        seq = df["Peptide Sequence"]
        
        n_df = pd.concat([seq,names],axis=1)
        n_df.to_csv("CapHLA-2_random_inputs.csv",index=False, header=False)
        print("Beginning Cap-HLA-2.0 processing...")
        CapHLA.main(gpu=False,BA=True,randYN=True)

#Finalize outputs:
print("Finalizing outputs...")
deep=pd.read_csv("best_random_DeepMHCII_results.csv",index_col=0)
cap = pd.read_csv("CapHLA-2_random_outputs.csv")
cap = cap.sort_values(["peptide","Allele Name"])
cap.reset_index(inplace = True, drop = True)
deep = deep.sort_values(["Peptide Sequence","HLA Allele"])
deep.reset_index(inplace = True, drop = True)
final = pd.concat([deep,cap],axis=1)
final.pop("peptide")
pep_score = (final["Maximum DeepMHCII Binding Score"] + final["presentation_score"] + final["affinity_score"])/3
final = pd.concat([final,pd.DataFrame(pep_score,columns=["PepScore"])],axis=1)
BA = pd.DataFrame()
for i in final["affinity_score"]:
    BA = pd.concat([BA,pd.DataFrame(data=[denormalize(i)],columns = ["EC50"])],axis=0,ignore_index=True)
final.pop("affinity_score")
final = pd.concat([final,BA],axis=1)

dmhc = pd.DataFrame()
for i in final["Maximum DeepMHCII Binding Score"]:
    dmhc = pd.concat([dmhc,pd.DataFrame(data=[denormalize(i)],columns = ["IC50"])],axis=0,ignore_index=True)
final.pop("Maximum DeepMHCII Binding Score")
final = pd.concat([final,dmhc],axis=1)
PB = final.pop("presentation_score")
PB = pd.DataFrame(PB)
PB.columns = ["PB"]
final = pd.concat([final,PB],axis=1)


#Prepare documentation of CompoundScore:
final_groups = final.groupby("Peptide Sequence")
compounds = pd.DataFrame()
cdf = dict()
for f in final_groups:
    d = pd.DataFrame([f[0],(np.average(f[1]["PepScore"]))]).T
    cdf[f[0]]=np.average(f[1]["PepScore"])
    d.columns = ["Peptide Sequence","CompoundScore"]
    compounds = pd.concat([compounds,d],axis=0,ignore_index=True)
compounds = compounds.sort_values(by="CompoundScore",ascending=False)
compounds.to_csv("rand_compound_scores.csv",index=False)

fdf = pd.DataFrame()
for i in range(final.shape[0]):
    row = pd.DataFrame(final.iloc[i,:]).T
    row.reset_index(inplace=True,drop=True)
    seq = row["Peptide Sequence"].iloc[0]
    nrow = pd.DataFrame(data=[cdf[seq]],columns=["CompoundScore"])
    nrow = pd.concat([row,nrow],axis=1)
    fdf = pd.concat([fdf,nrow],axis=0,ignore_index=True)
final = fdf.sort_values(by=["CompoundScore","PepScore"],ascending=False)
final.to_csv("rand_ensemble_output.csv",index=False)

print("Experimental data is available at 'final_ensemble_output.csv' and 'compound_scores.csv', and randomized control data is available at 'rand_ensemble_out.csv' and 'rand_compound_scores.csv'.")