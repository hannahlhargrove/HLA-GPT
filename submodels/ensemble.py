import core_GPT
import pandas as pd
import numpy as np
import logo
import sys
import random
import os
import datetime
alphabet = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]

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
    blank.to_csv(f"{os.getcwd()}/outputs/random_peptides.csv",header=True,mode="w",index=False)
    for pep in range(numpeps):
        seq = ""
        for acid in range(peplen):
            seq += random.choice(alphabet)
        seq = pd.DataFrame(data=[seq],columns=["Peptide_sequence"])
        mark = pd.DataFrame(data=[str(datetime.datetime.now())],columns=["Timestamp"])
        seq = pd.concat([seq,mark],axis=1)
        seq.to_csv(f"{os.getcwd()}/outputs/random_peptides.csv",header=False,mode="a",index=False)

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

logo.main()

answers = ["Y","N"] #Basic Y N answer list, makes notation easier to read. 

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
    df = pd.read_csv(path,sep=" ")
    n_df = pd.DataFrame(df.iloc[:,0])
    nn_df = pd.DataFrame()
    sp = pd.DataFrame()
    trip = False
    blank = pd.DataFrame(columns=range(0,36))
    printProgressBar(1,n_df.shape[0],prefix="Processing outputs...",suffix="",length=50)
    hold=pd.DataFrame(n_df.iloc[0,2:])
    for i in range(1,n_df.shape[0]):
        trip=False
        printProgressBar(i,n_df.shape[0],prefix="Processing outputs...",suffix="",length=50)
        try:
            look = n_df.iloc[i].iloc[0].split(",")[1]
            if look == "XXXX":
                trip = True
            if not trip:
                nsp = pd.DataFrame((n_df.iloc[i].iloc[0]).split(",")).T
                sp = pd.concat([sp,nsp],axis=1,ignore_index=True)
            else:
                trip=False
                nsp = pd.DataFrame((n_df.iloc[i].iloc[0]).split(",")).T
                sp = pd.concat([sp,nsp],axis=1,ignore_index=True)
                if sp.shape[1] > 3:#excludes the very first XXXX from being added as an empty point
                    sp =sp.dropna()
                    hold = pd.concat([hold,sp],axis=0,ignore_index=True)
                    sp = pd.DataFrame()
                
        except AttributeError:
            pass
    nn_df=hold
    mhc_name = pd.DataFrame(nn_df.iloc[:,1])
    mhc_name.columns = ["mhc_name"]
    pseud_mhc = pd.DataFrame(nn_df.iloc[:,3])
    pseud_mhc.columns = ["pep_seq"]
    one = pd.DataFrame(nn_df.iloc[:,11])
    one.columns = ["binding_core_1"]
    one_s = pd.DataFrame(nn_df.iloc[:,13])
    one_s.columns = ["binding_core_score_1"]
    two = pd.DataFrame(nn_df.iloc[:,15])
    two.columns = ["binding_core_2"]
    two_s = pd.DataFrame(nn_df.iloc[:,17])
    two_s.columns = ["binding_core_score_2"]
    three = pd.DataFrame(nn_df.iloc[:,19])
    three.columns = ["binding_core_3"]
    three_s = pd.DataFrame(nn_df.iloc[:,21])
    three_s.columns = ["binding_core_score_3"]
    four = pd.DataFrame(nn_df.iloc[:,23])
    four.columns = ["binding_core_4"]
    four_s = pd.DataFrame(nn_df.iloc[:,25])
    four_s.columns = ["binding_core_score_4"]
    five = pd.DataFrame(nn_df.iloc[:,27])
    five.columns = ["binding_core_5"]
    five_s = pd.DataFrame(nn_df.iloc[:,29])
    five_s.columns = ["binding_core_score_5"]
    six = pd.DataFrame(nn_df.iloc[:,31])
    six.columns = ["binding_core_6"]
    six_s = pd.DataFrame(nn_df.iloc[:,33])
    six_s.columns = ["binding_core_score_6"]
    print()
    pol_df = pd.concat([mhc_name,pseud_mhc,one,one_s,two,two_s,three,three_s,four,four_s,five,five_s,six,six_s],axis=1)
    pol_df.to_csv("polished_DeepMHCII_outputs.csv",index=False,header=True)
    #Find best score for each peptide:
    n_pol = pd.DataFrame()
    for i in range(pol_df.shape[0]):
        printProgressBar(i,n_df.shape[0],prefix="Polishing data...",suffix="",length=50)
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
        minimize_DeepMHC_output = ""
        while minimize_DeepMHC_output not in answers:
            minimize_DeepMHC_output = input("Would you like to set a threshold for inclusion for predicted DeepMHCII scores before CapHLA-2.0 processing begins? (This will significantly speed up processing times on long jobs) y/n: ").upper()
            if minimize_DeepMHC_output not in answers:
                print("Please only enter y or n: y to set a minimum DeepMHCII inclusion value, or n to test all generated peptides with CapHLA-2.0")
        if minimize_DeepMHC_output == "Y":
            #get the minimum and maximum predicted DeepMHCII values
            best = np.round(float(n_pol.iloc[0,-1]),2)
            worst = np.round(float(n_pol.iloc[-1,-1]),2)
            deep_inc_thresh = -1.0
            print(f"Note: the DeepMHCII scores for your experiment fall between {worst} and {best}.")
            while bool(deep_inc_thresh >= best or deep_inc_thresh <= worst):#while the threshold response is outside of the predicted range:
                deep_inc_thresh = input("Specify an inclusion DeepMHCII score threshold where scores greater than the specified value will be included: ")
                try:
                    deep_inc_thresh = float(deep_inc_thresh)
                    if bool(deep_inc_thresh >= best or deep_inc_thresh <= worst):
                        print(f"Please specify a DeepMHCII score inclusion threshold as a float between {worst} and {best}.")
                except ValueError:
                    print(f"Please specify a DeepMHCII score inclusion threshold as a float between {worst} and {best}.")

            #rename deep_inc_thresh to dit: 
            dit = deep_inc_thresh
            #from the n_pol set, reduce to only peps with normalized IC50 >= dit
            better_n_pol = pd.DataFrame()
            for i in range(n_pol.shape[0]):
                score = float(n_pol.iloc[i,-1])
                if score >= dit:
                    better_n_pol = pd.concat([better_n_pol,pd.DataFrame(n_pol.iloc[i,:]).T],axis=0, ignore_index=True)
            df = better_n_pol
        else:
            #Make CapHLA-readable outputs:
            df = n_pol
        names = df["HLA Allele"]
        seq = df["Peptide Sequence"]
        
        n_df = pd.concat([seq,names],axis=1)
        n_df.to_csv("CapHLA-2_inputs.csv",index=False, header=False)
        print("Beginning Cap-HLA-2.0 processing...")
        CapHLA.main(gpu=False,BA=True)

if start_DeepMHC and start_Cap:
    #Finalize outputs:
    print("Finalizing outputs...")
    if minimize_DeepMHC_output == "Y":
        deep = better_n_pol
    else:
        deep = n_pol
    cap = pd.read_csv("CapHLA-2_outputs.csv")
    cap = cap.sort_values(["peptide","Allele Name"])
    cap.reset_index(inplace = True, drop = True)
    deep = deep.sort_values(["Peptide Sequence","HLA Allele"])
    deep.reset_index(inplace = True, drop = True)
    printer = pd.concat([deep,cap],axis=1)
    allele = printer["HLA Allele"]
    seq = printer["peptide"]
    core = printer["Predicted Binding Core"]
    m_mhc = printer["Maximum DeepMHCII Binding Score"]
    EL = printer["presentation_score"]
    BA = printer["affinity_score"]
    count_alleles = len(np.unique(printer["HLA Allele"]))
    ps_scores = pd.DataFrame()
    peps = printer.groupby("Predicted Binding Core")
    for i in peps:
        name = i[0]
        data = i[1]
        data.reset_index(inplace = True, drop = True)
        alleles = data.groupby("HLA Allele")
        pep_scores = dict()
        dmhc_d = dict()
        cap_d = dict()
        pres_d = dict()
        for k in alleles:
            k0 = k[0]
            k1 = k[1]
            dmhc = 0
            cap = 0
            pres = 0
            for row in range(k1.shape[0]):
                dmhc += k1.iloc[row].loc["Maximum DeepMHCII Binding Score"]
                cap += k1.iloc[row].loc["affinity_score"]
                pres += k1.iloc[row].loc["presentation_score"]
            dmhc = dmhc/k1.shape[0]
            cap = cap/k1.shape[0]
            pres = pres/k1.shape[0]
            pep_scores[k0] = (dmhc + cap + pres)/3
            dmhc_d[k0] = dmhc
            cap_d[k0] = cap
            pres_d[k0] = pres
        if len(pep_scores.keys()) == count_alleles:
            dmhc_d2 = dict()
            cap_d2 = dict()
            pres_d2 = pres_d
            for k in dmhc_d.keys():
                dmhc_d2[k] = denormalize(dmhc_d[k])
                cap_d2[k] = denormalize(cap_d[k])
            dmhc = pd.DataFrame(data = dmhc_d2.items(),columns = ["HLA Allele", "DeepMHCII IC50 (nM)"])
            cap = pd.DataFrame(data = cap_d2.items(),columns = ["HLA Allele","CapHLA-2.0 EC50 (nM)"])
            pres = pd.DataFrame(data = pres_d2.items(),columns = ["HLA Allele", "CapHLA-2.0 PB (%)"])
            PepScore = pd.DataFrame(data = pep_scores.items(),columns = ["HLA Allele","PepScore"])
            CompoundScore = 0
            for l in range(PepScore.shape[0]):
                try:
                    CompoundScore += float(PepScore.iloc[l,1])
                except IndexError:
                    print(PepScore)
            CompoundScore = CompoundScore/PepScore.shape[0]
            PepScore = pd.concat([pd.DataFrame([name]*len(a_list),columns=["Peptide Binding Core"]),dmhc,cap,pres,PepScore,pd.DataFrame([CompoundScore]*len(a_list),columns=["CompoundScore"])],axis=1)
            ps_scores = pd.concat([ps_scores,PepScore],axis=0,ignore_index=True)
    final = ps_scores.loc[:,~ps_scores.columns.duplicated()].copy()

elif start_DeepMHC:
    #Finalize outputs:
    print("Finalizing outputs...")
    deep = pd.read_csv("best_DeepMHCII_results.csv",index_col=0)
    deep = deep.sort_values(["Peptide Sequence","HLA Allele"])
    deep.reset_index(inplace = True, drop = True)
    print(deep.head())
    allele = printer["HLA Allele"]
    seq = printer["Peptide Sequence"]
    core = printer["Predicted Binding Core"]
    m_mhc = printer["Maximum DeepMHCII Binding Score"]
    
    #De-normalize BA and DeepMHCII binding scores:
    nMHC = pd.DataFrame()
    for i in m_mhc:
        score = float(i)
        ic50=10**(-(score-1)*np.log10(50000))
        nMHC = pd.concat([nMHC,pd.DataFrame([ic50])],axis=0,ignore_index=True)
    
    m_mhc = nMHC
    
    
    final = pd.concat([allele,seq,core,m_mhc],axis=1)
    final.columns = ["HLA-II Allele", "Peptide Sequence", "Predicted Binding Core", "IC50 (DeepMHCII), nM"]
    
else:
    #Finalize outputs:
    print("Finalizing outputs...")
    final = pd.read_csv("outputs/generated_peptides.csv")
    final = pd.DataFrame(final["Peptide_sequence"])
    
try:
    final = final.sort_values(by=["CompoundScore","Peptide Binding Core","HLA Allele"],ascending=False)
except KeyError:
    try:
        final = final.sort_values(by=["DeepMHCII IC50 (nM)","Peptide Binding Core","HLA Allele"],ascending=True)
    except KeyError:
        final = final.sort_values(by=["Peptide Binding Core","HLA Allele"],ascending=True)
final.reset_index(inplace=True,drop=True)
final.to_csv("final_ensemble_output.csv",header=True,index=False)


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
    df = pd.read_csv(path,sep=" ")
    n_df = pd.DataFrame(df.iloc[:,0])
    nn_df = pd.DataFrame()
    sp = pd.DataFrame()
    trip = False
    blank = pd.DataFrame(columns=range(0,36))
    printProgressBar(1,n_df.shape[0],prefix="Processing outputs...",suffix="",length=50)
    hold=pd.DataFrame(n_df.iloc[0,2:])
    for i in range(1,n_df.shape[0]):
        trip=False
        printProgressBar(i,n_df.shape[0],prefix="Processing outputs...",suffix="",length=50)
        try:
            look = n_df.iloc[i].iloc[0].split(",")[1]
            if look == "XXXX":
                trip = True
            if not trip:
                nsp = pd.DataFrame((n_df.iloc[i].iloc[0]).split(",")).T
                sp = pd.concat([sp,nsp],axis=1,ignore_index=True)
            else:
                trip=False
                nsp = pd.DataFrame((n_df.iloc[i].iloc[0]).split(",")).T
                sp = pd.concat([sp,nsp],axis=1,ignore_index=True)
                if sp.shape[1] > 3:#excludes the very first XXXX from being added as an empty point
                    sp =sp.dropna()
                    hold = pd.concat([hold,sp],axis=0,ignore_index=True)
                    sp = pd.DataFrame()
                
        except AttributeError:
            pass
    nn_df=hold
    mhc_name = pd.DataFrame(nn_df.iloc[:,1])
    mhc_name.columns = ["mhc_name"]
    pseud_mhc = pd.DataFrame(nn_df.iloc[:,3])
    pseud_mhc.columns = ["pep_seq"]
    one = pd.DataFrame(nn_df.iloc[:,11])
    one.columns = ["binding_core_1"]
    one_s = pd.DataFrame(nn_df.iloc[:,13])
    one_s.columns = ["binding_core_score_1"]
    two = pd.DataFrame(nn_df.iloc[:,15])
    two.columns = ["binding_core_2"]
    two_s = pd.DataFrame(nn_df.iloc[:,17])
    two_s.columns = ["binding_core_score_2"]
    three = pd.DataFrame(nn_df.iloc[:,19])
    three.columns = ["binding_core_3"]
    three_s = pd.DataFrame(nn_df.iloc[:,21])
    three_s.columns = ["binding_core_score_3"]
    four = pd.DataFrame(nn_df.iloc[:,23])
    four.columns = ["binding_core_4"]
    four_s = pd.DataFrame(nn_df.iloc[:,25])
    four_s.columns = ["binding_core_score_4"]
    five = pd.DataFrame(nn_df.iloc[:,27])
    five.columns = ["binding_core_5"]
    five_s = pd.DataFrame(nn_df.iloc[:,29])
    five_s.columns = ["binding_core_score_5"]
    six = pd.DataFrame(nn_df.iloc[:,31])
    six.columns = ["binding_core_6"]
    six_s = pd.DataFrame(nn_df.iloc[:,33])
    six_s.columns = ["binding_core_score_6"]
    print()
    rand_pol_df = pd.concat([mhc_name,pseud_mhc,one,one_s,two,two_s,three,three_s,four,four_s,five,five_s,six,six_s],axis=1)
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
        df = rand_pol
        names = df["HLA Allele"]
        seq = df["Peptide Sequence"]
        
        n_df = pd.concat([seq,names],axis=1)
        n_df.to_csv("CapHLA-2_random_inputs.csv",index=False, header=False)
        print("Beginning Cap-HLA-2.0 processing...")
        CapHLA.main(gpu=False,BA=True,randYN=True)
        
if start_DeepMHC and start_Cap:
    deep = rand_pol
    cap = pd.read_csv("CapHLA-2_random_outputs.csv")
    cap = cap.sort_values(["peptide","Allele Name"])
    cap.reset_index(inplace = True, drop = True)
    deep = deep.sort_values(["Peptide Sequence","HLA Allele"])
    deep.reset_index(inplace = True, drop = True)
    printer = pd.concat([deep,cap],axis=1)
    allele = printer["HLA Allele"]
    seq = printer["peptide"]
    core = printer["Predicted Binding Core"]
    m_mhc = printer["Maximum DeepMHCII Binding Score"]
    EL = printer["presentation_score"]
    BA = printer["affinity_score"]
    count_alleles = len(np.unique(printer["HLA Allele"]))
    ps_scores = pd.DataFrame()
    peps = printer.groupby("Predicted Binding Core")
    for i in peps:
        name = i[0]
        data = i[1]
        data.reset_index(inplace = True, drop = True)
        alleles = data.groupby("HLA Allele")
        pep_scores = dict()
        dmhc_d = dict()
        cap_d = dict()
        pres_d = dict()
        for k in alleles:
            k0 = k[0]
            k1 = k[1]
            dmhc = 0
            cap = 0
            pres = 0
            for row in range(k1.shape[0]):
                dmhc += k1.iloc[row].loc["Maximum DeepMHCII Binding Score"]
                cap += k1.iloc[row].loc["affinity_score"]
                pres += k1.iloc[row].loc["presentation_score"]
            dmhc = dmhc/k1.shape[0]
            cap = cap/k1.shape[0]
            pres = pres/k1.shape[0]
            pep_scores[k0] = (dmhc + cap + pres)/3
            dmhc_d[k0] = dmhc
            cap_d[k0] = cap
            pres_d[k0] = pres
        if len(pep_scores.keys()) == count_alleles:
            dmhc_d2 = dict()
            cap_d2 = dict()
            pres_d2 = pres_d
            for k in dmhc_d.keys():
                dmhc_d2[k] = denormalize(dmhc_d[k])
                cap_d2[k] = denormalize(cap_d[k])
            dmhc = pd.DataFrame(data = dmhc_d2.items(),columns = ["HLA Allele", "DeepMHCII IC50 (nM)"])
            cap = pd.DataFrame(data = cap_d2.items(),columns = ["HLA Allele","CapHLA-2.0 EC50 (nM)"])
            pres = pd.DataFrame(data = pres_d2.items(),columns = ["HLA Allele", "CapHLA-2.0 PB (%)"])
            PepScore = pd.DataFrame(data = pep_scores.items(),columns = ["HLA Allele","PepScore"])
            CompoundScore = 0
            for l in range(PepScore.shape[0]):
                try:
                    CompoundScore += float(PepScore.iloc[l,1])
                except IndexError:
                    print(PepScore)
            CompoundScore = CompoundScore/PepScore.shape[0]
            PepScore = pd.concat([pd.DataFrame([name]*len(a_list),columns=["Peptide Binding Core"]),dmhc,cap,pres,PepScore,pd.DataFrame([CompoundScore]*len(a_list),columns=["CompoundScore"])],axis=1)
            ps_scores = pd.concat([ps_scores,PepScore],axis=0,ignore_index=True)
    rand = ps_scores.loc[:,~ps_scores.columns.duplicated()].copy()
    
elif start_DeepMHC:
    #Finalize outputs:
    print("Finalizing control outputs...")
    deep = rand_pol
    deep = deep.sort_values(["Peptide Sequence","HLA Allele"])
    deep.reset_index(inplace = True, drop = True)
    print(deep.head())
    allele = printer["HLA Allele"]
    seq = printer["Peptide Sequence"]
    core = printer["Predicted Binding Core"]
    m_mhc = printer["Maximum DeepMHCII Binding Score"]
    count_alleles = len(np.unique(printer["HLA Allele"]))
    ps_scores = pd.DataFrame()
    peps = printer.groupby("Predicted Binding Core")
    for i in peps:
        name = i[0]
        data = i[1]
        data.reset_index(inplace = True, drop = True)
        alleles = data.groupby("HLA Allele")
        dmhc_d = dict()
        for k in alleles:
            k0 = k[0]
            k1 = k[1]
            dmhc = 0
            for row in range(k1.shape[0]):
                dmhc += k1.iloc[row].loc["Maximum DeepMHCII Binding Score"]
            dmhc = dmhc/k1.shape[0]
            dmhc_d[k0] = dmhc
        if len(pep_scores.keys()) == count_alleles:
            dmhc_d2 = dict()
            for k in dmhc_d.keys():
                dmhc_d2[k] = denormalize(dmhc_d[k])
            dmhc = pd.DataFrame(data = dmhc_d2.items(),columns = ["HLA Allele", "DeepMHCII IC50 (nM)"])
            PepScore = pd.concat([pd.DataFrame([name]*len(a_list),columns=["Peptide Binding Core"]),dmhc],axis=1)
            ps_scores = pd.concat([ps_scores,PepScore],axis=0,ignore_index=True)
    rand = ps_scores.loc[:,~ps_scores.columns.duplicated()].copy()
    
    
else:
    #Finalize outputs:
    print("Finalizing control outputs...")
    rand = pd.read_csv("outputs/random_peptides.csv")
    rand = pd.DataFrame(rand["Peptide_sequence"])
      
try:
    rand = rand.sort_values(by=["CompoundScore","Peptide Binding Core","HLA Allele"],ascending=False)
except KeyError:
    try:
        rand = rand.sort_values(by=["DeepMHCII IC50 (nM)","Peptide Binding Core","HLA Allele"],ascending=True)
    except KeyError:
        rand = rand.sort_values(by=["Peptide Binding Core","HLA Allele"],ascending=True)
rand.reset_index(inplace=True,drop=True)
rand.to_csv("rand_ensemble_output.csv",header=True,index=False)

print("Experimental data is available at 'final_ensemble_output.csv', and randomized control data is available at 'rand_ensemble_out.csv'")
