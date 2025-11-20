import pandas as pd
import logo

logo.main()

thresh = -1
indic = ""

while indic =="" or( indic != "Y" and indic != "N"):
    indic = input("Would you like to manually set the inclusion threshold? (y/n): ").upper()
    if indic != "Y" and indic != "N":
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
            while checkpoint != "Y" and checkpoint != "N":
                checkpoint = input("Peptide count in excess of 1000 may lead to long generation times, proceed? (y/n): ").upper()
                if checkpoint == "Y":
                    pass
                elif checkpoint == "N":
                    num_peps=-1
                else:
                    print("Please select y or n.")
            
    except ValueError:
        print("Please enter an integer value.")

core_GPT.main(cutoff=thresh,tag=None,final_pep_count = num_peps)