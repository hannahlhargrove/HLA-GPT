import pandas as pd
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import datetime

def encode(prompt,stoi): #take in a string and returns token integers
    acids = get_acids(prompt)
    tokens = tokenize(acids)
    i_list = []
    for i in tokens:
        i_list.append(stoi[i])
    return i_list

def decode(prompt_list): #take in a list of integers and returns a completed string (no start/end tokens)
    st_list = []
    for i in prompt_list:
        st_list.append(itos[i])
        if itos[i] == "<end>":
            break
    epi = ""
    for i in st_list[1:-1]:#cut the <start> and <end> cues
        epi = epi + i
    return epi
    
def check_in_iedb(seq): #Checks if a sequence is in the IEDB or fully contained within a sequence that's in the IEDB.

    iedb = pd.read_csv("iedb_crosscheck/epitope_table_export_1757877035.csv",index_col=0)
    i_epis = list(set((iedb["Epitope"])))

    for i in i_epis:
        if seq in i:
            return True
    return False
    
def get_acids(prompt): #turns a peptide into a list of individual acids + <start> and <end> tokens
    p_list = ["<start>"]
    for p in list(prompt):
        p_list.append(p)
    p_list.append("<end>")
    
    return p_list

def tokenize(p_list): #tokenizes a peptide list prompt into N-grams of N= [1,2,3]
    tokens = []
    for i in range(len(p_list)):
        g = ""
        for j in range(1):
            g += p_list[i+j]
        tokens.append(g)

    for i in range(len(p_list)-1):
        g = ""
        for j in range(2):
            g += p_list[i+j]
        tokens.append(g)

    for i in range(len(p_list)-2):
        g = ""
        for j in range(3):
            g += p_list[i+j]
        tokens.append(g)
    return tokens
    
def get_batch(split, train_data, val_data, batch_size, block_size):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
    
@torch.no_grad()
def estimate_loss(device, model, train_data, val_data, vocab_size, criterion, eval_iters, batch_size, block_size): 
    model.eval()
    out = []
    losses = torch.zeros(2, eval_iters)
        
    @torch.no_grad()
    def evaluate(device, train_data, val_data, vocab_size, model, criterion, batch_size, block_size):#============================================
        model.eval()
        out = []
        with torch.no_grad():
            for split in ['train', 'val']:
                x, y = get_batch(split, train_data, val_data, batch_size, block_size)
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred.view(-1, vocab_size), y.view(-1))
                out.append(loss.item())
        return out
        
    for i in range(eval_iters):
        losses[0, i], losses[1, i] = evaluate(device, train_data, val_data, vocab_size, model, criterion, batch_size, block_size)
    return losses.mean(dim=1).tolist()
        

# training function
def train(device, model, train_data, val_data, vocab_size, eval_iters, criterion, optimizer, num_iterations, batch_size, block_size):#============================================
    model.train()
    losses = []
    holder_1=0
    holder_2=0
    output_losses = []
    i = 0
    early_stop=False
    while i in range(num_iterations) and early_stop != True:
        x, y = get_batch('train',train_data,val_data, batch_size, block_size)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)  # shape (B, T, vocab_size)
        loss = criterion(y_pred.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if i % eval_iters == 0:
            # mean training loss over the last 100 batches
            mean_train_loss = sum(losses[-eval_iters:]) / float(eval_iters)
            
                    
            train_loss, val_loss = estimate_loss(device, model, train_data, val_data, vocab_size, criterion, eval_iters, batch_size, block_size)
            output_losses.append((i,train_loss, val_loss))
            print(f"train:{train_loss},val:{val_loss}")
            if val_loss >= holder_1 and holder_1>=holder_2 and holder_1 != 0:
                early_stop = True
            else:
                holder_2 = holder_1
                holder_1 = val_loss
        i+=1
        
    return output_losses
@torch.no_grad()

#Utilizes the trained model to generate a single novel peptide sequence
def generate_text(device,stoi,itos,model, block_size, input, max_new_tokens = 50, temperature = 1.0,min_new_tokens = 1):
    thinking = True
    while thinking:
        model.eval()
        min_new_tokens += len("<start>") 
        max_new_tokens += len("<start>") 
        out = ''
        if isinstance(input, str):
            input = encode(input,stoi)
        input = torch.tensor(input, dtype=torch.long, device=device).unsqueeze(0)
        stopper = False
        while len(list(out))<max_new_tokens and stopper == False:
            with torch.no_grad():
                input = input[:, -block_size:]
                y = model(input)  # shape (B, T, vocab_size)
                # take the logits at the final time-step and scale by the temperature
                y = y[0, -1, :] / temperature  # shape (vocab_size,)
                y = torch.softmax(y, dim=-1)
                next_token = torch.multinomial(y, 1)
                #
                if (itos[next_token.item()])[:8].find("<start>") == -1 and len(out)<1:
                    pass
                elif itos[next_token.item()][-5:].find("<end>") != -1:
                    if len(list(out))<min_new_tokens:
                        pass
                    else:
                         out += itos[next_token.item()]
                         stopper=True
                elif itos[next_token.item()].find("<start>") != -1 and len(out)>=1: #if there is a stray "<start>" token, pass over it.
                    pass
                else:
                    out += itos[next_token.item()]
                    input = torch.cat((input, next_token.unsqueeze(0)), dim=1)
                    
        if out[-5:] != "<end>":
            out += "<end>"
        seq = out[8:-5] #sequence without leader and ender, to check iedb with:
        #thinking = check_in_iedb(seq)
        thinking=False
    return out

#Generates a specified number of novel peptide sequences
def mass_generate_peps(device,stoi,itos,model, block_size, input, max_new_tokens = 50, temperature = 1.0, min_new_tokens=3,num_peps=10):
    print(f"Peptides to be generated: {num_peps}")
    blank = pd.DataFrame(columns=["Peptide_sequence","Timestamp"])
    blank.to_csv(f"{os.getcwd()}/outputs/generated_peptides.csv",header=True,mode="w",index=False)
    for i in range(num_peps):
        printProgressBar(i,num_peps,prefix = "", suffix = '', length = 50)
        g = generate_text(device,stoi, itos,model, block_size, input, max_new_tokens = max_new_tokens, temperature = temperature, min_new_tokens=min_new_tokens)
        gen_pep = pd.DataFrame(data=[g[8:-5]],columns=["Peptide_sequence"])
        mark = pd.DataFrame(data=[str(datetime.datetime.now())],columns=["Timestamp"])
        gen_pep = pd.concat([gen_pep,mark],axis=1)
        gen_pep.to_csv(f"{os.getcwd()}/outputs/generated_peptides.csv",header=False,mode="a",index=False)

    printProgressBar(num_peps,num_peps,prefix = "", suffix = '', length = 50)

#================================================GPT ARCHITECTURE=====================================================================
class SelfAttentionHead(nn.Module):
    """ one head of self-attention """

    def __init__(self, embed_dim, block_size, head_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        try:
            B,T,C = x.shape
        except ValueError:
            print(x)
            B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class MultiHeadSelfAttention(nn.Module):
    """ a stack of self-attention layers """

    def __init__(self, embed_dim, block_size, num_heads, head_size, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(embed_dim, block_size, head_size, dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads*head_size, embed_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, channels)
        head_out = [head(x) for head in self.heads] # a list of tensors of shape (B, T, hs)
        out = torch.cat(head_out, dim=-1) # concatenate along the feature dimension
        out = self.projection(out) # project back to the original feature dimension
        out = self.dropout(out)
        return out
    
# Position embeddings
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Feed forward network

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, block_size, num_heads, head_size, ff_hidden_dim, dropout=0.2):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, block_size, num_heads, head_size, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_hidden_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # apply the self-attention mechanism
        att = self.attention(x) # shape (B, T, C)
        # add skip connection and apply layer normalization
        x = x + self.norm1(att)  
        # apply the feed-forward network
        ffn = self.ffn(x)  # shape (B, T, C)
        # add skip connection and apply layer normalization
        x = x + self.norm2(ffn)
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size,embed_dim, block_size, num_heads, head_size, ff_hidden_dim, num_layers, dropout=0.2):
        super().__init__()
        self.emd = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = PositionalEncoding(embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, block_size, num_heads, head_size, ff_hidden_dim, dropout=dropout) #For each of the layers, establish the architecture of that transformer block.
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.emd(x)
        x = self.pos_emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.head(x)
        return x

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)

#=======================================================Any code adjustments to the model for personal use should be made below this line, to minimize technical issues. =====================================================================
def main(cutoff=-1,tag=None,final_pep_count=-1):
    data = pd.read_csv(f"{os.getcwd()}/inputs/mhc_ligand_table_export.csv",index_col=0) #Input file
    #if no cutoff specified, set cutoff to default values:
    if cutoff == -1:
        if tag == "DP":
            cutoff = 2000
        elif tag == "DQ":
            cutoff = 1000
        elif tag == "DR":
            cutoff = 10
        else:
            cutoff= 10

    df = pd.DataFrame()
    if tag != None:
        for i in range(data.shape[0]):
            if i%(round(data.shape[0]/100,0))==0:
                printProgressBar(i,data.shape[0],prefix = "Processing input data...", suffix = '', length = 50)
            r = pd.DataFrame(data.iloc[i]).T
            targ = r["Quantitative measurement"].iloc[0]
            if targ <= cutoff and tag in data.iloc[i,-1]: #if target value is below the threshold for binding (aka it's strong binding) and tag is in the HLA-II subtype label
                df = pd.concat([df,r],axis=0,ignore_index=True)
        printProgressBar(data.shape[0],data.shape[0],prefix = "Processing input data...", suffix = '', length = 50)
        df = df.sample(frac=1.0) #scramble the df dataset to ensure randomized data 
    else: #if there's no tag, include all subtype data in DR, DP and DQ
        for i in range(data.shape[0]):
            if i%(round(data.shape[0]/100,0))==0:
                printProgressBar(i,data.shape[0],prefix = "Processing input data...", suffix = '', length = 50)
            r = pd.DataFrame(data.iloc[i]).T
            targ = r["Quantitative measurement"].iloc[0]
            if targ <= cutoff and ("DP" in data.iloc[i,-1] or "DQ" in data.iloc[i,-1] or "DR" in data.iloc[i,-1]): #if target value is below the threshold for binding (aka it's strong binding) and the HLA-II subtype label is DP, DQ or DR
                df = pd.concat([df,r],axis=0,ignore_index=True)
        df = df.sample(frac=1.0) #scramble the df dataset to ensure randomized data 
        printProgressBar(data.shape[0],data.shape[0],prefix = "Processing input data...", suffix = '', length = 50)
    print()
    #Select only the epitopes: make them into a list of epitopes
    epis = list(df["Name"])
    
    #Establish a list of the unique n-gram tokens:
    token_set = []
    for i in range(len(epis)):
        if i%(round(len(epis)/100,0))==0:
            printProgressBar(i,len(epis),prefix = "Forming vocabulary...", suffix = '', length = 50)
        acids = get_acids(epis[i])
        tokens = tokenize(acids)
        for j in tokens:
            token_set.append(j)
        token_set = list(set(token_set))

    printProgressBar(len(epis),len(epis),prefix = "Forming vocabulary...", suffix = '', length = 50)
    
    chars = sorted(list(set(token_set)))
    chars_df = pd.DataFrame(chars)
    chars_list = []
    for i in range(chars_df.shape[0]):
        chars_list.append(chars_df.iloc[i,0])
    print() 
    # create a mapping from characters to integers -- this allows for the encoding and decoding of peptide sequences
    stoi = { ch:i for i,ch in enumerate(chars_list) }
    itos = { i:ch for i,ch in enumerate(chars_list) }
    
    vocab_size = len(chars)
    
    # let's now encode the entire text dataset and store it into a torch.Tensor
    n_epis = pd.DataFrame(epis)
    n_epis = n_epis.sample(frac=1.0)
    epi_list = []
    for i in range(n_epis.shape[0]):
        epi_list.append(n_epis.iloc[i,0])

    print("")
    text = []
    for i in epi_list:
        e_list = encode(i,stoi)
        for e in e_list:
            text.append(e)
                
    data = torch.tensor(text, dtype=torch.long)
    # Let's now split up the data into train and validation sets
    
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    #-------------------------------------------------------------------------
    # Model Initialization and Number of Parameters 
    # model hyperparameters
    vocab_size = len(chars)
    embed_dim = 384
    num_heads = 2
    head_size = 32
    ff_hidden_dim = 4*embed_dim
    num_layers = 6
    block_size = 256 # the context window length. Note: this is the number of prior observed tokens which are considered at once. This can be longer than a single peptide's worth of tokens, which will lead to a list of peptides being considered simultaneously. 
    dropout = 0.3
            
    # training parameters
    lr = 3e-4
    batch_size = 64         # this is the number of independent sequences which will be processed in parallel
    num_iterations = 10000
    eval_iters = 50
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
            
    model = GPTLanguageModel(vocab_size, embed_dim, block_size, num_heads, head_size, ff_hidden_dim, num_layers, dropout=dropout)
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    model = model.to(device)
    # intiatizing loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
    # Number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters())
            
    # Let's now train our transformer model
    # we will use the cross-entropy loss and the Adam optimizer to train the model.
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print()      
    print("Training model...")
    
    #This is the command to train the model based on the input data
    ls = train(device, model, train_data, val_data, vocab_size, eval_iters, criterion, optimizer, num_iterations, batch_size, block_size)
    
    outlet = pd.DataFrame(ls)
    outlet.to_csv(f"{os.getcwd()}/losses/{tag}_{cutoff}nM_losses_{str(datetime.datetime.now()).split(' ')[0]}.csv")
    print()
    mass_generate_peps(device,stoi,itos,model, block_size, "AAA", max_new_tokens=15, min_new_tokens=15, temperature=0.8, num_peps=final_pep_count)
