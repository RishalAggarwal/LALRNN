from lark import Lark
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from enum import Enum
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter


class lalrDataset(torch.utils.data.Dataset):
    """LALR dataset."""

    def __init__(self, lines_s, lines_p):
        self.lines_s=lines_s.copy()
        self.lines_p=lines_p.copy()
        self.length_s=[]
        self.length_p=[]
        for line in self.lines_s:
            self.length_s.append(len(line))
        for line in self.lines_p:
            self.length_p.append(len(line))
        assert self.length_s == self.length_p
        max_lenght = np.max(self.length_s)
        '''sets for chars'''
        self.s_chars = ['_'] + sorted(set(''.join(self.lines_s)))
        self.p_chars = ['_'] + sorted(set(''.join(self.lines_p)))
        ''' padding '''
        for i in range(len(self.lines_s)):
            while len(self.lines_s[i]) < max_lenght+1:
                self.lines_s[i] += '_'
        for i in range(len(self.lines_p)):
            while len(self.lines_p[i]) < max_lenght+1:
                self.lines_p[i] += '_'
        ''' int2chars and chars2int dictionaries '''
        self.int2s_char = enumerate(self.s_chars)
        self.s_char2int = {char: ind for ind, char in self.int2s_char}
        self.s_int2char = {self.s_char2int[char]: char for char in self.s_char2int}
        self.int2p_char = enumerate(self.p_chars)
        self.p_char2int = {char: ind for ind, char in self.int2p_char}
        ''' creating index value arrays'''
        input_seq = []
        target_seq = []
        for i in range(len(self.lines_p)):
            input_seq.append([self.p_char2int[character] for character in self.lines_p[i]])
            target_seq.append([self.s_char2int[character] for character in self.lines_s[i]])
        '''one hot encoding'''
        dict_size = len(self.p_char2int)
        seq_len = max_lenght+1
        batch_size = len(self.lines_p)
        self.input_seq = self.one_hot_encode(input_seq, dict_size, seq_len, batch_size)
        self.input_seq=torch.tensor(self.input_seq).cuda(0)
        dict_size = len(self.s_char2int)
        self.target_seq = self.one_hot_encode(target_seq, dict_size, seq_len, batch_size)
        self.target_seq=torch.tensor(self.target_seq).cuda(0)

    def __len__(self):
        return self.input_seq.shape[0]

    def __getitem__(self, idx):
        return [self.input_seq[idx],self.target_seq[idx],]

    def lastpadindex(self,batch):
        "I'm sure there's a more clever way to do this.."
        input = batch[0]
        target = batch[1]
        for i in range(input.shape[1]):
            if (input[:, i].equal(input[:, -1])):
                return input[:, :i + 1], target[:, :i + 1]
        return input, target
    def get_s_chars(self):
        return set(self.s_chars)
    def get_s_char2int(self):
        return self.s_char2int
    def get_tokenmap(self):
        return [self.s_int2char[i] for i in sorted(self.s_int2char.keys())]
    def one_hot_encode(self,sequence, dict_size, seq_len, batch_size):
        ''' Creating a multi-dimensional array of zeros with the desired output shape '''
        features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)

        ''' Replacing the 0 at the relevant character index with a 1 to represent that character '''
        for i in range(batch_size):
            for u in range(seq_len):
                features[i, u, sequence[i][u]] = 1
        return features




class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, input):
        output, hidden = self.gru(input)
        return output, hidden

    def reinit(self):
        '''Reinitialize weights'''

        def weights_init(l):
            if hasattr(l, 'weight') and isinstance(l.weight, torch.Tensor):
                nn.init.xavier_uniform_(l.weight.data)
            if hasattr(l, 'bias') and isinstance(l.bias, torch.Tensor):
                nn.init.uniform_(l.bias)

        self.apply(weights_init)


'''5s_rna_data files'''
with open("/home/rishal/lalrnn/5s_data/5s_shortlisted/dbn.txt") as f:
    lines_s_5srna = [line.strip() for line in f.readlines()]
with open("/home/rishal/lalrnn/5s_data/5s_shortlisted/seq.txt") as f:
    lines_p_5srna = [line.strip() for line in f.readlines()]
f.close()

lines_p_5srna_train1, lines_p_5srna_test, lines_s_5srna_train1, lines_s_5srna_test = train_test_split(lines_p_5srna, lines_s_5srna, test_size=0.1, random_state=42)
lines_p_5srna_train, lines_p_5srna_val, lines_s_5srna_train, lines_s_5srna_val = train_test_split(lines_p_5srna_train1, lines_s_5srna_train1, test_size=0.22, random_state=42)

'''srp_rna_data files'''
with open("/home/rishal/lalrnn/srp_data/srp_shortlisted/dbn.txt") as f:
    lines_s_srprna = [line.strip() for line in f.readlines()]
with open("/home/rishal/lalrnn/srp_data/srp_shortlisted/seq.txt") as f:
    lines_p_srprna = [line.strip() for line in f.readlines()]
f.close()

lines_p_srprna_train1, lines_p_srprna_test, lines_s_srprna_train1, lines_s_srprna_test = train_test_split(lines_p_srprna, lines_s_srprna, test_size=0.1, random_state=42)
lines_p_srprna_train, lines_p_srprna_val, lines_s_srprna_train, lines_s_srprna_val = train_test_split(lines_p_srprna_train1, lines_s_srprna_train1, test_size=0.22, random_state=42)

'''trna_data files'''
with open("/home/rishal/lalrnn/trna_data/trna_shortlisted/dbn.txt") as f:
    lines_s_trna = [line.strip() for line in f.readlines()]
with open("/home/rishal/lalrnn/trna_data/trna_shortlisted/seq.txt") as f:
    lines_p_trna = [line.strip() for line in f.readlines()]
f.close()

lines_p_trna_train1, lines_p_trna_test, lines_s_trna_train1, lines_s_trna_test = train_test_split(lines_p_trna, lines_s_trna, test_size=0.1, random_state=42)
lines_p_trna_train, lines_p_trna_val, lines_s_trna_train, lines_s_trna_val = train_test_split(lines_p_trna_train1, lines_s_trna_train1, test_size=0.22, random_state=42)

grammar = '''?e: DOT
 | LPAREN RPAREN
 | e LPAREN e RPAREN 
 | e DOT
 | LPAREN e RPAREN
 | e LPAREN RPAREN
DOT: "."
LPAREN: "("
RPAREN: ")"
'''

test_data=lalrDataset(lines_s_srprna_test,lines_p_srprna_test)
testloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
parser = Lark(grammar, start='e', parser='lalr')
tokenmap = [str(t.pattern).replace(r'\\', '').strip("'") for t in parser.terminals]
tokenmap.append("_")
assert set(tokenmap)==test_data.get_s_chars()
tokenmap=test_data.get_tokenmap()

from lalrnn_6_canonical import SimpleGenerativeLALRNN
#decoder = DecoderRNN(100,4)
decoder = SimpleGenerativeLALRNN(grammar, 'e', tokenmap, '_', test_data.get_s_char2int())
encoder = EncoderRNN(5,100)
encoder.cuda(0)
decoder.cuda(0)

checkpoint=torch.load('/home/rishal/lalrnn/paper_th1_canon_same_grammar_latest.pth.tar')
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

ppv_list=[]
sensitivity_list=[]
valid_num=0
lalrnn=True
for i, data in enumerate(testloader):
    if lalrnn==True:
        decoder.set_test()
        input,target = test_data.lastpadindex(data)
        encoder_outputs, encoder_hidden = encoder(input)
        decoder_input=target
        encoder_hidden = encoder_hidden.squeeze(0)
        decoder_hidden = encoder_hidden
        decoder_output, _ = decoder(decoder_input, decoder_hidden,input)
    else:
        input, target = test_data.lastpadindex(data)
        encoder_outputs, encoder_hidden = encoder(input)
        decoder_output=decoder.predict(encoder_hidden,target.size(1))
    string1=torch.argmax(decoder_output[:, :target.size(1), :], dim=2)
    string1=string1.squeeze(0)
    string2=torch.argmax(target,dim=2)
    string2=string2.squeeze(0)
    str1=[]
    str2=[]
    for x in range(len(string1)):
        str1.append(tokenmap[string1[x]])
    str1=''.join(str1)
    print('predicted')
    print(str1)
    for x in range(len(string2)):
        str2.append(tokenmap[string2[x]])
    str2=''.join(str2)
    print('true')
    print(str2)
    str2=str2.replace('_','')
    str1=str1.replace('_','')
    stack_s1 = []
    stack_s2 = []
    pairs_s1 = set()
    pairs_s2 = set()
    str1 = list(str1)
    str2 = list(str2)
    assert len(str1) == len(str2)
    for i in range(len(str1)):
        if str1[i] == '(':
            stack_s1.append(i)
        if str2[i] == '(':
            stack_s2.append(i)
        if str1[i] == ')':
            pairs_s1.add((stack_s1[-1], i))
            stack_s1.pop()
        if str2[i] == ')':
            pairs_s2.add((stack_s2[-1], i))
            stack_s2.pop()
    TP = len(pairs_s1 & pairs_s2)
    FP = len(pairs_s1 - pairs_s2)
    FN = len(pairs_s2 - pairs_s1)
    #print(TP,FP,FN)
    if TP+FP==0:
        FP=1
    if TP+FN==0:
        FN=1
    ppv_list.append(TP/(TP+FP))
    #print(ppv_list)
    sensitivity_list.append(TP/(TP+FN))

print('validity ', valid_num/(i+1))
ppv_list=np.array(ppv_list)
sensitivity_list=np.array(sensitivity_list)
f1=(2*ppv_list*sensitivity_list)/(ppv_list+sensitivity_list)
f1=np.nan_to_num(f1)
print('ppv',ppv_list.mean())
print('sensitivity',sensitivity_list.mean())
print('f1',f1.mean())