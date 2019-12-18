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
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)

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


def exp_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer

def trainlalr(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,offset_target=True,max_length=3451, numclasses=4):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    encoder_outputs, encoder_hidden = encoder(input_tensor)

    if offset_target:
        decoder_input = F.pad(target_tensor, (0, 0, 1, 0, 0, 0))
    else:
        decoder_input = target_tensor
        encoder_hidden = (encoder_hidden[0] +
                          encoder_hidden[1])
#        encoder_hidden = encoder_hidden.squeeze(0)

    decoder_hidden = encoder_hidden
    decoder_output, _ = decoder(decoder_input, decoder_hidden, input_tensor)
    decoder_output=decoder_output[:,:target_tensor.size(1),:]
    #print(decoder_output)
    mask = (torch.argmax(target_tensor, dim=2) != 0).float().cuda(0)
    for k in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[k,j]==0:
                mask[k,j]=1
                break
    loss = criterion(decoder_output[:,:target_tensor.size(1),:],target_tensor)
    loss[mask==0,:]=0
    loss = loss.sum()/(mask.sum()*10)
    loss.backward()
    nn.utils.clip_grad_norm_(encoder.parameters(), 10)
    nn.utils.clip_grad_norm_(decoder.parameters(), 10)
    mask = (torch.argmax(target_tensor, dim=2) != 0).float().cuda(0)
    correct = (torch.argmax(decoder_output[:, :target_tensor.size(1), :], dim=2) == torch.argmax(target_tensor,dim=2)).float()
    correct=(correct*mask).sum()
    correct /= mask.sum()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss,correct

def testlalr(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,offset_target=True,max_length=3451, numclasses=4):
    with torch.no_grad():
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        if offset_target:
            decoder_input = F.pad(target_tensor, (0, 0, 1, 0, 0, 0))
        else:
            decoder_input = target_tensor

            encoder_hidden = (encoder_hidden[0] +
                          encoder_hidden[1])

        decoder_hidden = encoder_hidden
        decoder_output, _ = decoder(decoder_input, decoder_hidden, input_tensor)
        decoder_output=decoder_output[:,:target_tensor.size(1),:]
        mask = (torch.argmax(target_tensor, dim=2) != 0).float().cuda(0)
        for k in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[k,j]==0:
                    mask[k,j]=1
                    break
        loss = criterion(decoder_output[:,:target_tensor.size(1),:],target_tensor)
        loss[mask==0,:]=0
        loss = loss.sum()/(mask.sum()*10)
        mask = (torch.argmax(target_tensor, dim=2) != 0).float().cuda(0)
        correct = (torch.argmax(decoder_output[:, :target_tensor.size(1), :], dim=2) == torch.argmax(target_tensor,dim=2)).float()
        correct=(correct*mask).sum()
        correct /= mask.sum()
        return loss,correct

'''5s_rna_data files'''
with open("/home/rishal/lalrnn/5s_data/5s_shortlisted/dbn_can.txt") as f:
    lines_s_5srna = [line.strip() for line in f.readlines()]
with open("/home/rishal/lalrnn/5s_data/5s_shortlisted/seq_can.txt") as f:
    lines_p_5srna = [line.strip() for line in f.readlines()]
f.close()

lines_p_5srna_train1, lines_p_5srna_test, lines_s_5srna_train1, lines_s_5srna_test = train_test_split(lines_p_5srna, lines_s_5srna, test_size=0.1, random_state=42)
lines_p_5srna_train, lines_p_5srna_val, lines_s_5srna_train, lines_s_5srna_val = train_test_split(lines_p_5srna_train1, lines_s_5srna_train1, test_size=0.22, random_state=42)

'''srp_rna_data files'''
with open("/home/rishal/lalrnn/srp_data/srp_shortlisted/dbn_can.txt") as f:
    lines_s_srprna = [line.strip() for line in f.readlines()]
with open("/home/rishal/lalrnn/srp_data/srp_shortlisted/seq_can.txt") as f:
    lines_p_srprna = [line.strip() for line in f.readlines()]
f.close()

lines_p_srprna_train1, lines_p_srprna_test, lines_s_srprna_train1, lines_s_srprna_test = train_test_split(lines_p_srprna, lines_s_srprna, test_size=0.1, random_state=42)
lines_p_srprna_train, lines_p_srprna_val, lines_s_srprna_train, lines_s_srprna_val = train_test_split(lines_p_srprna_train1, lines_s_srprna_train1, test_size=0.22, random_state=42)

'''trna_data files'''
with open("/home/rishal/lalrnn/trna_data/trna_shortlisted/dbn_can.txt") as f:
    lines_s_trna = [line.strip() for line in f.readlines()]
with open("/home/rishal/lalrnn/trna_data/trna_shortlisted/seq_can.txt") as f:
    lines_p_trna = [line.strip() for line in f.readlines()]
f.close()

lines_p_trna_train1, lines_p_trna_test, lines_s_trna_train1, lines_s_trna_test = train_test_split(lines_p_trna, lines_s_trna, test_size=0.1, random_state=42)
lines_p_trna_train, lines_p_trna_val, lines_s_trna_train, lines_s_trna_val = train_test_split(lines_p_trna_train1, lines_s_trna_train1, test_size=0.22, random_state=42)

'''training data'''

lines_s_train=lines_s_5srna_train+lines_s_trna_train+lines_s_srprna_train
lines_p_train=lines_p_5srna_train+lines_p_trna_train+lines_p_srprna_train

'''testing data'''
lines_s_test=lines_s_5srna_test+lines_s_trna_test+lines_s_srprna_test
lines_p_test=lines_p_5srna_test+lines_p_trna_test+lines_p_srprna_test

'''validation data'''

lines_s_val=lines_s_5srna_val+lines_s_trna_val+lines_s_srprna_val
lines_p_val=lines_p_5srna_val+lines_p_trna_val+lines_p_srprna_val

train_data=lalrDataset(lines_s_train,lines_p_train)
val_data=lalrDataset(lines_s_val,lines_p_val)
test_data=lalrDataset(lines_s_test,lines_p_test)
grammar = '''?e: DOT
 | LPARENA RPARENU
 | LPARENC RPARENG
 | LPARENG RPARENC
 | LPARENG RPARENU
 | LPARENU RPARENG
 | LPARENU RPARENA
 | e LPARENA e RPARENU 
 | e LPARENC e RPARENG 
 | e LPARENG e RPARENC 
 | e LPARENG e RPARENU 
 | e LPARENU e RPARENG 
 | e LPARENU e RPARENA 
 | e DOT
 | LPARENA e RPARENU
 | LPARENC e RPARENG
 | LPARENG e RPARENC
 | LPARENG e RPARENU
 | LPARENU e RPARENG
 | LPARENU e RPARENA
 | e LPARENA RPARENU
 | e LPARENC RPARENG
 | e LPARENG RPARENC
 | e LPARENG RPARENU
 | e LPARENU RPARENG
 | e LPARENU RPARENA
DOT: "."
LPARENA: "A"
LPARENC: "C"
LPARENG: "G"
LPARENU: "U"
RPARENA: "a"
RPARENC: "c"
RPARENG: "g"
RPARENU: "u"
'''

parser = Lark(grammar, start='e', parser='lalr')
tokenmap = [str(t.pattern).replace(r'\\', '').strip("'") for t in parser.terminals]
tokenmap.append("_")
assert set(tokenmap)==train_data.get_s_chars()
tokenmap=train_data.get_tokenmap()
batch_size=5
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,shuffle=True, num_workers=0)
#from autoex import DecoderRNN
from lalrnn_all_lets import SimpleGenerativeLALRNN
#decoder = DecoderRNN(100,4)
decoder = SimpleGenerativeLALRNN(grammar, 'e', tokenmap, '_', train_data.get_s_char2int())
encoder = EncoderRNN(5,400)
encoder.cuda(0)
decoder.cuda(0)
criterion = nn.BCELoss(reduction='none')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.0001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.0001)
best_acc=0
best_loss=100
writer=SummaryWriter(log_dir='runs/lalr_lets_l400_bi_dropout3')
for e in range(350):
    for i,data in enumerate(trainloader):
        decoder.train()
        decoder.set_train()
        input,target = train_data.lastpadindex(data)
        loss,correct = trainlalr(input, target, encoder, decoder, encoder_optimizer, decoder_optimizer,criterion, offset_target=False)
        writer.add_scalar('Loss/train',loss,(e*(int(1372/batch_size))+(i+1)))
        writer.add_scalar('Accuracy/train',correct,(e*(int(1327/batch_size))+(i+1)))
        print("Epoch {},Iteration {}, loss: {:.3f},  Accuracy: {:.3f}".format(e+1,i + 1, loss, correct))
        #decoder_optimizer=exp_lr_scheduler(decoder_optimizer,e+1)
    if e%3==0:
        decoder.eval()
        test_loss=torch.tensor([]).cuda(0)
        accuracy=torch.tensor([]).cuda(0)
        for i, data in enumerate(valloader):
            input, target = train_data.lastpadindex(data)
            loss, correct = testlalr(input, target, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
                                      offset_target=False)
            test_loss=torch.cat((test_loss,loss.unsqueeze(0)))
            accuracy=torch.cat((accuracy,correct.unsqueeze(0)))
        writer.add_scalar('Loss/test', test_loss.mean(), int(e/3) +1)
        writer.add_scalar('Accuracy/test', accuracy.mean(), int(e/3) + 1)
        print("Epoch {},test, loss: {:.3f},  Accuracy: {:.3f}".format(e/3 + 1, test_loss.mean(), accuracy.mean()))
        torch.save({'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                    'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                    'epoch': e}, '/home/rishal/lalrnn/lets_l400_bi_dropout3_latest.pth.tar')
        if test_loss.mean()<best_loss:
            torch.save({'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                    'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                    'epoch': e}, '/home/rishal/lalrnn/lets_l400_bi_dropout3_best.pth.tar')
            best_loss=test_loss.mean()
for i in range(100):
    try:
        checkpoint=torch.load('/home/rishal/lalrnn/lets_l400_bi_dropout3_best.pth.tar')
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        break
    except:
        continue


testloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
valid_num=0
lalrnn=True
for i, data in enumerate(testloader):
    if lalrnn==True:
        decoder.set_test()
        decoder.eval()
        input,target = test_data.lastpadindex(data)
        encoder_outputs, encoder_hidden = encoder(input)
        decoder_input=target
        encoder_hidden = encoder_hidden.squeeze(0)
        decoder_hidden = encoder_hidden
        decoder_output, _ = decoder(decoder_input, decoder_hidden, input)
    else:
        input, target = test_data.lastpadindex(data)
        encoder_outputs, encoder_hidden = encoder(input)
        decoder_output=decoder.predict(encoder_hidden,target.size(1))
    string=torch.argmax(decoder_output[:, :target.size(1), :], dim=2)
    string=string.squeeze(0)
    str=[]
    for x in range(len(string)):
        str.append(tokenmap[string[x]])
    str=''.join(str)
    print(str)
    str=str.replace('_','')
    try:
        parser.parse(str)
        valid_num+=1
    except:
        continue
print('validity ', valid_num/(i+1))
