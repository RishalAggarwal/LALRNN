from lark import Lark
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from enum import Enum
import random
from collections import OrderedDict
import numpy as np

class IndicesExpansionLayer(nn.Module):
    '''Index vals by indices into a zero output vector of output_size.
    Input has batch index, output does not'''

    def __init__(self, indices, output_size):
        super(IndicesExpansionLayer, self).__init__()
        self.indices = indices
        self.output_size = output_size

    def forward(self, vals):
        x = torch.zeros(self.output_size, dtype=vals.dtype, device=vals.device)
        x[self.indices] = vals
        return x


class SimpleGenerativeLALRNN(nn.Module):
    '''NN generative architecture defined by LALR parse table using single Dense networks in each state'''

    class Action(Enum):
        SHIFT = 0
        REDUCE = 1
        END = 2

    class State:
        '''A single LALR state'''

        def _add_layer(self, l, is_output=False):
            '''adds the specified layer to the model'''
            i = len(self.model.state_layers)
            self.model.state_layers.append(l)
            setattr(self.model, "layer%i" % i, l)  # does pytorch need this
            if is_output:
                self.output_layers.append(l)
            else:
                self.layers.append(l)
            return l

        def __init__(self, state, model):
            '''take parser state and dict mapping terminal strings to (index,char) and construct our own state'''
            self.name=str(OrderedDict(sorted(state.items())))
            terminals = model.terminals
            n_latent = model.n_latent
            self.action = None  # shift or reduce
            self.output_index = []  # for shift, how to map token vector output to full tokens
            self.output_tokens = []  # array of tokens that can be output
            self.layers = []  # generate latent vector
            self.output_layers = []  # generate token vector
            # shift tables
            self.next = {}  # indexed by token
            self.goto = {}  # indexed by nonterminal
            # reduce values
            self.popcnt = 0
            self.lhs = None
            self.model = model

            Action = SimpleGenerativeLALRNN.Action

            keys = sorted(state.keys())
            if not keys:
                self.action = Action.END
            else:
                action = state[keys[0]][0].name
                if action == 'Shift':
                    self.action = Action.SHIFT
                elif action == 'Reduce':
                    self.action = Action.REDUCE
                else:
                    raise ValueError("Unknown action %s" % action)

            if self.action == Action.SHIFT or self.action == Action.END:
                if self.action == Action.END:
                    token = terminals['$END'][1]  # pad char
                    self.output_tokens.append(token)
                    self.output_index.append(terminals['$END'][0])
                else:  # SHIFT
                    # create next and goto tables (and verify no shift/reduce conflicts)
                    for k in keys:
                        kaction = state[k][0].name
                        if kaction != action:  # shift/reduce conflict
                            raise ValueError("Shift/reduce conflict in state %s" % state)
                        # k is the (non)terminal name
                        if k in terminals:  # next
                            token = terminals[k][1]
                            self.next[token] = state[k][1]
                            self.output_tokens.append(token)
                            self.output_index.append(terminals[k][0])
                        else:  # a nonterminal - goto
                            self.goto[k] = state[k][1]
                # define layers, must build them manually so they are all properly initialized
                if len(self.output_tokens)==1:
                    self._add_layer(IndicesExpansionLayer(self.output_index, len(terminals)), is_output=True)
                else:
                    self._add_layer(nn.Linear(n_latent, n_latent))
                    self._add_layer(nn.Dropout(0.3))
                    self._add_layer(nn.Tanh())
                    self._add_layer(nn.Linear(n_latent, len(self.output_tokens)), is_output=True)
                    self._add_layer(nn.Dropout(0.3))
                    self._add_layer(nn.Softmax(dim=1), is_output=True)
                    self._add_layer(IndicesExpansionLayer(self.output_index, len(terminals)), is_output=True)

            else:  # reduce
                # set popcnt and lhs
                rule = state[keys[0]][1]
                self.popcnt = len(rule.expansion)
                self.lhs = rule.origin.name

                # verify all rules are the same (and no shift/reduce conflicts)
                for k in keys[1:]:
                    if rule != state[k][1]:
                        raise ValueError("Different reduce rules in state %s" % state)

                # define layers - concat is done by caller
                self._add_layer(nn.Linear(n_latent * self.popcnt, n_latent))
                self._add_layer(nn.Dropout(0.3))
                self._add_layer(nn.Tanh())

        def call(self, inputs):
            '''Apply state to inputs'''
            output = []
            x = inputs
            if len(self.output_tokens)==1:
                output.append(x)
                for l in self.output_layers:
                    output.append(l(torch.tensor([1],dtype=torch.float).cuda(0)))
                return output
            for l in self.layers:
                x=l(x)
            output.append(x)
            if self.output_layers:
                for l in self.output_layers:
                    x = l(x)
                x[x==1]=0.99
                if self.model.steps_count!=0:
                    x[self.model.token2index['_']]=0
                    x = self.model.correct_base_prediction(x)
                else:
                    x[self.model.token2index['_']]=1
                    output.append(x)
                    return output
                if len(self.model.base_pair_index_stack) > 0:#update end indices
                    self.model.base_pair_index_stack[-1] = self.model.base_pair_index_stack[-1][
                        self.model.base_pair_index_stack[-1] >= self.model.current_index]
                if len(self.model.base_pair_index_stack)!=0 and self.model.base_pair_index_stack[-1].shape[0]==1 and self.model.base_pair_index_stack[-1][0]==self.model.current_index: #last base pair possible
                    x[self.model.token2index[self.model.sequence[self.model.current_index].lower()]] = 1
                    output.append(x)
                    return output

                if self.model.tokens[torch.argmax(x)].isupper():
                    if len(self.model.base_pair_index_stack)==0:
                        self.model.start_indices=[len(self.model.sequence)]
                    else:
                        self.model.start_indices=self.model.base_pair_index_stack[-1]
                    if self.model.check_base_open(self.model.start_indices,self.model.sequence[self.model.current_index],self.model.current_index): #base pair possible?
                        output.append(x)

                        return output
                    else:
                        x[self.model.token2index[self.model.tokens[torch.argmax(x)]]] = 0
                #print(self.model.base_pair_index_stack,self.model.open_brack_stack)
                if self.model.tokens[torch.argmax(x)].islower():  #probably not necessary
                    output.append(x)
                    return output

                output.append(x)
            return output

    def __init__(self, grammar, start, tokens, padchar,token_dict,n_latent=400):
        '''construct LALRNN model from grammar. tokens must match tokens of the grammar in be in
        the correct order for the input one-hot encoding.  These need to include the specified pad
        character which is _not_ in the grammar'''
        super(SimpleGenerativeLALRNN, self).__init__()
        self.n_latent = n_latent
        self.parser = Lark(grammar, start=start, parser='lalr')
        self.lalr = self.parser.parser.parser.parser
        self.origstates = self.lalr.states
        self.start = self.lalr.start_states
        self.end = self.lalr.end_states
        self.tokens=tokens  # these are actual characters in one-hot order
        self.token2index = {}  # indexed by character, return one hot position
        self.state_layers = []
        self.test=False #prediction flag (no teacher forcing)
        self.open_brack_stack=[]
        self.steps_count=0
        self.current_index=0
        self.base_open_flag=False #flag to check if base pair opening possible
        self.sequence=[] #primary sequence of RNA
        self.base_pair_index_stack=[] #locations for possible base pairs
        self.base_pairs={'A':['U'],'G':['C','U'],'C':['G'],'U':['A','G']} #all possible base pairs (canonical)
        self.start_indices=[]
        # map from char to name
        terminaldict = {str(t.pattern).replace(r'\\', '').strip("'"): t.name for t in self.parser.terminals}
        # sanity check specified tokens vs tokens in grammar
        self.terminals = {}  # map from parser terminal names to one-hot position and character
        for t in token_dict:
            self.token2index[t] = token_dict[t]
            if t == padchar:
                self.terminals['$END'] = (token_dict[t], t)
                continue
            if t not in terminaldict:
                raise ValueError("%s not found as terminal in grammar" % t)
            self.terminals[terminaldict[t]] = (token_dict[t], t)
        for t in terminaldict.keys():
            if t not in tokens:
                raise ValueError("%s is terminal in grammar, but not specifed as a token" % t)

        # for each parser state, create our version of the state
        self.states = {i: self.State(state, self) for (i, state) in self.origstates.items()}

    def reinit(self):
        '''Reinitialize weights'''

        def weights_init(l):
            if hasattr(l, 'weight'):
                nn.init.xavier_uniform_(l.weight.data)
            if hasattr(l, 'bias'):
                nn.init.uniform_(l.bias)

        self.apply(weights_init)

    def forward(self, inputs, hidden,input_seqs):
        '''Execute one pass of generation during training.
        inputs should be [initial latent vector, one-hot encoding of outputs]
        The true outputs are used to control the state machine during training.
        '''
        current_state = self.start
        SHIFT = self.Action.SHIFT
        REDUCE = self.Action.REDUCE
        END = self.Action.END

        latent_vecs = hidden
        true_outputs = inputs
        output_shape = true_outputs.shape
        timesteps = output_shape[1]
        if output_shape[2] != len(self.tokens):
            raise ValueError(
                "incompatible lengths of one-hot encoded output: %d vs %d" % (output_shape[2], self.tokens))

        # must process each member of the batch individually
        outputs = []

        for (latent_vec, true_output,input_seq) in zip(latent_vecs, true_outputs, input_seqs):
            self.current_index = 0
            latent_stack = latent_vec.unsqueeze(0).clone().unsqueeze(0) # we need to make everything have a batch dimension of length 1
            state_stack = [self.start]
            self.sequence.clear()
            for j in range(true_output.shape[0]):
                if torch.argmax(input_seq[j])==1:
                    self.sequence.append('A')
                elif torch.argmax(input_seq[j])==2:
                    self.sequence.append('C')
                elif torch.argmax(input_seq[j])==3:
                    self.sequence.append('G')
                elif torch.argmax(input_seq[j])==4:
                    self.sequence.append('U')
                if true_output[j].equal(true_output[-1]):
                    self.steps_count=j
                    break
            self.open_brack_stack=[]
            self.base_pair_index_stack=[]
            predicted_output = []
            while self.current_index < timesteps:
                #print(len(latent_stack))
                if state_stack[-1] == self.start:
                    statenum = state_stack[-1]
                    statenum = statenum[list(statenum.keys())[0]]
                    state = self.states[statenum]
                else:
                    statenum = state_stack[-1]
                    state = self.states[statenum]
                if state.action == END or state.action == SHIFT:
                    # apply
                    (newlatent, newtoken) = state.call(latent_stack[-1])
                    # save predicted token (encoded)
                    predicted_output.append(newtoken)
                    true_token = self.tokens[torch.argmax(true_output[self.current_index])]
                    test_token = self.tokens[torch.argmax(newtoken)]

                   # print(test_token,len(self.open_brack_stack),self.current_index-1)
                    if self.test:
                        if test_token.isupper(): #append to open bracket stack
                            if len(self.base_pair_index_stack) == 0:
                                self.start_indices = [len(self.sequence)]
                            else:
                                self.start_indices = self.base_pair_index_stack[-1]
                            end_indices = self.get_base_pair_indices(self.start_indices,
                                                                           self.sequence[
                                                                               self.current_index],
                                                                           self.current_index)
                            self.base_pair_index_stack.append(np.asarray(end_indices))
                            self.open_brack_stack.append(self.sequence[self.current_index])
                        if test_token.islower():
                            #print(self.base_pair_index_stack, self.current_index)
                            self.base_pair_index_stack.pop()
                            #print(self.base_pair_index_stack)
                            self.open_brack_stack.pop()
                    if not self.test:
                        if true_token.isupper(): #append to open bracket stack
                            if len(self.base_pair_index_stack) == 0:
                                self.start_indices = [len(self.sequence)]
                            else:
                                self.start_indices = self.base_pair_index_stack[-1]
                            end_indices = self.get_base_pair_indices(self.start_indices,
                                                                           self.sequence[
                                                                               self.current_index],
                                                                           self.current_index)
                            self.base_pair_index_stack.append(np.asarray(end_indices))
                            self.open_brack_stack.append(self.sequence[self.current_index])
                        if true_token.islower():

                            self.base_pair_index_stack.pop()

                            self.open_brack_stack.pop()

                    # print(self.open_brack_stack)
                    if state.action == SHIFT:
                        # get correct next state and push
                        if self.test:
                            #print(self.open_brack_stack,test_token)

                            state_stack.append(state.next[test_token])
                        else:
                            state_stack.append(state.next[true_token])
                        latent_stack = torch.cat((latent_stack, newlatent.unsqueeze(0)), dim=0)
                        # push vec
                    self.current_index += 1  # we have output something
                    self.steps_count -= 1


                    # else END - stay here

                elif state.action == REDUCE:
                    popped_latent = latent_stack[-state.popcnt:]
                    popped_latent=popped_latent.squeeze(1)
                    latent_stack=latent_stack[:-state.popcnt]
                    #del latent_stack[-state.popcnt:]
                    del state_stack[-state.popcnt:]
                    # merge latent vectors
                    (newlatent,) = state.call(popped_latent.view(popped_latent.shape[0]*popped_latent.shape[1]))
                    # figure out next state
                    if state_stack[-1] == self.start:
                        statenum = state_stack[-1]
                        statenum = statenum[list(statenum.keys())[0]]
                    else:
                        statenum = state_stack[-1]
                    gotostate = self.states[statenum]
                    nextstate = gotostate.goto[state.lhs]
                    # push
                    state_stack.append(nextstate)
                    #print(latent_stack.shape,newlatent.shape)
                    latent_stack = torch.cat((latent_stack, newlatent.unsqueeze(0).unsqueeze(0)),dim=0)
            #print(latent_stack.shape)
            outputs.append(torch.stack(predicted_output))
        return torch.stack(outputs), latent_stack[-1]
    def set_test(self):
        self.test=True
    def set_train(self):
        self.test=False
    def check_base_open(self,indices,base,current_index):

        for i in indices:
            for j in range(i-1,current_index,-1):
                if self.sequence[j] in self.base_pairs[base]:

                    return True
        return False
    def get_base_pair_indices(self,indices,base,curent_index):
        end_indices=[]
        for i in indices:
            new_indices=self.get_end_indices(i,base,curent_index)
            end_indices=end_indices+new_indices
       # print(list(set(end_indices)), base, i, curent_index)
        if len(end_indices)>0:
            return list(set(end_indices))
        return None
    def get_end_indices(self,index,base,current_index):
        indices=[]
        for j in range(index-1,current_index,-1):
            if self.sequence[j] in self.base_pairs[base]:
                indices.append(j)
        return indices
    def correct_base_prediction(self,x):
        base=self.sequence[self.current_index]
        base=[base.lower(),base.upper()]
        for c in self.tokens :
            if c not in base and c.isalpha():
                x[self.token2index[c]]=0
        return x





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