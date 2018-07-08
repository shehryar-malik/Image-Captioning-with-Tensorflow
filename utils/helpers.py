import numpy as np

class Tokenizer():
    def __init__(self, descriptions):
        self.desc = descriptions
        self.dictionary = self.create_tokenizer()
    
    # Create tokenizer
    def create_tokenizer(self):
        dic = {}
        idx = 1;
        lines = to_list(self.desc)
        for line in lines:
            split = line.split()
            for word in split:
                if word not in dic:
                    dic[word] = idx
                    idx += 1
        return dic
    
    # Create a sequence of integers for a given text
    def text_to_sequences(self, text):
        seq = []
        for word in text.split():
            seq.append(self.dictionary[word])
        
        return seq
    

#### Helper functions ####

# Create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, images, descriptions, vocab_size):
    X1, X2, y = list(), list(), list()
    all_out_seq = []
    max_len = max_length(descriptions)
    # walk through each image identifier
    for key, desc_list in descriptions.items():
        # walk through each description for the image
        for desc in desc_list:
            # encode the sequence
            seq = tokenizer.text_to_sequences(desc)
            # split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences(in_seq, max_len)
                # encode output sequence
                all_out_seq.append(out_seq) 
                # store
                X1.append(images[key][0])
                X2.append(in_seq)
                #y.append(out_seq)
    all_out_seq = np.array(all_out_seq)
    y = one_hot(all_out_seq, vocab_size)
    return np.array(X1), np.array(X2) , np.array(y)

# Convert a dictionary to a list
def to_list(dictionary):
    lst = list()
    for key in dictionary.keys():
        [lst.append(d) for d in dictionary[key]]
    return lst

def max_length(descriptions):
    lines = to_list(descriptions)
    return max(len(d.split()) for d in lines)

# Pre-pad a sequence to a maximum length
def pad_sequences(seq, max_len, value=0):
    add_zeros = max_len - len(seq)
    new_seq = [0 for i in range(add_zeros)]
    new_seq.extend(seq)
    
    return new_seq

# Create a one-hot vector
def one_hot(seq, num_classes):
    vec = np.zeros([seq.shape[0], num_classes])
    vec[np.arange(0,seq.shape[0]),seq] = 1
    
    return vec

    
    
    
    
