import numpy as np
import codecs
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

def get_embeddings():
    with codecs.open("/home/embeddings", "r", "utf-8") as f:
            elems = f.readline().strip().split()
            if len(elems) == 2:
                header = True
                dim = int(elems[1])
            else:
                header = False
                dim = len(elems)-1
    words = []
    word2vec = {}
    with codecs.open("/home/embeddings", "r", "utf-8") as f:
        line_count = 0
        if header:
            f.readline()
            line_count = 1
        for line in f:
            line_count += 1
            elems = line.strip().split()
            if len(elems) == dim + 1:
                word = elems[0]
                try:
                    vec = np.asarray(elems[1:], dtype=np.float32)
                    words.append(word)
                    word2vec[word] = vec
                except ValueError as e:
                    print("ValueError: Skipping line {}".format(line_count))
            else:
                msg = "Error: Skipping line {}. ".format(line_count)
                msg += "Expected {} elements, found {}.".format(dim+1, len(elems))
        print(line_count)
                
    return words, word2vec
words, word2vec = get_embeddings()
embeddings = []
for embed in word2vec.values():
    embeddings.append(list(embed))
embeddings = np.array(embeddings)
word_mapping = {}
for i in range(len(words)):
    word_mapping[words[i]] = i
def subtask_file_mapping(subtask):
    mapping ={"1A": "1A.english", "2A": "2A.medical", "2B": "2B.music", "1B": "1B.italian", "1C": "1C.spanish"}
    return mapping[subtask]
def load_data(subtask, datadir, split):
    if(split == "train"):
        split = "training"
    data = []
    gold = {}
    data_path = f'{datadir}/{split}/data/{subtask_file_mapping(subtask)}.{split}.data.txt'
    gold_path = f'{datadir}/{split}/gold/{subtask_file_mapping(subtask)}.{split}.gold.txt'
    with open(data_path, 'r') as f:
        lines = f.read().split('\n')
        print(len(lines))
        for line in lines:
            word = line.split('\t')[0].lower()
            word = word.replace(" ", "_")
            data.append(word)
    with open(gold_path, 'r') as f:
        lines = f.read().split('\n')
        for i,line in enumerate(lines):
            gold_data = []
            golds = line.lower().split('\t') 
            for gold_word in golds:
                gold_data.append(gold_word.replace(" ", "_"))
            gold[data[i]] = gold_data
            
    return data, gold
test_data, test_gold = load_data("1A", "SemEval2018-Task9", "test")
train_data, train_gold  = load_data("1A", "SemEval2018-Task9", "train")
def make_embedding_matrix(word2vec, words, seed=0):
    np.random.seed(seed)
    dim = 200
    dtype = np.float32
    matrix = np.zeros((len(words), dim), dtype=dtype)
    count = 0
    for (i,word) in enumerate(words):
        if word in word2vec.keys():
            matrix[i] = word2vec[word]
        else:
            count += 1
            matrix[i] = np.random.uniform(low=-0.5, high=0.5) / dim
    return matrix
def get_vocabulary():
    with open('./SemEval2018-Task9/vocabulary/1A.english.vocabulary.txt') as f:
        words = f.read().split('\n')
        vocab = []
        for i,word in enumerate(words):
            word = word.lower()
            word = word.replace(' ', '_')
            vocab.append(word)
        return vocab
        

vocab = get_vocabulary()


class CustomDataset(Dataset):
    def __init__(self, data, vocab,  word2vec,num_negative_samples=100):
        self.data = data
        self.vocab = vocab
        self.word2vec = word2vec
        self.num_negative_samples = num_negative_samples
        self.num_items = len(vocab)
        self.item_indices = list(range(self.num_items))
        self.samples = set()
        for word in self.data.keys():
            for positive_item in self.data[word]:
                self.samples.add((word, positive_item, 1))                
                negative_items = self.negative_sample(positive_item)
                for negative_item in negative_items:
                    self.samples.add((word, negative_item, 0))
        self.samples = list(self.samples)      
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return1 = [],
        return2 = []
        if self.samples[idx][0] in self.word2vec.keys():
            return1 = torch.Tensor(self.word2vec[self.samples[idx][0]])
        else:
            return1 = torch.Tensor(self.word2vec['hi'])
        if self.samples[idx][1] in self.word2vec.keys():
            return2 = torch.Tensor(self.word2vec[self.samples[idx][1]])
        else:
            return2 = torch.Tensor(self.word2vec['hi'])
        return return1,return2, self.samples[idx][2]

    
    def negative_sample(self, positive_item):
        negative_samples = []
        for _ in range(self.num_negative_samples):
            negative_item = np.random.randint(self.num_items)
            while negative_item == positive_item:
                negative_item = np.random.randint(self.num_items)
            negative_samples.append(self.vocab[negative_item])
        
        return negative_samples
traindata = CustomDataset(train_gold,vocab,word2vec,100)
device = "cuda:1"

class HypernymPredictionModel(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size, output_size):
        super(HypernymPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size1 + input_size2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc4 = nn.Linear(hidden_size//4, hidden_size//2)
        self.fc5 = nn.Linear(hidden_size//2, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1) 
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return self.sigmoid(x)
model = HypernymPredictionModel(200, 200, 200,1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()
model = model.to(device)
data_loader = DataLoader(traindata, batch_size=512, shuffle=True)

num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0.0
    for i,(word,sample,label) in enumerate(data_loader):
        word =word.to(device)
        sample =sample.to(device)
        label = label.to(device)
        outputs = model(word,sample)
        loss = criterion(outputs.squeeze(), label.float())
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(data_loader):.4f}')


class HypernymDataset(Dataset):
    def __init__(self, word, candidates,  word2vec):
        self.word = word
        self.candidates = candidates
        self.samples = set()
        self.word2vec = word2vec
        for candidate in candidates:
            if word == candidate:
                continue
            else:
                self.samples.add((word,candidate))
        self.samples = list(self.samples)
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return1 = [],
        return2 = []
        
        if self.samples[idx][0] in self.word2vec.keys():
            return1 = torch.Tensor(self.word2vec[self.samples[idx][0]])
        else:
            return1 = torch.Tensor(self.word2vec['hi'])
        if self.samples[idx][1] in self.word2vec.keys():
            return2 = torch.Tensor(self.word2vec[self.samples[idx][1]])
        else:
            return2 = torch.Tensor(self.word2vec['hi'])
        return return1,return2, self.samples[idx][1]
model.eval()
results = []

with torch.no_grad():
    for idx in range(len(test_data)):
        probabilities = []
        candidates = []
        print(idx)
        hypernymdata = HypernymDataset(test_data[idx],vocab,word2vec)
        hypernymloader = DataLoader(hypernymdata, batch_size=len(test_data))
        for i, (word, candidate, candidate_word) in enumerate(hypernymloader):
            word = word.to(device)
            candidate = candidate.to(device)
            probability = model(word,candidate)
            probability = list(probability.cpu().numpy().reshape(probability.shape[0],))
            probabilities.extend(list(probability))
            candidates.extend(candidate_word)
        probabilities = np.array(probabilities)
        candidates = np.array(candidates)
        results.append(list(candidates[np.flip(np.argsort(probabilities))][:15]))
with open('Neural_predicted_100.txt', 'w') as f:
    for result in results:
        string = "\t".join(result)
        f.write(string + "\n")

torch.save(model.to("cpu"), "neural_100.pt")