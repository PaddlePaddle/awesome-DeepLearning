import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD
import torch.nn.functional as F

CONTEXT_SIZE = 4
EMBEDDING_DIM = 300
EPOCH = 20
VERVOSE = 5

corpus_text = "This tutorial will walk you through the key ideas of deep learning programming using Pytorch." \
              " Many of the concepts (such as the computation graph abstraction and autograd) " \
              "are not unique to Pytorch and are relevant to any deep learning tool kit out there.".split(' ')


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_size, context_size):
        super(CBOW, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        # return vector size will be context_size*2*embedding_size
        self.lin1 = nn.Linear(self.context_size * 2 * self.embedding_size, 512)
        self.lin2 = nn.Linear(512, self.vocab_size)

    def forward(self, inp):
        out = self.embeddings(inp).view(1, -1)
        out = out.view(1, -1)
        out = self.lin1(out)
        out = F.relu(out)
        out = self.lin2(out)
        out = F.log_softmax(out, dim=1)
        return out

    def get_word_vector(self, word_idx):
        word = Variable(torch.LongTensor([word_idx]))
        return self.embeddings(word).view(1, -1)


def train_cbow(data, unique_vocab, word_to_idx):
    cbow = CBOW(len(unique_vocab), EMBEDDING_DIM, CONTEXT_SIZE)

    nll_loss = nn.NLLLoss()  # loss function
    optimizer = SGD(cbow.parameters(), lr=0.001)

    print(len(data))

    for epoch in range(EPOCH):
        total_loss = 0
        for context, target in data:
            inp_var = Variable(torch.LongTensor([word_to_idx[word] for word in context]))
            target_var = Variable(torch.LongTensor([word_to_idx[target]]))

            cbow.zero_grad()
            log_prob = cbow(inp_var)
            loss = nll_loss(log_prob, target_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.data

        if epoch % VERVOSE == 0:
            loss_avg = float(total_loss / len(data))
            print("{}/{} loss {:.2f}".format(epoch, EPOCH, loss_avg))
    return cbow


def test_cbow(cbow, unique_vocab, word_to_idx):
    # test word similarity
    word_1 = unique_vocab[2]
    word_2 = unique_vocab[3]

    word_1_vec = cbow.get_word_vector(word_to_idx[word_1]).squeeze(dim=0)
    word_2_vec = cbow.get_word_vector(word_to_idx[word_2]).squeeze(dim=0)



    word_similarity = (word_1_vec.dot(word_2_vec) / (torch.norm(word_1_vec) * torch.norm(word_2_vec))).data.numpy()
    print("Similarity between '{}' & '{}' : {:0.4f}".format(word_1, word_2, word_similarity))


def main():
    # content processed as context/target
    # consider 2*CONTEXT_SIZE as context window where middle word as target
    data = list()
    for i in range(CONTEXT_SIZE, len(corpus_text) - CONTEXT_SIZE):
        data_context = list()
        for j in range(CONTEXT_SIZE):
            data_context.append(corpus_text[i - CONTEXT_SIZE + j])

        for j in range(1, CONTEXT_SIZE + 1):
            data_context.append(corpus_text[i + j])
        data_target = corpus_text[i]
        data.append((data_context, data_target))

    print("Some data: ", data[:3])

    unique_vocab = list(set(corpus_text))

    # mapping to index
    word_to_idx = {w: i for i, w in enumerate(unique_vocab)}

    # train model- changed global variable if needed
    cbow = train_cbow(data, unique_vocab, word_to_idx)

    # get two words similarity
    test_cbow(cbow, unique_vocab, word_to_idx)


if __name__ == "__main__":
    main()