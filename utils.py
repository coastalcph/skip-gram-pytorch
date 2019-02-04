import numpy as np

def load_embeddings_from_file(fname, max_vocab=-1):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []

    # load pretrained embeddings
    with open(fname) as f:
        for i, line in enumerate(f):
            if i == 0 and len(line.split()) == 2:
                continue
            else:
                word, vect = line.rstrip().split(' ', 1)

                vect = np.fromstring(vect, sep=' ')
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                assert word not in word2id
                word2id[word] = len(word2id)
                vectors.append(vect[None])
            if max_vocab > 0 and i >= max_vocab:
                break

    # compute new vocabulary / embeddings
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.concatenate(vectors, 0)

    return embeddings, word2id, id2word
