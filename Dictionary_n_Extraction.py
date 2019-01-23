import os, numpy as np
from collections import Counter

# Defining our dictionary making function
def make_Dict(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for msg in emails:
        with open(msg) as m:
            for i, line in enumerate(m):
                if i == 2:  # We are interested in 3rd line of text file only
                    words = line.split()
                    all_words += words

    dict = Counter(all_words)
    to_remove = dict.keys()
    for i in to_remove:
        if i.isalpha() == False:
            del dict[i]
        elif len(i) == 1:
            del dict[i]
    dict = dict.most_common(3000)
    return dict

#Preparing our dictionary
train_dir = 'train-mails'
dict = make_Dict(train_dir)


#Defining our mails extracting function
def extract(mail_dir):
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    for fil in files:
      with open(fil) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dict):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1
    return features_matrix