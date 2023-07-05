import lmdb
import pandas as pd
BGC_train_dataset_inbalence = pd.read_csv('./BGC_train_dataset_inbalence.csv')
words = BGC_train_dataset_inbalence['TDsentence'].apply(lambda x:x.split())
train_words_list = []
for i in words:
    train_words_list += i
env_all = lmdb.open('../lmdb_BGC', readonly=True)
env_train = lmdb.open('../lmdb_train', subdir=True, map_size=307374182400, readonly=False, meminit=False, map_async=True)
with env_all.begin(write=False) as txn:
    with env_train.begin(write=True) as txn_train:
        for word in train_words_list:
            txn_train.put(word.encode('ascii'), txn.get(word.encode('ascii')))
