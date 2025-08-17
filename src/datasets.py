import pandas as pd
import logging
import os
import torch
from torch.utils.data import Dataset, DataLoader


class MovieLens100KLoader:

    def __init__(self, base_path):
        self.base_path = base_path

    def load_ratings(self):
        return pd.read_csv(
            os.path.join(self.base_path, "u.data"),
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"]
        )
    
    def load_info(self):
        return pd.read_csv(
            os.path.join(self.base_path, "u.item"),
            sep="|",
            header=None,
            encoding="ISO-8859-1",  # prevents UnicodeDecodeError
            names=[
                "item_id", "title", "release_date", "video_release_date", "IMDb_URL"
                ] + [f"genre_{i}" for i in range(19)]  # ignore genre columns for now
            )
        
    def name(self):
        return "MovieLens100K"


class MovieLens1MLoader:

    def __init__(self, base_path):
        self.base_path = base_path

    def load_ratings(self):
        ratings_columns = ['user_id', 'item_id', 'rating', 'timestamp']
        return pd.read_csv(os.path.join(self.base_path,'ratings.dat'), 
            sep='::', 
            names=ratings_columns, 
            engine='python')        

    def load_info(self):
        return pd.read_csv(os.path.join(self.base_path,'movies.dat'),
            sep='::', 
            names=['item_id', 'title', 'genre'], 
            engine='python', 
            encoding='iso-8859-1')
        
    def name(self):
        return "MovieLens1M"


class CsvLoader:
    def __init__(self, name, path):
        self.path = path
        self.dataset_name = name

    def load_ratings(self):
        return pd.read_csv(self.path)
    
    def load_info(self):
        return None
    
    def name(self):
        return self.dataset_name
        

class UserItemDataset(Dataset):
    def __init__(self, user_ids, item_ids, labels):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.labels = labels

    def __len__(self):
        return self.user_ids.size(0)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.labels[idx]
    
class UserItemPairsDataset(Dataset):
    def __init__(self, watch_matrix, time_matrix):
        self.watch_matrix = watch_matrix
        self.time_matrix = time_matrix

    def __len__(self):
        return self.watch_matrix.shape[0]

    def __getitem__(self, idx):
        return (idx+1, self.watch_matrix[idx,0], 
                self.watch_matrix[idx,1], 
                self.time_matrix[idx,0], 
                self.time_matrix[idx,1])


class MovieLensData:

    def __init__(self, loader):
        self.loader = loader
        self._ratings = None
        self._movie_info = None

    @property
    def ratings(self):
        if self._ratings is None:
            logging.info("loading ratings")
            self._ratings = self.loader.load_ratings()
        return self._ratings

    @property
    def info(self):
        if self._movie_info is None:
            logging.info("loading info")
            self._movie_info = self.loader.load_info()
        return self._movie_info

    @property 
    def name(self):
        return self.loader.name

    def get_watch_matrix(self, ratings=None, timestamps=False):

        if ratings is None:
            ratings = self.ratings

        # Get the number of users and movies
        n_users = ratings['user_id'].max()
        n_items = ratings['item_id'].max()

        # Create a binary watch matrix: [n_users x n_movies]
        wdtype = (torch.float32 if timestamps else torch.int32)
        watch_matrix = torch.zeros((n_users, n_items), dtype=wdtype)
        
        watch_vec =  watch_matrix.reshape(n_users * n_items)
        indexes = torch.tensor(((ratings["user_id"]-1)*n_items + (ratings["item_id"]-1)).to_numpy(), dtype=torch.long)
        if timestamps:
            watch_vec[indexes] = torch.tensor(ratings["timestamp"].to_numpy(), dtype=wdtype)
        else:
            watch_vec[indexes] = 1
        watch_matrix = watch_vec.reshape((n_users, n_items))
        return watch_matrix

    def get_dataset(self):
        wm = self.get_watch_matrix()
        base = torch.zeros((self.num_users,self.num_items), dtype=torch.int32)
        uidx = torch.arange(self.num_users).unsqueeze(1) + base
        iidx = torch.arange(self.num_items).unsqueeze(0) + base
        return UserItemDataset(uidx.flatten(), iidx.flatten(), wm.flatten().float())


    def get_pairs_dataset(self, first_id, second_id):
        df = self.ratings
        parts = []
        for idx, item_id in [(1,first_id), (2,second_id)]:
            pdf = df[df["item_id"]==item_id].copy()
            pdf["item_id"] = idx
            parts.append(pdf)
        pair_ratings = pd.concat(parts, axis=0)

        watch_mat = self.get_watch_matrix(pair_ratings)
        time_mat = self.get_watch_matrix(pair_ratings, timestamps=True)
        return UserItemPairsDataset(watch_mat, time_mat)

    def verify_ids(self, ids):
        diff = set(ids)-set(range(1,len(ids)+1))
        assert len(diff)==0

    def verify(self):
        self.verify_ids(self.ratings["user_id"])
        self.verify_ids(self.ratings["item_id"])

    @property
    def num_users(self):
        return int(self.ratings["user_id"].max())

    @property
    def num_items(self):
        return int(self.ratings["item_id"].max())


def enrich_cause_indexes(csdf, info):
    # Drop existing columns if present
    csdf = csdf.drop(columns=['treatment_idx', 'resp_idx'], errors='ignore')

    # Merge for treatment_idx
    csdf = csdf.merge(
        info[['title', 'item_id']],
        left_on='treatment_title',
        right_on='title',
        how='inner'
    ).rename(columns={'item_id': 'treatment_idx'}).drop(columns=['title'])

    # Merge for resp_idx
    csdf = csdf.merge(
        info[['title', 'item_id']],
        left_on='resp_title',
        right_on='title',
        how='inner'
    ).rename(columns={'item_id': 'resp_idx'}).drop(columns=['title'])

    return csdf


