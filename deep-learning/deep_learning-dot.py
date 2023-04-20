#!/usr/bin/env python
# coding: utf-8

# In[3]:

import os
movie_titles_path = '../prize_dataset/movie_titles.csv'
movie_metadata_path = '../IMDB_data/merge4.csv'
combined_data_1_path = '../prize_dataset/combined_data_1.txt'
bellkor_requirements_path = './BellkorAlgorithm/requirements.txt'
bellkor_import_path = 'BellkorAlgorithm/Bellkor'
google_save_path = os.path.expanduser('~/google-drive/CSC 422/CSC422 Class Project/codes/checkpoints/')

# ## Load Data

# ### Load Movie Tiles w/o metadata

# In[5]:


import pandas as pd
import numpy as np
from io import StringIO
import re

for_pd = StringIO()
with open(movie_titles_path, encoding = 'ISO-8859-1') as movie_titles:
    for line in movie_titles:
        new_line = re.sub(r',', '|', line.rstrip(), count=2)
        print (new_line, file=for_pd)

for_pd.seek(0)

movie_titles = pd.read_csv(for_pd, sep='|', header=None, names=['Id', 'Year', 'Name']).set_index('Id')
del for_pd

print('Shape Movie-Titles:\t{}'.format(movie_titles.shape))
movie_titles.sample(5)


# In[6]:


from collections import deque

# Load single data-file
df_raw = pd.read_csv(combined_data_1_path, header=None, names=['User', 'Rating', 'Date'], usecols=[0, 1, 2])


# Find empty rows to slice dataframe for each movie
tmp_movies = df_raw[df_raw['Rating'].isna()]['User'].reset_index()
movie_indices = [[index, int(movie[:-1])] for index, movie in tmp_movies.values]

# Shift the movie_indices by one to get start and endpoints of all movies
shifted_movie_indices = deque(movie_indices)
shifted_movie_indices.rotate(-1)


# Gather all dataframes
user_data = []

# Iterate over all movies
for [df_id_1, movie_id], [df_id_2, next_movie_id] in zip(movie_indices, shifted_movie_indices):
    
    # Check if it is the last movie in the file
    if df_id_1<df_id_2:
        tmp_df = df_raw.loc[df_id_1+1:df_id_2-1].copy()
    else:
        tmp_df = df_raw.loc[df_id_1+1:].copy()
        
    # Create movie_id column
    tmp_df['Movie'] = movie_id
    
    # Append dataframe to list
    user_data.append(tmp_df)

# Combine all dataframes
df = pd.concat(user_data)
del user_data, df_raw, tmp_movies, tmp_df, shifted_movie_indices, movie_indices, df_id_1, movie_id, df_id_2, next_movie_id
print('Shape User-Ratings:\t{}'.format(df.shape))
df.sample(5)


# #### More formatting for user-data and only use X of the users (choose users with the most ratings) from (1/4) of the total data. Number subject to change.

# In[7]:


unique_movies = df['Movie'].nunique()
unique_users = df['User'].nunique()

print(f'Number of unique users:\t{unique_users}')
print(f'Number of unique movies:\t{unique_movies}')

pct_movies = unique_movies
pct_users = int(unique_users * 0.01)

filter_movies = df['Movie'].value_counts().sort_values(ascending=False)[:pct_movies].index

filter_users = df['User'].value_counts().sort_values(ascending=False)[:pct_users].index

df_filtered = df[df["Movie"].isin(filter_movies) & df["User"].isin(filter_users)]
del filter_movies, filter_users, df

# rename the users and movies with new ids start from 0
df_filtered['User'] = df_filtered['User'].astype("category")
df_filtered['Movie'] = df_filtered['Movie'].astype("category")
df_filtered['User'] = df_filtered['User'].cat.codes.values
df_filtered['Movie'] = df_filtered['Movie'].cat.codes.values

# make user the index and sort the index
df_filtered.set_index('User', inplace=True)
df_filtered.sort_index(inplace=True)

print(f'Number users: {df_filtered.index.nunique()}')
print(f'Number movies: {df_filtered["Movie"].nunique()}')
print(f'Shape: {df_filtered.shape}')
df_filtered.head(5)


# ### Shuffle the filtered dataframe and split into train and test set

# In[8]:


# Shuffle DataFrame
df_filtered = df_filtered.sample(frac=1).reset_index()

percent_test = .2

# create random seed
import random
seed = random.seed(42)


# Split train and set set based on percentage
df_train = df_filtered.sample(frac=1-percent_test, random_state=seed).reset_index(drop=True)
df_test = df_filtered.drop(df_train.index).reset_index(drop=True)

# split into X and y
X_train = df_train.drop('Rating', axis=1)
y_train = df_train['Rating']

X_test = df_test.drop('Rating', axis=1)
y_test = df_test['Rating']

df_train.head(10)


# ## Machine Learning Models

# #### General Setup

# In[9]:


df_filtered = df_filtered.drop('Date', axis=1)
X_train = X_train.drop('Date', axis=1)
X_test = X_test.drop('Date', axis=1)


# ### Matrix Factorization (Dot Product) w/ hidden layers
# Uses embeddings to represent users and movies. The dot product of user embeddings (n_users x e_dims) and movie embedding matrix (n_movies x e_dims) is a good approx of rating from user to movie.

# ##### Setup

# In[27]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
import numpy as np

class MovieDataset(Dataset):
    def __init__(self, users, movie_ids, ratings):
        self.users = torch.tensor(users, dtype=torch.int)
        self.movies = torch.tensor(movie_ids, dtype=torch.int)
        self.ratings = torch.tensor(ratings, dtype=torch.float)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

class RecommenderModel(nn.Module):
    def __init__(self, n_users, n_movies, embedding_size):
        super(RecommenderModel, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_size)
        self.movie_embedding = nn.Embedding(n_movies, embedding_size)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(2 * embedding_size + 1, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

    def forward(self, user, movie):
        user_vector = self.user_embedding(user).squeeze(1)
        movie_vector = self.movie_embedding(movie).squeeze(1)
        # mul and sum are needed because dot only works with 1D tensors
        x = torch.mul(user_vector, movie_vector).sum(1).unsqueeze(1)
        cat = torch.cat([user_vector, movie_vector, x], dim=1)
        dense = torch.relu(self.fc1(cat))
        dense = self.dropout(dense)
        dense = torch.relu(self.fc2(dense))
        dense = self.dropout(dense)
        dense = torch.relu(self.fc3(dense))
        dense = self.dropout(dense)
        dense = torch.relu(self.fc4(dense))
        dense = self.dropout(dense)
        y = self.fc5(dense)
        return y.flatten()

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for user, movie, rating in dataloader:
        user, movie, rating = user.to(device), movie.to(device), rating.to(device)
        optimizer.zero_grad()
        outputs = model(user, movie)
        loss = criterion(outputs, rating)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * user.size(0)
    return running_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion, device):
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for user, movie, rating in dataloader:
                user, movie, rating = user.to(device), movie.to(device), rating.to(device)
                outputs = model(user, movie)
                loss = criterion(outputs, rating)
                running_loss += loss.item() * user.size(0)
        return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    with torch.no_grad():
        model.eval()
        predictions = []
        ground_truth = []
        with torch.no_grad():
            for user, movie, rating in dataloader:
                user, movie, rating = user.to(device), movie.to(device), rating.to(device)
                outputs = model(user, movie)
                predictions.extend(outputs.view(-1).cpu().numpy())
                ground_truth.extend(rating.view(-1).cpu().numpy())
        return np.sqrt(mean_squared_error(ground_truth, predictions))

def training_loop(n_epochs, optimizer, model, criterion, train_dataloader, val_dataloader, device, model_name, restore_state=False):
    if restore_state:
        checkpoint = torch.load(f"{google_save_path}/{model_name}-checkpoint.pt")
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict']).to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        start_epoch = 0

    best_val_loss = float('inf')
    bad_epochs = 0

    print("Starting Training...")
    for epoch in range(start_epoch, n_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        val_loss = validate(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss}, Val Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{google_save_path}/{model_name}-best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= 5:
                print("Early stopping")
                break
        # save the model checkpoint
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss
            }, f"{google_save_path}/{model_name}-checkpoint.pt")


# In[28]:


# Training
n_users = df_filtered['User'].nunique()
n_movies = df_filtered['Movie'].nunique()
embedding_size = 100
batch_size = 2048

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RecommenderModel(n_users, n_movies, embedding_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

train_dataset = MovieDataset(X_train['User'], X_train['Movie'], y_train)
val_dataset = MovieDataset(X_test['User'], X_test['Movie'], y_test)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

n_epochs = 300

training_loop(n_epochs, optimizer, model, criterion, train_dataloader, val_dataloader, device, "nn-dot-embed100-batch2048")