#!/usr/bin/env python
# coding: utf-8

import os
movie_titles_path = '../prize_dataset/movie_titles.csv'
movie_metadata_path = '../IMDB_data/merge4.csv'
combined_data_1_path = '../prize_dataset/combined_data_1.txt'
bellkor_requirements_path = './BellkorAlgorithm/requirements.txt'
bellkor_import_path = 'BellkorAlgorithm/Bellkor'
google_save_path = os.path.expanduser('~/google-drive/CSC 422/CSC422 Class Project/codes/checkpoints/')

# In[70]:


import pandas as pd
import numpy as np

movie_metadata_raw = pd.read_csv(movie_metadata_path)

movie_metadata = movie_metadata_raw[movie_metadata_raw['imdbID'].notnull()].set_index('Name').drop('imdbID', axis=1)
del movie_metadata_raw

na_count = movie_metadata.isna().sum()
print('Number of missing values in each column:\n{}'.format(na_count))

movie_metadata = movie_metadata.set_index('MovieID').sort_index()
# movie_metadata.head(5)


# In[71]:


from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer

# Impute missing values using the mean for continuous variables
mean_imputer = SimpleImputer(strategy='mean')
movie_metadata[['Year', 'NumRating', 'duration', 'AggregateAverageRating']] = mean_imputer.fit_transform(movie_metadata[['Year', 'NumRating', 'duration', 'AggregateAverageRating']])

# Fill missing values in 'ContentRating' with the mode value
mode_content_rating = movie_metadata['contentRating'].mode().iloc[0]
movie_metadata['contentRating'].fillna(mode_content_rating, inplace=True)

# Encode 'ContentRating' column using LabelEncoder
le_content_rating = LabelEncoder()
movie_metadata['contentRating'] = le_content_rating.fit_transform(movie_metadata['contentRating'])

# Replace NaN values with an empty string
movie_metadata['Genre'].fillna('', inplace=True)
movie_metadata['actors'].fillna('', inplace=True)
movie_metadata['Directors'].fillna('', inplace=True)
movie_metadata['creators'].fillna('', inplace=True)
movie_metadata['Keywords'].fillna('', inplace=True)
movie_metadata['description'].fillna('', inplace=True)

# Split comma-separated values into lists
movie_metadata['Genre'] = movie_metadata['Genre'].apply(lambda x: str(x).split(', '))
movie_metadata['actors'] = movie_metadata['actors'].apply(lambda x: str(x).split(', '))
movie_metadata['Directors'] = movie_metadata['Directors'].apply(lambda x: str(x).split(', '))
movie_metadata['creators'] = movie_metadata['creators'].apply(lambda x: str(x).split(', '))

# Encode multi-value columns using MultiLabelBinarizer
mlb_encoders = {}
multi_label_columns = ['Genre', 'actors', 'Directors', 'creators']
min_rows_with_1 = int(movie_metadata.shape[0] * 0.005)
for col in multi_label_columns:
    mlb = MultiLabelBinarizer()
    encoded_col = mlb.fit_transform(movie_metadata[col])
    encoded_df = pd.DataFrame(encoded_col, columns=[f"{col}_{c}" for c in mlb.classes_], index=movie_metadata.index)
    
    # Filter columns with at least 100 rows containing a 1
    cols_to_keep = encoded_df.columns[encoded_df.sum(axis=0) >= min_rows_with_1]
    encoded_df = encoded_df[cols_to_keep]

    movie_metadata = pd.concat([movie_metadata.drop(col, axis=1), encoded_df], axis=1)
    mlb_encoders[col] = mlb

# Encode text columns using TfidfVectorizer
tfidf_encoders = {}
max_features = 500
text_columns = ['Keywords', 'description']
for col in text_columns:
    tfidf = TfidfVectorizer(max_features=max_features)
    encoded_col = tfidf.fit_transform(movie_metadata[col].astype(str)).toarray()
    encoded_df = pd.DataFrame(encoded_col, columns=[f"{col}_{c}" for c in tfidf.get_feature_names_out()], index=movie_metadata.index)
    movie_metadata = pd.concat([movie_metadata.drop(col, axis=1), encoded_df], axis=1)
    tfidf_encoders[col] = tfidf

movie_metadata.head(5)


# In[72]:


import torch

# Create a dense PyTorch tensor
dense_tensor = torch.from_numpy(movie_metadata.to_numpy())

# Calculate the number of non-zero elements in the tensor
num_nonzero = torch.nonzero(dense_tensor).size(0)

# Calculate the total number of elements in the tensor
total_elements = dense_tensor.numel()

# Calculate the sparsity ratio
sparsity_ratio = 1.0 - (num_nonzero / total_elements)

# Print the sparsity ratio
print("Sparsity ratio: {:.2f}%".format(sparsity_ratio * 100))


# ### Load user-data structure (1/4 to save memory + speed up compute) and preprocess to extract all rating to form a matrix. File structure is messy mix of json and csv.

# In[73]:


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

# In[74]:


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

# In[75]:


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


# In[76]:


df_filtered = df_filtered.drop('Date', axis=1)
X_train = X_train.drop('Date', axis=1)
X_test = X_test.drop('Date', axis=1)


# ### Matrix Factorization (Dot Product) w/ hidden layers
# Uses embeddings to represent users and movies. The dot product of user embeddings (n_users x e_dims) and movie embedding matrix (n_movies x e_dims) is a good approx of rating from user to movie.

# ##### Setup

# In[24]:


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
            bad_epochs = 0
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


# In[101]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class HybridMovieDataset(Dataset):
    def __init__(self, users, movie_ids, movie_metadata, ratings):
        self.users = torch.tensor(users, dtype=torch.int)
        self.movie_ids = torch.tensor(movie_ids, dtype=torch.int)
        self.movie_metadata = torch.tensor(movie_metadata, dtype=torch.float)
        self.ratings = torch.tensor(ratings, dtype=torch.float)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.movie_metadata[self.movie_ids[idx]], self.ratings[idx]

class HybridRecommenderModel(nn.Module):
    def __init__(self, n_users, n_movies, embedding_size, metadata_size):
        super(HybridRecommenderModel, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_size)
        self.movie_embedding = nn.Embedding(n_movies, embedding_size)
        self.movie_metadata = nn.Linear(metadata_size, embedding_size)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(2 * embedding_size + 1, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.m0 = nn.Linear(embedding_size, 256)
        self.m1 = nn.Linear(128, embedding_size)
        self.fc5 = nn.Linear(32, 1)

    def forward(self, user, movie_metadata):
        user_vector = self.user_embedding(user).squeeze(1)
        metadata_vector = self.movie_metadata(movie_metadata)

        metadata_vector = torch.relu(self.m0(metadata_vector))
        metadata_vector = self.dropout(metadata_vector)
        metadata_vector = torch.relu(self.fc2(metadata_vector))
        metadata_vector = self.dropout(metadata_vector)
        metadata_vector = self.m1(metadata_vector)
        metadata_vector_norm = nn.functional.normalize(metadata_vector, p=2, dim=1)

        # mul and sum are needed because dot only works with 1D tensors
        x = torch.mul(user_vector, metadata_vector_norm).sum(1).unsqueeze(1)
        cat = torch.cat([user_vector, metadata_vector_norm, x], dim=1)
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


# In[102]:


# Training
n_users = df_filtered['User'].nunique()
n_movies = df_filtered['Movie'].nunique()
embedding_size = 20
batch_size = 1024
# get the columns of the metadata
n_features_metadata = movie_metadata.shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridRecommenderModel(n_users, n_movies, embedding_size, n_features_metadata).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

train_dataset = HybridMovieDataset(X_train['User'], X_train['Movie'], movie_metadata.values, y_train)
val_dataset = HybridMovieDataset(X_test['User'], X_test['Movie'], movie_metadata.values, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

n_epochs = 50

training_loop(n_epochs, optimizer, model, criterion, train_dataloader, val_dataloader, device, "nn-hybrid")

