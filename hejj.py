import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

# Загрузка данных о рейтингах фильмов
ratings_data = pd.read_csv('ratings.csv')

# Преобразование идентификаторов пользователей и фильмов в уникальные целые значения
user_ids = ratings_data['userId'].unique()
movie_ids = ratings_data['movieId'].unique()
user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
ratings_data['user_idx'] = ratings_data['userId'].map(user_to_idx)
ratings_data['movie_idx'] = ratings_data['movieId'].map(movie_to_idx)

# Разделение данных на обучающую и тестовую выборки
train_data = ratings_data.sample(frac=0.8, random_state=42)
test_data = ratings_data.drop(train_data.index)

# Количество уникальных пользователей и фильмов
num_users = len(user_ids)
num_movies = len(movie_ids)

# Создание модели рекомендательной системы на основе глубокого обучения
user_input = Input(shape=(1,))
movie_input = Input(shape=(1,))
user_embedding = Embedding(num_users, 10)(user_input)
movie_embedding = Embedding(num_movies, 10)(movie_input)
user_flatten = Flatten()(user_embedding)
movie_flatten = Flatten()(movie_embedding)
concat = Concatenate()([user_flatten, movie_flatten])
dense1 = Dense(128, activation='relu')(concat)
dense2 = Dense(64, activation='relu')(dense1)
output = Dense(1)(dense2)
model = Model(inputs=[user_input, movie_input], outputs=output)

# Компиляция и обучение модели
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit([train_data['user_idx'], train_data['movie_idx']], train_data['rating'], epochs=10, batch_size=64, verbose=1)

# Оценка модели на тестовой выборке
loss = model.evaluate([test_data['user_idx'], test_data['movie_idx']], test_data['rating'], verbose=0)
print(f"Потери на тестовой выборке: {loss}")

# Получение рекомендаций для пользователя
user_id = 1
user_idx = user_to_idx[user_id]
user_movies = ratings_data[ratings_data['user_idx'] == user_idx]['movie_idx'].unique()
unrated_movies = np.setdiff1d(np.arange(num_movies), user_movies)
user_input = np.full_like(unrated_movies, user_idx)
movie_input = unrated_movies
predictions = model.predict([user_input, movie_input]).flatten()
top_movie_indices = np.argsort(-predictions)[:10]
recommended_movie_ids = [list(movie_to_idx.keys())[list(movie_to_idx.values()).index(movie_idx)] for movie_idx in top_movie_indices]
print(f"Рекомендации для пользователя с ID {user_id}:")
print(recommended_movie_ids)
