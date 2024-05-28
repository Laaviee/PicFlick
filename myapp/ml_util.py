import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import linear_kernel
import random
import os

lemmatizer = WordNetLemmatizer()

a = pd.read_csv('all1.csv')
a['Genre'].fillna('nil')
movies = a[['Name','Genre','Description','Lang','Cast', 'Director', 'Year', 'Run Time', 'img_L', 'Rating']]
movies['Genre'].fillna('nil')

stop_words = set(stopwords.words('english'))

# ******************* DATA PREPROCESSING ******************************
def preprocess_text(text):
    words = word_tokenize(text.lower())  # Convert text to lowercase and tokenize
    my_sent=[lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    finalsent = ' '.join(my_sent)
    symbols = '!\"#$%&(),*+-./:;<=>?@[\]^_`{|}~\n'

    finalsent = finalsent.replace("n't", " not")
    finalsent = finalsent.replace("'m", " am")
    finalsent = finalsent.replace("'s", " is")
    finalsent = finalsent.replace("'re", " are")
    finalsent = finalsent.replace("'ll", " will")
    finalsent = finalsent.replace("'ve", " have")
    finalsent = finalsent.replace("'d", " would")
    for i in symbols:
        finalsent = np.char.replace(finalsent, i, ' ')
    return finalsent

movies.loc[:, 'Genre'] = movies['Genre'].str.lower().str.split()
movies['Processed_Plot'] = movies['Description'].apply(preprocess_text)

# TF-IDF vectorization
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['Processed_Plot'])

# Calculate similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ****************** RECOMMEND MOVIE USING GENRE AND LANG ********************
'''def Lang_gen(genres,lang, tfidf_matrix, movies):
    # Transform user input genres to lowercase
    #tfidf = TfidfVectorizer()
    genres = [genre.lower() for genre in genres]
    # Compute TF-IDF for user input genres
    user_tfidf = tfidf.transform([' '.join(genres)])
    # Calculate cosine similarity between user input and movie descriptions
    cosine_sim = linear_kernel(user_tfidf, tfidf_matrix)
    # Get indices of movies sorted by similarity
    sim_scores = list(enumerate(cosine_sim[0]))
    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Filter movies by language
    sim_scores_filtered = [(idx, score) for idx, score in sim_scores if movies.iloc[idx]['Lang'] in lang]
    # Get top recommendations
    top_recommendations = sim_scores_filtered[:5]  # Top 5 recommendations
    # Return recommended movie titles
    recommended_movies = [movies.iloc[idx]['Name'] for idx, _ in top_recommendations]'''

# ************************* Similar Movie Recommendation
'''def recommend_by_similar(title,movies, cosine_sim=cosine_sim):
    idx = movies[movies['Name'] == title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:]
    print(sim_scores)
    # Get the top 5 most similar movies
    movie_indices = [i[0] for i in sim_scores]
    mov = movies['Name'].iloc[movie_indices]
    return mov'''
'''def recommend_by_similar(movie_title, movies):
    # Check if DataFrame is empty

    # Check if movie with specified title exists
    if movie_title not in movies['Name'].values:
        return f"Movie '{movie_title}' not found."

    # Get index of movie with specified title
    idx = movies[movies['Name'] == movie_title].index[0]

    # Calculate similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:]  # Exclude self-similarity

    # Get indices of top similar movies
    movie_indices = [i[0] for i in sim_scores]

    # Get names of top similar movies
    similar_movies = movies['Name'].iloc[movie_indices]

    return similar_movies
'''

# ***************** RECOMMEND MOVIE BASED ON GENRE *******************************
import random
from sklearn.metrics.pairwise import linear_kernel

def recommend_by_genre(genres, tfidf_matrix, movies):
    k = 10
    genres = [genre.lower() for genre in genres]
    user_tfidf = tfidf.transform([' '.join(genres)])
    cosine_sim = linear_kernel(user_tfidf, tfidf_matrix)
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_recommendations = [sim for sim in sim_scores if sim[1] > 0.13]
    
    # Ensure that k does not exceed the length of top_recommendations
    k = min(k, len(top_recommendations))
    
    if k > 0:
        top_recommendations = random.sample(top_recommendations, k)
        recommended_movies = [{
            'Name': movies.iloc[idx]['Name'],
            'Actors': movies.iloc[idx]['Cast'],
            'Director': movies.iloc[idx]['Director'],
            'IMDB Rating': movies.iloc[idx]['Rating'],
            'Runtime': movies.iloc[idx]['Run Time'],
            'Release Year': movies.iloc[idx]['Year'],
            'Language': movies.iloc[idx]['Lang']

        } for idx, _ in top_recommendations]
    else:
        recommended_movies = []  # No recommendations if top_recommendations is empty
    
    return recommended_movies


# ********************** RECOMMEND MOVIE BASED ON A LANGUAGE ********************
def recommend_by_language(lan, movies):
  lan = str(lan)
  mov = movies[movies['Lang'].str.contains(lan, case = False, na=False)]
  mov = mov['Name']
  if len(mov) >= 5:
        return mov.sample(n=5)  # Take a sample of 5 movies
  else:
        return mov


  '''def Siml_D(words,tfidf_matrix,cosine_sim):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)
    tfidf_vectorizer = TfidfVectorizer()
    user_tfidf_vector = tfidf_vectorizer.transform([[desc]])
    cosine_sim = cosine_similarity(user_tfidf_vector, tfidf_matrix)
    similar_movies_indices = cosine_sim.argsort()[0][-5:][::-1]
    top_similar_movies = movies.iloc[similar_movies_indices]
    recommended_movies_array = top_similar_movies.to_numpy()'''