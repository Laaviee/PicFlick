from django.shortcuts import render, redirect
from .ml_util import movies, recommend_by_language
from .ml_util import movies, recommend_by_genre
from .ml_util import recommend_by_similar
from .ml_util import tfidf_matrix
def index(request):
    return render(request, 'index.html')
def preferences(request):
    return render(request, 'preferences.html')

def language(request):
    if request.method == 'POST':
        selected_language = request.POST.get('language')
        recommended_movies = recommend_by_language(selected_language, movies)
        return render(request, 'recommendations.html', {'recommended_movies': recommended_movies})
    else:
        return render(request, 'language.html')

def genre(request):
    if request.method == 'POST':
        selected_genre = request.POST.getlist('genre')
        recommended_movies = recommend_by_genre(selected_genre, tfidf_matrix, movies)
        return render(request, 'recommendations.html', {'recommended_movies': recommended_movies})
        
    # Render the genre selection page if it's a GET request
    else:
        # Render the genre selection page for GET requests
        return render(request, 'genre.html')
'''def SimilarContents(request):
    if request.method == 'POST':
        movie_title = request.POST.get('SimilarContents')
        movie_description = request.POST.get('SimilarContents')
        recommended_movies = recommend_by_similar(movie_title,movies)
        return render(request, 'recommendations.html', {'recommended_movies': recommended_movies})
    else:
        return render(request, 'SimilarContents.html')'''
def recommendations(request):
    context = {}
    if request.method == 'POST':
        if 'genres' in request.POST:
            context['recommended_movies'] = request.POST.getlist('genres')
        elif 'language' in request.POST:
            selected_language = request.POST.get('language')
            recommended_movies = recommend_by_language(selected_language, movies)
            context['recommended_movies'] = recommended_movies
        '''elif 'SimilarContents' in request.POST:
            context['recommended_movies'] = request.POST.getlist('SimilarContents')'''# Example: Get list of selected genres from form
    return render(request, 'recommendations.html', context)


def user1(request):
    return redirect('http://127.0.0.1:8000/admin/login/?next=/admin/')