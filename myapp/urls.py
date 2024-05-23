from turtle import home
from xml.etree.ElementInclude import include
from django.urls import path, re_path
from django.conf import settings
from . import views
from django.conf.urls.static import static
urlpatterns = [
    path('',views.index, name='index'),
    path('preferences.html', views.preferences, name ='preferences'),
    path('language.html', views.language, name='language'),
    path('genre.html',views.genre, name='genre'),
    path('SimilarContents.html', views.SimilarContents, name='SimilarContents'),
    path('recommendations.html',views.recommendations, name='recommendations'),
    path('user1/', views.user1, name='user1')    
    
]