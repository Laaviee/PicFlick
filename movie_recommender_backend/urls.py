from django.contrib import admin
from django.urls import path, include,re_path
from django.conf import settings
from django.conf.urls.static import static
from myapp import views

from myapp import urls

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('myapp.urls')),
    path('index/', include('myapp.urls')),
    path('preferences/', include('myapp.urls')),
    path('language/', include('myapp.urls')),
    path('genre/', include('myapp.urls')),
    path('SimilarContents/', include('myapp.urls')),
    path('recommendations/', include('myapp.urls')),
    path('user1/', include('myapp.urls'))
]
