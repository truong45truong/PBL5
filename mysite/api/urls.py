from django.urls import URLPattern, path
from . import views
app_name = 'api'
urlpatterns=[
    path('Gym_squats', views.livefe_Gym_squats, name='video_feed'),
    path('Gym_crunch', views.livefe_Gym_crunch, name='video_feed'),
    path('Gym_dumbbellcurl', views.livefe_Gym_dumbbellcurl, name='video_feed'),
    path('Ex_lession1', views.livefe_Ex_lession1, name='video_feed'),
    path('Ex_lession2', views.livefe_Ex_lession2, name='video_feed'),
    path('Ex_lession3', views.livefe_Ex_lession3, name='video_feed'),
    path('Ex_lession4', views.livefe_Ex_lession4, name='video_feed'),
    path('Yoga_lession1', views.livefe_Yoga_lession1, name='video_feed'),   
    path('Yoga_lession2', views.livefe_Yoga_lession2, name='video_feed'),  
    path('Yoga_lession3', views.livefe_Yoga_lession3, name='video_feed'),  
    path('Yoga_lession4', views.livefe_Yoga_lession4, name='video_feed'),  
]