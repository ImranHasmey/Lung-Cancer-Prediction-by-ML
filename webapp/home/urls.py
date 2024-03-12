from django.urls import path
from . import views


urlpatterns = [
    path("", views.index, name='index'),
    path('upload', views.upload_dataset, name='upload'),
    path('train', views.train, name='train'),
    path('predictions', views.predictions, name='predictions'),
    path('Accuracy',views.Accuracy, name='Accuracy')
    
]