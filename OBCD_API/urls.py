from django.conf.urls import url
from .views import *
from django.urls import path

urlpatterns = [
    url(r'^run/$', OBCDAPIView.as_view(), name='run'),

]