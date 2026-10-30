"""
shalbin/urls.py  (the INNER shalbin folder, next to settings.py)
=================================================================
Put this in place of the default urls.py Django generates.
"""

from django.contrib import admin
from django.urls import path
from app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('predict/', views.predict, name='predict'),          # ESP32 posts breath data here
    path('record-dose/', views.record_dose, name='record_dose'),  # ESP32 posts dose events here
    path('results/', views.results, name='results'),          # open this in a browser to see the table
    path('manual/', views.manual_entry, name='manual_entry'),  # type in values by hand
]