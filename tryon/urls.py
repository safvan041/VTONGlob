"""URL configuration for the ``tryon`` app.

Include this in the project ``urls.py`` with::

    path('tryon/', include('tryon.urls'))
"""

from django.urls import path
from . import views

urlpatterns = [
    path("", views.tryon_view, name="tryon"),
]
