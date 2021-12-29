from django.urls import path, include

from chat import views

urlpatterns = [
    path(r'', views.answer),
    path(r'todo', views.test_todo_list),
    path(r'suggestion', views.test_suggestion_list),
]