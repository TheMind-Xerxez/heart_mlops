from django.contrib import admin

# Register your models here.

from .models import heart_disease

admin.site.register(heart_disease)
