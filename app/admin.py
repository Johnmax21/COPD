from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import BreathRecord


@admin.register(BreathRecord)
class BreathRecordAdmin(admin.ModelAdmin):
    list_display = ["id", "timestamp", "true_label", "prediction", "confidence",
                     "mq135_mean", "mq137_mean"]
    list_editable = ["true_label"]   # lets you edit the label right in the list view
    list_filter = ["true_label", "prediction"]
    ordering = ["-timestamp"]