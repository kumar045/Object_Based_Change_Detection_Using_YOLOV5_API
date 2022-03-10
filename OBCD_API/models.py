from django.db import models

# Create your models here.
class OBCD(models.Model):
    tif_images_folder_path = models.CharField(max_length=1000)
