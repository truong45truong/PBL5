from django.db import models

# Create your models here.


class UploadGym(models.Model):
    date=models.CharField(max_length=255,primary_key=True)
    squat=models.CharField(max_length=4)
    crunch=models.CharField(max_length=4)
    yoga=models.CharField(max_length=4)

    def __str__(self):
        return self.date

class UploadEX(models.Model):
    date=models.CharField(max_length=255,primary_key=True)
    number_session=models.CharField(max_length=4)
    def __str__(self):
        return self.date
class UploadYoga(models.Model):
    date=models.CharField(max_length=255,primary_key=True)
    number_session=models.CharField(max_length=4)
    def __str__(self):
        return self.date