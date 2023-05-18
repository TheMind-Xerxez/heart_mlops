# Generated by Django 4.2.1 on 2023-05-18 18:11

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="heart_disease",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("age", models.FloatField()),
                ("sex", models.FloatField()),
                ("cp", models.FloatField()),
                ("trestbps", models.FloatField()),
                ("chol", models.FloatField()),
                ("fbs", models.FloatField()),
                ("restecg", models.FloatField()),
                ("thalach", models.FloatField()),
                ("exang", models.FloatField()),
                ("oldpeak", models.FloatField()),
                ("slope", models.FloatField()),
                ("ca", models.FloatField()),
                ("thal", models.FloatField()),
                ("target", models.FloatField()),
            ],
        ),
    ]
