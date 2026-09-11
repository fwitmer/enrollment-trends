from django.db import models
from django.utils import timezone
from datetime import datetime

# Create your models here.
# File: main/models.py
class Course(models.Model):
    term = models.CharField(max_length=10)
    code = models.CharField(max_length=20)
    title = models.CharField(max_length=255)
    enrolled = models.IntegerField()
    #updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.code} - {self.title} ({self.enrolled})"

    @classmethod
    def save_courses(cls, subj="CSCE"):
        from .csce_scraper import schedule_scraper, build_term_codes_past_years

        # Get all terms for the past 5 years
        all_terms = build_term_codes_past_years(years=5)

        # Determine which terms are already in the database
        existing_terms = set(Course.objects.values_list('term', flat=True))

        # Filter only missing terms
        missing_terms = [term for term, label in all_terms if term not in existing_terms]

        if not missing_terms:
            print("All semesters are already in there. No scraping needed.")
            return

        for term_code in missing_terms:
            print(f"Scraping missing term: {term_code}")
            results = schedule_scraper(term=term_code, subj=subj)
            print(f"Found {len(results)} courses for {term_code}")

            for code, title, enrolled in results:
                if code.strip() in {"CSCE A381", "CSCE A412"}:
                    continue
                if code.strip().upper() == "CSCE A490": # Change so CSCE 490 course title are labels "Speical Topics"
                    title = "Special Topics"
                cls.objects.update_or_create(
                    code=code,
                    term=term_code,
                    defaults={'title': title, 'enrolled': enrolled}
                )


############################################################################################
############################################################################################
#Prerequisite
class Prerequisite(models.Model):
    course_code = models.CharField(max_length=20, unique=True)
    prereq_1 = models.CharField(max_length=20, null=True, blank=True)
    prereq_2 = models.CharField(max_length=20, null=True, blank=True)

    def __str__(self):
        return f"{self.course_code} → {', '.join(filter(None, [self.prereq_1, self.prereq_2]))}"

    @classmethod
    def save_prereq_data(cls, prereq_map):
        """
        Save a dict of prerequisite data to the database.
        """
        count = 0
        for course, prereqs in prereq_map.items():
            prereq_1 = prereqs[0] if len(prereqs) > 0 and prereqs[0].strip() else None
            prereq_2 = prereqs[1] if len(prereqs) > 1 and prereqs[1].strip() else None
            cls.objects.update_or_create(
                course_code=course,
                defaults={
                    "prereq_1": prereq_1,
                    "prereq_2": prereq_2,
                }
            )
            count += 1
        return count

    @classmethod
    def scrape_if_empty(cls, scraper_func):
        """
        Check if table is empty. If empty, run scraper and save data.
        """
        if cls.objects.exists():
            print(f"Table already has data ({cls.objects.count()} records). Skipping scrape.")
            return 0

        print("Table is empty. Running scraper...")
        prereq_map = scraper_func()  # your scraper function returns a dict
        count = cls.save_prereq_data(prereq_map)
        print(f"Scraper saved {count} records at {datetime.now()}")
        return count

#Highschool Graduation Model
class GraduationData(models.Model):
    year = models.IntegerField(unique=True)
    graduates = models.IntegerField()

    def __str__(self):
        return f"{self.year} - {self.graduates}"
    
    
