from django.shortcuts import render, redirect
from django.http import HttpResponse
import csv
from django.utils import timezone
from django.conf import settings
from django.contrib import messages
from datetime import timedelta
from .models import Course, GraduationData
from .csce_scraper import build_term_codes_past_years
from django.contrib.auth.hashers import check_password
from .forms import GraduationForm
from datetime import datetime
import json
import os
import sys
import subprocess
import io


def home(request):
    # Build all expected terms
    all_terms = [term for term, label in build_term_codes_past_years(years=5)]

    # Look at which terms we actually have in the DB
    existing_terms = set(Course.objects.values_list('term', flat=True))

    # Identify missing terms by TERM CODE only
    missing_terms = [t for t in all_terms if t not in existing_terms]

    if missing_terms:
        print(f"Scraping missing terms: {missing_terms}")
        Course.save_courses(subj="CSCE")
    else:
        print("No missing terms. All caught up.")

    courses = Course.objects.values_list('code', flat=True).distinct().order_by('code')

    forecast_path = os.path.join(settings.BASE_DIR, "main", "forecast_data.json")
    if os.path.exists(forecast_path):
        with open(forecast_path, "r") as f:
            course_data = f.read()
    else:
        from .arima import generate_all_forecasts
        data = generate_all_forecasts()
        course_data = json.dumps(data)

    return render(request, 'home.html', {
        'courses': courses,
        'course_data': course_data
    })



def rescrape_data(request):
    if request.method == 'POST':

        password = request.POST.get('rescrape_password', '').strip()

        # Validate hashed password
        if not check_password(password, settings.UPLOAD_PASSWORD_HASH):

            # Store error message for the template
            messages.error(request, "Invalid password for rescraping.")

            # Redirect back to data page (where the form is)
            return redirect('data')

        # Password correct → continue normally
        Course.objects.all().delete()

        all_terms = [term for term, label in build_term_codes_past_years(years=5)]
        Course.save_courses(subj="CSCE")

        script_path = os.path.join(settings.BASE_DIR, "main", "arima.py")
        subprocess.run([sys.executable, script_path], check=True)

        messages.success(
            request,
            f"All courses removed and rescraped for terms: {', '.join(all_terms)}"
        )

    return redirect('home')


def graduate_data(request):
    error = None

    # Auto-fill missing years
    current_year = datetime.now().year
    for year in range(2021, current_year + 1):
        GraduationData.objects.get_or_create(year=year, defaults={'graduates': 0})

    if request.method == 'POST':
        password = request.POST.get('graduate_password', '').strip()

        #Use the ONE hashed password stored in settings.py
        if not check_password(password, settings.UPLOAD_PASSWORD_HASH):
            error = "Invalid password."
            form = GraduationForm(request.POST)

        else:
            form = GraduationForm(request.POST)

            if form.is_valid():
                year = form.cleaned_data['year']
                graduates = form.cleaned_data['graduates']

                GraduationData.objects.update_or_create(
                    year=year,
                    defaults={'graduates': graduates}
                )

                # run ARIMA after change
                script_path = os.path.join(settings.BASE_DIR, "main", "arima.py")
                subprocess.run([sys.executable, script_path], check=True)

                return redirect('graduates')

    else:
        form = GraduationForm()

    data = GraduationData.objects.all().order_by('-year')

    return render(request, "graduates.html", {
        'form': form,
        'data': data,
        'error': error
    })


# ==============================
# CSV Helpers + Upload
# ==============================
def sanitize_csv_value(value):
    if value and value[0] in ('=', '+', '-', '@'):
        value = "'" + value
    return value.strip()


def data(request):
    message = None
    error = None

    if request.method == 'POST' and 'csv_file' in request.FILES:
        password = request.POST.get('upload_password', '').strip()

        # hashed password check for CSV too
        if not check_password(password, settings.UPLOAD_PASSWORD_HASH):
            error = "Invalid password."
        else:
            csv_file = request.FILES['csv_file']

            if not csv_file.name.endswith('.csv'):
                error = "Error: File is not CSV type."

            else:
                try:
                    data_set = csv_file.read().decode('UTF-8')
                    io_string = io.StringIO(data_set)
                    reader = csv.DictReader(io_string)

                    if not reader.fieldnames or len(reader.fieldnames) < 3:
                        error = "CSV must have at least Code, Title, and one Term column."

                    else:
                        term_columns = reader.fieldnames[2:]
                        updated_count = 0
                        skipped_count = 0

                        for row in reader:
                            code = sanitize_csv_value(row.get('Code', ''))
                            title = sanitize_csv_value(row.get('Title', ''))

                            if not code or not title:
                                skipped_count += 1
                                continue

                            for term in term_columns:
                                enrolled = row.get(term, '').strip()

                                if enrolled in ('', '-'):
                                    continue

                                try:
                                    enrolled = int(enrolled)
                                except ValueError:
                                    skipped_count += 1
                                    continue

                                Course.objects.update_or_create(
                                    code=code,
                                    term=term,
                                    defaults={'title': title, 'enrolled': enrolled}
                                )
                                updated_count += 1

                        script_path = os.path.join(settings.BASE_DIR, "main", "arima.py")
                        subprocess.run([sys.executable, script_path], check=True)

                        message = f"CSV uploaded successfully. {updated_count} updated, {skipped_count} skipped."

                except Exception as e:
                    error = f"Error processing CSV: {str(e)}"

    all_terms = sorted(Course.objects.values_list('term', flat=True).distinct(), reverse=True)
    courses_dict = {}

    for c in Course.objects.all().order_by('code'):
        if c.code not in courses_dict:
            courses_dict[c.code] = {'title': c.title, 'enrolled_list': []}

    for code, info in courses_dict.items():
        enrolled_list = []
        for term in all_terms:
            course = Course.objects.filter(code=code, term=term).first()
            enrolled_list.append(course.enrolled if course else '-')
        info['enrolled_list'] = enrolled_list

    return render(request, 'data.html', {
        'courses_dict': courses_dict,
        'all_terms': all_terms,
        'message': message,
        'error': error
    })


def download_data(request):
    all_terms = sorted(Course.objects.values_list('term', flat=True).distinct(), reverse=True)
    courses_dict = {}

    for c in Course.objects.all().order_by('code'):
        if c.code not in courses_dict:
            courses_dict[c.code] = {'title': c.title, 'enrolled_list': []}

    for code, info in courses_dict.items():
        enrolled_list = []
        for term in all_terms:
            course = Course.objects.filter(code=code, term=term).first()
            enrolled_list.append(course.enrolled if course else '-')
        info['enrolled_list'] = enrolled_list

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="course_data.csv"'
    writer = csv.writer(response)

    writer.writerow(['Code', 'Title'] + all_terms)

    for code, course in courses_dict.items():
        writer.writerow([code, course['title']] + course['enrolled_list'])

    return response


def model_info(request):
    forecast_path = os.path.join("main", "forecast_data.json")
    if os.path.exists(forecast_path):
        with open(forecast_path, "r") as f:
            results_json = json.dumps(json.load(f))
    else:
        results_json = "[]"

    return render(request, 'model_info.html', {
        "results_json": results_json
    })
