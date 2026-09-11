import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import re
from datetime import datetime


def build_term_codes_past_years(years=5, include_current=True):
    now = datetime.now()    #get the current semester
    y_now = now.year
    m = now.month #determine whether its spring/summer/fall
    
    # Determine current semester codes: Spring (01) starts Jan 1, Summer (02) is May-Jul, Fall (03) starts Aug 1
    if m < 5:
        sem_now = "01" 
    elif m < 8:
        sem_now = "02"
    else:
        sem_now = "03"
    cutoff = int(str(y_now) + sem_now)  #e.g., 202503

    semester_map = {"01": "Spring", "02": "Summer", "03": "Fall"}   #setup term labels
    terms = []
    start_year = y_now - (years - 1)    #e.g. if years = 5 and it's 2025 then start_year = 2021

    #Build all terms for the last n years
    for year in range(start_year, y_now + 1):
        for sem in ("01", "02", "03"):
            code = str(year) + sem
            code_int = int(code)
            #excludes future zero enrolled terms
            if code_int > cutoff: 
                continue
            label = semester_map[sem] + " " + str(year)
            terms.append((code, label))  #oldest -> newest
    return terms


def get_current_calendar_term(now=None):
    """Returns calendar term code: Fall (03) if month >= 8, else Spring (01)."""
    if now is None:
        now = datetime.now()
    y = now.year
    m = now.month
    if m >= 8:
        return y * 100 + 3
    else:
        return y * 100 + 1


def term_name_from_code(term):
    """Converts numeric term code (e.g. 202503) to label (e.g. 'Fall 2025')."""
    term = int(term)
    year, sem = divmod(term, 100)
    names = {1: "Spring", 2: "Summer", 3: "Fall"}
    return f"{names.get(sem, 'Unknown')} {year}"


def get_course_offered_pattern(terms):
    """Determines if a course is fall_only, spring_only, or both based on historical term codes."""
    sems = set(int(t) % 100 for t in terms)
    has_spring = 1 in sems
    has_fall = 3 in sems
    if has_fall and not has_spring:
        return "fall_only"
    elif has_spring and not has_fall:
        return "spring_only"
    else:
        return "both"


def get_course_prediction_targets(terms, now=None):
    """
    Returns a dict with 'prior', 'current', and 'future' term codes (int)
    for a course based on its historical offering pattern and the current date.
    """
    cal_term = get_current_calendar_term(now)
    cal_year, cal_sem = divmod(cal_term, 100)
    pattern = get_course_offered_pattern(terms)

    if pattern == "fall_only":
        if cal_sem == 3:  # Fall
            prior = (cal_year - 1) * 100 + 3
            current = cal_year * 100 + 3
            future = (cal_year + 1) * 100 + 3
        else:  # Spring
            prior = (cal_year - 2) * 100 + 3
            current = (cal_year - 1) * 100 + 3
            future = cal_year * 100 + 3
    elif pattern == "spring_only":
        if cal_sem == 1:  # Spring
            prior = (cal_year - 1) * 100 + 1
            current = cal_year * 100 + 1
            future = (cal_year + 1) * 100 + 1
        else:  # Fall
            prior = (cal_year - 1) * 100 + 1
            current = cal_year * 100 + 1
            future = (cal_year + 1) * 100 + 1
    else:  # both
        if cal_sem == 3:  # Fall
            prior = cal_year * 100 + 1
            current = cal_year * 100 + 3
            future = (cal_year + 1) * 100 + 1
        else:  # Spring
            prior = (cal_year - 1) * 100 + 3
            current = cal_year * 100 + 1
            future = cal_year * 100 + 3

    return {
        "prior": prior,
        "current": current,
        "future": future
    }


def schedule_scraper(term="202503", subj="CSCE"):
    url = "https://curric.uaa.alaska.edu/ajax/ajaxScheduleSearch.php"   #API url might change in the near future a point of failure
    params = {"term": term, "subj": subj}
    
    try:
        response = requests.get(url, params=params, timeout=20) #time out after 20s 
        response.raise_for_status()  
    except requests.RequestException as e:
        # API unavailable, slow, wrong URL, network error, etc.
        return {"error": f"API request failed: {e}"}

    rows = response.json()  #convert json array of objects to python list of dictionaries
    totals = defaultdict(int)   #totals keep track of total students per course, each new key value is set to 0
    for row in rows:    #iterates through the dictonaries to extract subject, course, title, and enrollment number
        if row.get("subj").upper() == "CSCE":   #only for CSCE
            crs = row["crs"].upper()
            if not crs.endswith("L"):   #if it is a lab it skips (since the same students in the lab class are in the normal class)
                num = int(re.search(r'\d+', crs).group())  #regular expression that finds the string of numbers, and converts to integers
                if 100 <= num < 500:    #since we are only gathering data for 100-400 level courses
                    code = row['subj'].upper() + " " + row['crs']   #concatenates the subject and the course (e.g. CSCE A101)
                    title = row['title']
                    enrolled = int(row.get("enrolled")) 
                    totals[code, title] += enrolled  #sums up the enrollment for one course
    results = []
    for key, total in sorted(totals.items()):
        code, title = key
        results.append((code, title, total))
    return results

