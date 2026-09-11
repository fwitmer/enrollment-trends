import os
import sys
import django
import pandas as pd
import warnings
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
import json
try:
    from csce_scraper import (
        build_term_codes_past_years,
        term_name_from_code,
        get_course_prediction_targets,
        get_course_offered_pattern,
    )
except ImportError:
    from main.csce_scraper import (
        build_term_codes_past_years,
        term_name_from_code,
        get_course_prediction_targets,
        get_course_offered_pattern,
    )
warnings.filterwarnings("ignore")

#django setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Capstone.settings')
django.setup()

from main.models import Course, Prerequisite
from main.models import GraduationData

hs_qs = GraduationData.objects.all().values('year', 'graduates')
hs_map = {row['year']: row['graduates'] for row in hs_qs}

def hs_value_for_term(term):
    term = int(term)
    year, sem = divmod(term, 100)

    #Spring term uses previous year's graduates
    if sem == 1:
        hs_year = year - 1
    else:
        hs_year = year

    return hs_map.get(hs_year, np.nan)


#counts semester steps between last_term and target_term
def count_steps_between(last_term, target_term, course_num, yearly_course):
    last_term = int(last_term)
    target_term = int(target_term)
    if last_term >= target_term:
        return 1

    if yearly_course:
        y1 = last_term // 100
        y2 = target_term // 100
        return max(1, y2 - y1)

    cur = last_term
    steps = 0
    while cur < target_term and steps < 10:
        y, s = divmod(cur, 100)
        if course_num <= 201:
            if s == 1:
                cur = y * 100 + 2
            elif s == 2:
                cur = y * 100 + 3
            else:
                cur = (y + 1) * 100 + 1
        else:
            if s == 1:
                cur = y * 100 + 3
            else:
                cur = (y + 1) * 100 + 1
        steps += 1
    return max(1, steps)


#calculates previous term code        
def previous_term_code(term_code):
    term_code = int(term_code)
    year, sem = divmod(term_code, 100)
    if sem == 1:   #Spring → previous Fall
        return (year - 1) * 100 + 3
    elif sem == 2: #Summer → previous Spring
        return year * 100 + 1
    elif sem == 3: #Fall → same year Summer
        return year * 100 + 2

#get previous term code with option to skip summers for upper-level courses
def get_previous_term(term, steps_back, upper_level=False):
    lag_term = term
    for _ in range(steps_back):
        lag_term = previous_term_code(lag_term)

        #For upper-level courses (200+), skip *only* summers if they're not in the dataset
        if upper_level and (lag_term % 100 == 2):
            #Only skip summer if previous Fall or Spring exists
            lag_term = previous_term_code(lag_term)
    return lag_term

#load data from the Course model
qs = Course.objects.all().values('code', 'term', 'enrolled', 'title')
df = pd.DataFrame(qs)

#load prerequisite data into a DataFrame
prereq_qs = Prerequisite.objects.all().values('course_code', 'prereq_1', 'prereq_2')
prereq_df = pd.DataFrame(prereq_qs)

#clean + sort chronologically
df['term'] = df['term'].astype(int)
df['term_name'] = df['term'].apply(term_name_from_code)
#Add numeric course number column
df['course_num'] = df['code'].str.extract(r'A(\d+)').astype(int)

#Remove Summer terms for upper-level courses (A211+)
df = df[~(
    (df['course_num'] > 201) &
    (df['term_name'].str.contains("Summer", case=False, na=False))
)]

df = df.sort_values(['code', 'term'])
prereq_map = prereq_df.set_index('course_code')[['prereq_1', 'prereq_2']].to_dict('index') if not prereq_df.empty else {}


def build_exog_data(code, course_num, train_group, target_term, steps, prereq_map, hs_map, full_df):
    prereqs = prereq_map.get(code, {})
    pr1, pr2 = prereqs.get('prereq_1'), prereqs.get('prereq_2')

    exog_values = []
    # Add HS grads as exog ONLY for A101
    if code == "CSCE A101":
        hs_vals = []
        for term in train_group['term']:
            hs_raw = hs_value_for_term(term)
            hs_vals.append(hs_raw / 50 if not np.isnan(hs_raw) else 0.0)
        exog_values.append(hs_vals)

    # Prereq 1 and 2
    for i, prereq_code in enumerate([pr1, pr2]):
        if prereq_code:
            lagged = []
            for term in train_group['term']:
                lag_term = get_previous_term(term, i + 1, course_num > 201)
                enroll_row = full_df[(full_df['code'] == prereq_code) & (full_df['term'] == lag_term)]
                val = float(enroll_row['enrolled'].values[0]) if not enroll_row.empty else 0.0
                lagged.append(val)
            exog_values.append(lagged)
        else:
            exog_values.append([0.0] * len(train_group))

    exog = np.column_stack(exog_values) if exog_values else np.zeros((len(train_group), 1))
    exog = np.nan_to_num(exog, nan=0.0)

    # Build future exog for forecasting target_term
    cur_term = train_group['term'].iloc[-1]
    step_terms = []
    temp_term = cur_term
    for _ in range(steps):
        y, s = divmod(temp_term, 100)
        if course_num <= 201:
            temp_term = y * 100 + 2 if s == 1 else (y * 100 + 3 if s == 2 else (y + 1) * 100 + 1)
        else:
            temp_term = y * 100 + 3 if s == 1 else (y + 1) * 100 + 1
        step_terms.append(temp_term)

    target_exog_rows = []
    for st in step_terms:
        row_vals = []
        if code == "CSCE A101":
            hs_raw = hs_value_for_term(st)
            row_vals.append(hs_raw / 50 if not np.isnan(hs_raw) else 0.0)
        for i, prereq_code in enumerate([pr1, pr2]):
            if prereq_code:
                lag_t = get_previous_term(st, i + 1, course_num > 201)
                enroll_row = full_df[(full_df['code'] == prereq_code) & (full_df['term'] == lag_t)]
                val = float(enroll_row['enrolled'].values[0]) if not enroll_row.empty else 0.0
                row_vals.append(val)
            else:
                row_vals.append(0.0)
        target_exog_rows.append(row_vals)

    next_exog = np.array(target_exog_rows) if target_exog_rows else np.zeros((steps, 1))
    next_exog = np.nan_to_num(next_exog, nan=0.0)
    return exog, next_exog


def fit_target_forecast(code, course_num, full_group, target_term, prereq_map, hs_map, full_df, yearly_course):
    target_term = int(target_term)
    target_name = term_name_from_code(target_term)

    # Actual enrollment in DB for target_term (if exists and > 0)
    actual_rows = full_df[(full_df['code'] == code) & (full_df['term'] == target_term)]
    actual_enrolled = None
    if not actual_rows.empty:
        enr = int(actual_rows['enrolled'].values[0])
        if enr > 0:
            actual_enrolled = enr

    # Training data: strictly prior to target_term
    train_group = full_group[full_group['term'] < target_term].sort_values('term')
    if len(train_group) < 4:
        return {
            "term": target_term,
            "term_name": target_name,
            "actual_enrolled": actual_enrolled,
            "arima_forecast": None,
            "sarima_forecast": None,
            "arimax_forecast": None,
            "sarimax_forecast": None,
            "arima_mae": None,
            "sarima_mae": None,
            "arimax_mae": None,
            "sarimax_mae": None,
            "best_accuracy": None,
            "arima_val_terms": None,
            "arima_val_preds": None,
            "sarima_val_terms": None,
            "sarima_val_preds": None,
            "arimax_val_terms": None,
            "arimax_val_preds": None,
            "sarimax_val_terms": None,
            "sarimax_val_preds": None,
        }

    y = train_group['enrolled'].astype(float).values
    terms = train_group['term'].astype(int).values
    last_term = terms[-1]
    steps = count_steps_between(last_term, target_term, course_num, yearly_course)

    m = 3 if course_num <= 201 else 2
    exog, next_exog = build_exog_data(code, course_num, train_group, target_term, steps, prereq_map, hs_map, full_df)

    arima_forecast = None
    sarima_forecast = None
    arimax_forecast = None
    sarimax_forecast = None

    arima_mae = None
    sarima_mae = None
    arimax_mae = None
    sarimax_mae = None

    arima_val_terms, arima_val_preds = None, None
    sarima_val_terms, sarima_val_preds = None, None
    arimax_val_terms, arimax_val_preds = None, None
    sarimax_val_terms, sarimax_val_preds = None, None

    models_accuracy = []

    # 1. ARIMA
    try:
        model = ARIMA(y, order=(1, 1, 1)).fit()
        preds = np.ravel(model.forecast(steps=steps))
        arima_forecast = max(0.0, round(float(preds[-1]), 0))

        if len(y) >= 4:
            try:
                tr, te = y[:-2], y[-2:]
                tr_terms, te_terms = terms[:-2], terms[-2:]
                val_fit = ARIMA(tr, order=(1, 1, 1)).fit()
                val_preds = np.ravel(val_fit.forecast(steps=len(te)))
                arima_mae = round(float(mean_absolute_error(te, val_preds)), 2)
                arima_val_terms = te_terms.tolist()
                arima_val_preds = [max(0.0, round(float(p), 0)) for p in val_preds]
                acc = 100 - (arima_mae / te.mean() * 100) if te.mean() > 0 else 0
                models_accuracy.append(acc)
            except Exception:
                pass
    except Exception as e:
        print(f"ARIMA error for {code}: {e}")

    # 2. SARIMA (non-yearly courses only)
    if not yearly_course:
        try:
            model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, m)).fit(disp=False)
            preds = np.ravel(model.forecast(steps=steps))
            sarima_forecast = max(0.0, round(float(preds[-1]), 0))

            if len(y) >= 4:
                try:
                    tr, te = y[:-2], y[-2:]
                    tr_terms, te_terms = terms[:-2], terms[-2:]
                    val_fit = SARIMAX(tr, order=(1, 1, 1), seasonal_order=(1, 1, 1, m)).fit(disp=False)
                    val_preds = np.ravel(val_fit.forecast(steps=len(te)))
                    sarima_mae = round(float(mean_absolute_error(te, val_preds)), 2)
                    sarima_val_terms = te_terms.tolist()
                    sarima_val_preds = [max(0.0, round(float(p), 0)) for p in val_preds]
                    acc = 100 - (sarima_mae / te.mean() * 100) if te.mean() > 0 else 0
                    models_accuracy.append(acc)
                except Exception:
                    pass
        except Exception as e:
            print(f"SARIMA error for {code}: {e}")

    # 3. ARIMAX
    try:
        model = ARIMA(y, exog=exog, order=(1, 1, 1)).fit()
        preds = np.ravel(model.forecast(steps=steps, exog=next_exog))
        arimax_forecast = max(0.0, round(float(preds[-1]), 0))

        if len(y) >= 4:
            try:
                tr, te = y[:-2], y[-2:]
                tr_exog, te_exog = exog[:-2], exog[-2:]
                tr_terms, te_terms = terms[:-2], terms[-2:]
                val_fit = ARIMA(tr, exog=tr_exog, order=(1, 1, 1)).fit()
                val_preds = np.ravel(val_fit.forecast(steps=len(te), exog=te_exog))
                arimax_mae = round(float(mean_absolute_error(te, val_preds)), 2)
                arimax_val_terms = te_terms.tolist()
                arimax_val_preds = [max(0.0, round(float(p), 0)) for p in val_preds]
                acc = 100 - (arimax_mae / te.mean() * 100) if te.mean() > 0 else 0
                models_accuracy.append(acc)
            except Exception:
                pass
    except Exception as e:
        print(f"ARIMAX error for {code}: {e}")

    # 4. SARIMAX (non-yearly courses only)
    if not yearly_course:
        try:
            seasonal_order = (1, 0, 1, 2) if code == "CSCE A201" else (1, 1, 1, m)
            model = SARIMAX(y, exog=exog, order=(1, 1, 1), seasonal_order=seasonal_order).fit(disp=False)
            preds = np.ravel(model.forecast(steps=steps, exog=next_exog))
            sarimax_forecast = max(0.0, round(float(preds[-1]), 0))

            if len(y) >= 4:
                try:
                    tr, te = y[:-2], y[-2:]
                    tr_exog, te_exog = exog[:-2], exog[-2:]
                    tr_terms, te_terms = terms[:-2], terms[-2:]
                    val_fit = SARIMAX(tr, exog=tr_exog, order=(1, 1, 1), seasonal_order=seasonal_order).fit(disp=False)
                    val_preds = np.ravel(val_fit.forecast(steps=len(te), exog=te_exog))
                    sarimax_mae = round(float(mean_absolute_error(te, val_preds)), 2)
                    sarimax_val_terms = te_terms.tolist()
                    sarimax_val_preds = [max(0.0, round(float(p), 0)) for p in val_preds]
                    acc = 100 - (sarimax_mae / te.mean() * 100) if te.mean() > 0 else 0
                    models_accuracy.append(acc)
                except Exception:
                    pass
        except Exception as e:
            print(f"SARIMAX error for {code}: {e}")


    best_accuracy = round(max(models_accuracy), 1) if models_accuracy else None

    return {
        "term": target_term,
        "term_name": target_name,
        "actual_enrolled": actual_enrolled,
        "arima_forecast": arima_forecast,
        "sarima_forecast": sarima_forecast,
        "arimax_forecast": arimax_forecast,
        "sarimax_forecast": sarimax_forecast,
        "arima_mae": arima_mae,
        "sarima_mae": sarima_mae,
        "arimax_mae": arimax_mae,
        "sarimax_mae": sarimax_mae,
        "best_accuracy": best_accuracy,
        "arima_val_terms": arima_val_terms,
        "arima_val_preds": arima_val_preds,
        "sarima_val_terms": sarima_val_terms,
        "sarima_val_preds": sarima_val_preds,
        "arimax_val_terms": arimax_val_terms,
        "arimax_val_preds": arimax_val_preds,
        "sarimax_val_terms": sarimax_val_terms,
        "sarimax_val_preds": sarimax_val_preds,
    }


def generate_all_forecasts():
    qs = Course.objects.all().values('code', 'term', 'enrolled', 'title')
    df_all = pd.DataFrame(qs)
    df_all['term'] = df_all['term'].astype(int)
    df_all['term_name'] = df_all['term'].apply(term_name_from_code)
    df_all['course_num'] = df_all['code'].str.extract(r'A(\d+)').astype(int)

    # Filter out Summer terms for A211+ only (CSCE A101 and A201 retain all summer data)
    df_filtered = df_all[~(
        (df_all['course_num'] > 201) &
        (df_all['term_name'].str.contains("Summer", case=False, na=False))
    )].copy()
    df_filtered = df_filtered.sort_values(['code', 'term'])

    historical_records = []
    for _, row in df_filtered.iterrows():
        historical_records.append({
            "code": row['code'],
            "term": int(row['term']),
            "term_name": row['term_name'],
            "enrolled": int(row['enrolled']),
            "title": row['title']
        })

    predictions_by_course = {}

    for code, group in df_filtered.groupby('code'):
        course_num = int(code.split('A')[-1])
        title = group['title'].iloc[0]
        terms = group['term'].tolist()

        # Check if yearly course
        spring_count = sum(1 for t in terms if t % 100 == 1)
        summer_count = sum(1 for t in terms if t % 100 == 2)
        fall_count = sum(1 for t in terms if t % 100 == 3)
        yearly_course = (spring_count > 0 and summer_count == 0 and fall_count == 0) or \
                        (spring_count == 0 and summer_count == 0 and fall_count > 0)

        targets = get_course_prediction_targets(terms)
        pattern = get_course_offered_pattern(terms)

        target_forecasts = {}
        for mode in ['prior', 'current', 'future']:
            tgt_term = targets[mode]
            forecast_dict = fit_target_forecast(
                code=code,
                course_num=course_num,
                full_group=group,
                target_term=tgt_term,
                prereq_map=prereq_map,
                hs_map=hs_map,
                full_df=df_all,
                yearly_course=yearly_course
            )
            target_forecasts[mode] = forecast_dict

        predictions_by_course[code] = {
            "course_name": code,
            "title": title,
            "yearly_course": yearly_course,
            "offered_pattern": pattern,
            "targets": target_forecasts
        }

    output_data = {
        "historical": historical_records,
        "predictions": predictions_by_course
    }

    output_path = os.path.join(BASE_DIR, "main", "forecast_data.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Successfully generated forecasts for {len(predictions_by_course)} courses and saved to {output_path}")
    return output_data


if __name__ == "__main__":
    generate_all_forecasts()

