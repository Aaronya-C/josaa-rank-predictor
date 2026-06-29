from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import os
import re

app = Flask(__name__, static_folder='../frontend')
CORS(app)

# ── Linear Regression from scratch (CS229 Normal Equation) ──
class LinearRegression:
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        X_b = np.hstack([np.ones((len(X), 1)), X])
        self.theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        return self

    def predict(self, X):
        X_b = np.hstack([np.ones((len(X), 1)), X])
        return X_b @ self.theta


# ── Load and clean data once at startup ──
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'josaa_data.csv')

print("Loading data...")
df = pd.read_csv(DATA_PATH)

# ────────────────────────────────────────────────────
# CHANGE 5 — normalize whitespace in key text columns.
#
# Different years' JoSAA CSVs are formatted slightly
# differently (double spaces, trailing spaces, etc.)
# inside the SAME institute/program names. Left as-is,
# this silently:
#   1) splits one real seat into two different Seat_Keys
#      (each gets its own, smaller history → two different
#      predicted closing ranks for "the same" seat), and
#   2) breaks the substring check in get_inst_type() if the
#      stray whitespace lands inside the matched phrase
#      (e.g. "Indian  Institute of Technology" no longer
#      contains "indian institute of technology" → silently
#      falls through to the GFTI default).
#
# Fix: collapse all whitespace runs to a single space and
# strip ends, for every text column used in Seat_Key or
# classification, BEFORE anything groups or filters on them.
# ────────────────────────────────────────────────────
def _normalize_whitespace(s):
    if pd.isna(s):
        return s
    return re.sub(r'\s+', ' ', str(s)).strip()

for _col in ['Institute', 'Academic Program Name', 'Quota', 'Seat Type', 'Gender']:
    df[_col] = df[_col].apply(_normalize_whitespace)

# ── Remove Architecture / Planning (JEE Paper-2 programs — different rank scale) ──
PAPER2_KEYWORDS = ['Architecture', 'Planning']
paper2_mask = df['Academic Program Name'].str.contains('|'.join(PAPER2_KEYWORDS), case=False, na=False)
print(f"Removed {paper2_mask.sum()} Paper-2 rows (Architecture/Planning).")
df = df[~paper2_mask].reset_index(drop=True)

df_final = df[df['Round'] == df.groupby('Year')['Round'].transform('max')]
df_clean  = df_final.dropna(
    subset=['Closing Rank', 'Opening Rank', 'Gender', 'Seat Type', 'Quota']
).copy()

df_clean['Seat_Key'] = (
    df_clean['Institute']             + ' | ' +
    df_clean['Academic Program Name'] + ' | ' +
    df_clean['Quota']                 + ' | ' +
    df_clean['Seat Type']             + ' | ' +
    df_clean['Gender']
)

# Keep ALL seats — even those with only 1 or 2 years of data
# We handle sparse seats differently inside the predictor
df_model = df_clean.copy()
print(f"Data ready. {df_model['Seat_Key'].nunique()} unique seats loaded.")


# ── Helper: classify institute type ──
def get_inst_type(name):
    n = re.sub(r'\s+', ' ', (name or '').strip()).lower()
    if 'indian institute of technology' in n:                          return 'IIT'
    if 'national institute of technology' in n:                        return 'NIT'
    if 'indian institute of information technology' in n or 'iiit' in n: return 'IIIT'
    return 'GFTI'


# ── Core prediction function ──
def predict_for_student(student_rank, category, gender, quota,
                        inst_type_filter='ALL', branch_filter='',
                        predict_year=2026, top_n=50):

    # ── Filter by category / gender / quota ──
    filtered = df_model[
        (df_model['Seat Type'] == category) &
        (df_model['Gender']    == gender)   &
        (df_model['Quota']     == quota)
    ]

    if inst_type_filter != 'ALL':
        filtered = filtered[
            filtered['Institute'].apply(lambda x: get_inst_type(x) == inst_type_filter)
        ]

    if branch_filter:
        filtered = filtered[
            filtered['Academic Program Name']
            .str.lower()
            .str.contains(branch_filter.lower(), na=False)
        ]

    results = []

    for seat_key, seat_data in filtered.groupby('Seat_Key'):
        seat_data     = seat_data.sort_values('Year')
        closing_ranks = seat_data['Closing Rank'].values.astype(float)
        years         = seat_data['Year'].values.astype(float)
        n_years       = len(closing_ranks)

        hist_mean = float(np.mean(closing_ranks))
        hist_std  = float(np.std(closing_ranks))

        # ────────────────────────────────────────────────────
        # CHANGE 2 — seats with <= 3 years: skip regression,
        #            just use the historical average as the
        #            predicted closing rank directly.
        # ────────────────────────────────────────────────────
        if n_years <= 3:
            predicted_closing = hist_mean
            method = 'average'

        else:
            # ── Stage 1: Linear Regression ──
            X      = years.reshape(-1, 1)
            X_mean = X.mean();  X_std = X.std() or 1.0
            X_norm = (X - X_mean) / X_std

            model = LinearRegression()
            model.fit(X_norm, closing_ranks)

            X_fut           = np.array([[(predict_year - X_mean) / X_std]])
            predicted_closing = float(model.predict(X_fut)[0])

            # ────────────────────────────────────────────────
            # CHANGE 1 — if regression prediction deviates
            #            more than 5 % of the historical mean,
            #            replace it with the rounded mean.
            # ────────────────────────────────────────────────
            tolerance = 0.05 * hist_mean          # 5 % of mean
            if abs(predicted_closing - hist_mean) > tolerance:
                predicted_closing = round(hist_mean)
                method = 'average'
            else:
                method = 'regression'

        # Always ensure a valid positive rank
        predicted_closing = max(1.0, predicted_closing)

        # ────────────────────────────────────────────────────
        # CHANGE 4 — probability from RELATIVE gap, not each
        #            seat's own historical std.
        #
        # Why CHANGE 3 still inverted: std_used was clipped to
        # 12%–20% of THAT SEAT'S OWN mean. Two seats can land
        # at opposite ends of that band (a ~1.7x difference in
        # curve steepness) purely from data noise — enough, on
        # its own, to make a harder seat (lower closing rank)
        # outscore an easier one (higher closing rank).
        #
        # Fix: every seat now uses the SAME fixed scale, applied
        # to the gap as a fraction of that seat's predicted
        # closing rank. This makes probability a strictly
        # increasing function of predicted_closing for a fixed
        # student rank — a higher closing rank can never score
        # lower than a lower closing rank again, regardless of
        # how noisy either seat's history is.
        # ────────────────────────────────────────────────────
        relative_gap = (predicted_closing - student_rank) / predicted_closing
        PROB_SCALE   = 7.0   # tuned so ~15% rank buffer ≈ 80% probability
        probability  = float(1 / (1 + np.exp(-np.clip(relative_gap * PROB_SCALE, -6, 6)))) * 100

        # Historical volatility is now ONLY a display confidence
        # label — it can no longer affect the probability number
        # or the ranking.
        if n_years <= 3:
            confidence = 'low'
        elif hist_mean and (hist_std / hist_mean) > 0.20:
            confidence = 'medium'
        else:
            confidence = 'high'

        parts = seat_key.split(' | ')
        results.append({
            'institute':        parts[0],
            'program':          parts[1],
            'quota':            parts[2],
            'seat_type':        parts[3],
            'gender':           parts[4],
            'inst_type':        get_inst_type(parts[0]),
            'predicted_closing': round(predicted_closing),
            'prob':             round(probability, 1),
            'method':           method,          # 'regression' or 'average'
            'confidence':       confidence,       # 'low' / 'medium' / 'high' — data-quality label only, doesn't affect prob/sort
            'n_years':          n_years,
            'historical':       [int(r) for r in closing_ranks.tolist()],
            'years':            [int(y) for y in years.tolist()]
        })

    # Sort by probability descending
    results.sort(key=lambda x: x['predicted_closing'])

    if top_n != 'ALL':
        results = results[:int(top_n)]

    return results


# ── API routes ──
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        quota = data.get('quota', '')
        if quota == 'AI':
            rank = int(data.get('adv_rank') or data.get('mains_rank', 0))
        else:
            rank = int(data.get('mains_rank', 0))

        if rank <= 0:
            return jsonify({'error': 'Please enter a valid rank.'}), 400

        results = predict_for_student(
            student_rank   = rank,
            category       = data.get('category', 'OPEN'),
            gender         = data.get('gender', 'Gender-Neutral'),
            quota          = quota,
            inst_type_filter = data.get('inst_type', 'ALL'),
            branch_filter  = data.get('branch_filter', ''),
            predict_year   = 2026,
            top_n          = data.get('top_n', 50)
        )
        return jsonify({'results': results, 'rank_used': rank})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/institutes', methods=['GET'])
def get_institutes():
    institutes = sorted(df_model['Institute'].unique().tolist())
    return jsonify({'institutes': institutes})


@app.route('/api/programs', methods=['GET'])
def get_programs():
    programs = sorted(df_model['Academic Program Name'].unique().tolist())
    return jsonify({'programs': programs})


@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)