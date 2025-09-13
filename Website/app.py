from flask import Flask, render_template, request, jsonify
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# ---------- Data config ----------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

FILES = {
    "odi":  DATA_DIR / "odi_batting_stats.xlsx",
    "t20":  DATA_DIR / "t20_batting_stats.xlsx",
    "test": DATA_DIR / "test_batting_stats.xlsx",   # note: 'test' lowercase (your preference)
}

# Required columns we’ll display (others are ignored if present)
FIELDS = [
    "runs_scored", "balls_faced", "high_score",
    "fours", "sixes", "strike_rate",
    "fifties", "hundreds", "double_hundreds",
    "batting_average", "boundary_pct", "balls_per_boundary"
]

# Load all three dataframes (empty if file missing)
dfs = {}
for fmt, path in FILES.items():
    if path.exists():
        dfs[fmt] = pd.read_excel(path)
    else:
        # empty df with needed columns
        dfs[fmt] = pd.DataFrame(columns=["batter"] + FIELDS)

# All unique batter names across formats (sorted case-insensitively)
names = set()
for fmt, df in dfs.items():
    if "batter" in df.columns:
        names.update(df["batter"].dropna().astype(str).tolist())
BATTER_NAMES = sorted(names, key=lambda s: s.lower())

def suggest_batters(q: str, limit: int = 10):
    """Unified suggestions from all formats: prefix matches first, then substring."""
    if not q:
        return BATTER_NAMES[:limit]
    ql = q.lower()
    prefix = [n for n in BATTER_NAMES if n.lower().startswith(ql)]
    other  = [n for n in BATTER_NAMES if (ql in n.lower()) and (not n.lower().startswith(ql))]
    return (prefix + other)[:limit]

def get_row(df: pd.DataFrame, name: str):
    if df.empty or "batter" not in df.columns or not name:
        return None
    m = df["batter"].astype(str).str.lower() == name.strip().lower()
    if m.any():
        return df.loc[m].iloc[0].to_dict()
    return None

def normalize_player(row: dict, batter_name: str):
    """Return a dict with all FIELDS present. Missing -> 0. Floats rounded where sensible."""
    def num(x, nd=0):
        if x is None or x == "" or (isinstance(x, float) and pd.isna(x)):
            return 0
        try:
            v = float(x)
            return round(v, nd) if nd > 0 else int(round(v))
        except:
            return 0

    if row is None:
        # all zeros
        return {
            "batter": batter_name,
            "runs_scored": 0,
            "balls_faced": 0,
            "high_score": 0,
            "fours": 0,
            "sixes": 0,
            "strike_rate": 0.0,
            "fifties": 0,
            "hundreds": 0,
            "double_hundreds": 0,
            "batting_average": 0.0,
            "boundary_pct": 0.0,
            "balls_per_boundary": 0.0,
        }

    return {
        "batter": batter_name,
        "runs_scored": num(row.get("runs_scored")),
        "balls_faced": num(row.get("balls_faced")),
        "high_score": num(row.get("high_score")),
        "fours": num(row.get("fours")),
        "sixes": num(row.get("sixes")),
        "strike_rate": num(row.get("strike_rate"), nd=2),
        "fifties": num(row.get("fifties")),
        "hundreds": num(row.get("hundreds")),
        "double_hundreds": num(row.get("double_hundreds")),
        "batting_average": num(row.get("batting_average"), nd=2),
        "boundary_pct": num(row.get("boundary_pct"), nd=2),
        "balls_per_boundary": num(row.get("balls_per_boundary"), nd=2),
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/batting")
def batting():
    q = request.args.get("q", "").strip()
    batter_name = None
    stats = None

    if q:
        # if name exists in our union list, use its canonical casing (first match)
        # otherwise keep user input for display
        matches = [n for n in BATTER_NAMES if n.lower() == q.lower()]
        batter_name = matches[0] if matches else q

        # Build per-format stats (zeros if missing)
        stats = {}
        for fmt, df in dfs.items():
            row = get_row(df, batter_name)
            stats[fmt] = normalize_player(row, batter_name)

    return render_template("batting.html", q=q, batter_name=batter_name, stats=stats)

@app.route("/api/batters")
def api_batters():
    q = request.args.get("q", "")
    return jsonify({"names": suggest_batters(q, limit=10)})

BASE_DIR = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"

BOWL_FILES = {
    "odi":  DATA_DIR / "odi_bowling_stats.xlsx",
    "t20":  DATA_DIR / "t20_bowling_stats.xlsx",
    "test": DATA_DIR / "test_bowling_stats.xlsx",
}

BOWL_FIELDS = [
    "balls_bowled", "runs_conceded", "wickets",
    "fours_conceded", "sixes_conceded", "maidens",
    "five_wicket_hauls", "wickets_std",
    "best_innings_wickets", "best_innings_runs_conceded",
    "bowling_average", "strike_rate", "economy",
    "balls_per_wicket", "boundary_pct"
]

# Load 3 bowling dataframes (gracefully handle missing files)
dfs_bowl = {}
for fmt, path in BOWL_FILES.items():
    if path.exists():
        dfs_bowl[fmt] = pd.read_excel(path)
    else:
        dfs_bowl[fmt] = pd.DataFrame(columns=["bowler"] + BOWL_FIELDS)

# Union of bowler names (sorted, case-insensitive)
BOWLER_NAMES = sorted(
    set().union(
        *[set(df["bowler"].dropna().astype(str)) for df in dfs_bowl.values() if "bowler" in df.columns]
    ),
    key=lambda s: s.lower()
)

def suggest_bowlers(q: str, limit: int = 10):
    if not q:
        return BOWLER_NAMES[:limit]
    ql = q.lower()
    prefix = [n for n in BOWLER_NAMES if n.lower().startswith(ql)]
    other  = [n for n in BOWLER_NAMES if (ql in n.lower()) and (not n.lower().startswith(ql))]
    return (prefix + other)[:limit]

def _row_by_name(df: pd.DataFrame, name: str, col="bowler"):
    if df.empty or col not in df.columns or not name:
        return None
    m = df[col].astype(str).str.lower() == name.strip().lower()
    return df.loc[m].iloc[0].to_dict() if m.any() else None

def normalize_bowler(row: dict, name: str):
    def num(x, nd=0):
        if x is None or x == "" or (isinstance(x, float) and pd.isna(x)): return 0
        try:
            v = float(x)
            return round(v, nd) if nd > 0 else int(round(v))
        except: return 0

    if row is None:
        return {
            "bowler": name,
            "balls_bowled": 0, "runs_conceded": 0, "wickets": 0,
            "fours_conceded": 0, "sixes_conceded": 0, "maidens": 0,
            "five_wicket_hauls": 0, "wickets_std": 0,
            "best_innings_wickets": 0, "best_innings_runs_conceded": 0,
            "bowling_average": 0.0, "strike_rate": 0.0, "economy": 0.0,
            "balls_per_wicket": 0.0, "boundary_pct": 0.0
        }

    return {
        "bowler": name,
        "balls_bowled": num(row.get("balls_bowled")),
        "runs_conceded": num(row.get("runs_conceded")),
        "wickets": num(row.get("wickets")),
        "fours_conceded": num(row.get("fours_conceded")),
        "sixes_conceded": num(row.get("sixes_conceded")),
        "maidens": num(row.get("maidens")),
        "five_wicket_hauls": num(row.get("five_wicket_hauls")),
        "wickets_std": num(row.get("wickets_std"), nd=2),
        "best_innings_wickets": num(row.get("best_innings_wickets")),
        "best_innings_runs_conceded": num(row.get("best_innings_runs_conceded")),
        "bowling_average": num(row.get("bowling_average"), nd=2),
        "strike_rate": num(row.get("strike_rate"), nd=2),
        "economy": num(row.get("economy"), nd=2),
        "balls_per_wicket": num(row.get("balls_per_wicket"), nd=2),
        "boundary_pct": num(row.get("boundary_pct"), nd=2),
    }

# ----------------------- NEW: Bowling routes -----------------------
@app.route("/bowling")
def bowling():
    q = request.args.get("q", "").strip()
    bowler_name = None
    stats = None

    if q:
        matches = [n for n in BOWLER_NAMES if n.lower() == q.lower()]
        bowler_name = matches[0] if matches else q

        stats = {}
        for fmt, df in dfs_bowl.items():
            row = _row_by_name(df, bowler_name, col="bowler")
            stats[fmt] = normalize_bowler(row, bowler_name)

    return render_template("bowling.html", q=q, bowler_name=bowler_name, stats=stats)

@app.route("/api/bowlers")
def api_bowlers():
    q = request.args.get("q", "")
    return jsonify({"names": suggest_bowlers(q, limit=10)})

@app.route("/prediction")
def prediction():
    # e.g. /prediction?format=ODI&player=Virat%20Kohli
    return render_template(
        "prediction.html",
        initial_format=request.args.get("format",""),
        initial_player=request.args.get("player",""),
        initial_venue=request.args.get("venue",""),
        initial_opposition=request.args.get("opposition",""),
    )
# === PREDICTION (multi-format) ===
from flask import request, jsonify
import pandas as pd, numpy as np, joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
UNK = -1

# ---------- helpers ----------
def find_first(*cands):
    for p in cands:
        if p and Path(p).exists():
            return Path(p)
    return None

def read_any(path: Path) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    return pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)

def norm_fmt(s: str) -> str:
    s = (s or "").strip().lower()
    if s in ("odi",): return "odi"
    if s in ("t20", "t-20", "twenty20"): return "t20"
    return "test"

def get_lgbm_booster(model):
    mod = type(model).__module__.lower()
    name = type(model).__name__.lower()
    if "lightgbm" in mod and name == "booster":
        return model
    if hasattr(model, "booster_") and model.booster_ is not None:
        return model.booster_
    if hasattr(model, "_Booster") and model._Booster is not None:
        return model._Booster
    return None

def align_features_for_booster(booster, X: pd.DataFrame) -> pd.DataFrame:
    try:
        feats = booster.feature_name()
    except Exception:
        feats = None
    if feats:
        for f in feats:
            if f not in X.columns: X[f] = 0
        X = X[[f for f in feats if f in X.columns]]
    return X.astype("float32")

def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

def encode_like_training(df: pd.DataFrame, maps: dict) -> pd.DataFrame:
    df = df.copy()
    for c, mapping in maps.items():
        if c in df.columns:
            s = df[c].astype(str)
            df[c] = s.map(mapping).fillna(UNK).astype(int)    # unseen → -1
    return df

def force_numeric(X: pd.DataFrame, maps: dict) -> pd.DataFrame:
    # Encode any leftover object cols with maps if available, otherwise UNK
    obj_cols = [c for c in X.columns if X[c].dtype == "object"]
    for c in obj_cols:
        if c in maps:
            X[c] = X[c].astype(str).map(maps[c]).fillna(UNK).astype(int)
        else:
            X[c] = UNK
    # ensure numeric
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = UNK
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    return X

# ---------- load ALL formats up-front ----------
FMTS = ("odi", "t20", "test")

# Batting sources per format
BAT_DF_PATH = {
    "odi":  find_first(DATA_DIR/"odi_bat_model_df.csv",  DATA_DIR/"odi_bat_model_df.csv"),
    "t20":  find_first(DATA_DIR/"T20_bat_model_df.csv",   DATA_DIR/"t20_bat_model_df.csv"),
    "test": find_first(DATA_DIR/"test_bat_model_df.csv", DATA_DIR/"test_bat_model_df.csv"),
}
BAT_MAP_PATH = {
    "odi":  MODEL_DIR/"label_maps_odi_batters.joblib",
    "t20":  MODEL_DIR/"label_maps_t20_batters.joblib",
    "test": MODEL_DIR/"label_maps_test_batters.joblib",
}
BAT_MODEL_PATH = {
    "odi":  MODEL_DIR/"lgb_reg_odi_bat.joblib",
    "t20":  MODEL_DIR/"lgb_reg_t20_bat.joblib",
    "test": MODEL_DIR/"lgb_reg_test_bat.joblib",
}

# Bowling sources per format
BOWL_DF_PATH = {
    "odi":  find_first(DATA_DIR/"odi_bowl_model_df.csv",  DATA_DIR/"odi_bowl_model_df.csv"),
    "t20":  find_first(DATA_DIR/"T20_bowl_model_df.csv",   DATA_DIR/"t20_bowl_model_df.csv"),
    "test": find_first(DATA_DIR/"test_bowl_model_df.csv", DATA_DIR/"test_bowl_model_df.csv"),
}
BOWL_MAP_PATH = {
    "odi":  MODEL_DIR/"label_maps_odi_bowl.joblib",
    "t20":  MODEL_DIR/"label_maps_t20_bowl.joblib",
    "test": MODEL_DIR/"label_maps_test_bowlers.joblib",
}
BOWL_MODEL_PATH = {
    "odi":  MODEL_DIR/"lgb_reg_odi_bowl.joblib",
    "t20":  MODEL_DIR/"lgb_reg_t20_bowl.joblib",
    "test": MODEL_DIR/"lgb_test_bowl.joblib",
}

# Load into memory
BAT = {}
for f in FMTS:
    df = read_any(BAT_DF_PATH[f]) if BAT_DF_PATH[f] else pd.DataFrame()
    BAT[f] = {
        "df": df,
        "maps": joblib.load(BAT_MAP_PATH[f]),
        "model": joblib.load(BAT_MODEL_PATH[f]),
        # dynamic column picks per-format
        "COL_PLAYER": pick_col(df, ["batter","player","batsman","striker","bowler"]),
        "COL_FORMAT": pick_col(df, ["format","match_type","match_format","type"]),
        "COL_VENUE":  pick_col(df, ["venue","venue_name","ground","stadium"]),
        "COL_OPP":    pick_col(df, ["opposition","opponent","bowling_team","against_team"]),
    }

BOWL = {}
for f in FMTS:
    df = read_any(BOWL_DF_PATH[f]) if BOWL_DF_PATH[f] else pd.DataFrame()
    BOWL[f] = {
        "df": df,
        "maps": joblib.load(BOWL_MAP_PATH[f]),
        "model": joblib.load(BOWL_MODEL_PATH[f]),
        "COL_BOWLER": pick_col(df, ["bowler","player","batter","batsman","striker"]),
        "COL_FORMAT": pick_col(df, ["format","match_type","match_format","type"]),
        "COL_VENUE":  pick_col(df, ["venue","venue_name","ground","stadium"]),
        "COL_OPP":    pick_col(df, ["opposition","opponent","batting_team","against_team"]),
    }

# ---------- common suggest ----------
def suggest_from(df: pd.DataFrame, col: str, q: str = "", fmt_col: str | None = None,
                 selected_fmt: str | None = None, limit: int = 12):
    cur = df
    if selected_fmt and fmt_col in df.columns:
        cur = cur[cur[fmt_col].astype(str).str.lower() == selected_fmt]
    if not col or col not in cur.columns:
        return []
    vals = cur[col].dropna().astype(str).unique().tolist()
    if not q:
        return sorted(vals, key=str.lower)[:limit]
    ql = q.lower()
    prefix = [v for v in vals if v.lower().startswith(ql)]
    other  = [v for v in vals if ql in v.lower() and not v.lower().startswith(ql)]
    seen, out = set(), []
    for v in prefix + other:
        k = v.lower()
        if k not in seen:
            out.append(v); seen.add(k)
        if len(out) >= limit: break
    return out

# ====================== Batting endpoints (multi-format) ======================
@app.get("/api/predict/batting/options")
def api_batting_options():
    field  = (request.args.get("field") or "").strip().lower()
    q      = (request.args.get("q") or "").strip()
    fmt    = norm_fmt(request.args.get("format", "test"))

    df  = BAT[fmt]["df"]
    CF  = BAT[fmt]["COL_FORMAT"]; CP = BAT[fmt]["COL_PLAYER"]; CV = BAT[fmt]["COL_VENUE"]; CO = BAT[fmt]["COL_OPP"]

    if field == "format":
        return jsonify({"options": ["ODI","T20","Test"]})
    if df.empty:
        return jsonify({"options": []})

    if field == "player":
        opts = suggest_from(df, CP, q, CF, fmt)
    elif field == "venue":
        opts = suggest_from(df, CV, q, CF, fmt)
    elif field == "opposition":
        opts = suggest_from(df, CO, q, CF, fmt)
    else:
        opts = []
    return jsonify({"options": opts})

@app.post("/api/predict/batting")
def api_predict_batting():
    payload = request.get_json(force=True) or {}
    fmt = norm_fmt(payload.get("format"))
    player = (payload.get("player") or "").strip()
    venue  = (payload.get("venue") or "").strip()
    opp    = (payload.get("opposition") or "").strip()

    ctx = BAT[fmt]
    df, maps, model = ctx["df"], ctx["maps"], ctx["model"]
    CF, CP, CV, CO   = ctx["COL_FORMAT"], ctx["COL_PLAYER"], ctx["COL_VENUE"], ctx["COL_OPP"]

    if df.empty:
        return jsonify({"ok": False, "message": f"No batting DF loaded for {fmt.upper()}."}), 500

    def match(col, val):
        if not val or not col or col not in df.columns: return pd.Series(True, index=df.index)
        return df[col].astype(str).str.lower() == val.lower()

    mask = match(CP, player) & match(CV, venue) & match(CO, opp)
    if CF in df.columns: mask &= match(CF, fmt)
    filtered = df.loc[mask].copy()
    if filtered.empty:
        return jsonify({"ok": False, "message": "No rows found for those filters."}), 404

    # clean
    drop_cols = ['batsman_runs','Match_ID','batter','player_id']
    drop_cols = [c for c in drop_cols if c in filtered.columns]
    X = filtered.drop(columns=drop_cols)

    # encode + force numeric
    X = encode_like_training(X, maps)
    X = force_numeric(X, maps)

    # predict (LightGBM-safe)
    booster = get_lgbm_booster(model)
    try:
        if booster is not None:
            Xb = align_features_for_booster(booster, X)
            y_pred = booster.predict(Xb.values)
        else:
            if hasattr(model, "feature_names_in_"):
                need = list(model.feature_names_in_)
                for m in need:
                    if m not in X.columns: X[m] = 0
                X = X[need]
            y_pred = model.predict(X)

        return jsonify({"ok": True, "n_rows": int(len(X)), "predicted_runs": round(float(np.mean(y_pred)), 0)})
    except Exception as e:
        return jsonify({"ok": False, "message": f"Model prediction error: {e}"}), 500

# ====================== Bowling endpoints (multi-format) ======================
@app.get("/api/predict/bowling/options")
def api_bowling_options():
    field  = (request.args.get("field") or "").strip().lower()
    q      = (request.args.get("q") or "").strip()
    fmt    = norm_fmt(request.args.get("format", "test"))

    df  = BOWL[fmt]["df"]
    CF  = BOWL[fmt]["COL_FORMAT"]; CB = BOWL[fmt]["COL_BOWLER"]; CV = BOWL[fmt]["COL_VENUE"]; CO = BOWL[fmt]["COL_OPP"]

    if field == "format":
        return jsonify({"options": ["ODI","T20","Test"]})
    if df.empty:
        return jsonify({"options": []})

    if field == "player":      # bowler
        opts = suggest_from(df, CB, q, CF, fmt)
    elif field == "venue":
        opts = suggest_from(df, CV, q, CF, fmt)
    elif field == "opposition":
        opts = suggest_from(df, CO, q, CF, fmt)
    else:
        opts = []
    return jsonify({"options": opts})

@app.post("/api/predict/bowling")
def api_predict_bowling():
    payload = request.get_json(force=True) or {}
    fmt = norm_fmt(payload.get("format"))
    bowler = (payload.get("player") or "").strip()
    venue  = (payload.get("venue") or "").strip()
    opp    = (payload.get("opposition") or "").strip()

    ctx = BOWL[fmt]
    df, maps, model = ctx["df"], ctx["maps"], ctx["model"]
    CF, CB, CV, CO   = ctx["COL_FORMAT"], ctx["COL_BOWLER"], ctx["COL_VENUE"], ctx["COL_OPP"]

    if df.empty:
        return jsonify({"ok": False, "message": f"No bowling DF loaded for {fmt.upper()}."}), 500

    def match(col, val):
        if not val or not col or col not in df.columns: return pd.Series(True, index=df.index)
        return df[col].astype(str).str.lower() == val.lower()

    mask = match(CB, bowler) & match(CV, venue) & match(CO, opp)
    if CF in df.columns: mask &= match(CF, fmt)
    filtered = df.loc[mask].copy()
    if filtered.empty:
        return jsonify({"ok": False, "message": "No rows found for those filters."}), 404

    # clean (your exact drop list)
    drop_cols = [
        'wickets_in_match','Match_ID','player_id','bowler','match_date',
        'balls_bowled_in_match','runs_conceded_in_match','extras_in_match',
        'maidens_in_match','econ_rate_in_match','strike_rate_in_match',
        'innings_runs_conceded','innings_wickets','innings_balls_bowled','innings_batting_team',
        'outcome.winner','outcome.by.runs','outcome.by.wickets','outcome.result','outcome.summary'
    ]
    drop_cols = [c for c in drop_cols if c in filtered.columns]
    X = filtered.drop(columns=drop_cols)

    # encode + force numeric
    X = encode_like_training(X, maps)
    X = force_numeric(X, maps)

    # predict (LightGBM-safe)
    booster = get_lgbm_booster(model)
    try:
        if booster is not None:
            Xb = align_features_for_booster(booster, X)
            y_pred = booster.predict(Xb.values)
        else:
            if hasattr(model, "feature_names_in_"):
                need = list(model.feature_names_in_)
                for m in need:
                    if m not in X.columns: X[m] = 0
                X = X[need]
            y_pred = model.predict(X)

        return jsonify({"ok": True, "n_rows": int(len(X)), "predicted_wickets": round(float(np.mean(y_pred)), 0)})
    except Exception as e:
        return jsonify({"ok": False, "message": f"Model prediction error: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
