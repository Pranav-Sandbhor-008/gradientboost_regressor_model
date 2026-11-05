"""
Salary Predictor - Flask app for Vercel
Model : gradientboost.pkl (sklearn GradientBoostingRegressor)
Input : Name, Department, Experience_Years, Education_Level, Age, Gender, City
"""
import json
import os
import pickle
import re
import warnings

import numpy as np
from flask import Flask, Response, jsonify, request

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ----------------------------------------------------------------------------
# 1) CONFIG  -->  EDIT THIS SECTION TO MATCH YOUR TRAINING DATA
# ----------------------------------------------------------------------------
# Your model was trained on LABEL-ENCODED numbers (0, 1, 2 ...), not text.
# The order below must be the SAME order your LabelEncoder used while training.
# LabelEncoder sorts alphabetically, so these are alphabetical guesses based on
# the number of categories found inside the model. Replace with your real values
# (check with:  list(le.classes_) in your training notebook).
DEPARTMENTS = ["Engineering", "Finance", "HR", "Marketing", "Sales"]      # 0..4
EDUCATION   = ["Bachelors", "High School", "Masters", "PhD"]              # 0..3
GENDERS     = ["Female", "Male"]                                          # 0..1
CITIES      = ["Bangalore", "Delhi", "Hyderabad", "Mumbai", "Pune"]       # 0..4

DEFAULT_NAME_CODE = 25     # 'Name' was used as a numeric code by the model (0-49)
CURRENCY = "₹"             # change to "$" etc.
SALARY_PERIOD = "per year" # or "per month"
MODEL_FILE = "gradientboost.pkl"
# ----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, MODEL_FILE), "rb") as f:
    MODEL = pickle.load(f)

FEATURES = list(getattr(MODEL, "feature_names_in_",
                        ["Name", "Department", "Experience_Years",
                         "Education_Level", "Age", "Gender", "City"]))
try:
    BASELINE = float(np.ravel(MODEL.init_.constant_)[0])   # mean salary in training data
except Exception:
    BASELINE = float(MODEL.predict(np.array([[DEFAULT_NAME_CODE, 0, 5, 0, 30, 0, 0]], float))[0])

IMPORTANCE = {n: float(v) for n, v in zip(FEATURES, MODEL.feature_importances_)}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def money(n):
    n = int(round(n))
    s = str(abs(n))
    if CURRENCY == "₹" and len(s) > 3:                      # Indian digit grouping
        s = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", s[:-3]) + "," + s[-3:]
    else:
        s = f"{abs(n):,}"
    return f"{'-' if n < 0 else ''}{CURRENCY}{s}"


def vector(d):
    return [d["name_code"], d["department"], d["experience"],
            d["education"], d["age"], d["gender"], d["city"]]


def predict_one(d):
    return float(MODEL.predict(np.array([vector(d)], dtype=float))[0])


def parse(payload):
    def num(key, lo, hi, cast=float):
        try:
            v = cast(payload.get(key))
        except (TypeError, ValueError):
            raise ValueError(f"'{key}' must be a number")
        if not lo <= v <= hi:
            raise ValueError(f"'{key}' must be between {lo} and {hi}")
        return v

    d = {
        "name": str(payload.get("name", "")).strip()[:40],
        "name_code": num("name_code", 0, 200, int) if payload.get("name_code") not in (None, "") else DEFAULT_NAME_CODE,
        "department": num("department", 0, len(DEPARTMENTS) - 1, int),
        "education": num("education", 0, len(EDUCATION) - 1, int),
        "gender": num("gender", 0, len(GENDERS) - 1, int),
        "city": num("city", 0, len(CITIES) - 1, int),
        "experience": num("experience", 0, 45),
        "age": num("age", 18, 70),
    }
    if d["age"] - d["experience"] < 16:
        raise ValueError("Experience is too high for this age. Please check both values.")
    return d


def level_text(ratio):
    if ratio < 0.80:  return "Below average", "low"
    if ratio < 0.95:  return "Slightly below average", "low"
    if ratio <= 1.05: return "Around average", "mid"
    if ratio <= 1.25: return "Above average", "high"
    if ratio <= 1.50: return "Well above average", "high"
    return "Top of the range", "top"


def stage_text(exp):
    if exp < 2:  return "an entry-level"
    if exp < 5:  return "an early-career"
    if exp < 10: return "a mid-career"
    if exp < 18: return "a senior"
    return "a highly experienced"


def build_response(d):
    pred = predict_one(d)
    ratio = pred / BASELINE
    label, tone = level_text(ratio)
    diff_pct = (ratio - 1) * 100

    # curve: salary vs experience (age moves together with experience)
    xs = list(range(0, 31))
    rows = []
    for e in xs:
        c = dict(d)
        c["experience"] = float(e)
        c["age"] = float(min(70, max(18, d["age"] + (e - d["experience"]))))
        rows.append(vector(c))
    curve = [round(float(v), 2) for v in MODEL.predict(np.array(rows, dtype=float))]

    # insight: one more year of experience
    plus = dict(d); plus["experience"] = d["experience"] + 1; plus["age"] = min(70, d["age"] + 1)
    per_year = predict_one(plus) - pred

    # insight: best department for the same profile
    dept_scores = []
    for i, name in enumerate(DEPARTMENTS):
        c = dict(d); c["department"] = i
        dept_scores.append((name, predict_one(c)))
    dept_scores.sort(key=lambda x: -x[1])
    best_dept, best_val = dept_scores[0]

    # insight: best city for the same profile
    city_scores = []
    for i, name in enumerate(CITIES):
        c = dict(d); c["city"] = i
        city_scores.append((name, predict_one(c)))
    city_scores.sort(key=lambda x: -x[1])
    best_city, best_city_val = city_scores[0]

    who = d["name"] or "This profile"
    summary = (
        f"{who} is {stage_text(d['experience'])} professional in {DEPARTMENTS[d['department']]} "
        f"with {d['experience']:g} year{'s' if d['experience'] != 1 else ''} of experience, "
        f"a {EDUCATION[d['education']]} qualification, based in {CITIES[d['city']]}. "
        f"The model estimates a salary of {money(pred)} {SALARY_PERIOD}, which is "
        f"{abs(diff_pct):.0f}% {'above' if diff_pct >= 0 else 'below'} the average of "
        f"{money(BASELINE)} seen in the training data."
    )

    insights = []
    if abs(per_year) >= 1:
        insights.append(
            f"One more year of experience changes the estimate by about "
            f"{'+' if per_year >= 0 else '-'}{money(abs(per_year))}.")
    if best_dept != DEPARTMENTS[d["department"]] and best_val - pred > 1:
        insights.append(
            f"With the same profile, {best_dept} shows the highest estimate "
            f"({money(best_val)}, {money(best_val - pred)} more).")
    else:
        insights.append(f"{DEPARTMENTS[d['department']]} already gives the highest estimate for this profile.")
    if best_city != CITIES[d["city"]] and best_city_val - pred > 1:
        insights.append(
            f"Among the cities, {best_city} gives the highest estimate ({money(best_city_val)}).")
    else:
        insights.append(f"{CITIES[d['city']]} already gives the highest estimate for this profile.")

    return {
        "prediction": round(pred, 2),
        "formatted": money(pred),
        "baseline": round(BASELINE, 2),
        "baseline_formatted": money(BASELINE),
        "ratio": round(ratio, 4),
        "level": label,
        "tone": tone,
        "diff_pct": round(diff_pct, 1),
        "summary": summary,
        "insights": insights,
        "curve": curve,
        "curve_x": xs,
        "current_exp": d["experience"],
        "period": SALARY_PERIOD,
        "currency": CURRENCY,
    }


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/")
def index():
    cfg = {
        "departments": DEPARTMENTS, "education": EDUCATION, "genders": GENDERS,
        "cities": CITIES, "name_code": DEFAULT_NAME_CODE, "currency": CURRENCY,
        "period": SALARY_PERIOD,
        "importance": [
            {"feature": k.replace("_", " "), "value": round(v * 100, 1)}
            for k, v in sorted(IMPORTANCE.items(), key=lambda x: -x[1])
        ],
    }
    html = INDEX_HTML.replace("__CONFIG__", json.dumps(cfg))
    return Response(html, mimetype="text/html")


@app.post("/api/predict")
def api_predict():
    try:
        data = parse(request.get_json(force=True, silent=True) or {})
        return jsonify(ok=True, **build_response(data))
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 400
    except Exception as e:  # pragma: no cover
        return jsonify(ok=False, error=f"Prediction failed: {e}"), 500


@app.get("/health")
def health():
    return jsonify(status="ok", features=FEATURES)


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Salary Predictor · AI Estimate</title>
<meta name="color-scheme" content="light dark">
<style>
:root{
  --bg:#f4f6fb; --bg2:#eaeefb; --card:#ffffff; --text:#0f172a; --muted:#64748b;
  --border:#e2e8f0; --accent:#4f46e5; --accent2:#06b6d4; --accent-soft:#eef2ff;
  --good:#16a34a; --warn:#f59e0b; --bad:#ef4444; --shadow:0 10px 30px rgba(15,23,42,.08);
  --input:#f8fafc;
}
[data-theme="dark"]{
  --bg:#0b1020; --bg2:#10172e; --card:#141b34; --text:#e6eaf7; --muted:#93a0c0;
  --border:#232c4d; --accent:#818cf8; --accent2:#22d3ee; --accent-soft:#1b2347;
  --good:#4ade80; --warn:#fbbf24; --bad:#f87171; --shadow:0 10px 30px rgba(0,0,0,.45);
  --input:#0f1630;
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{font-family:Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:var(--text);
  background:radial-gradient(1100px 500px at 10% -10%,var(--bg2),transparent),var(--bg);
  min-height:100vh;transition:background .3s,color .3s;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding:20px}
header{display:flex;align-items:center;justify-content:space-between;padding:10px 0 22px}
.brand{display:flex;gap:12px;align-items:center}
.logo{width:42px;height:42px;border-radius:12px;display:grid;place-items:center;color:#fff;font-size:20px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));box-shadow:0 6px 18px rgba(79,70,229,.35)}
.brand h1{font-size:1.25rem;letter-spacing:-.2px}
.brand p{font-size:.8rem;color:var(--muted)}
.icon-btn{border:1px solid var(--border);background:var(--card);color:var(--text);width:42px;height:42px;
  border-radius:12px;cursor:pointer;font-size:18px;transition:.2s;box-shadow:var(--shadow)}
.icon-btn:hover{transform:translateY(-2px);border-color:var(--accent)}
.grid{display:grid;grid-template-columns:420px 1fr;gap:22px;align-items:start}
@media(max-width:940px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:22px;
  box-shadow:var(--shadow);transition:background .3s,border-color .3s}
.card h2{font-size:1rem;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.field{margin-bottom:16px}
label{display:flex;justify-content:space-between;font-size:.8rem;font-weight:600;color:var(--muted);margin-bottom:6px}
label output{color:var(--accent);font-weight:700}
input[type=text],input[type=number],select{width:100%;padding:11px 12px;border-radius:12px;border:1px solid var(--border);
  background:var(--input);color:var(--text);font-size:.95rem;outline:none;transition:.2s}
input:focus,select:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
input[type=range]{width:100%;accent-color:var(--accent);height:6px;cursor:pointer}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.seg{display:grid;grid-template-columns:1fr 1fr;background:var(--input);border:1px solid var(--border);border-radius:12px;padding:4px;gap:4px}
.seg button{border:0;background:transparent;color:var(--muted);padding:9px;border-radius:9px;cursor:pointer;font-weight:600;transition:.2s}
.seg button.on{background:var(--card);color:var(--accent);box-shadow:0 2px 8px rgba(0,0,0,.12)}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{border:1px solid var(--border);background:var(--input);color:var(--text);padding:7px 12px;border-radius:999px;
  font-size:.82rem;cursor:pointer;transition:.2s}
.chip.on{background:var(--accent);border-color:var(--accent);color:#fff}
details{margin:6px 0 16px;font-size:.85rem;color:var(--muted)}
summary{cursor:pointer;font-weight:600;margin-bottom:8px}
.actions{display:flex;gap:10px;margin-top:6px}
.btn{flex:1;border:0;border-radius:12px;padding:13px;font-weight:700;cursor:pointer;font-size:.95rem;transition:.2s}
.btn.primary{color:#fff;background:linear-gradient(135deg,var(--accent),var(--accent2));box-shadow:0 8px 20px rgba(79,70,229,.3)}
.btn.primary:hover{transform:translateY(-2px)}
.btn.ghost{background:var(--input);color:var(--text);border:1px solid var(--border);flex:0 0 auto;padding:13px 16px}
.btn.ghost:hover{border-color:var(--accent)}
.toggle{display:flex;align-items:center;gap:8px;font-size:.82rem;color:var(--muted);margin-top:14px}
.err{background:rgba(239,68,68,.1);border:1px solid var(--bad);color:var(--bad);padding:10px 12px;border-radius:12px;
  font-size:.85rem;margin-bottom:14px;display:none}
.hero{position:relative;overflow:hidden;border-radius:20px;padding:26px;color:#fff;
  background:linear-gradient(135deg,#4f46e5,#06b6d4);margin-bottom:18px}
.hero::after{content:"";position:absolute;right:-60px;top:-60px;width:220px;height:220px;border-radius:50%;background:rgba(255,255,255,.12)}
.hero small{opacity:.85;font-weight:600;letter-spacing:.4px;text-transform:uppercase;font-size:.72rem}
.hero .amount{font-size:clamp(2rem,6vw,3.2rem);font-weight:800;letter-spacing:-1px;margin:4px 0}
.hero .per{opacity:.85;font-size:.9rem}
.badge{display:inline-block;margin-top:12px;padding:6px 12px;border-radius:999px;background:rgba(255,255,255,.2);font-weight:700;font-size:.82rem;backdrop-filter:blur(6px)}
.gauge{margin:6px 0 4px}
.gbar{position:relative;height:12px;border-radius:99px;background:linear-gradient(90deg,var(--bad),var(--warn),var(--good))}
.gdot{position:absolute;top:50%;width:22px;height:22px;border-radius:50%;background:var(--card);border:4px solid var(--accent);
  transform:translate(-50%,-50%);transition:left .7s cubic-bezier(.2,.8,.2,1);box-shadow:0 2px 10px rgba(0,0,0,.3)}
.gleg{display:flex;justify-content:space-between;font-size:.72rem;color:var(--muted);margin-top:8px}
.summary{font-size:.98rem;padding:14px 16px;border-left:4px solid var(--accent);background:var(--accent-soft);border-radius:0 12px 12px 0;margin:14px 0}
.insights{list-style:none;display:grid;gap:8px}
.insights li{display:flex;gap:10px;align-items:flex-start;font-size:.9rem;padding:10px 12px;border:1px solid var(--border);border-radius:12px;background:var(--input)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}
@media(max-width:700px){.two{grid-template-columns:1fr}}
canvas{width:100%;height:230px;display:block}
.imp{display:grid;gap:10px}
.imp .r{display:grid;grid-template-columns:120px 1fr 44px;gap:10px;align-items:center;font-size:.82rem}
.imp .t{height:8px;border-radius:99px;background:var(--border);overflow:hidden}
.imp .t i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2));border-radius:99px;transition:width .8s}
.hist{display:grid;gap:8px;margin-top:6px}
.hist button{display:flex;justify-content:space-between;gap:8px;text-align:left;border:1px solid var(--border);background:var(--input);
  color:var(--text);border-radius:10px;padding:9px 12px;cursor:pointer;font-size:.82rem}
.hist button:hover{border-color:var(--accent)}
.muted{color:var(--muted);font-size:.8rem}
.skeleton{opacity:.55}
.toast{position:fixed;left:50%;bottom:26px;transform:translate(-50%,20px);background:var(--text);color:var(--bg);padding:10px 16px;
  border-radius:12px;font-size:.85rem;opacity:0;pointer-events:none;transition:.3s}
.toast.show{opacity:1;transform:translate(-50%,0)}
footer{text-align:center;color:var(--muted);font-size:.78rem;padding:26px 0 8px}
.head-actions{display:flex;gap:8px;margin-bottom:14px}
.small-btn{border:1px solid var(--border);background:var(--input);color:var(--text);padding:7px 12px;border-radius:10px;font-size:.78rem;cursor:pointer}
.small-btn:hover{border-color:var(--accent)}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">
      <div class="logo">💼</div>
      <div><h1>Salary Predictor</h1><p>Gradient Boosting · instant AI estimate</p></div>
    </div>
    <button class="icon-btn" id="themeBtn" title="Toggle theme" aria-label="Toggle theme">🌙</button>
  </header>

  <div class="grid">
    <!-- FORM -->
    <section class="card">
      <h2>🧾 Employee profile</h2>
      <div class="err" id="err"></div>

      <div class="field">
        <label for="name">Name (optional)</label>
        <input type="text" id="name" placeholder="e.g. Aarav Sharma" maxlength="40" autocomplete="off">
      </div>

      <div class="field">
        <label>Department</label>
        <div class="chips" id="deptChips"></div>
      </div>

      <div class="field">
        <label for="exp">Experience <output id="expOut">5 yrs</output></label>
        <input type="range" id="exp" min="0" max="40" step="0.5" value="5">
      </div>

      <div class="field">
        <label for="age">Age <output id="ageOut">30 yrs</output></label>
        <input type="range" id="age" min="18" max="65" step="1" value="30">
      </div>

      <div class="row">
        <div class="field">
          <label for="edu">Education</label>
          <select id="edu"></select>
        </div>
        <div class="field">
          <label for="city">City</label>
          <select id="city"></select>
        </div>
      </div>

      <div class="field">
        <label>Gender</label>
        <div class="seg" id="genderSeg"></div>
      </div>

      <details>
        <summary>⚙️ Advanced</summary>
        <label for="nameCode">Name code used by the model</label>
        <input type="number" id="nameCode" min="0" max="200" value="25">
        <p class="muted" style="margin-top:6px">Your model was trained with Name as a number. Leave the default unless you know the code.</p>
      </details>

      <div class="actions">
        <button class="btn primary" id="predictBtn">✨ Predict salary</button>
        <button class="btn ghost" id="randomBtn" title="Random profile">🎲</button>
        <button class="btn ghost" id="resetBtn" title="Reset">↺</button>
      </div>
      <label class="toggle"><input type="checkbox" id="live" checked> Live update while I change values</label>
    </section>

    <!-- RESULT -->
    <section>
      <div class="hero" id="hero">
        <small>Estimated salary</small>
        <div class="amount" id="amount">—</div>
        <div class="per" id="per"></div>
        <span class="badge" id="badge">Waiting for prediction…</span>
      </div>

      <div class="card">
        <h2>🗣️ In plain words</h2>
        <div class="head-actions">
          <button class="small-btn" id="copyBtn">📋 Copy summary</button>
          <button class="small-btn" id="printBtn">🖨️ Print</button>
        </div>
        <div class="summary" id="summary">Fill in the profile and press Predict.</div>

        <div class="gauge">
          <div class="gbar"><div class="gdot" id="gdot" style="left:50%"></div></div>
          <div class="gleg"><span>Lower</span><span id="avgLabel">Average</span><span>Higher</span></div>
        </div>

        <ul class="insights" id="insights" style="margin-top:16px"></ul>

        <div class="two">
          <div>
            <h2>📈 Salary vs experience</h2>
            <canvas id="chart"></canvas>
            <p class="muted" id="chartTip">Hover the chart to explore.</p>
          </div>
          <div>
            <h2>🎯 What the model cares about</h2>
            <div class="imp" id="imp"></div>
            <h2 style="margin-top:20px">🕘 Recent predictions</h2>
            <div class="hist" id="hist"></div>
          </div>
        </div>
      </div>
    </section>
  </div>
  <footer>Estimates are statistical predictions from a machine-learning model, not a guarantee of actual pay.</footer>
</div>
<div class="toast" id="toast"></div>

<script>
const CFG = __CONFIG__;
const $ = id => document.getElementById(id);
const state = {department:0, gender:0, last:null, hoverIdx:null};

/* ---------- theme ---------- */
function setTheme(t){
  document.documentElement.setAttribute("data-theme", t);
  $("themeBtn").textContent = t === "dark" ? "☀️" : "🌙";
  try{localStorage.setItem("theme", t)}catch(e){}
  if(state.last) drawChart();
}
(function(){
  let t = null; try{t = localStorage.getItem("theme")}catch(e){}
  if(!t) t = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  setTheme(t);
})();
$("themeBtn").onclick = () => setTheme(document.documentElement.getAttribute("data-theme")==="dark" ? "light" : "dark");

/* ---------- build form ---------- */
CFG.departments.forEach((d,i)=>{
  const b=document.createElement("button"); b.type="button"; b.className="chip"+(i===0?" on":""); b.textContent=d;
  b.onclick=()=>{state.department=i;[...$("deptChips").children].forEach((c,j)=>c.classList.toggle("on",j===i));trigger()};
  $("deptChips").appendChild(b);
});
CFG.education.forEach((e,i)=>$("edu").add(new Option(e,i)));
CFG.cities.forEach((c,i)=>$("city").add(new Option(c,i)));
CFG.genders.forEach((g,i)=>{
  const b=document.createElement("button"); b.type="button"; b.textContent=g; b.className=i===0?"on":"";
  b.onclick=()=>{state.gender=i;[...$("genderSeg").children].forEach((c,j)=>c.classList.toggle("on",j===i));trigger()};
  $("genderSeg").appendChild(b);
});
$("genderSeg").style.gridTemplateColumns=`repeat(${CFG.genders.length},1fr)`;
$("nameCode").value = CFG.name_code;

CFG.importance.forEach(r=>{
  const d=document.createElement("div"); d.className="r";
  d.innerHTML=`<span>${r.feature}</span><div class="t"><i style="width:0"></i></div><b>${r.value}%</b>`;
  $("imp").appendChild(d);
  requestAnimationFrame(()=>setTimeout(()=>d.querySelector("i").style.width=Math.min(100,r.value*2.2)+"%",50));
});

const syncOut=()=>{ $("expOut").textContent=$("exp").value+" yrs"; $("ageOut").textContent=$("age").value+" yrs"; };
["exp","age"].forEach(id=>$(id).addEventListener("input",()=>{syncOut();trigger()}));
["edu","city"].forEach(id=>$(id).addEventListener("change",trigger));
$("name").addEventListener("input",trigger); $("nameCode").addEventListener("input",trigger);

let timer; function trigger(){ if(!$("live").checked) return; clearTimeout(timer); timer=setTimeout(predict,280); }

function payload(){
  return {name:$("name").value, name_code:$("nameCode").value, department:state.department,
    experience:$("exp").value, education:$("edu").value, age:$("age").value, gender:state.gender, city:$("city").value};
}

/* ---------- predict ---------- */
async function predict(){
  const btn=$("predictBtn"); btn.textContent="Predicting…"; $("hero").classList.add("skeleton");
  try{
    const res = await fetch("/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload())});
    const data = await res.json();
    if(!data.ok){ showErr(data.error); return; }
    showErr(null); state.last=data; render(data); saveHistory(data);
  }catch(e){ showErr("Could not reach the server. Please try again."); }
  finally{ btn.textContent="✨ Predict salary"; $("hero").classList.remove("skeleton"); }
}
function showErr(m){ const e=$("err"); e.style.display=m?"block":"none"; e.textContent=m||""; }

let animFrom=0;
function countUp(to){
  const from=animFrom, t0=performance.now(), dur=700;
  (function step(t){
    const p=Math.min(1,(t-t0)/dur), v=from+(to-from)*(1-Math.pow(1-p,3));
    $("amount").textContent=CFG.currency+Math.round(v).toLocaleString(CFG.currency==="₹"?"en-IN":"en-US");
    if(p<1) requestAnimationFrame(step); else animFrom=to;
  })(t0);
}
function render(d){
  countUp(d.prediction);
  $("per").textContent=d.period+" · typical average "+d.baseline_formatted;
  $("badge").textContent=(d.diff_pct>=0?"▲ ":"▼ ")+Math.abs(d.diff_pct)+"% vs average · "+d.level;
  $("summary").textContent=d.summary;
  $("avgLabel").textContent="Average "+d.baseline_formatted;
  const pos=Math.max(4,Math.min(96,50+(d.ratio-1)*100));
  $("gdot").style.left=pos+"%";
  $("insights").innerHTML=d.insights.map(t=>`<li><span>💡</span><span>${t}</span></li>`).join("");
  drawChart();
}

/* ---------- chart ---------- */
function drawChart(){
  const d=state.last; if(!d) return;
  const cv=$("chart"), dpr=window.devicePixelRatio||1, W=cv.clientWidth, H=cv.clientHeight;
  cv.width=W*dpr; cv.height=H*dpr; const c=cv.getContext("2d"); c.scale(dpr,dpr);
  const css=getComputedStyle(document.documentElement);
  const col=n=>css.getPropertyValue(n).trim();
  const ys=d.curve, xs=d.curve_x, pad={l:52,r:12,t:12,b:26};
  const mn=Math.min(...ys)*0.97, mx=Math.max(...ys)*1.03;
  const X=x=>pad.l+(x-xs[0])/(xs[xs.length-1]-xs[0])*(W-pad.l-pad.r);
  const Y=y=>pad.t+(1-(y-mn)/(mx-mn))*(H-pad.t-pad.b);
  c.clearRect(0,0,W,H); c.font="11px system-ui"; c.fillStyle=col("--muted"); c.strokeStyle=col("--border");
  for(let i=0;i<=4;i++){const v=mn+(mx-mn)*i/4,y=Y(v);c.beginPath();c.moveTo(pad.l,y);c.lineTo(W-pad.r,y);c.stroke();
    c.textAlign="right";c.fillText(CFG.currency+(v>=1000?Math.round(v/1000)+"k":Math.round(v)),pad.l-6,y+4);}
  c.textAlign="center"; [0,10,20,30].forEach(x=>c.fillText(x+"y",X(x),H-8));
  const g=c.createLinearGradient(0,pad.t,0,H-pad.b); g.addColorStop(0,col("--accent")+"55"); g.addColorStop(1,col("--accent")+"00");
  c.beginPath(); xs.forEach((x,i)=>i?c.lineTo(X(x),Y(ys[i])):c.moveTo(X(x),Y(ys[i])));
  c.lineTo(X(xs[xs.length-1]),H-pad.b); c.lineTo(X(xs[0]),H-pad.b); c.fillStyle=g; c.fill();
  c.beginPath(); xs.forEach((x,i)=>i?c.lineTo(X(x),Y(ys[i])):c.moveTo(X(x),Y(ys[i])));
  c.strokeStyle=col("--accent"); c.lineWidth=2.5; c.lineJoin="round"; c.stroke();
  const cx=Math.min(30,d.current_exp), ci=Math.round(cx);
  c.beginPath(); c.arc(X(cx),Y(ys[ci]),6,0,7); c.fillStyle=col("--card"); c.fill(); c.lineWidth=3; c.strokeStyle=col("--accent2"); c.stroke();
  if(state.hoverIdx!=null){ const i=state.hoverIdx;
    c.strokeStyle=col("--muted"); c.lineWidth=1; c.setLineDash([4,4]); c.beginPath(); c.moveTo(X(xs[i]),pad.t); c.lineTo(X(xs[i]),H-pad.b); c.stroke(); c.setLineDash([]);
    c.beginPath(); c.arc(X(xs[i]),Y(ys[i]),5,0,7); c.fillStyle=col("--accent"); c.fill();
    $("chartTip").textContent=`At ${xs[i]} years of experience → about ${CFG.currency}${Math.round(ys[i]).toLocaleString(CFG.currency==="₹"?"en-IN":"en-US")}`;
  }
  cv._map={X,pad,W};
}
$("chart").addEventListener("mousemove",e=>{
  const d=state.last, cv=$("chart"); if(!d||!cv._map) return;
  const r=cv.getBoundingClientRect(), x=e.clientX-r.left, {pad,W}=cv._map;
  const frac=(x-pad.l)/(W-pad.l-pad.r); state.hoverIdx=Math.max(0,Math.min(d.curve_x.length-1,Math.round(frac*(d.curve_x.length-1)))); drawChart();
});
$("chart").addEventListener("mouseleave",()=>{state.hoverIdx=null;$("chartTip").textContent="Hover the chart to explore.";drawChart()});
window.addEventListener("resize",drawChart);

/* ---------- history ---------- */
function getHist(){ try{return JSON.parse(localStorage.getItem("hist")||"[]")}catch(e){return[]} }
function saveHistory(d){
  const p=payload(), h=getHist(); const key=JSON.stringify(p);
  if(h[0]&&h[0].key===key) return;
  h.unshift({key,p,label:d.formatted,title:`${CFG.departments[p.department]} · ${p.experience}y · ${CFG.cities[p.city]}`});
  try{localStorage.setItem("hist",JSON.stringify(h.slice(0,5)))}catch(e){}
  renderHist();
}
function renderHist(){
  const h=getHist(), box=$("hist");
  box.innerHTML=h.length?"":'<span class="muted">No predictions yet.</span>';
  h.forEach(x=>{
    const b=document.createElement("button"); b.innerHTML=`<span>${x.title}</span><b>${x.label}</b>`;
    b.onclick=()=>{ applyProfile(x.p); predict(); }; box.appendChild(b);
  });
}
function applyProfile(p){
  $("name").value=p.name||""; $("exp").value=p.experience; $("age").value=p.age; $("edu").value=p.education; $("city").value=p.city;
  $("nameCode").value=p.name_code; state.department=+p.department; state.gender=+p.gender;
  [...$("deptChips").children].forEach((c,j)=>c.classList.toggle("on",j===state.department));
  [...$("genderSeg").children].forEach((c,j)=>c.classList.toggle("on",j===state.gender)); syncOut();
}

/* ---------- buttons ---------- */
$("predictBtn").onclick=predict;
$("resetBtn").onclick=()=>{applyProfile({name:"",experience:5,age:30,education:0,city:0,name_code:CFG.name_code,department:0,gender:0});predict()};
$("randomBtn").onclick=()=>{
  const r=n=>Math.floor(Math.random()*n), exp=r(25), age=Math.min(65,22+exp+r(6));
  applyProfile({name:"",experience:exp,age,education:r(CFG.education.length),city:r(CFG.cities.length),
    name_code:CFG.name_code,department:r(CFG.departments.length),gender:r(CFG.genders.length)}); predict();
};
$("copyBtn").onclick=async()=>{
  if(!state.last) return;
  try{await navigator.clipboard.writeText(state.last.summary); toast("Summary copied ✅")}catch(e){toast("Copy not supported")}
};
$("printBtn").onclick=()=>window.print();
function toast(m){const t=$("toast");t.textContent=m;t.classList.add("show");setTimeout(()=>t.classList.remove("show"),1800)}

syncOut(); renderHist(); predict();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, port=5000)