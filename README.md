# 💼 Salary Predictor

A modern, responsive web app that predicts employee salary using a **Gradient Boosting Regressor** (scikit-learn). Built with **Flask** and deployed on **Vercel**. The prediction is shown in plain language, with an interactive chart and tips.

🔗 **Live Demo:** [gradientboostregressormodel.vercel.app](https://gradientboostregressormodel.vercel.app/)

---

## ✨ Features

- 🤖 **ML-powered predictions** using a trained `GradientBoostingRegressor`
- 🗣️ **Human-readable results**: a plain-English summary, for example *"mid-career professional in Engineering… 20% above average"*
- 🌗 **Light and dark mode**: follows the system theme and can be toggled manually
- ⚡ **Live updates**: the prediction changes as you move the sliders
- 📈 **Interactive chart** of salary vs experience, with hover details
- 🎯 **Feature importance** bars that show what the model relies on
- 💡 **Smart insights**: effect of one more year of experience, and which department or city gives a higher estimate
- 🕘 **Recent predictions** saved in the browser and restorable with one click
- 🎲 Random profile, ↺ reset, 📋 copy summary and 🖨️ print
- 📱 Fully responsive design for mobile, tablet and desktop
- ✅ Input validation with clear error messages

---

## 🧰 Tech Stack

| Layer    | Technology                                                          |
|----------|---------------------------------------------------------------------|
| Backend  | Python, Flask                                                       |
| ML Model | scikit-learn (GradientBoostingRegressor)                            |
| Frontend | HTML, CSS, vanilla JavaScript (single file, no external libraries)  |
| Hosting  | Vercel                                                              |

---

## 📁 Project Structure

```
.
├── app.py               # Flask app (backend + UI)
├── gradientboost.pkl    # Trained ML model
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 🧠 Model Inputs

The model expects 7 features, all **label-encoded** as numbers:

| Feature            | Description                           |
|--------------------|---------------------------------------|
| `Name`             | Encoded numeric code (see note below) |
| `Department`       | Encoded department                    |
| `Experience_Years` | Years of experience                   |
| `Education_Level`  | Encoded education level               |
| `Age`              | Age in years                          |
| `Gender`           | Encoded gender                        |
| `City`             | Encoded city                          |

> ⚠️ **Note:** The category lists in `app.py` (`DEPARTMENTS`, `EDUCATION`, `GENDERS`, `CITIES`) must match the **order used by your `LabelEncoder` during training**. In your training notebook, run `list(le.classes_)` for each encoder and update the lists in the CONFIG section of `app.py`.

---

## 🚀 Run Locally

**1. Clone the repository**

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Start the app**

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## ☁️ Deploy on Vercel

1. Push this project to GitHub (make sure `app.py`, `requirements.txt` and `gradientboost.pkl` are in the **root** folder).
2. Go to [vercel.com](https://vercel.com) and click **Add New → Project**.
3. Import your GitHub repository.
4. Click **Deploy**. Vercel detects Flask automatically.

Or use the Vercel CLI:

```bash
npm i -g vercel
vercel
```

---

## ⚙️ Configuration

Open `app.py` and edit the **CONFIG** section at the top:

```python
DEPARTMENTS = ["Engineering", "Finance", "HR", "Marketing", "Sales"]
EDUCATION   = ["Bachelors", "High School", "Masters", "PhD"]
GENDERS     = ["Female", "Male"]
CITIES      = ["Bangalore", "Delhi", "Hyderabad", "Mumbai", "Pune"]

DEFAULT_NAME_CODE = 25
CURRENCY = "₹"              # e.g. "$"
SALARY_PERIOD = "per year"  # or "per month"
```

---

## 🔌 API Reference

### `POST /api/predict`

**Request body**

```json
{
  "name": "Aarav",
  "name_code": 25,
  "department": 0,
  "experience": 6,
  "education": 2,
  "age": 31,
  "gender": 1,
  "city": 4
}
```

**Success response (shortened)**

```json
{
  "ok": true,
  "prediction": 102963.0,
  "formatted": "₹1,02,963",
  "level": "Above average",
  "diff_pct": 20.0,
  "summary": "Aarav is a mid-career professional in Engineering…",
  "insights": ["…"],
  "curve": ["…"]
}
```

**Error response**

```json
{ "ok": false, "error": "Experience is too high for this age. Please check both values." }
```

### `GET /health`

Returns the app status and the model's feature names.

---

## ⚠️ Notes and Limitations

- **Pickle compatibility:** the model was saved with scikit-learn **1.9.1**, so `requirements.txt` pins that version. Use the same version your model was trained with.
- **`Name` feature:** the model was trained with `Name` as a numeric feature, which is not meaningful for predicting salary. Retraining without it is recommended.
- **Estimates only:** predictions are statistical estimates and do not guarantee actual salary.
- **Security:** only load `.pkl` files you created or trust, because pickle files can run code when loaded.

---

## 🛣️ Future Improvements

- Retrain the model without the `Name` feature
- Add model evaluation metrics (MAE, RMSE, R²) to the UI
- Batch prediction through CSV upload
- Prediction ranges instead of a single value

---

## 🤝 Contributing

Contributions are welcome. Fork the repo, create a feature branch and open a pull request.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 👤 Author

**Pranav Sandbhor**

- 💼 LinkedIn: [linkedin.com/in/pranavsandbhor](https://www.linkedin.com/in/pranavsandbhor/)
- 🐙 GitHub: [@your-username](https://github.com/your-username)
- 🌐 Live Demo: [gradientboostregressormodel.vercel.app](https://gradientboostregressormodel.vercel.app/)

⭐ If you found this project useful, please consider giving it a star on GitHub.
