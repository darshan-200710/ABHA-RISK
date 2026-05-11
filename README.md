# ABHA Model - Investor Risk Dashboard

ABHA is a simple local web app that estimates whether an investor may be at risk of panic-selling during a market drop.

It includes:
- A dashboard you can open in your browser.
- Three local machine-learning models: Random Forest, Gradient Boosting, and Logistic Regression.
- Sliders for age, market drop, app logins, and SIP stopped status.
- Batch testing for multiple example investor scenarios.

This project runs fully on your computer. It does not need a Hugging Face token or any paid API key.

## Important Note

This is an educational MVP, not financial advice. The app uses synthetic sample data and simple model training when the server starts.

## What You Need

Install these first:

- Python 3.10 or newer
- Git, if you want to upload or download from GitHub

To check Python:

```powershell
python --version
```

## Start The App On Windows

Open PowerShell in this project folder:

```powershell
cd D:\projects\folder
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Turn it on:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

Start the app:

```powershell
python main.py
```

Open this in your browser:

```text
http://127.0.0.1:8000/
```

## If Port 8000 Is Busy

Use another port, for example 8001:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

Then open:

```text
http://127.0.0.1:8001/
```

## Useful Pages

- Dashboard: `http://127.0.0.1:8000/`
- Health check: `http://127.0.0.1:8000/health`
- API docs: `http://127.0.0.1:8000/docs`

If you run on port 8001, replace `8000` with `8001`.

## How To Use The Dashboard

1. Choose one of the three AI models.
2. Use the quick market scenarios, or adjust the sliders manually.
3. Click **Analyze This Investor**.
4. Read the risk tier and suggested advisor action.
5. Use **Run Batch Test** to test several sample investors at once.

## API Example

You can also test the API directly:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/evaluate_risk `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"age":35,"market_drop_30_days":12,"app_logins_7_days":10,"sip_stopped":false}'
```

```powershell
pip install -r requirements.txt
```
