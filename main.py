"""
AI + Behavioral Hybrid Advisory Engine (ABHA Model)
A FastAPI-based MVP for identifying panic-driven financial decisions in Indian retail investors.

Author: FinTech AI Team
Date: 2026
"""

import random
from typing import List, Dict
from datetime import datetime
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import uvicorn


# ============================================================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE VALIDATION
# ============================================================================

class InvestorData(BaseModel):
    """Mock investor data model."""
    investor_id: str
    age: int
    market_drop_30_days: float
    app_logins_7_days: int
    sip_stopped: bool
    target_panic_sold: int = 0  # 0 or 1


class RiskEvaluationRequest(BaseModel):
    """Request model for risk evaluation endpoint."""
    age: int
    market_drop_30_days: float
    app_logins_7_days: int
    sip_stopped: bool


class RiskEvaluationResponse(BaseModel):
    """Response model for risk evaluation endpoint."""
    risk_tier: str
    ai_panic_probability: float
    recommended_action: str
    timestamp: str


class BatchScenarioRequest(BaseModel):
    """Request for batch scenario testing."""
    scenarios: List[Dict]


class ScenarioResult(BaseModel):
    """Individual scenario result."""
    scenario_id: int
    age: int
    market_drop_30_days: float
    app_logins_7_days: int
    sip_stopped: bool
    model_used: str
    ai_panic_probability: float
    risk_tier: str
    recommended_action: str


# ============================================================================
# MOCK DATA GENERATION
# ============================================================================

def generate_mock_investors(count: int = 50) -> List[InvestorData]:
    """
    Generate synthetic dataset of Indian retail investors.
    
    The dataset includes:
    - investor_id: Unique identifier (IND-001, IND-002, etc.)
    - age: Between 25 and 65
    - market_drop_30_days: Percentage drop (2.0 to 20.0)
    - app_logins_7_days: Number of portfolio checks (1 to 25, correlates with anxiety)
    - sip_stopped: Whether they cancelled an SIP
    - target_panic_sold: Historical panic selling behavior (correlated with other factors)
    
    Args:
        count: Number of investors to generate
        
    Returns:
        List of InvestorData objects
    """
    investors = []
    
    for i in range(1, count + 1):
        # Base investor attributes
        age = random.randint(25, 65)
        market_drop = round(random.uniform(2.0, 20.0), 2)
        
        # Behavioral signals: higher market drops lead to more logins and higher panic sell likelihood
        # Create logical correlation
        base_logins = random.randint(1, 5)
        logins_adjustment = int(market_drop * 1.2)  # More drops = more anxious checks
        app_logins = min(25, base_logins + logins_adjustment)
        
        # SIP cancellation is more likely if market drop is high
        sip_stopped = random.random() < (market_drop / 20.0) * 0.6
        
        # Target: panic sold correlates with market drop + high logins
        panic_likelihood = (market_drop / 20.0) * 0.5 + (app_logins / 25.0) * 0.4 + (0.1 if sip_stopped else 0)
        target_panic_sold = 1 if random.random() < panic_likelihood else 0
        
        investor = InvestorData(
            investor_id=f"IND-{i:03d}",
            age=age,
            market_drop_30_days=market_drop,
            app_logins_7_days=app_logins,
            sip_stopped=sip_stopped,
            target_panic_sold=target_panic_sold
        )
        investors.append(investor)
    
    return investors


# ============================================================================
# MACHINE LEARNING MODEL TRAINING
# ============================================================================

def train_panic_predictor_model(investors: List[InvestorData], model_type: str = "random_forest") -> tuple:
    """
    Train a classifier to predict panic-selling behavior.

    Supports multiple model types:
    - random_forest: RandomForestClassifier (default, most balanced)
    - gradient_boosting: GradientBoostingClassifier (more aggressive)
    - logistic_regression: LogisticRegression (simpler, explainable)
    """
    X = np.array([
        [
            investor.age,
            investor.market_drop_30_days,
            investor.app_logins_7_days,
            int(investor.sip_stopped)
        ]
        for investor in investors
    ])

    y = np.array([investor.target_panic_sold for investor in investors])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            learning_rate=0.1
        )
    elif model_type == "logistic_regression":
        model = LogisticRegression(random_state=42, max_iter=1000)
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2
        )

    model.fit(X_scaled, y)

    return model, scaler, model_type


# ============================================================================
# BEHAVIORAL RISK SCORE (BRS) RULE ENGINE
# ============================================================================

def calculate_brs_tier(
    ai_panic_probability: float,
    market_drop_30_days: float,
    app_logins_7_days: int,
    sip_stopped: bool
) -> tuple:
    """
    Deterministic rule engine for Behavioral Risk Score (BRS) tier assignment.
    
    This function applies explicit business logic on top of the ML probability
    to ensure explainability and auditability in financial advisory.
    
    Rules (evaluated in order):
    1. CRITICAL: AI probability > 70% AND market drop > 15% AND SIP stopped
    2. RED: AI probability > 50% AND market drop > 10% AND logins >= 10
    3. AMBER: Market drop > 8% AND logins >= 5
    4. NORMAL: All other conditions
    
    Args:
        ai_panic_probability: ML model output (0.0 to 1.0)
        market_drop_30_days: Market drop percentage
        app_logins_7_days: Number of app logins in 7 days
        sip_stopped: Whether SIP was cancelled
        
    Returns:
        Tuple of (risk_tier, recommended_action)
    """
    
    # CRITICAL: Highest risk - immediate intervention
    if ai_panic_probability > 0.70 and market_drop_30_days > 15 and sip_stopped:
        return (
            "CRITICAL",
            "URGENT: Schedule immediate call with investor. High panic indicators detected. "
            "Emphasize long-term investment philosophy and provide historical volatility context."
        )
    
    # RED: High risk - proactive outreach
    if ai_panic_probability > 0.50 and market_drop_30_days > 10 and app_logins_7_days >= 10:
        return (
            "RED",
            "HIGH PRIORITY: Initiate advisor contact within 24 hours. Provide portfolio rebalancing options. "
            "Share market commentary and expected recovery timeline."
        )
    
    # AMBER: Moderate risk - monitoring
    if market_drop_30_days > 8 and app_logins_7_days >= 5:
        return (
            "AMBER",
            "MONITOR: Send educational content on market volatility. Offer portfolio review session. "
            "Ensure investor understands diversification benefits."
        )
    
    # NORMAL: Low risk - routine management
    return (
        "NORMAL",
        "ROUTINE: Continue standard advisory services. No immediate action required."
    )


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

# Global state for trained model and data
MODEL_STATE = {
    "model": None,
    "scaler": None,
    "investors": None,
    "model_type": "random_forest",
    "available_models": ["random_forest", "gradient_boosting", "logistic_regression"]
}


async def startup_event():
    """
    Initialize the application on startup:
    1. Generate 50 mock investors
    2. Train the active local ML model
    3. Keep model and data in memory
    """
    print("[STARTUP] Generating 50 mock Indian investors...")
    investors = generate_mock_investors(count=50)
    MODEL_STATE["investors"] = investors

    print(f"[STARTUP] Training {MODEL_STATE['model_type']} panic predictor model...")
    model, scaler, model_type = train_panic_predictor_model(investors, model_type=MODEL_STATE["model_type"])
    MODEL_STATE["model"] = model
    MODEL_STATE["scaler"] = scaler
    MODEL_STATE["model_type"] = model_type

    print("[STARTUP] ABHA Model initialized and ready for predictions.")
    print(f"[STARTUP] Total investors loaded: {len(investors)}")
    print(f"[STARTUP] Active Model: {model_type.upper()}")
    print(f"[STARTUP] Available Models: {', '.join(MODEL_STATE['available_models'])}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_event()
    yield


app = FastAPI(
    title="ABHA Model - AI + Behavioral Hybrid Advisory Engine",
    description="FastAPI MVP for identifying panic-driven investment decisions",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", include_in_schema=False)
async def root():
    """Open the dashboard when users visit the base URL."""
    return RedirectResponse(url="/admin/dashboard")


@app.get("/api/v1/mock_investors", response_model=List[InvestorData])
async def get_mock_investors():
    """
    Endpoint to retrieve the list of generated mock investors.
    
    Returns:
        List of all 50 mock InvestorData objects
    """
    if MODEL_STATE["investors"] is None:
        return {"error": "Model not initialized"}
    
    return MODEL_STATE["investors"]


@app.post("/api/v1/evaluate_risk", response_model=RiskEvaluationResponse)
async def evaluate_risk(request: RiskEvaluationRequest):
    """
    Endpoint to evaluate panic-selling risk for an investor.
    
    Process:
    1. Extract investor features from request
    2. Scale features using the trained scaler
    3. Generate ML probability from the active model
    4. Pass probability and features to BRS rule engine
    5. Return risk tier and recommended action
    
    Args:
        request: RiskEvaluationRequest containing investor metrics
        
    Returns:
        RiskEvaluationResponse with risk_tier, probability, and action
    """
    if MODEL_STATE["model"] is None or MODEL_STATE["scaler"] is None:
        return {
            "error": "Model not initialized",
            "risk_tier": "UNKNOWN",
            "ai_panic_probability": 0.0,
            "recommended_action": "System error. Please retry."
        }

    features = np.array([[
        request.age,
        request.market_drop_30_days,
        request.app_logins_7_days,
        int(request.sip_stopped)
    ]])

    features_scaled = MODEL_STATE["scaler"].transform(features)
    probabilities = MODEL_STATE["model"].predict_proba(features_scaled)[0]
    panic_probability = float(probabilities[1])
    
    # Apply rule engine
    risk_tier, recommended_action = calculate_brs_tier(
        ai_panic_probability=panic_probability,
        market_drop_30_days=request.market_drop_30_days,
        app_logins_7_days=request.app_logins_7_days,
        sip_stopped=request.sip_stopped
    )
    
    return RiskEvaluationResponse(
        risk_tier=risk_tier,
        ai_panic_probability=round(panic_probability * 100, 2),  # Convert to percentage
        recommended_action=recommended_action,
        timestamp=datetime.now().isoformat()
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": MODEL_STATE["model"] is not None,
        "investors_loaded": MODEL_STATE["investors"] is not None,
        "active_model": MODEL_STATE["model_type"],
        "available_models": MODEL_STATE["available_models"]
    }


@app.post("/api/v1/batch_evaluate")
async def batch_evaluate_risk(batch_request: BatchScenarioRequest):
    """
    Batch evaluate risk for multiple scenarios.
    Useful for admin testing different market conditions.
    
    Args:
        batch_request: List of scenarios to evaluate
        
    Returns:
        List of results for each scenario
    """
    if MODEL_STATE["model"] is None or MODEL_STATE["scaler"] is None:
        return {"error": "Model not initialized"}
    
    results = []
    
    for idx, scenario in enumerate(batch_request.scenarios):
        try:
            features = np.array([[
                scenario.get("age", 35),
                scenario.get("market_drop_30_days", 10),
                scenario.get("app_logins_7_days", 7),
                int(scenario.get("sip_stopped", False))
            ]])

            features_scaled = MODEL_STATE["scaler"].transform(features)
            probabilities = MODEL_STATE["model"].predict_proba(features_scaled)[0]
            panic_probability = float(probabilities[1])
            
            risk_tier, recommended_action = calculate_brs_tier(
                ai_panic_probability=panic_probability,
                market_drop_30_days=scenario.get("market_drop_30_days", 10),
                app_logins_7_days=scenario.get("app_logins_7_days", 7),
                sip_stopped=scenario.get("sip_stopped", False)
            )
            
            results.append({
                "scenario_id": idx + 1,
                "market_drop_30_days": scenario.get("market_drop_30_days", 10),
                "app_logins_7_days": scenario.get("app_logins_7_days", 7),
                "sip_stopped": scenario.get("sip_stopped", False),
                "model_used": MODEL_STATE["model_type"],
                "ai_panic_probability": round(panic_probability * 100, 2),
                "risk_tier": risk_tier,
                "recommended_action": recommended_action
            })
        except Exception as e:
            results.append({
                "scenario_id": idx + 1,
                "error": str(e)
            })
    
    return {
        "total_scenarios": len(results),
        "model_used": MODEL_STATE["model_type"],
        "results": results
    }


@app.post("/api/v1/switch_model/{new_model}")
async def switch_model(new_model: str):
    """
    Switch the active ML model for predictions.

    Available models:
    - random_forest: Balanced performance (default)
    - gradient_boosting: More aggressive predictions
    - logistic_regression: Simple and explainable
    
    Args:
        new_model: Name of the model to switch to
        
    Returns:
        Status of model switch
    """
    if new_model not in MODEL_STATE["available_models"]:
        return {
            "error": f"Model '{new_model}' not available",
            "available_models": MODEL_STATE["available_models"]
        }

    if MODEL_STATE["investors"] is None:
        return {"error": "Investors data not loaded"}

    print(f"[ADMIN] Switching model from {MODEL_STATE['model_type']} to {new_model}")
    model, scaler, model_type = train_panic_predictor_model(
        MODEL_STATE["investors"],
        model_type=new_model
    )

    MODEL_STATE["model"] = model
    MODEL_STATE["scaler"] = scaler
    MODEL_STATE["model_type"] = model_type

    return {
        "status": "success",
        "new_model": new_model,
        "message": f"Model successfully switched to {new_model.upper()}"
    }


@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard():
    """
    User-friendly admin dashboard for non-technical users.
    Simplified UI with large buttons, sliders, and visual feedback.
    """
    return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ABHA Market Risk Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            min-height: 100vh;
            background: #0a0604;
            padding: 20px;
            overflow-x: hidden;
            position: relative;
        }

        /* ===== MESH GRADIENT BACKGROUND ===== */
        .mesh-bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            overflow: hidden;
            pointer-events: none;
        }

        .mesh-blob {
            position: absolute;
            border-radius: 50%;
            filter: blur(90px);
            opacity: 0.35;
            will-change: transform;
        }

        .blob-1 {
            width: 700px;
            height: 700px;
            background: radial-gradient(circle, #d97706, #fbbf24);
            top: -10%;
            left: -10%;
            animation: floatBlob1 25s ease-in-out infinite;
        }

        .blob-2 {
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, #dc2626, #991b1b);
            bottom: -15%;
            right: -8%;
            animation: floatBlob2 30s ease-in-out infinite;
        }

        .blob-3 {
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, #b91c1c, #7f1d1d);
            bottom: 20%;
            left: -5%;
            animation: floatBlob3 22s ease-in-out infinite;
        }

        .blob-4 {
            width: 550px;
            height: 550px;
            background: radial-gradient(circle, #f59e0b, #fcd34d);
            top: 30%;
            right: -10%;
            animation: floatBlob4 28s ease-in-out infinite;
        }

        .blob-5 {
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, #92400e, #78350f);
            top: 50%;
            left: 40%;
            animation: floatBlob5 20s ease-in-out infinite;
            opacity: 0.2;
        }

        @keyframes floatBlob1 {
            0%, 100% { transform: translate(0, 0) scale(1); }
            33% { transform: translate(80px, 60px) scale(1.05); }
            66% { transform: translate(-40px, 100px) scale(0.95); }
        }
        @keyframes floatBlob2 {
            0%, 100% { transform: translate(0, 0) scale(1); }
            33% { transform: translate(-70px, -50px) scale(1.08); }
            66% { transform: translate(50px, -80px) scale(0.92); }
        }
        @keyframes floatBlob3 {
            0%, 100% { transform: translate(0, 0) scale(1); }
            33% { transform: translate(60px, -40px) scale(0.95); }
            66% { transform: translate(-50px, 30px) scale(1.1); }
        }
        @keyframes floatBlob4 {
            0%, 100% { transform: translate(0, 0) scale(1); }
            33% { transform: translate(-30px, 70px) scale(1.06); }
            66% { transform: translate(60px, -30px) scale(0.94); }
        }
        @keyframes floatBlob5 {
            0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.2; }
            50% { transform: translate(40px, 40px) scale(1.15); opacity: 0.3; }
        }

        /* ===== PAGE OVERLAY & WRAPPER ===== */
        .bg-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(ellipse at center, transparent 0%, #0a0604 70%);
            z-index: 1;
            pointer-events: none;
        }

        .wrapper {
            max-width: 1400px;
            margin: 0 auto;
            position: relative;
            z-index: 2;
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* ===== GLASS CARD MIXIN STYLES ===== */
        .glass {
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }

        .glass-light {
            background: rgba(255, 255, 255, 0.06);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 16px;
        }

        /* ===== HEADER ===== */
        header {
            padding: 40px 45px;
            text-align: center;
            margin-bottom: 30px;
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 24px;
            box-shadow: 0 8px 40px rgba(0, 0, 0, 0.25);
            position: relative;
            overflow: hidden;
        }

        header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, #d97706, #f59e0b, #dc2626, transparent);
        }

        h1 {
            font-size: 2.6em;
            font-weight: 800;
            background: linear-gradient(135deg, #fbbf24, #d97706, #f59e0b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
            letter-spacing: -1px;
        }

        .subtitle {
            color: rgba(255, 255, 255, 0.5);
            font-size: 1.05em;
            font-weight: 400;
            margin-bottom: 24px;
            letter-spacing: 0.3px;
        }

        .status-bar {
            display: flex;
            justify-content: center;
            gap: 16px;
            flex-wrap: wrap;
        }

        .status-item {
            background: rgba(255, 255, 255, 0.05);
            padding: 14px 28px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            transition: all 0.3s ease;
        }

        .status-item:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(217, 119, 6, 0.3);
            transform: translateY(-2px);
        }

        .status-label {
            font-size: 0.8em;
            color: rgba(255, 255, 255, 0.35);
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 500;
        }

        .status-value {
            font-size: 1.3em;
            font-weight: 700;
            color: #fff;
            margin-top: 4px;
        }

        /* ===== CARDS ===== */
        .main-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }

        .card {
            padding: 32px;
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
            transition: all 0.3s ease;
        }

        .card:hover {
            border-color: rgba(255, 255, 255, 0.12);
            box-shadow: 0 12px 48px rgba(0, 0, 0, 0.35);
        }

        .card h2 {
            color: #fff;
            margin-bottom: 20px;
            font-size: 1.55em;
            font-weight: 700;
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .card h2 .icon {
            font-size: 1.2em;
        }

        .card > p {
            color: rgba(255, 255, 255, 0.4);
            margin-bottom: 24px;
            font-size: 0.95em;
            line-height: 1.6;
        }

        /* ===== MODEL BUTTONS ===== */
        .model-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 12px;
        }

        .model-button {
            padding: 18px 22px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(255, 255, 255, 0.03);
            border-radius: 14px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: all 0.3s ease;
            text-align: left;
            color: rgba(255, 255, 255, 0.7);
            font-family: 'Inter', sans-serif;
            position: relative;
            overflow: hidden;
        }

        .model-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(217, 119, 6, 0.1), transparent);
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .model-button:hover {
            border-color: rgba(217, 119, 6, 0.4);
            background: rgba(217, 119, 6, 0.08);
            transform: translateX(4px);
            color: #fff;
        }

        .model-button:hover::before {
            opacity: 1;
        }

        .model-button.active {
            background: linear-gradient(135deg, rgba(217, 119, 6, 0.25), rgba(245, 158, 11, 0.1));
            border-color: rgba(217, 119, 6, 0.5);
            color: #fff;
            box-shadow: 0 0 30px rgba(217, 119, 6, 0.15), inset 0 1px 0 rgba(255,255,255,0.1);
        }

        .model-button.active::after {
            content: '';
            position: absolute;
            left: 0;
            top: 10%;
            height: 80%;
            width: 3px;
            background: linear-gradient(180deg, #d97706, #f59e0b);
            border-radius: 0 3px 3px 0;
        }

        .model-desc {
            font-size: 0.82em;
            opacity: 0.5;
            margin-top: 6px;
            font-weight: 400;
        }

        .model-button.active .model-desc {
            opacity: 0.7;
        }

        /* ===== SCENARIO BUTTONS ===== */
        .scenario-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }

        .scenario-button {
            padding: 24px 18px;
            border: none;
            border-radius: 14px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.35s ease;
            color: #fff;
            font-family: 'Inter', sans-serif;
            position: relative;
            overflow: hidden;
        }

        .scenario-button::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(180deg, rgba(255,255,255,0.15) 0%, transparent 50%);
            pointer-events: none;
        }

        .scenario-mild {
            background: linear-gradient(135deg, #059669, #34d399);
            box-shadow: 0 8px 24px rgba(5, 150, 105, 0.3);
        }

        .scenario-moderate {
            background: linear-gradient(135deg, #fdcb6e, #e17055);
            box-shadow: 0 8px 24px rgba(225, 112, 85, 0.3);
        }

        .scenario-severe {
            background: linear-gradient(135deg, #e17055, #d63031);
            box-shadow: 0 8px 24px rgba(214, 48, 49, 0.3);
        }

        .scenario-critical {
            background: linear-gradient(135deg, #d63031, #b71c1c);
            box-shadow: 0 8px 24px rgba(183, 28, 28, 0.35);
        }

        .scenario-button:hover {
            transform: translateY(-4px) scale(1.02);
            box-shadow: 0 12px 36px rgba(0,0,0,0.4);
        }

        .scenario-button:active {
            transform: translateY(-1px) scale(0.98);
        }

        .scenario-desc {
            font-size: 0.8em;
            opacity: 0.75;
            margin-top: 6px;
            font-weight: 400;
        }

        /* ===== INPUT SECTION ===== */
        .input-section {
            padding: 24px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 16px;
            margin-bottom: 20px;
        }

        .input-group {
            margin-bottom: 22px;
        }

        .input-group:last-of-type {
            margin-bottom: 20px;
        }

        .input-label {
            font-weight: 600;
            color: rgba(255, 255, 255, 0.8);
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .input-value {
            color: #fbbf24;
            font-size: 1.2em;
            font-weight: 700;
        }

        input[type="range"] {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: rgba(255, 255, 255, 0.1);
            outline: none;
            -webkit-appearance: none;
            cursor: pointer;
            transition: background 0.3s;
        }

        input[type="range"]:hover {
            background: rgba(255, 255, 255, 0.15);
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: linear-gradient(135deg, #d97706, #fbbf24);
            cursor: pointer;
            box-shadow: 0 0 20px rgba(217, 119, 6, 0.4), 0 2px 8px rgba(0,0,0,0.3);
            border: 2px solid rgba(255,255,255,0.15);
            transition: all 0.2s;
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.15);
            box-shadow: 0 0 30px rgba(217, 119, 6, 0.6);
        }

        input[type="range"]::-moz-range-thumb {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: linear-gradient(135deg, #d97706, #fbbf24);
            cursor: pointer;
            border: 2px solid rgba(255,255,255,0.15);
            box-shadow: 0 0 20px rgba(217, 119, 6, 0.4);
        }

        input[type="range"]::-moz-range-track {
            height: 6px;
            border-radius: 3px;
            background: rgba(255, 255, 255, 0.1);
        }

        .input-group small {
            display: block;
            color: rgba(255, 255, 255, 0.3);
            margin-top: 8px;
            font-size: 0.82em;
        }

        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 14px;
            font-size: 1em;
            cursor: pointer;
            color: rgba(255, 255, 255, 0.8);
            font-weight: 500;
        }

        .checkbox-group input[type="checkbox"] {
            width: 22px;
            height: 22px;
            cursor: pointer;
            accent-color: #d97706;
            border-radius: 6px;
            transition: all 0.2s;
        }

        /* ===== BUTTONS ===== */
        .button-group {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }

        .btn-primary {
            padding: 18px 24px;
            background: linear-gradient(135deg, #d97706, #fbbf24);
            color: #fff;
            border: none;
            border-radius: 12px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 6px 24px rgba(217, 119, 6, 0.3);
            font-family: 'Inter', sans-serif;
            position: relative;
            overflow: hidden;
        }

        .btn-primary::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(180deg, rgba(255,255,255,0.15) 0%, transparent 60%);
            pointer-events: none;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 32px rgba(217, 119, 6, 0.45);
        }

        .btn-primary:active {
            transform: translateY(0);
        }

        .btn-secondary {
            padding: 18px 24px;
            background: rgba(255, 255, 255, 0.05);
            color: rgba(255, 255, 255, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            font-family: 'Inter', sans-serif;
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.2);
            color: #fff;
        }

        /* ===== RESULTS ===== */
        .result-section {
            padding: 24px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 16px;
            margin-top: 20px;
            display: none;
        }

        .result-section.show {
            display: block;
            animation: slideIn 0.4s ease-out;
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateY(-12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-header {
            font-size: 1.3em;
            font-weight: 700;
            margin-bottom: 20px;
            color: #fff;
            text-align: center;
        }

        .risk-display {
            display: flex;
            justify-content: center;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }

        .risk-box {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.06);
            backdrop-filter: blur(8px);
            padding: 20px 28px;
            border-radius: 14px;
            text-align: center;
            flex: 1;
            min-width: 160px;
            transition: all 0.3s;
        }

        .risk-box:hover {
            background: rgba(255, 255, 255, 0.06);
            border-color: rgba(217, 119, 6, 0.2);
        }

        .risk-label {
            color: rgba(255, 255, 255, 0.4);
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 500;
            margin-bottom: 8px;
        }

        .risk-value {
            font-size: 2em;
            font-weight: 700;
            color: #fff;
        }

        .risk-tier-badge {
            display: inline-block;
            padding: 14px 36px;
            border-radius: 50px;
            font-size: 1.2em;
            font-weight: 700;
            color: white;
            margin-bottom: 20px;
            letter-spacing: 1px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }

        .tier-normal {
            background: linear-gradient(135deg, #059669, #34d399);
            box-shadow: 0 8px 32px rgba(5, 150, 105, 0.3);
        }
        .tier-amber {
            background: linear-gradient(135deg, #fdcb6e, #e17055);
            box-shadow: 0 8px 32px rgba(225, 112, 85, 0.3);
        }
        .tier-red {
            background: linear-gradient(135deg, #e17055, #d63031);
            box-shadow: 0 8px 32px rgba(214, 48, 49, 0.3);
        }
        .tier-critical {
            background: linear-gradient(135deg, #d63031, #b71c1c);
            box-shadow: 0 8px 32px rgba(183, 28, 28, 0.35);
            animation: pulseDanger 1.5s ease-in-out infinite;
        }

        @keyframes pulseDanger {
            0%, 100% { box-shadow: 0 8px 32px rgba(183, 28, 28, 0.35); }
            50% { box-shadow: 0 8px 48px rgba(183, 28, 28, 0.6), 0 0 60px rgba(183, 28, 28, 0.2); }
        }

        .action-box {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 14px;
            padding: 20px;
            border-left: 3px solid #d97706;
            margin-top: 16px;
        }

        .action-title {
            font-weight: 600;
            color: rgba(255, 255, 255, 0.8);
            margin-bottom: 10px;
            font-size: 0.95em;
        }

        .action-text {
            color: rgba(255, 255, 255, 0.55);
            line-height: 1.7;
            font-size: 0.92em;
        }

        /* ===== BATCH RESULTS ===== */
        .batch-results {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 14px;
            padding: 8px;
            margin-top: 16px;
            max-height: 400px;
            overflow-y: auto;
        }

        .batch-results::-webkit-scrollbar {
            width: 6px;
        }

        .batch-results::-webkit-scrollbar-track {
            background: transparent;
        }

        .batch-results::-webkit-scrollbar-thumb {
            background: rgba(217, 119, 6, 0.3);
            border-radius: 3px;
        }

        .batch-row {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr 1fr;
            gap: 10px;
            padding: 14px 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            font-size: 0.9em;
            color: rgba(255, 255, 255, 0.6);
            transition: background 0.2s;
            align-items: center;
        }

        .batch-row:last-child {
            border-bottom: none;
        }

        .batch-row:hover {
            background: rgba(255, 255, 255, 0.04);
        }

        .batch-header {
            font-weight: 600;
            background: rgba(217, 119, 6, 0.15);
            color: #fbbf24;
            border-radius: 8px;
            border-bottom: none;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .batch-header:hover {
            background: rgba(217, 119, 6, 0.15);
        }

        /* ===== LOADING ===== */
        .loading {
            text-align: center;
            color: #fbbf24;
            font-size: 1.05em;
            padding: 30px;
        }

        .loading-spinner {
            display: inline-block;
            width: 28px;
            height: 28px;
            border: 3px solid rgba(217, 119, 6, 0.15);
            border-top-color: #d97706;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 12px;
            vertical-align: middle;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* ===== RESPONSIVE ===== */
        @media (max-width: 768px) {
            body { padding: 12px; }
            header { padding: 28px 20px; }
            h1 { font-size: 1.8em; }
            .main-container { grid-template-columns: 1fr; }
            .scenario-grid { grid-template-columns: 1fr; }
            .card { padding: 24px; }
            .button-group { grid-template-columns: 1fr; }
            .status-bar { gap: 10px; }
            .status-item { padding: 12px 18px; flex: 1; min-width: 120px; }
            .risk-display { flex-direction: column; }
            .batch-row { grid-template-columns: 1fr 1fr; gap: 6px; font-size: 0.82em; }
        }

        @media (max-width: 480px) {
            h1 { font-size: 1.4em; }
            .card h2 { font-size: 1.2em; }
            .scenario-button { padding: 18px 14px; }
        }
    </style>
</head>
<body>

    <!-- ===== MESH GRADIENT BACKGROUND ===== -->
    <div class="mesh-bg">
        <div class="mesh-blob blob-1"></div>
        <div class="mesh-blob blob-2"></div>
        <div class="mesh-blob blob-3"></div>
        <div class="mesh-blob blob-4"></div>
        <div class="mesh-blob blob-5"></div>
    </div>
    <div class="bg-overlay"></div>

    <div class="wrapper">

        <!-- ===== HEADER ===== -->
        <header>
            <h1>Investor Risk Dashboard</h1>
            <p class="subtitle">AI-powered market scenario testing & behavioral risk analysis</p>
            <div class="status-bar">
                <div class="status-item">
                    <div class="status-label">System</div>
                    <div class="status-value">● Ready</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Active Model</div>
                    <div class="status-value" id="modelStatus">Loading...</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Investors</div>
                    <div class="status-value" id="investorCount">50</div>
                </div>
            </div>
        </header>

        <!-- ===== MAIN GRID ===== -->
        <div class="main-container">

            <!-- Model Selection -->
            <div class="card">
                <h2><span class="icon">&#9670;</span> Select AI Model</h2>
                <p>Choose which AI model to use for panic-sell risk analysis</p>
                <div class="model-grid">
                    <button class="model-button active" onclick="switchModel(this, 'random_forest')">
                        Random Forest
                        <div class="model-desc">Balanced performance &mdash; recommended for most cases</div>
                    </button>
                    <button class="model-button" onclick="switchModel(this, 'gradient_boosting')">
                        Gradient Boosting
                        <div class="model-desc">More sensitive &mdash; detects early warning signs</div>
                    </button>
                    <button class="model-button" onclick="switchModel(this, 'logistic_regression')">
                        Logistic Regression
                        <div class="model-desc">Simple &mdash; best for regulatory explainability</div>
                    </button>
                </div>
            </div>

            <!-- Quick Scenarios -->
            <div class="card">
                <h2><span class="icon">&#9889;</span> Quick Scenarios</h2>
                <p>Test investor reactions to different market conditions instantly</p>
                <div class="scenario-grid">
                    <button class="scenario-button scenario-mild" onclick="runScenario('mild')">
                        Mild Drop
                        <div class="scenario-desc">10% market correction</div>
                    </button>
                    <button class="scenario-button scenario-moderate" onclick="runScenario('moderate')">
                        Moderate Drop
                        <div class="scenario-desc">15% market decline</div>
                    </button>
                    <button class="scenario-button scenario-severe" onclick="runScenario('severe')">
                        Severe Drop
                        <div class="scenario-desc">18% market sell-off</div>
                    </button>
                    <button class="scenario-button scenario-critical" onclick="runScenario('critical')">
                        Critical Drop
                        <div class="scenario-desc">20%+ market crash</div>
                    </button>
                </div>
            </div>

        </div>

        <!-- ===== CUSTOM ANALYSIS ===== -->
        <div class="card" style="margin-bottom: 24px;">
            <h2><span class="icon">&#9998;</span> Custom Investor Analysis</h2>
            <p>Adjust the sliders to build a custom investor profile and evaluate risk</p>

            <div class="input-section">
                <div class="input-group">
                    <div class="input-label">
                        <span>Age</span>
                        <span><span class="input-value" id="ageDisplay">35</span> years</span>
                    </div>
                    <input type="range" id="age" min="25" max="65" value="35" oninput="updateDisplay()">
                    <small>Investor age range &mdash; typically 25&ndash;65</small>
                </div>

                <div class="input-group">
                    <div class="input-label">
                        <span>Market Drop</span>
                        <span><span class="input-value" id="dropDisplay">10.0</span>%</span>
                    </div>
                    <input type="range" id="marketDrop" min="0" max="40" step="0.5" value="10" oninput="updateDisplay()">
                    <small>Magnitude of the recent market decline</small>
                </div>

                <div class="input-group">
                    <div class="input-label">
                        <span>App Logins (7 days)</span>
                        <span><span class="input-value" id="loginsDisplay">7</span> times</span>
                    </div>
                    <input type="range" id="appLogins" min="1" max="25" value="7" oninput="updateDisplay()">
                    <small>Higher login frequency often correlates with anxiety</small>
                </div>

                <div class="input-group">
                    <div class="checkbox-group">
                        <input type="checkbox" id="sipStopped" onchange="updateDisplay()">
                        <label for="sipStopped">SIP (auto-investment) has been stopped</label>
                    </div>
                    <small style="display: block; margin-top: 8px;">A red flag &mdash; indicates potential panic behavior</small>
                </div>

                <div class="button-group">
                    <button class="btn-primary" onclick="evaluateRisk()">
                        Analyze This Investor
                    </button>
                    <button class="btn-secondary" onclick="resetForm()">
                        Reset Fields
                    </button>
                </div>
            </div>

            <!-- Results -->
            <div class="result-section" id="resultSection"></div>

        </div>

        <!-- ===== BATCH TESTING ===== -->
        <div class="card">
            <h2><span class="icon">&#9776;</span> Batch Scenario Test</h2>
            <p>Analyze five different investor profiles side-by-side in a single run</p>

            <button class="btn-primary" onclick="runBatchTest()" style="width: 100%;">
                Run Batch Test
            </button>

            <div class="result-section" id="batchResultSection"></div>
        </div>

    </div>

    <script>
        const API_BASE = window.location.origin;

        const scenarios = {
            mild:     { age: 35, drop: 10, logins: 7,  sip: false },
            moderate: { age: 40, drop: 15, logins: 15, sip: false },
            severe:   { age: 38, drop: 18, logins: 18, sip: true  },
            critical: { age: 45, drop: 20, logins: 22, sip: true  }
        };

        function updateDisplay() {
            document.getElementById('ageDisplay').textContent = document.getElementById('age').value;
            document.getElementById('dropDisplay').textContent = document.getElementById('marketDrop').value;
            document.getElementById('loginsDisplay').textContent = document.getElementById('appLogins').value;
        }

        function getTierClass(tier) {
            switch (tier.toUpperCase()) {
                case 'CRITICAL': return 'tier-critical';
                case 'RED':      return 'tier-red';
                case 'AMBER':    return 'tier-amber';
                default:         return 'tier-normal';
            }
        }

        function switchModel(btn, modelName) {
            fetch(API_BASE + '/api/v1/switch_model/' + modelName, { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    document.querySelectorAll('.model-button').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    document.getElementById('modelStatus').textContent =
                        modelName.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
                });
        }

        function runScenario(name) {
            const s = scenarios[name];
            document.getElementById('age').value = s.age;
            document.getElementById('marketDrop').value = s.drop;
            document.getElementById('appLogins').value = s.logins;
            document.getElementById('sipStopped').checked = s.sip;
            updateDisplay();
            setTimeout(evaluateRisk, 150);
        }

        function evaluateRisk() {
            const payload = {
                age: parseInt(document.getElementById('age').value),
                market_drop_30_days: parseFloat(document.getElementById('marketDrop').value),
                app_logins_7_days: parseInt(document.getElementById('appLogins').value),
                sip_stopped: document.getElementById('sipStopped').checked
            };

            const container = document.getElementById('resultSection');
            container.innerHTML = '<div class="loading"><span class="loading-spinner"></span>Analyzing investor profile...</div>';
            container.classList.add('show');

            fetch(API_BASE + '/api/v1/evaluate_risk', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
            .then(r => r.json())
            .then(data => {
                const tier = data.risk_tier;
                const prob = data.ai_panic_probability;
                container.innerHTML = `
                    <div class="result-header">Analysis Complete</div>
                    <div style="text-align: center;">
                        <div class="risk-tier-badge ${getTierClass(tier)}">${tier} RISK</div>
                    </div>
                    <div class="risk-display">
                        <div class="risk-box">
                            <div class="risk-label">Panic Probability</div>
                            <div class="risk-value">${prob}%</div>
                        </div>
                    </div>
                    <div class="action-box">
                        <div class="action-title">Recommended Advisor Action</div>
                        <div class="action-text">${data.recommended_action}</div>
                    </div>
                `;
            });
        }

        function runBatchTest() {
            const container = document.getElementById('batchResultSection');
            container.innerHTML = '<div class="loading"><span class="loading-spinner"></span>Running batch evaluation...</div>';
            container.classList.add('show');

            const testScenarios = [
                { age: 30, market_drop_30_days: 5,  app_logins_7_days: 3,  sip_stopped: false },
                { age: 35, market_drop_30_days: 10, app_logins_7_days: 7,  sip_stopped: false },
                { age: 40, market_drop_30_days: 15, app_logins_7_days: 15, sip_stopped: false },
                { age: 45, market_drop_30_days: 18, app_logins_7_days: 20, sip_stopped: true  },
                { age: 50, market_drop_30_days: 22, app_logins_7_days: 25, sip_stopped: true  }
            ];

            fetch(API_BASE + '/api/v1/batch_evaluate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scenarios: testScenarios })
            })
            .then(r => r.json())
            .then(data => {
                let html = '<div class="batch-results">';
                html += '<div class="batch-row batch-header"><div>Investor</div><div>Market Drop</div><div>Risk Tier</div><div>Probability</div></div>';
                data.results.forEach((r, i) => {
                    html += `
                        <div class="batch-row">
                            <div><strong>#${i + 1}</strong></div>
                            <div>${r.market_drop_30_days}%</div>
                            <div>${r.risk_tier}</div>
                            <div>${r.ai_panic_probability}%</div>
                        </div>
                    `;
                });
                html += '</div>';
                container.innerHTML = html;
            });
        }

        function resetForm() {
            document.getElementById('age').value = 35;
            document.getElementById('marketDrop').value = 10;
            document.getElementById('appLogins').value = 7;
            document.getElementById('sipStopped').checked = false;
            document.getElementById('resultSection').classList.remove('show');
            updateDisplay();
        }

        // Initialise on load
        window.onload = () => {
            updateDisplay();
            fetch(API_BASE + '/health')
                .then(r => r.json())
                .then(d => {
                    const m = (d.active_model || 'unknown').replace(/_/g, ' ');
                    document.getElementById('modelStatus').textContent =
                        m.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
                });
        };
    </script>
</body>
</html>"""


@app.get("/api/v1/model_info")
async def get_model_info():
    """Get information about the current model and available models."""
    return {
        "active_model": MODEL_STATE["model_type"],
        "available_models": MODEL_STATE["available_models"],
        "investors_loaded": len(MODEL_STATE["investors"]) if MODEL_STATE["investors"] else 0,
        "model_descriptions": {
            "random_forest": "Balanced performance with good interpretability. Recommended for production.",
            "gradient_boosting": "More aggressive predictions. Better for detecting early panic signals.",
            "logistic_regression": "Simple and highly explainable. Best for regulatory compliance."
        }
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("""
    -------------------------------------------------------------------------------
        AI + Behavioral Hybrid Advisory Engine (ABHA Model) - MVP v2.0
        Fintech Behavioral Finance Platform
    |                     WITH ADMIN DASHBOARD & MODEL SWITCHING                    |
    -------------------------------------------------------------------------------
    
    Starting FastAPI application...
    
    ADMIN DASHBOARD:
    - http://127.0.0.1:8000/admin/dashboard
       -> Test market scenarios with dashboard UI
       -> Switch between ML models in real-time
       -> Batch test multiple scenarios
    
    API ENDPOINTS:
    
    MOCK DATA:
    - GET  /api/v1/mock_investors         -> Get all 50 generated mock investors
    
    RISK EVALUATION:
    - POST /api/v1/evaluate_risk          -> Single investor risk assessment
    - POST /api/v1/batch_evaluate         -> Batch evaluate multiple scenarios
    
    MODEL MANAGEMENT:
    - POST /api/v1/switch_model/{model}   -> Switch to different ML model
       - random_forest (default)
       - gradient_boosting
       - logistic_regression
    - GET  /api/v1/model_info             -> Get current model info
    
    SYSTEM:
    - GET  /health                        -> Health check with model status
    - GET  /docs                          -> Swagger UI (full API docs)
    - GET  /redoc                         -> ReDoc documentation
    
    -------------------------------------------------------------------------------
    """)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
