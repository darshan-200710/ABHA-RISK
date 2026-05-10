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
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ABHA Market Risk Dashboard</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            body { 
                font-family: 'Arial', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .wrapper {
                max-width: 1400px;
                margin: 0 auto;
            }
            
            header {
                background: white;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            
            h1 {
                font-size: 2.8em;
                color: #667eea;
                margin-bottom: 10px;
            }
            
            .subtitle {
                color: #666;
                font-size: 1.1em;
                margin-bottom: 20px;
            }
            
            .status-bar {
                display: flex;
                justify-content: center;
                gap: 30px;
                flex-wrap: wrap;
            }
            
            .status-item {
                background: #f8f9ff;
                padding: 15px 25px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }
            
            .status-label {
                font-size: 0.9em;
                color: #999;
            }
            
            .status-value {
                font-size: 1.4em;
                font-weight: bold;
                color: #667eea;
            }
            
            .main-container {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 25px;
                margin-bottom: 30px;
            }
            
            .card {
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            
            .card h2 {
                color: #333;
                margin-bottom: 25px;
                font-size: 1.8em;
                border-bottom: 3px solid #667eea;
                padding-bottom: 15px;
            }
            
            /* Model Selection */
            .model-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 12px;
            }
            
            .model-button {
                padding: 20px;
                border: 3px solid #ddd;
                background: white;
                border-radius: 10px;
                cursor: pointer;
                font-size: 1.1em;
                font-weight: bold;
                transition: all 0.3s;
                text-align: left;
            }
            
            .model-button:hover {
                border-color: #667eea;
                background: #f8f9ff;
                transform: translateX(5px);
            }
            
            .model-button.active {
                background: #667eea;
                color: white;
                border-color: #667eea;
            }
            
            .model-desc {
                font-size: 0.85em;
                opacity: 0.7;
                margin-top: 8px;
            }
            
            /* Scenario Presets */
            .scenario-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
            }
            
            .scenario-button {
                padding: 25px 15px;
                border: none;
                border-radius: 10px;
                font-size: 1em;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s;
                color: white;
                box-shadow: 0 5px 15px rgba(0,0,0,0.15);
            }
            
            .scenario-mild {
                background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
            }
            
            .scenario-moderate {
                background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            }
            
            .scenario-severe {
                background: linear-gradient(135deg, #ff9a56 0%, #ff6a88 100%);
            }
            
            .scenario-critical {
                background: linear-gradient(135deg, #ff6b6b 0%, #cc5555 100%);
            }
            
            .scenario-button:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 20px rgba(0,0,0,0.25);
            }
            
            .scenario-desc {
                font-size: 0.8em;
                opacity: 0.9;
                margin-top: 5px;
            }
            
            /* Manual Input Section */
            .input-section {
                background: #f8f9ff;
                padding: 25px;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            
            .input-group {
                margin-bottom: 25px;
            }
            
            .input-label {
                font-weight: bold;
                color: #333;
                margin-bottom: 8px;
                display: flex;
                justify-content: space-between;
            }
            
            .input-value {
                color: #667eea;
                font-size: 1.2em;
            }
            
            input[type="range"] {
                width: 100%;
                height: 8px;
                border-radius: 5px;
                background: #ddd;
                outline: none;
                -webkit-appearance: none;
                cursor: pointer;
            }
            
            input[type="range"]::-webkit-slider-thumb {
                -webkit-appearance: none;
                appearance: none;
                width: 25px;
                height: 25px;
                border-radius: 50%;
                background: #667eea;
                cursor: pointer;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }
            
            input[type="range"]::-moz-range-thumb {
                width: 25px;
                height: 25px;
                border-radius: 50%;
                background: #667eea;
                cursor: pointer;
                border: none;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }
            
            .checkbox-group {
                display: flex;
                align-items: center;
                gap: 12px;
                font-size: 1.1em;
                cursor: pointer;
            }
            
            .checkbox-group input[type="checkbox"] {
                width: 24px;
                height: 24px;
                cursor: pointer;
            }
            
            .button-group {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
            }
            
            .btn-primary {
                padding: 18px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 1.1em;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s;
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
            }
            
            .btn-primary:hover {
                background: #764ba2;
                transform: translateY(-2px);
                box-shadow: 0 7px 20px rgba(102, 126, 234, 0.5);
            }
            
            .btn-secondary {
                padding: 18px;
                background: #f0f0f0;
                color: #333;
                border: 2px solid #ddd;
                border-radius: 10px;
                font-size: 1.1em;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s;
            }
            
            .btn-secondary:hover {
                background: #e8e8e8;
                border-color: #667eea;
            }
            
            /* Results Display */
            .result-section {
                background: #f8f9ff;
                padding: 25px;
                border-radius: 10px;
                margin-top: 20px;
                display: none;
            }
            
            .result-section.show {
                display: block;
                animation: slideIn 0.3s ease-out;
            }
            
            @keyframes slideIn {
                from { opacity: 0; transform: translateY(-10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .result-header {
                font-size: 1.4em;
                font-weight: bold;
                margin-bottom: 15px;
                color: #333;
            }
            
            .risk-display {
                display: flex;
                justify-content: space-around;
                gap: 15px;
                flex-wrap: wrap;
                margin-bottom: 20px;
            }
            
            .risk-box {
                background: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                flex: 1;
                min-width: 150px;
            }
            
            .risk-label {
                color: #999;
                font-size: 0.9em;
                margin-bottom: 8px;
            }
            
            .risk-value {
                font-size: 2.2em;
                font-weight: bold;
                color: #667eea;
            }
            
            .risk-tier-badge {
                display: inline-block;
                padding: 15px 30px;
                border-radius: 50px;
                font-size: 1.3em;
                font-weight: bold;
                color: white;
                margin-bottom: 20px;
            }
            
            .tier-normal { background: #00cc00; }
            .tier-amber { background: #ffaa00; }
            .tier-red { background: #ff8800; }
            .tier-critical { background: #ff4444; }
            
            .action-box {
                background: white;
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #667eea;
                margin-top: 15px;
            }
            
            .action-title {
                font-weight: bold;
                color: #333;
                margin-bottom: 8px;
            }
            
            .action-text {
                color: #666;
                line-height: 1.6;
            }
            
            .batch-results {
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin-top: 15px;
                max-height: 400px;
                overflow-y: auto;
            }
            
            .batch-row {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr 1fr;
                gap: 10px;
                padding: 12px;
                border-bottom: 1px solid #eee;
                font-size: 0.9em;
            }
            
            .batch-row:hover {
                background: #f8f9ff;
            }
            
            .batch-header {
                font-weight: bold;
                background: #667eea;
                color: white;
                padding: 12px;
                border-radius: 5px;
            }
            
            .loading {
                text-align: center;
                color: #667eea;
                font-size: 1.1em;
                padding: 20px;
            }
            
            .loading::after {
                content: '';
                animation: dots 1.5s steps(4, end) infinite;
            }
            
            @keyframes dots {
                0%, 20% { content: ''; }
                40% { content: '.'; }
                60% { content: '..'; }
                80%, 100% { content: '...'; }
            }
            
            @media (max-width: 768px) {
                .main-container {
                    grid-template-columns: 1fr;
                }
                
                .scenario-grid {
                    grid-template-columns: 1fr;
                }
                
                h1 {
                    font-size: 1.8em;
                }
            }
        </style>
    </head>
    <body>
        <div class="wrapper">
            
            <!-- Header -->
            <header>
                <h1>Investor Risk Dashboard</h1>
                <p class="subtitle">Simple Market Scenario Testing & Risk Analysis</p>
                <div class="status-bar">
                    <div class="status-item">
                        <div class="status-label">System Status</div>
                        <div class="status-value">Ready</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Current Model</div>
                        <div class="status-value" id="modelStatus">Loading...</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Investors Loaded</div>
                        <div class="status-value" id="investorCount">50</div>
                    </div>
                </div>
            </header>
            
            <!-- Main Content -->
            <div class="main-container">
                
                <!-- Left Panel: Model Selection -->
                <div class="card">
                    <h2>Select AI Model</h2>
                    <p style="margin-bottom: 20px; color: #666;">Choose which AI model to use for analysis</p>
                    
                    <div class="model-grid">
                        <button class="model-button active" onclick="switchModel('random_forest')">
                            Random Forest
                            <div class="model-desc">Recommended for most cases</div>
                        </button>
                        <button class="model-button" onclick="switchModel('gradient_boosting')">
                            Gradient Boosting
                            <div class="model-desc">More sensitive to early warning signs</div>
                        </button>
                        <button class="model-button" onclick="switchModel('logistic_regression')">
                            Logistic Regression
                            <div class="model-desc">Simple and easy to explain</div>
                        </button>
                    </div>
                </div>
                
                <!-- Right Panel: Quick Scenarios -->
                <div class="card">
                    <h2>Quick Market Scenarios</h2>
                    <p style="margin-bottom: 20px; color: #666;">Test how investors react to different market drops</p>
                    
                    <div class="scenario-grid">
                        <button class="scenario-button scenario-mild" onclick="runScenario('mild')">
                            Mild Drop
                            <div class="scenario-desc">10% Market Fall</div>
                        </button>
                        <button class="scenario-button scenario-moderate" onclick="runScenario('moderate')">
                            Moderate Drop
                            <div class="scenario-desc">15% Market Fall</div>
                        </button>
                        <button class="scenario-button scenario-severe" onclick="runScenario('severe')">
                            Severe Drop
                            <div class="scenario-desc">18% Market Fall</div>
                        </button>
                        <button class="scenario-button scenario-critical" onclick="runScenario('critical')">
                            Critical Drop
                            <div class="scenario-desc">20%+ Market Fall</div>
                        </button>
                    </div>
                </div>
                
            </div>
            
            <!-- Manual Testing -->
            <div class="card">
                <h2>Custom Investor Analysis</h2>
                <p style="margin-bottom: 20px; color: #666;">Adjust the sliders to create a custom investor profile</p>
                
                <div class="input-section">
                    <div class="input-group">
                        <div class="input-label">
                            <span>Age: <span class="input-value" id="ageDisplay">35</span> years</span>
                        </div>
                        <input type="range" id="age" min="25" max="65" value="35" oninput="updateDisplay()">
                        <small style="color: #999;">Investor's age (typically 25-65)</small>
                    </div>
                    
                    <div class="input-group">
                        <div class="input-label">
                            <span>Market Drop: <span class="input-value" id="dropDisplay">10.0</span>%</span>
                        </div>
                        <input type="range" id="marketDrop" min="0" max="40" step="0.5" value="10" oninput="updateDisplay()">
                        <small style="color: #999;">How much the market has fallen</small>
                    </div>
                    
                    <div class="input-group">
                        <div class="input-label">
                            <span>App Logins: <span class="input-value" id="loginsDisplay">7</span> times</span>
                        </div>
                        <input type="range" id="appLogins" min="1" max="25" value="7" oninput="updateDisplay()">
                        <small style="color: #999;">How many times they checked their portfolio in 7 days (Higher = More Anxious)</small>
                    </div>
                    
                    <div class="input-group">
                        <div class="checkbox-group">
                            <input type="checkbox" id="sipStopped" onchange="updateDisplay()">
                            <label for="sipStopped">SIP Has Been Stopped</label>
                        </div>
                        <small style="color: #999; display: block; margin-top: 8px;">Did they cancel their automatic investments?</small>
                    </div>
                    
                    <div class="button-group">
                        <button class="btn-primary" onclick="evaluateRisk()">
                            Analyze This Investor
                        </button>
                        <button class="btn-secondary" onclick="resetForm()">
                            Reset
                        </button>
                    </div>
                </div>
                
                <!-- Results -->
                <div class="result-section" id="resultSection">
                    <div class="result-header">Analysis Complete</div>
                    
                    <div style="text-align: center;">
                        <div class="risk-tier-badge" id="riskTier"></div>
                    </div>
                    
                    <div class="risk-display">
                        <div class="risk-box">
                            <div class="risk-label">Risk Probability</div>
                            <div class="risk-value" id="probability">--</div>
                        </div>
                        <div class="risk-box">
                            <div class="risk-label">Recommended Action</div>
                            <div class="risk-value" id="actionBadge" style="font-size: 1.4em;">--</div>
                        </div>
                    </div>
                    
                    <div class="action-box">
                        <div class="action-title">What Advisor Should Do:</div>
                        <div class="action-text" id="actionText"></div>
                    </div>
                </div>
                
            </div>
            
            <!-- Batch Testing -->
            <div class="card">
                <h2>Test Multiple Scenarios (Batch)</h2>
                <p style="margin-bottom: 20px; color: #666;">Analyze 5 different investor types at once</p>
                
                <button class="btn-primary" onclick="runBatchTest()" style="width: 100%; padding: 20px; font-size: 1.2em;">
                    Run Batch Test
                </button>
                
                <div class="result-section" id="batchResultSection">
                    <div class="batch-results">
                        <div class="batch-row batch-header">
                            <div>Scenario</div>
                            <div>Market Drop</div>
                            <div>Risk</div>
                            <div>Probability</div>
                        </div>
                        <div id="batchResults"></div>
                    </div>
                </div>
                
            </div>
            
        </div>

        <script>
            const API_BASE = window.location.origin;
            
            const scenarios = {
                mild: { age: 35, drop: 10, logins: 7, sip: false },
                moderate: { age: 40, drop: 15, logins: 15, sip: false },
                severe: { age: 38, drop: 18, logins: 18, sip: true },
                critical: { age: 45, drop: 20, logins: 22, sip: true }
            };
            
            function updateDisplay() {
                document.getElementById('ageDisplay').textContent = document.getElementById('age').value;
                document.getElementById('dropDisplay').textContent = document.getElementById('marketDrop').value;
                document.getElementById('loginsDisplay').textContent = document.getElementById('appLogins').value;
            }
            
            function getTierClass(tier) {
                switch(tier.toUpperCase()) {
                    case 'CRITICAL': return 'tier-critical';
                    case 'RED': return 'tier-red';
                    case 'AMBER': return 'tier-amber';
                    default: return 'tier-normal';
                }
            }
            
            function getTierEmoji(tier) {
                switch(tier.toUpperCase()) {
                    case 'CRITICAL': return 'CRITICAL';
                    case 'RED': return 'RED';
                    case 'AMBER': return 'AMBER';
                    default: return 'NORMAL';
                }
            }
            
            function switchModel(modelName) {
                fetch(API_BASE + '/api/v1/switch_model/' + modelName, {method: 'POST'})
                    .then(r => r.json())
                    .then(data => {
                        document.querySelectorAll('.model-button').forEach(b => b.classList.remove('active'));
                        event.target.closest('.model-button').classList.add('active');
                        document.getElementById('modelStatus').textContent = modelName.replace('_', ' ').toUpperCase();
                    });
            }
            
            function runScenario(name) {
                const s = scenarios[name];
                document.getElementById('age').value = s.age;
                document.getElementById('marketDrop').value = s.drop;
                document.getElementById('appLogins').value = s.logins;
                document.getElementById('sipStopped').checked = s.sip;
                updateDisplay();
                setTimeout(evaluateRisk, 100);
            }
            
            function evaluateRisk() {
                const payload = {
                    age: parseInt(document.getElementById('age').value),
                    market_drop_30_days: parseFloat(document.getElementById('marketDrop').value),
                    app_logins_7_days: parseInt(document.getElementById('appLogins').value),
                    sip_stopped: document.getElementById('sipStopped').checked
                };
                
                const resultSection = document.getElementById('resultSection');
                resultSection.innerHTML = '<div class="loading">Analyzing Investor</div>';
                resultSection.classList.add('show');
                
                fetch(API_BASE + '/api/v1/evaluate_risk', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                })
                .then(r => r.json())
                .then(data => {
                    const tier = data.risk_tier;
                    const prob = data.ai_panic_probability;
                    
                    resultSection.innerHTML = `
                        <div class="result-header">Analysis Complete</div>
                        <div style="text-align: center;">
                            <div class="risk-tier-badge ${getTierClass(tier)}">
                                ${tier} RISK
                            </div>
                        </div>
                        <div class="risk-display">
                            <div class="risk-box">
                                <div class="risk-label">Panic Probability</div>
                                <div class="risk-value">${prob}%</div>
                            </div>
                        </div>
                        <div class="action-box">
                            <div class="action-title">What You Should Do:</div>
                            <div class="action-text">${data.recommended_action}</div>
                        </div>
                    `;
                });
            }
            
            function runBatchTest() {
                const batchSection = document.getElementById('batchResultSection');
                batchSection.innerHTML = '<div class="loading">Running Batch Test</div>';
                batchSection.classList.add('show');
                
                const testScenarios = [
                    {age: 30, market_drop_30_days: 5, app_logins_7_days: 3, sip_stopped: false},
                    {age: 35, market_drop_30_days: 10, app_logins_7_days: 7, sip_stopped: false},
                    {age: 40, market_drop_30_days: 15, app_logins_7_days: 15, sip_stopped: false},
                    {age: 45, market_drop_30_days: 18, app_logins_7_days: 20, sip_stopped: true},
                    {age: 50, market_drop_30_days: 22, app_logins_7_days: 25, sip_stopped: true}
                ];
                
                fetch(API_BASE + '/api/v1/batch_evaluate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({scenarios: testScenarios})
                })
                .then(r => r.json())
                .then(data => {
                    let html = '<div class="batch-row batch-header"><div>Scenario</div><div>Market Drop</div><div>Risk</div><div>Probability</div></div>';
                    data.results.forEach((r, i) => {
                        html += `
                            <div class="batch-row">
                                <div>Investor ${i+1}</div>
                                <div>${r.market_drop_30_days}%</div>
                                <div>${r.risk_tier}</div>
                                <div>${r.ai_panic_probability}%</div>
                            </div>
                        `;
                    });
                    batchSection.innerHTML = `<div class="batch-results">${html}</div>`;
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
            
            // Initialize on page load
            window.onload = () => {
                updateDisplay();
                fetch(API_BASE + '/health')
                    .then(r => r.json())
                    .then(d => {
                        document.getElementById('modelStatus').textContent = (d.active_model || 'unknown').replace('_', ' ').toUpperCase();
                    });
            };
        </script>
    </body>
    </html>
    """


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
