from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import random

app = Flask(__name__)

# Configure CORS properly
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:5173"],
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Mock data generators (same logic as your React components)
def generate_mock_classification_data():
    return [
        {'category': 'Relevant', 'count': 11543},
        {'category': 'Irrelevant', 'count': 1304},
        {'category': 'Advertisement', 'count': 304},
        {'category': 'Rant without Visit', 'count': 696}
    ]

def generate_time_series_data(metric_type="Review Volume", time_range="Last 6 Months"):
    if time_range == "Last 30 Days":
        periods = 30
    elif time_range == "Last 3 Months":
        periods = 90
    elif time_range == "Last 6 Months":
        periods = 180
    else:
        periods = 365
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq='D')
    
    if metric_type == "Review Volume":
        values = np.random.normal(1200, 200, periods)
    elif metric_type == "Violations":
        values = np.random.normal(300, 50, periods)
    elif metric_type == "Accuracy":
        values = np.random.normal(0.94, 0.02, periods)
    else:  # Processing Time
        values = np.random.normal(2.5, 0.5, periods)
    
    return [{'date': date.strftime('%Y-%m-%d'), 'value': float(value)} 
            for date, value in zip(dates, values)]

def generate_performance_metrics():
    return [
        {'category': 'Relevant', 'precision': 0.96, 'recall': 0.94, 'f1Score': 0.95, 'support': 8543},
        {'category': 'Advertisement', 'precision': 0.93, 'recall': 0.88, 'f1Score': 0.90, 'support': 1876},
        {'category': 'Irrelevant', 'precision': 0.89, 'recall': 0.92, 'f1Score': 0.90, 'support': 2104},
        {'category': 'Rant without Visit', 'precision': 0.87, 'recall': 0.85, 'f1Score': 0.86, 'support': 324}
    ]

def predict_review(text, language="English"):
    """Mock prediction function"""
    categories = ['Relevant', 'Advertisement', 'Irrelevant', 'Rant without Visit']
    random_category = random.choice(categories)
    confidence = 0.7 + random.random() * 0.3
    
    policies = {
        'advertisement': random.random(),
        'inappropriate_content': random.random(),
        'spam': random.random(),
        'rant_without_visit': random.random()
    }
    
    violations = sum(1 for score in policies.values() if score > 0.7)
    risk_level = 'High' if violations > 2 else 'Medium' if violations > 0 else 'Low'
    
    return {
        'category': random_category,
        'confidence': confidence,
        'policies': policies,
        'violations': violations,
        'riskLevel': risk_level,
        'explanation': f"This review appears to be {random_category.lower()} based on language patterns and content analysis. The model detected {violations} potential policy violations."
    }

# API Routes
@app.route('/api/overview/metrics')
def get_overview_metrics():
    return jsonify({
        'totalReviews': "12,847",
        'relevantReviews': "11,543",
        'violations': "304",
        'modelAccuracy': "94.2%"
    })

@app.route('/api/overview/classification-data')
def get_classification_data():
    return jsonify(generate_mock_classification_data())

@app.route('/api/overview/time-series')
def get_time_series():
    return jsonify(generate_time_series_data())

@app.route('/api/overview/recent-activity')
def get_recent_activity():
    return jsonify([
        {'time': '2 minutes ago', 'action': 'Review classified', 'details': 'Classified as Relevant'},
        {'time': '5 minutes ago', 'action': 'Violation detected', 'details': 'Advertisement detected'},
        {'time': '12 minutes ago', 'action': 'Model retrained', 'details': 'Accuracy improved to 94.2%'},
        {'time': '18 minutes ago', 'action': 'New data ingested', 'details': '1,234 new reviews'},
        {'time': '25 minutes ago', 'action': 'Review classified', 'details': 'Classified as Irrelevant'}
    ])

@app.route('/api/analytics/time-series')
def get_analytics_time_series():
    metric_type = request.args.get('metric', 'Review Volume')
    time_range = request.args.get('range', 'Last 6 Months')
    return jsonify(generate_time_series_data(metric_type, time_range))

@app.route('/api/classification/performance')
def get_performance_metrics():
    return jsonify(generate_performance_metrics())

@app.route('/api/classification/confidence-data')
def get_confidence_data():
    categories = ['Relevant', 'Advertisement', 'Irrelevant', 'Rant without Visit']
    data = []
    
    for category in categories:
        for _ in range(100):
            confidence = np.random.beta(8, 2)
            data.append({
                'category': category,
                'confidence': float(confidence),
                'bin': round(confidence, 1)
            })
    
    return jsonify(data)

@app.route('/api/prediction/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    language = data.get('language', 'English')
    
    if not text.strip():
        return jsonify({'error': 'Text is required'}), 400
    
    result = predict_review(text, language)
    return jsonify(result)

@app.route('/api/violations/analyze', methods=['POST'])
def analyze_violations():
    data = request.json
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({'error': 'Text is required'}), 400
    
    # Mock analysis
    categories = ['Relevant', 'Advertisement', 'Inappropriate Content', 'Spam', 'Rant without Visit']
    random_category = random.choice(categories)
    confidence = 0.7 + random.random() * 0.3
    
    policies = {
        'advertisement': random.random(),
        'inappropriate_content': random.random(),
        'spam': random.random(),
        'rant_without_visit': random.random()
    }
    
    high_risk_violations = [policy.replace('_', ' ').upper() for policy, score in policies.items() if score > 0.8]
    medium_risk_violations = [policy.replace('_', ' ').upper() for policy, score in policies.items() if 0.5 < score <= 0.8]
    
    total_violations = len(high_risk_violations) + len(medium_risk_violations)
    risk_level = 'High' if high_risk_violations else 'Medium' if medium_risk_violations else 'Low'
    
    recommendation = "No action required."
    if risk_level == 'High':
        recommendation = "Immediate review required. Consider removing or flagging this content."
    elif risk_level == 'Medium':
        recommendation = "Manual review recommended to verify policy compliance."
    
    return jsonify({
        'category': random_category,
        'confidence': confidence,
        'violations': total_violations,
        'riskLevel': risk_level,
        'highRiskViolations': high_risk_violations,
        'mediumRiskViolations': medium_risk_violations,
        'explanation': f"Analysis detected {total_violations} potential policy violations. The content appears to be {random_category.lower()} with {confidence*100:.1f}% confidence.",
        'recommendation': recommendation
    })

@app.route('/api/violations/statistics')
def get_violation_statistics():
    return jsonify({
        'advertisement': {'count': 156, 'severity': 'High'},
        'inappropriate_content': {'count': 23, 'severity': 'Medium'},
        'spam': {'count': 89, 'severity': 'Medium'},
        'rant_without_visit': {'count': 67, 'severity': 'Low'}
    })

# Health check
@app.route('/api/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Server will be available at: http://localhost:5001")
    print("API endpoints available at: http://localhost:5001/api/")
    app.run(debug=True, port=5001, host='127.0.0.1', threaded=True)
