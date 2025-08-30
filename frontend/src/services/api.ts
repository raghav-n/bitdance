const API_BASE_URL = 'http://localhost:5001/api';

// Type definitions
interface OverviewMetrics {
  totalReviews: string;
  relevantReviews: string;
  violations: string;
  bestModel: {
    name: string;
    family: string;
    f1_score: string;
  };
  modelComparison: {
    total_models: number;
    avg_f1: string;
    best_f1: string;
    worst_f1: string;
    encoder_count: number;
    sft_count: number;
  };
}

interface ModelSummary {
  id: string;
  name: string;
  family: string;
  f1_score: number;
  precision: number;
  recall: number;
}

interface ModelDetails {
  model: string;
  name: string;
  family: string;
  n_samples: number;
  threshold: number;
  overall_metrics: {
    precision: number;
    recall: number;
    f1_score: number;
    macro_precision: number;
    macro_recall: number;
    macro_f1: number;
  };
  per_category: Array<{
    category: string;
    precision: number;
    recall: number;
    f1_score: number;
    support: number;
    ap: number;
    confusion_matrix: {
      tn: number;
      fp: number;
      fn: number;
      tp: number;
    };
  }>;
}

interface ViolationStats {
  [key: string]: {
    count: number;
    severity: 'High' | 'Medium' | 'Low';
    examples?: string[];
  };
}

interface ClassificationPerformance {
  model: string;
  family: string;
  accuracy: number;
  f1_score: number;
  precision: number;
  recall: number;
}

interface ConfidenceData {
  category: string;
  confidence: number;
  count: number;
}

interface ModelInsights {
  model: string;
  family: string;
  insights: {
    strengths: string[];
    weaknesses: string[];
    recommendations: string[];
  };
}

interface ClassificationData {
  category: string;
  count: number;
}

interface TimeSeriesData {
  date: string;
  value: number;
}

interface RecentActivity {
  time: string;
  action: string;
  details: string;
}

class ApiService {
  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }

    return response.json();
  }

  // Overview API
  async getOverviewMetrics(): Promise<OverviewMetrics> {
    return this.request<OverviewMetrics>('/overview/metrics');
  }

  async getClassificationData(): Promise<ClassificationData[]> {
    return this.request<ClassificationData[]>('/overview/classification-data');
  }

  async getTimeSeriesData(): Promise<TimeSeriesData[]> {
    return this.request<TimeSeriesData[]>('/overview/time-series');
  }

  async getRecentActivity(): Promise<RecentActivity[]> {
    return this.request<RecentActivity[]>('/overview/recent-activity');
  }

  // Analytics API
  async getAnalyticsTimeSeries(metric: string, range: string): Promise<TimeSeriesData[]> {
    return this.request<TimeSeriesData[]>(`/analytics/time-series?metric=${encodeURIComponent(metric)}&range=${encodeURIComponent(range)}`);
  }

  // Classification API
  async getClassificationPerformance(): Promise<ClassificationPerformance[]> {
    return this.request<ClassificationPerformance[]>('/classification/performance');
  }

  async getConfidenceData(): Promise<ConfidenceData[]> {
    return this.request<ConfidenceData[]>('/classification/confidence-data');
  }

  async getModelInsights(): Promise<ModelInsights[]> {
    return this.request<ModelInsights[]>('/classification/model-insights');
  }

  // Prediction API
  async predictText(text: string, language: string): Promise<any> {
    return this.request('/prediction/predict', {
      method: 'POST',
      body: JSON.stringify({ text, language }),
    });
  }

  async getBatchPredictions(): Promise<any> {
    return this.request('/prediction/batch');
  }

  // Violations API
  async getViolationStats(): Promise<ViolationStats> {
    return this.request<ViolationStats>('/violations/statistics');
  }

  async getViolationExamples(category: string): Promise<any> {
    return this.request(`/violations/examples?category=${encodeURIComponent(category)}`);
  }

  async analyzeViolations(text: string, modelPath: string = "encoder/enc-distilbert-base", threshold: number = 0.5): Promise<any> {
    try {
      return await this.request('/inference/predict', {
        method: 'POST',
        body: JSON.stringify({
          model_path: modelPath,
          family: "encoder", // Default to encoder family
          text: text,
          threshold: threshold,
          batch_size: 8
        }),
      });
    } catch (error) {
      // If inference fails, provide a meaningful mock response indicating models need training
      console.warn('Model inference failed, providing mock response:', error);
      
      // Generate mock response based on simple keyword analysis
      const lowerText = text.toLowerCase();
      const mockPredictions = {
        probs: [
          lowerText.includes('irrelevant') || lowerText.includes('useless') ? 0.8 : 0.2,
          lowerText.includes('call') || lowerText.includes('website') || lowerText.includes('promotion') ? 0.9 : 0.1,
          lowerText.includes('heard') || lowerText.includes('friend said') || lowerText.includes('someone told') ? 0.7 : 0.3
        ],
        pred: [
          lowerText.includes('irrelevant') || lowerText.includes('useless') ? 1 : 0,
          lowerText.includes('call') || lowerText.includes('website') || lowerText.includes('promotion') ? 1 : 0,
          lowerText.includes('heard') || lowerText.includes('friend said') || lowerText.includes('someone told') ? 1 : 0
        ]
      };
      
      return {
        model: "Mock Analysis (Models need training)",
        family: "mock",
        labels: ["irrelevant_content", "advertisement", "review_without_visit"],
        threshold: threshold,
        n: 1,
        predictions: [mockPredictions],
        _isMockResponse: true
      };
    }
  }

  // Models API
  async getModelList(): Promise<ModelSummary[]> {
    return this.request<ModelSummary[]>('/models/list');
  }

  async getModelDetails(modelId: string): Promise<ModelDetails> {
    return this.request<ModelDetails>(`/models/${encodeURIComponent(modelId)}/details`);
  }
}

export const apiService = new ApiService();
export type { 
  OverviewMetrics, 
  ClassificationData, 
  TimeSeriesData, 
  RecentActivity, 
  ModelSummary, 
  ModelDetails, 
  ViolationStats,
  ClassificationPerformance,
  ConfidenceData,
  ModelInsights
};


