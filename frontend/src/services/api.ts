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


