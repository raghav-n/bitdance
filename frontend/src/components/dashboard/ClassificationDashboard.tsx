import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
} from "recharts";
import { ChevronDown } from "lucide-react";
import { useState, useEffect } from "react";
import { 
  apiService, 
  type ClassificationPerformance, 
  type ConfidenceData, 
  type ModelInsights,
  type ModelSummary 
} from "@/services/api";

const ClassificationDashboard = () => {
  const [isRecommendationsOpen, setIsRecommendationsOpen] = useState(false);
  const [performanceData, setPerformanceData] = useState<ClassificationPerformance[]>([]);
  const [confidenceData, setConfidenceData] = useState<ConfidenceData[]>([]);
  const [modelInsights, setModelInsights] = useState<ModelInsights[]>([]);
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [selectedConfidenceModel, setSelectedConfidenceModel] = useState<string>('');
  const [selectedCategoryModel, setSelectedCategoryModel] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch real classification data from API
  useEffect(() => {
    const fetchClassificationData = async () => {
      try {
        setLoading(true);
        const [performance, confidence, insights, modelList] = await Promise.all([
          apiService.getClassificationPerformance(),
          apiService.getConfidenceData(),
          apiService.getModelInsights(),
          apiService.getModelList()
        ]);
        
        setPerformanceData(performance);
        setConfidenceData(confidence);
        setModelInsights(insights);
        setModels(modelList);
        
        // Set the best model as default selection
        if (modelList.length > 0) {
          setSelectedModel(modelList[0].id);
          setSelectedConfidenceModel(modelList[0].id);
          setSelectedCategoryModel(modelList[0].id);
        }
        
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch classification data');
        console.error('Error fetching classification data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchClassificationData();
  }, []);

  // Process confidence data for histogram (filtered by selected model)
  const selectedModelName = selectedConfidenceModel ? selectedConfidenceModel.split('/')[1] : '';
  const filteredConfidenceData = confidenceData.filter(item => 
    selectedModelName ? item.category === selectedModelName : true
  );
  
  const confidenceBins = filteredConfidenceData.reduce((acc, item) => {
    const bin = Math.floor(item.confidence * 10) / 10;
    const key = `${bin.toFixed(1)}`;
    if (!acc[key]) {
      acc[key] = { bin: bin.toFixed(1), count: 0 };
    }
    acc[key].count += item.count;
    return acc;
  }, {});

  const histogramData = Object.values(confidenceBins);

  // Convert performance data to match UI format
  const performanceMetrics = performanceData.map(p => ({
    category: p.model,
    precision: p.precision / 100, // Convert percentage to decimal
    recall: p.recall / 100,
    f1Score: p.f1_score / 100,
    support: 211 // All models tested on same dataset
  }));

  // Generate category stats from confidence data for Category Analysis (filtered by selected category model)
  const selectedCategoryModelName = selectedCategoryModel ? selectedCategoryModel.split('/')[1] : '';
  const filteredCategoryData = confidenceData.filter(item => 
    selectedCategoryModelName ? item.category === selectedCategoryModelName : true
  );
  
  const categoryStats = filteredCategoryData.reduce((acc, item) => {
    const label = item.label || 'Unknown';
    if (!acc[label]) {
      acc[label] = { category: label, confidences: [], sampleCount: 0 };
    }
    acc[label].confidences.push(item.confidence);
    acc[label].sampleCount += item.count;
    return acc;
  }, {} as Record<string, { category: string; confidences: number[]; sampleCount: number }>);

  const categoryStatsArray = Object.values(categoryStats).map(stat => ({
    category: stat.category,
    meanConfidence: stat.confidences.length > 0 ? stat.confidences.reduce((sum, c) => sum + c, 0) / stat.confidences.length : 0,
    stdConfidence: stat.confidences.length > 0 ? Math.sqrt(stat.confidences.reduce((sum, c) => sum + Math.pow(c - (stat.confidences.reduce((s, x) => s + x, 0) / stat.confidences.length), 2), 0) / stat.confidences.length) : 0,
    sampleCount: stat.sampleCount
  }));

  const chartColors = ['#B3D9FF', '#80C1FF', '#4DA6FF', '#99E6FF'];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900 mx-auto mb-2"></div>
          <p>Loading classification data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <p className="text-red-600 mb-2">Error: {error}</p>
          <button 
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold">Classification Analysis</h1>
        <p className="text-gray-600 mt-2">Real performance metrics from your 5-model analysis</p>
      </div>

      {/* Model Comparison - All 5 Models */}
      <Card>
        <CardHeader>
          <CardTitle>Model Comparison</CardTitle>
          <CardDescription>
            Performance across all {models.length} models: {models.filter(m => m.family === 'encoder').length} BERT variants + {models.filter(m => m.family === 'sft').length} SFT models
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            {models.map((model) => {
              const isBest = model.f1_score === Math.max(...models.map(m => m.f1_score));
              return (
                <Card key={model.id} className={`${isBest ? 'ring-2 ring-blue-500 bg-blue-50' : 'bg-white'} transition-all`}>
                  <CardContent className="p-4 text-center">
                    <div className="space-y-2">
                      <div className={`text-lg font-bold ${isBest ? 'text-blue-700' : 'text-gray-900'}`}>
                        {(model.f1_score * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-gray-600">F1 Score</div>
                      <div className="text-xs font-medium text-gray-800">{model.name}</div>
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        model.family === 'encoder' ? 'bg-blue-100 text-blue-800' : 'bg-green-100 text-green-800'
                      }`}>
                        {model.family === 'encoder' ? 'BERT' : 'SFT'}
                      </span>
                      {isBest && (
                        <div className="text-xs text-blue-600 font-semibold">🏆 Best</div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Performance Metrics Table */}
      <Card>
        <CardHeader>
          <CardTitle>Performance Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Category</TableHead>
                <TableHead>Precision</TableHead>
                <TableHead>Recall</TableHead>
                <TableHead>F1-Score</TableHead>
                <TableHead>Support</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {performanceMetrics.map((metric) => (
                <TableRow key={metric.category}>
                  <TableCell className="font-medium">{metric.category}</TableCell>
                  <TableCell>{metric.precision.toFixed(3)}</TableCell>
                  <TableCell>{metric.recall.toFixed(3)}</TableCell>
                  <TableCell>{metric.f1Score.toFixed(3)}</TableCell>
                  <TableCell>{metric.support.toLocaleString()}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Confidence Distribution */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Confidence Distribution</CardTitle>
              <CardDescription>
                Distribution of prediction confidence scores for selected model
              </CardDescription>
            </div>
            <div className="w-64">
              <Select value={selectedConfidenceModel} onValueChange={setSelectedConfidenceModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a model..." />
                </SelectTrigger>
                <SelectContent>
                  {models.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      <div className="flex items-center justify-between w-full">
                        <span className="font-medium">{model.name}</span>
                        <div className="flex items-center space-x-2 ml-4">
                          <span className={`px-2 py-1 text-xs rounded-full ${
                            model.family === 'encoder' ? 'bg-blue-100 text-blue-800' : 'bg-green-100 text-green-800'
                          }`}>
                            {model.family === 'encoder' ? 'BERT' : 'SFT'}
                          </span>
                          <span className="text-sm font-mono">F1: {(model.f1_score * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={histogramData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis 
                  dataKey="bin" 
                  label={{ value: 'Confidence Score', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  label={{ value: 'Count', angle: -90, position: 'insideLeft' }}
                />
                <RechartsTooltip />
                <Bar dataKey="count" fill="#B3D9FF" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          {selectedConfidenceModel && (
            <div className="mt-4 text-sm text-gray-600 text-center">
              Showing confidence distribution for: <span className="font-semibold">
                {models.find(m => m.id === selectedConfidenceModel)?.name}
              </span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Model Insights */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <CardContent className="p-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">Relevant</div>
              <div className="text-sm text-gray-600 mt-1">Best Performing Category</div>
              <div className="text-xs text-green-600 mt-1">96% Precision</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">Irrelevant</div>
              <div className="text-sm text-gray-600 mt-1">Most Challenging Category</div>
              <div className="text-xs text-orange-600 mt-1">89% Precision</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">0.91</div>
              <div className="text-sm text-gray-600 mt-1">Overall F1-Score</div>
              <div className="text-xs text-green-600 mt-1">+0.03</div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Model Recommendations */}
      <Collapsible open={isRecommendationsOpen} onOpenChange={setIsRecommendationsOpen}>
        <Card>
          <CollapsibleTrigger asChild>
            <CardHeader className="cursor-pointer hover:bg-gray-50">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  📋 Model Recommendations
                </CardTitle>
                <ChevronDown className={`h-4 w-4 transition-transform ${isRecommendationsOpen ? 'rotate-180' : ''}`} />
              </div>
            </CardHeader>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold mb-2">Current Model Performance:</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
                    <li>The model shows excellent performance on Relevant reviews (96% precision)</li>
                    <li>Advertisement detection is highly accurate (93% precision, 88% recall)</li>
                    <li>Irrelevant category needs improvement (89% precision, 92% recall)</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Recommendations:</h4>
                  <ol className="list-decimal list-inside space-y-1 text-sm text-gray-600">
                    <li><strong>Collect more training data</strong> for the Irrelevant category</li>
                    <li><strong>Feature engineering</strong> to better distinguish irrelevant content</li>
                    <li><strong>Active learning</strong> to focus on challenging cases</li>
                    <li><strong>Ensemble methods</strong> to improve overall robustness</li>
                  </ol>
                </div>
              </div>
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>

      {/* Category Analysis */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Category Analysis</CardTitle>
              <CardDescription>
                Detailed confidence statistics per category for selected model
              </CardDescription>
            </div>
            <div className="w-64">
              <Select value={selectedCategoryModel} onValueChange={setSelectedCategoryModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a model..." />
                </SelectTrigger>
                <SelectContent>
                  {models.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      <div className="flex items-center justify-between w-full">
                        <span className="font-medium">{model.name}</span>
                        <div className="flex items-center space-x-2 ml-4">
                          <span className={`px-2 py-1 text-xs rounded-full ${
                            model.family === 'encoder' ? 'bg-blue-100 text-blue-800' : 'bg-green-100 text-green-800'
                          }`}>
                            {model.family === 'encoder' ? 'BERT' : 'SFT'}
                          </span>
                          <span className="text-sm font-mono">F1: {(model.f1_score * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Category</TableHead>
                <TableHead>Mean Confidence</TableHead>
                <TableHead>Std Confidence</TableHead>
                <TableHead>Sample Count</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {categoryStatsArray.map((stat) => (
                <TableRow key={stat.category}>
                  <TableCell className="font-medium">{stat.category}</TableCell>
                  <TableCell>{stat.meanConfidence.toFixed(3)}</TableCell>
                  <TableCell>{stat.stdConfidence.toFixed(3)}</TableCell>
                  <TableCell>{stat.sampleCount.toLocaleString()}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          {selectedCategoryModel && (
            <div className="mt-4 text-sm text-gray-600 text-center">
              Showing category analysis for: <span className="font-semibold">
                {models.find(m => m.id === selectedCategoryModel)?.name}
              </span>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default ClassificationDashboard;
