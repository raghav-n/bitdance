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
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line
} from "recharts";
import { useState, useEffect } from "react";
import { apiService, type OverviewMetrics, type ClassificationData, type TimeSeriesData, type RecentActivity, type ModelSummary, type ModelDetails } from "@/services/api";

const OverviewDashboard = () => {
  // State for data with proper types
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [keyMetrics, setKeyMetrics] = useState<OverviewMetrics | null>(null);
  const [classificationData, setClassificationData] = useState<ClassificationData[]>([]);
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesData[]>([]);
  const [recentActivity, setRecentActivity] = useState<RecentActivity[]>([]);
  
  // Model comparison state
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [modelDetails, setModelDetails] = useState<ModelDetails | null>(null);
  const [modelLoading, setModelLoading] = useState(false);

  // Chart colors matching your theme
  const chartColors = ['#B3D9FF', '#80C1FF', '#4DA6FF', '#99E6FF'];

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [metrics, classification, timeSeries, activity, modelList] = await Promise.all([
          apiService.getOverviewMetrics(),
          apiService.getClassificationData(),
          apiService.getTimeSeriesData(),
          apiService.getRecentActivity(),
          apiService.getModelList()
        ]);

        setKeyMetrics(metrics);
        setClassificationData(classification);
        setTimeSeriesData(timeSeries);
        setRecentActivity(activity);
        setModels(modelList);
        
        // Set the best model as default selection
        if (modelList.length > 0) {
          setSelectedModel(modelList[0].id);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Load model details when selection changes
  useEffect(() => {
    const fetchModelDetails = async () => {
      if (!selectedModel) return;
      
      try {
        setModelLoading(true);
        const details = await apiService.getModelDetails(selectedModel);
        setModelDetails(details);
      } catch (err) {
        console.error('Failed to fetch model details:', err);
      } finally {
        setModelLoading(false);
      }
    };

    fetchModelDetails();
  }, [selectedModel]);

  if (loading) return <div className="flex items-center justify-center h-64">Loading...</div>;
  if (error) return <div className="flex items-center justify-center h-64 text-red-600">Error: {error}</div>;

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold">Overview</h1>
        <p className="text-gray-600 mt-2">Review quality metrics and system performance</p>
      </div>

      {/* Key Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardContent className="p-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900">{keyMetrics?.totalReviews}</div>
              <div className="text-sm text-gray-600 mt-1">Total Reviews</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{keyMetrics?.relevantReviews}</div>
              <div className="text-sm text-gray-600 mt-1">Relevant Reviews</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">{keyMetrics?.violations}</div>
              <div className="text-sm text-gray-600 mt-1">Violations</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{keyMetrics?.bestModel.f1_score}</div>
              <div className="text-sm text-gray-600 mt-1">Best Model F1</div>
              <div className="text-xs text-gray-500 mt-1">{keyMetrics?.bestModel.name}</div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Model Comparison Section */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Model Performance Comparison</CardTitle>
              <CardDescription>
                Compare performance across all {keyMetrics?.modelComparison.total_models} models: {keyMetrics?.modelComparison.encoder_count} BERT variants + {keyMetrics?.modelComparison.sft_count} SFT models
              </CardDescription>
            </div>
            <div className="w-64">
              <Select value={selectedModel} onValueChange={setSelectedModel}>
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
          {modelLoading ? (
            <div className="flex items-center justify-center h-32">Loading model details...</div>
          ) : modelDetails ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {/* Overall Metrics */}
              <div className="space-y-4">
                <h4 className="font-semibold text-gray-900">Overall Performance</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">F1 Score:</span>
                    <span className="text-sm font-medium">{(modelDetails.overall_metrics.f1_score * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Precision:</span>
                    <span className="text-sm font-medium">{(modelDetails.overall_metrics.precision * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Recall:</span>
                    <span className="text-sm font-medium">{(modelDetails.overall_metrics.recall * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>

              {/* Per-Category Performance */}
              {modelDetails.per_category.map((category, index) => (
                <div key={category.category} className="space-y-4">
                  <h4 className="font-semibold text-gray-900">{category.category}</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">F1:</span>
                      <span className="text-sm font-medium">{(category.f1_score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Precision:</span>
                      <span className="text-sm font-medium">{(category.precision * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Recall:</span>
                      <span className="text-sm font-medium">{(category.recall * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Support:</span>
                      <span className="text-sm font-medium">{category.support}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="flex items-center justify-center h-32 text-gray-500">
              Select a model to view detailed performance metrics
            </div>
          )}
        </CardContent>
      </Card>

      {/* Model Comparison Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <CardContent className="p-6">
            <div className="text-center">
              <div className="text-xl font-bold text-green-600">{keyMetrics?.modelComparison.best_f1}</div>
              <div className="text-sm text-gray-600 mt-1">Best Performance</div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-6">
            <div className="text-center">
              <div className="text-xl font-bold text-blue-600">{keyMetrics?.modelComparison.avg_f1}</div>
              <div className="text-sm text-gray-600 mt-1">Average F1</div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-6">
            <div className="text-center">
              <div className="text-xl font-bold text-orange-600">{keyMetrics?.modelComparison.worst_f1}</div>
              <div className="text-sm text-gray-600 mt-1">Lowest Performance</div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Review Distribution Pie Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Review Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={classificationData}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    innerRadius={0}
                    dataKey="count"
                    nameKey="category"
                    label={({ cx, cy, midAngle, innerRadius, outerRadius, name, percent }) => {
                      const RADIAN = Math.PI / 180;
                      const radius = outerRadius + 30;
                      const x = cx + radius * Math.cos(-midAngle * RADIAN);
                      const y = cy + radius * Math.sin(-midAngle * RADIAN);

                      return (
                        <text 
                          x={x} 
                          y={y} 
                          fill="#374151" 
                          textAnchor={x > cx ? 'start' : 'end'} 
                          dominantBaseline="central"
                          fontSize="12"
                          fontWeight="500"
                        >
                          <tspan x={x} dy="-0.5em" fontSize="13" fontWeight="600">{name}</tspan>
                          <tspan x={x} dy="1.2em" fontSize="12" fill="#6B7280">{`${(percent * 100).toFixed(1)}%`}</tspan>
                        </text>
                      );
                    }}
                    labelLine={false}
                  >
                    {classificationData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={chartColors[index % chartColors.length]} />
                    ))}
                  </Pie>
                  <RechartsTooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Violations Trend Line Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Violations Trend</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timeSeriesData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis 
                    dataKey="date" 
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis 
                    axisLine={false}
                    tickLine={false}
                    label={{ value: 'Violations', angle: -90, position: 'insideLeft' }}
                  />
                  <RechartsTooltip />
                  <Line 
                    type="monotone" 
                    dataKey="value" 
                    stroke="#B3D9FF" 
                    strokeWidth={3}
                    dot={{ fill: '#B3D9FF', strokeWidth: 2, r: 4 }}
                    activeDot={{ r: 6, fill: '#4DA6FF' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Activity Section */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Activity</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {recentActivity.map((item, index) => (
              <div key={index} className="flex items-start space-x-4 p-3 rounded-lg hover:bg-gray-50 transition-colors">
                <div className="flex-shrink-0">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                </div>
                <div className="flex-grow min-w-0">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-medium text-gray-900">{item.action}</p>
                    <p className="text-xs text-gray-500">{item.time}</p>
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{item.details}</p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default OverviewDashboard;
