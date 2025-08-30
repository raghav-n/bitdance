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
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell
} from "recharts";
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";

const AnalyticsDashboard = () => {
  const [timeRange, setTimeRange] = useState("Last 6 Months");
  const [metricType, setMetricType] = useState("Review Volume");

  // Mock data generators
  const generateTimeSeriesData = (metric: string, range: string) => {
    const dataPoints = range === "Last 30 Days" ? 30 : range === "Last 3 Months" ? 90 : range === "Last 6 Months" ? 180 : 365;
    const baseValue = metric === "Review Volume" ? 1200 : metric === "Violations" ? 300 : metric === "Accuracy" ? 0.94 : 2.5;
    
    return Array.from({ length: dataPoints }, (_, i) => ({
      date: new Date(Date.now() - (dataPoints - i) * 24 * 60 * 60 * 1000).toLocaleDateString(),
      value: baseValue + Math.random() * baseValue * 0.3 - baseValue * 0.15
    }));
  };

  const generateCategoryData = () => [
    { category: 'Relevant', count: 8543, confidence: 0.94 },
    { category: 'Irrelevant', count: 2104, confidence: 0.89 },
    { category: 'Advertisement', count: 1876, confidence: 0.96 },
    { category: 'Rant', count: 324, confidence: 0.87 }
  ];

  const generatePerformanceData = () => {
    const weeks = 12;
    return Array.from({ length: weeks }, (_, i) => ({
      week: `W${i + 1}`,
      precision: 0.92 + Math.random() * 0.06,
      recall: 0.89 + Math.random() * 0.08,
      f1Score: 0.90 + Math.random() * 0.05
    }));
  };

  const timeSeriesData = generateTimeSeriesData(metricType, timeRange);
  const categoryData = generateCategoryData();
  const performanceData = generatePerformanceData();

  // Calculate metrics
  const avgValue = timeSeriesData.reduce((sum, d) => sum + d.value, 0) / timeSeriesData.length;
  const trend = timeSeriesData[timeSeriesData.length - 1].value > timeSeriesData[0].value ? "↗️" : "↘️";
  const trendPercent = Math.abs(((timeSeriesData[timeSeriesData.length - 1].value - timeSeriesData[0].value) / timeSeriesData[0].value) * 100);
  const peakValue = Math.max(...timeSeriesData.map(d => d.value));
  const volatility = Math.sqrt(timeSeriesData.reduce((sum, d) => sum + Math.pow(d.value - avgValue, 2), 0) / timeSeriesData.length) / avgValue;

  const formatValue = (value: number) => {
    if (metricType === "Accuracy") return `${(value * 100).toFixed(1)}%`;
    if (metricType === "Processing Time") return `${value.toFixed(1)}s`;
    return Math.round(value).toString();
  };

  const getYAxisTitle = () => {
    switch (metricType) {
      case "Review Volume": return "Reviews per Day";
      case "Violations": return "Violations per Day";
      case "Accuracy": return "Model Accuracy (%)";
      case "Processing Time": return "Processing Time (seconds)";
      default: return "Value";
    }
  };

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold">Analytics</h1>
        <p className="text-gray-600 mt-2">Deep insights into review patterns and model performance</p>
      </div>

      {/* Controls */}
      <div className="flex gap-4">
        <div className="flex-1">
          <label className="text-sm font-medium mb-2 block">Time Range:</label>
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="Last 30 Days">Last 30 Days</SelectItem>
              <SelectItem value="Last 3 Months">Last 3 Months</SelectItem>
              <SelectItem value="Last 6 Months">Last 6 Months</SelectItem>
              <SelectItem value="Last Year">Last Year</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="flex-1">
          <label className="text-sm font-medium mb-2 block">Metric:</label>
          <Select value={metricType} onValueChange={setMetricType}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="Review Volume">Review Volume</SelectItem>
              <SelectItem value="Violations">Violations</SelectItem>
              <SelectItem value="Accuracy">Accuracy</SelectItem>
              <SelectItem value="Processing Time">Processing Time</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="flex-2"></div>
      </div>

      {/* Main Time Series Chart */}
      <Card>
        <CardHeader>
          <CardTitle>{metricType} Over Time</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={timeSeriesData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="date" />
                <YAxis label={{ value: getYAxisTitle(), angle: -90, position: 'insideLeft' }} />
                <RechartsTooltip />
                <Line 
                  type="monotone" 
                  dataKey="value" 
                  stroke="#B3D9FF" 
                  strokeWidth={2}
                  dot={{ fill: '#B3D9FF', strokeWidth: 2, r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Key Insights */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardContent className="p-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900">{formatValue(avgValue)}</div>
              <div className="text-sm text-gray-600 mt-1">Average {metricType}</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{trend} {trendPercent.toFixed(1)}%</div>
              <div className="text-sm text-gray-600 mt-1">Trend</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{formatValue(peakValue)}</div>
              <div className="text-sm text-gray-600 mt-1">Peak {metricType}</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">{volatility.toFixed(2)}</div>
              <div className="text-sm text-gray-600 mt-1">Volatility</div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Analytics Tabs */}
      <Tabs defaultValue="category" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="category">📊 Category Analysis</TabsTrigger>
          <TabsTrigger value="performance">🎯 Performance</TabsTrigger>
          <TabsTrigger value="patterns">🔍 Patterns</TabsTrigger>
          <TabsTrigger value="forecasting">📈 Forecasting</TabsTrigger>
        </TabsList>

        <TabsContent value="category" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Review Category Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="h-80">
                  <h4 className="text-sm font-medium mb-4">Reviews by Category</h4>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={categoryData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="category" />
                      <YAxis />
                      <RechartsTooltip />
                      <Bar dataKey="count" fill="#B3D9FF" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="h-80">
                  <h4 className="text-sm font-medium mb-4">Confidence by Category</h4>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={categoryData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="category" />
                      <YAxis />
                      <RechartsTooltip />
                      <Bar dataKey="confidence" fill="#80C1FF" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Model Performance Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="week" />
                    <YAxis />
                    <RechartsTooltip />
                    <Legend />
                    <Line type="monotone" dataKey="precision" stroke="#FFB3E6" strokeWidth={2} />
                    <Line type="monotone" dataKey="recall" stroke="#B3D9FF" strokeWidth={2} />
                    <Line type="monotone" dataKey="f1Score" stroke="#D4B3FF" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="grid grid-cols-3 gap-4 mt-6">
                <Card>
                  <CardContent className="p-4 text-center">
                    <div className="text-lg font-bold">0.943</div>
                    <div className="text-sm text-gray-600">Current Precision</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4 text-center">
                    <div className="text-lg font-bold">0.912</div>
                    <div className="text-sm text-gray-600">Current Recall</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4 text-center">
                    <div className="text-lg font-bold">0.927</div>
                    <div className="text-sm text-gray-600">Current F1-Score</div>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="patterns" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Pattern Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold mb-2">📈 Seasonal Trends:</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
                    <li>Review volume peaks during weekends and holidays</li>
                    <li>Violation rates increase during promotional periods</li>
                    <li>Model accuracy is highest during regular business hours</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">🎯 Performance Patterns:</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
                    <li>Advertisement detection performs best (93% precision)</li>
                    <li>Relevant reviews have highest confidence scores</li>
                    <li>Multilingual reviews show 5% lower accuracy</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">⚠️ Risk Patterns:</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
                    <li>Policy violations cluster around specific time periods</li>
                    <li>New business locations show higher violation rates initially</li>
                    <li>Review length correlates with classification confidence</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="forecasting" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>30-Day Forecast</CardTitle>
              <CardDescription>Based on historical trends and current patterns</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={timeSeriesData.slice(-30)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <RechartsTooltip />
                    <Line type="monotone" dataKey="value" stroke="#B3D9FF" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="grid grid-cols-3 gap-4 mt-6">
                <Card>
                  <CardContent className="p-4 text-center">
                    <div className="text-lg font-bold">{formatValue(avgValue * 1.05)}</div>
                    <div className="text-sm text-gray-600">Predicted Average</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4 text-center">
                    <div className="text-lg font-bold">±{formatValue(avgValue * 0.1)}</div>
                    <div className="text-sm text-gray-600">Confidence Interval</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4 text-center">
                    <div className="text-lg font-bold">+5.2%</div>
                    <div className="text-sm text-gray-600">Trend vs Last Week</div>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

    </div>
  );
};

export default AnalyticsDashboard;
