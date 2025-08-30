import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
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
import { apiService, type OverviewMetrics, type ClassificationData, type TimeSeriesData, type RecentActivity } from "@/services/api";

const OverviewDashboard = () => {
  // State for data with proper types
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [keyMetrics, setKeyMetrics] = useState<OverviewMetrics | null>(null);
  const [classificationData, setClassificationData] = useState<ClassificationData[]>([]);
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesData[]>([]);
  const [recentActivity, setRecentActivity] = useState<RecentActivity[]>([]);

  // Chart colors matching your theme
  const chartColors = ['#B3D9FF', '#80C1FF', '#4DA6FF', '#99E6FF'];

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [metrics, classification, timeSeries, activity] = await Promise.all([
          apiService.getOverviewMetrics(),
          apiService.getClassificationData(),
          apiService.getTimeSeriesData(),
          apiService.getRecentActivity()
        ]);

        setKeyMetrics(metrics);
        setClassificationData(classification);
        setTimeSeriesData(timeSeries);
        setRecentActivity(activity);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

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
              <div className="text-2xl font-bold text-blue-600">{keyMetrics?.modelAccuracy}</div>
              <div className="text-sm text-gray-600 mt-1">Model Accuracy</div>
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
