import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
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
import { useState } from "react";

const ClassificationDashboard = () => {
  const [isRecommendationsOpen, setIsRecommendationsOpen] = useState(false);

  // Mock data generators
  const generatePerformanceMetrics = () => [
    { category: 'Relevant', precision: 0.96, recall: 0.94, f1Score: 0.95, support: 8543 },
    { category: 'Advertisement', precision: 0.93, recall: 0.88, f1Score: 0.90, support: 1876 },
    { category: 'Irrelevant', precision: 0.89, recall: 0.92, f1Score: 0.90, support: 2104 },
    { category: 'Rant', precision: 0.87, recall: 0.85, f1Score: 0.86, support: 324 }
  ];

  const generateConfidenceData = () => {
    const categories = ['Relevant', 'Advertisement', 'Irrelevant', 'Rant'];
    const data = [];
    
    categories.forEach(category => {
      for (let i = 0; i < 100; i++) {
        const confidence = Math.random();
        data.push({
          category,
          confidence: confidence,
          bin: Math.floor(confidence * 10) / 10
        });
      }
    });
    
    return data;
  };

  const generateCategoryStats = () => [
    { category: 'Relevant', meanConfidence: 0.943, stdConfidence: 0.087, sampleCount: 8543 },
    { category: 'Advertisement', meanConfidence: 0.912, stdConfidence: 0.094, sampleCount: 1876 },
    { category: 'Irrelevant', meanConfidence: 0.889, stdConfidence: 0.112, sampleCount: 2104 },
    { category: 'Rant', meanConfidence: 0.876, stdConfidence: 0.098, sampleCount: 324 }
  ];

  const performanceMetrics = generatePerformanceMetrics();
  const confidenceData = generateConfidenceData();
  const categoryStats = generateCategoryStats();

  // Process confidence data for histogram
  const confidenceBins = confidenceData.reduce((acc, item) => {
    const key = `${item.bin.toFixed(1)}-${item.category}`;
    if (!acc[key]) {
      acc[key] = { bin: item.bin.toFixed(1), category: item.category, count: 0 };
    }
    acc[key].count++;
    return acc;
  }, {});

  const histogramData = Object.values(confidenceBins);

  const chartColors = ['#B3D9FF', '#80C1FF', '#4DA6FF', '#99E6FF'];

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold">Classification Analysis</h1>
        <p className="text-gray-600 mt-2">Model performance metrics and classification insights</p>
      </div>

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
          <CardTitle>Confidence Distribution</CardTitle>
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
          <CardTitle>Category Analysis</CardTitle>
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
              {categoryStats.map((stat) => (
                <TableRow key={stat.category}>
                  <TableCell className="font-medium">{stat.category}</TableCell>
                  <TableCell>{stat.meanConfidence.toFixed(3)}</TableCell>
                  <TableCell>{stat.stdConfidence.toFixed(3)}</TableCell>
                  <TableCell>{stat.sampleCount.toLocaleString()}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
};

export default ClassificationDashboard;
