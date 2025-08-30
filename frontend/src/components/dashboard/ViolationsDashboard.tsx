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
    Collapsible,
    CollapsibleContent,
    CollapsibleTrigger,
  } from "@/components/ui/collapsible";
  import { Button } from "@/components/ui/button";
  import { Textarea } from "@/components/ui/textarea";
  import { Alert, AlertDescription } from "@/components/ui/alert";
  import { Badge } from "@/components/ui/badge";
  import { ChevronDown, AlertTriangle, CheckCircle, XCircle, RefreshCw } from "lucide-react";
  import { useState, useEffect } from "react";
  import { apiService, type ViolationStats } from "@/services/api";
  
  const ViolationsDashboard = () => {
    const [testText, setTestText] = useState("");
    const [analysisResult, setAnalysisResult] = useState(null);
    const [selectedViolationType, setSelectedViolationType] = useState("irrelevant_content");
    const [isPolicyGuidelinesOpen, setIsPolicyGuidelinesOpen] = useState(false);
    const [violationsData, setViolationsData] = useState<ViolationStats | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
  
    // Fetch real violation data from API
    useEffect(() => {
      const fetchViolationData = async () => {
        try {
          setLoading(true);
          const data = await apiService.getViolationStats();
          setViolationsData(data);
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Failed to fetch violation data');
          console.error('Error fetching violation data:', err);
        } finally {
          setLoading(false);
        }
      };

      fetchViolationData();
    }, []);

    // Refresh function for the refresh button
    const handleRefresh = async () => {
      try {
        setLoading(true);
        const data = await apiService.getViolationStats();
        setViolationsData(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to refresh violation data');
        console.error('Error refreshing violation data:', err);
      } finally {
        setLoading(false);
      }
    };

    // Fallback mock data (only used if API fails) - matches API response format
    const fallbackViolationData: ViolationStats = {
      irrelevant_content: {
        count: 17,
        severity: 'Medium',
        examples: [
          "This content doesn't provide useful review information.",
          "Not related to the actual restaurant experience.",
          "Generic comment without specific details."
        ]
      },
      advertisement: {
        count: 17,
        severity: 'High',
        examples: [
          "Best restaurant in town! Call us at 555-0123 for reservations!",
          "Visit our website for 50% off all meals this week!",
          "Looking for catering? Contact us today for the best service!"
        ]
      },
      review_without_visit: {
        count: 12,
        severity: 'Low',
        examples: [
          "I heard from my friend that this place has terrible food.",
          "My neighbor told me they got food poisoning here.",
          "Someone on social media said this restaurant is overpriced."
        ]
      }
    };
  
    // Mock analysis function
    const analyzeViolations = (text: string) => {
      const categories = ['Relevant', 'Advertisement', 'Inappropriate Content', 'Spam', 'Rant without Visit'];
      const randomCategory = categories[Math.floor(Math.random() * categories.length)];
      const confidence = 0.7 + Math.random() * 0.3;
      
      const policies = {
        advertisement: Math.random(),
        inappropriate_content: Math.random(),
        spam: Math.random(),
        rant_without_visit: Math.random()
      };
  
      const highRiskViolations = Object.entries(policies).filter(([_, score]) => score > 0.8);
      const mediumRiskViolations = Object.entries(policies).filter(([_, score]) => score > 0.5 && score <= 0.8);
      
      const totalViolations = highRiskViolations.length + mediumRiskViolations.length;
      const riskLevel = highRiskViolations.length > 0 ? 'High' : mediumRiskViolations.length > 0 ? 'Medium' : 'Low';
  
      let recommendation = "No action required.";
      if (riskLevel === 'High') {
        recommendation = "Immediate review required. Consider removing or flagging this content.";
      } else if (riskLevel === 'Medium') {
        recommendation = "Manual review recommended to verify policy compliance.";
      }
  
      return {
        category: randomCategory,
        confidence,
        violations: totalViolations,
        riskLevel,
        highRiskViolations: highRiskViolations.map(([policy, _]) => policy.replace('_', ' ').toUpperCase()),
        mediumRiskViolations: mediumRiskViolations.map(([policy, _]) => policy.replace('_', ' ').toUpperCase()),
        explanation: `Analysis detected ${totalViolations} potential policy violations. The content appears to be ${randomCategory.toLowerCase()} with ${(confidence * 100).toFixed(1)}% confidence.`,
        recommendation
      };
    };
  
    const handleAnalyze = () => {
      if (testText.trim()) {
        const result = analyzeViolations(testText);
        setAnalysisResult(result);
      }
    };
  
    // Use real data from API, fallback to mock data if API fails
    const currentViolationsData = violationsData || fallbackViolationData;
  
    const getRiskColor = (level: string) => {
      switch (level) {
        case 'High': return 'text-red-600';
        case 'Medium': return 'text-yellow-600';
        case 'Low': return 'text-green-600';
        default: return 'text-gray-600';
      }
    };
  
    const getRiskIcon = (level: string) => {
      switch (level) {
        case 'High': return '🔴';
        case 'Medium': return '🟡';
        case 'Low': return '🟢';
        default: return '⚪';
      }
    };
  
    const getSeverityVariant = (severity: string) => {
      switch (severity) {
        case 'High': return 'destructive';
        case 'Medium': return 'default';
        case 'Low': return 'secondary';
        default: return 'outline';
      }
    };
  
    if (loading) {
      return (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-2" />
            <p>Loading violation data...</p>
          </div>
        </div>
      );
    }

    if (error) {
      return (
        <div className="flex items-center justify-center h-64">
          <Alert className="max-w-md">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              Failed to load violation data: {error}
              <br />
              <Button 
                variant="outline" 
                size="sm" 
                className="mt-2"
                onClick={() => window.location.reload()}
              >
                Retry
              </Button>
            </AlertDescription>
          </Alert>
        </div>
      );
    }

    return (
      <div className="space-y-8">
        {/* Page Header */}
        <div>
          <h1 className="text-3xl font-bold">Policy Violations</h1>
          <p className="text-gray-600 mt-2">Real violation data from 5 model analysis</p>
        </div>
  
        {/* Violation Testing Section */}
        <Card>
          <CardHeader>
            <CardTitle>Test Review for Violations</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Textarea
              placeholder="Type here to check for policy violations..."
              value={testText}
              onChange={(e) => setTestText(e.target.value)}
              className="min-h-[120px]"
            />
            
            <Button onClick={handleAnalyze} disabled={!testText.trim()}>
              Analyze
            </Button>
  
            {/* Analysis Results */}
            {analysisResult && (
              <div className="space-y-4 pt-4 border-t">
                <h3 className="text-lg font-semibold">Analysis Results</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <Card>
                    <CardContent className="p-4 text-center">
                      <div className="text-lg font-bold">{analysisResult.category}</div>
                      <div className="text-sm text-gray-600">Classification</div>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardContent className="p-4 text-center">
                      <div className="text-lg font-bold">{(analysisResult.confidence * 100).toFixed(1)}%</div>
                      <div className="text-sm text-gray-600">Confidence</div>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardContent className="p-4 text-center">
                      <div className="text-lg font-bold">{analysisResult.violations}</div>
                      <div className="text-sm text-gray-600">Violations Found</div>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardContent className="p-4 text-center">
                      <div className={`text-lg font-bold ${getRiskColor(analysisResult.riskLevel)}`}>
                        {getRiskIcon(analysisResult.riskLevel)} {analysisResult.riskLevel}
                      </div>
                      <div className="text-sm text-gray-600">Risk Level</div>
                    </CardContent>
                  </Card>
                </div>
  
                {/* Risk Alerts */}
                {analysisResult.highRiskViolations.length > 0 && (
                  <Alert className="border-red-200 bg-red-50">
                    <XCircle className="h-4 w-4 text-red-600" />
                    <AlertDescription className="text-red-800">
                      ⚠️ <strong>High Risk Violations:</strong> {analysisResult.highRiskViolations.join(', ')}
                    </AlertDescription>
                  </Alert>
                )}
                
                {analysisResult.mediumRiskViolations.length > 0 && (
                  <Alert className="border-yellow-200 bg-yellow-50">
                    <AlertTriangle className="h-4 w-4 text-yellow-600" />
                    <AlertDescription className="text-yellow-800">
                      🟡 <strong>Medium Risk Violations:</strong> {analysisResult.mediumRiskViolations.join(', ')}
                    </AlertDescription>
                  </Alert>
                )}
                
                {analysisResult.violations === 0 && (
                  <Alert className="border-green-200 bg-green-50">
                    <CheckCircle className="h-4 w-4 text-green-600" />
                    <AlertDescription className="text-green-800">
                      ✅ <strong>No significant policy violations detected</strong>
                    </AlertDescription>
                  </Alert>
                )}
  
                <Alert>
                  <AlertDescription>
                    <strong>Explanation:</strong> {analysisResult.explanation}
                  </AlertDescription>
                </Alert>
  
                <Alert>
                  <AlertDescription>
                    <strong>Recommendation:</strong> {analysisResult.recommendation}
                  </AlertDescription>
                </Alert>
              </div>
            )}
          </CardContent>
        </Card>
  
        {/* Violation Statistics */}
        <Card>
          <CardHeader>
            <div className="flex justify-between items-center">
              <div>
                <CardTitle>Violation Statistics</CardTitle>
              </div>
              <Button variant="outline" size="sm" onClick={handleRefresh} disabled={loading}>
                <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {Object.entries(currentViolationsData).map(([violationType, data]) => (
                <Card key={violationType} className="flex-1">
                  <CardContent className="p-6 text-center">
                    <div className="text-3xl font-bold mb-2">{data.count}</div>
                    <div className="text-base text-gray-700 mb-3 font-medium">
                      {violationType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </div>
                    <Badge variant={getSeverityVariant(data.severity)} className="text-sm px-3 py-1">
                      {getRiskIcon(data.severity)} {data.severity}
                    </Badge>
                  </CardContent>
                </Card>
              ))}
            </div>
          </CardContent>
        </Card>
  
        {/* Violation Examples */}
        <Card>
          <CardHeader>
            <CardTitle>Violation Examples</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Select value={selectedViolationType} onValueChange={setSelectedViolationType}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Object.keys(currentViolationsData).map(type => (
                  <SelectItem key={type} value={type}>
                    {type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
  
            {selectedViolationType && (
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <h4 className="font-semibold">
                    {selectedViolationType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </h4>
                  <Badge variant={getSeverityVariant(currentViolationsData[selectedViolationType].severity)}>
                    Severity: {currentViolationsData[selectedViolationType].severity}
                  </Badge>
                </div>
                
                <div className="space-y-2">
                  {currentViolationsData[selectedViolationType]?.examples ? (
                    currentViolationsData[selectedViolationType].examples!.map((example, i) => (
                      <div key={i} className="p-3 bg-gray-50 rounded-lg">
                        <span className="text-sm font-medium">{i + 1}.</span>
                        <span className="text-sm italic ml-2">"{example}"</span>
                      </div>
                    ))
                  ) : (
                    <div className="p-3 bg-gray-50 rounded-lg">
                      <span className="text-sm text-gray-500 italic">
                        No example violations available for this category
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
  
        {/* Policy Guidelines */}
        <Collapsible open={isPolicyGuidelinesOpen} onOpenChange={setIsPolicyGuidelinesOpen}>
          <Card>
            <CollapsibleTrigger asChild>
              <CardHeader className="cursor-pointer hover:bg-gray-50">
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center gap-2">
                    📋 Policy Guidelines
                  </CardTitle>
                  <ChevronDown className={`h-4 w-4 transition-transform ${isPolicyGuidelinesOpen ? 'rotate-180' : ''}`} />
                </div>
              </CardHeader>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <CardContent>
                <div className="space-y-6">
                  <div>
                    <h4 className="font-semibold mb-2 text-red-600">📺 Advertisement</h4>
                    <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
                      <li>Reviews promoting business services, deals, or contact information</li>
                      <li>Contains promotional language or marketing content</li>
                      <li>Includes website links, phone numbers, or promotional codes</li>
                    </ul>
                  </div>
                  
                  <div>
                    <h4 className="font-semibold mb-2 text-yellow-600">🗣️ Rant without Visit</h4>
                    <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
                      <li>Negative reviews based on hearsay or second-hand information</li>
                      <li>Reviews stating "I heard..." or "My friend said..."</li>
                      <li>Opinions without actual experience at the establishment</li>
                    </ul>
                  </div>
                  
                  <div>
                    <h4 className="font-semibold mb-2 text-orange-600">❌ Inappropriate Content</h4>
                    <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
                      <li>Reviews containing offensive, abusive, or discriminatory language</li>
                      <li>Personal attacks on staff or other customers</li>
                      <li>Content violating community standards</li>
                    </ul>
                  </div>
                  
                  <div>
                    <h4 className="font-semibold mb-2 text-blue-600">📧 Spam</h4>
                    <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
                      <li>Repetitive content or excessive use of same phrases</li>
                      <li>Reviews that appear to be automated or fake</li>
                      <li>Multiple similar reviews from same source</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      </div>
    );
  };
  
  export default ViolationsDashboard;


