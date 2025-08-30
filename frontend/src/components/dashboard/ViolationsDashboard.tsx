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
  import { useState } from "react";
  
  const ViolationsDashboard = () => {
    const [testText, setTestText] = useState("");
    const [analysisResult, setAnalysisResult] = useState(null);
    const [selectedViolationType, setSelectedViolationType] = useState("advertisement");
    const [isPolicyGuidelinesOpen, setIsPolicyGuidelinesOpen] = useState(false);
  
    // Mock violation data generator
    const generateViolationData = () => ({
      advertisement: {
        count: 156,
        severity: 'High',
        examples: [
          "Best restaurant in town! Call us at 555-0123 for reservations and special deals!",
          "Visit our website at example.com for 50% off all meals this week!",
          "Looking for catering? We provide the best service. Contact us today!"
        ]
      },
      inappropriate_content: {
        count: 23,
        severity: 'Medium',
        examples: [
          "The staff here are absolutely terrible and should be fired immediately.",
          "This place is disgusting and the owner is a complete idiot.",
          "Worst service ever, these people don't deserve to have jobs."
        ]
      },
      spam: {
        count: 89,
        severity: 'Medium',
        examples: [
          "Great food great food great food amazing service amazing service!",
          "Best restaurant best restaurant best restaurant in the city!",
          "Excellent excellent excellent food and service and atmosphere!"
        ]
      },
      rant_without_visit: {
        count: 67,
        severity: 'Low',
        examples: [
          "I heard from my friend that this place has terrible food and service.",
          "My neighbor told me they got food poisoning here last month.",
          "Someone on social media said this restaurant is overpriced and dirty."
        ]
      }
    });
  
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
  
    const violationsData = generateViolationData();
  
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
  
    return (
      <div className="space-y-8">
        {/* Page Header */}
        <div>
          <h1 className="text-3xl font-bold">Policy Violations</h1>
          <p className="text-gray-600 mt-2">Monitor and analyze policy violations in reviews</p>
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
              <Button variant="outline" size="sm">
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {Object.entries(violationsData).map(([violationType, data]) => (
                <Card key={violationType}>
                  <CardContent className="p-4 text-center">
                    <div className="text-2xl font-bold">{data.count}</div>
                    <div className="text-sm text-gray-600 mb-2">
                      {violationType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </div>
                    <Badge variant={getSeverityVariant(data.severity)}>
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
                {Object.keys(violationsData).map(type => (
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
                  <Badge variant={getSeverityVariant(violationsData[selectedViolationType].severity)}>
                    Severity: {violationsData[selectedViolationType].severity}
                  </Badge>
                </div>
                
                <div className="space-y-2">
                  {violationsData[selectedViolationType].examples.map((example, i) => (
                    <div key={i} className="p-3 bg-gray-50 rounded-lg">
                      <span className="text-sm font-medium">{i + 1}.</span>
                      <span className="text-sm italic ml-2">"{example}"</span>
                    </div>
                  ))}
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


