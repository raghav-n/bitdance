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
    Collapsible,
    CollapsibleContent,
    CollapsibleTrigger,
  } from "@/components/ui/collapsible";
  import { Button } from "@/components/ui/button";
  import { Textarea } from "@/components/ui/textarea";
  import { Alert, AlertDescription } from "@/components/ui/alert";
  import { Badge } from "@/components/ui/badge";
  import { ChevronDown, AlertTriangle, CheckCircle, XCircle } from "lucide-react";
  import { useState } from "react";
  
  // Add type definitions at the top
  interface PolicyScores {
    advertisement: number;
    inappropriate_content: number;
    spam: number;
    rant_without_visit: number;
  }

  interface PredictionResult {
    category: string;
    confidence: number;
    policies: PolicyScores;
    violations: number;
    riskLevel: string;
    explanation: string;
  }
  
  const PredictionDashboard = () => {
    const [customText, setCustomText] = useState("");
    const [selectedLanguage, setSelectedLanguage] = useState("English");
    const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
    const [isModelInfoOpen, setIsModelInfoOpen] = useState(false);
  
    // Mock prediction function
    const predictReview = (text: string, language: string): PredictionResult => {
      const categories = ['Relevant', 'Advertisement', 'Irrelevant', 'Rant without Visit'];
      const randomCategory = categories[Math.floor(Math.random() * categories.length)];
      const confidence = 0.7 + Math.random() * 0.3;
      
      const policies: PolicyScores = {
        advertisement: Math.random(),
        inappropriate_content: Math.random(),
        spam: Math.random(),
        rant_without_visit: Math.random()
      };
  
      const violations = Object.entries(policies).filter(([_, score]) => (score as number) > 0.7);
      const riskLevel = violations.length > 2 ? 'High' : violations.length > 0 ? 'Medium' : 'Low';
  
      return {
        category: randomCategory,
        confidence,
        policies,
        violations: violations.length,
        riskLevel,
        explanation: `This review appears to be ${randomCategory.toLowerCase()} based on language patterns and content analysis. The model detected ${violations.length} potential policy violations.`
      };
    };
  
    const handlePredict = () => {
      if (customText.trim()) {
        const result = predictReview(customText, selectedLanguage);
        setPredictionResult(result);
      }
    };
  
    const getLanguageOptions = () => [
      'English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 
      'Chinese (Simplified)', 'Japanese', 'Korean', 'Arabic', 'Russian'
    ];
  
    const getExampleReviews = () => ({
      relevant: [
        "Great food and excellent service! The pasta was perfectly cooked and the staff was very friendly. Will definitely come back!",
        "Amazing atmosphere and delicious pizza. The wait time was reasonable and the prices are fair for the quality.",
        "Visited last week with my family. Kids loved the menu options and we had a wonderful dining experience."
      ],
      advertisement: [
        "Best restaurant in town! Call us at 555-0123 for reservations. Visit our website at example.com for special deals!",
        "Grand opening special! 50% off all meals this week. Don't miss out on our amazing offers!",
        "Looking for catering services? We provide the best food for your events. Contact us today!"
      ],
      rant: [
        "I heard from my friend that this place is terrible. Never been there myself but apparently the service is awful.",
        "My neighbor told me the food gave them food poisoning. I would never eat there based on what I heard.",
        "Someone on social media said this restaurant is overpriced. I'm writing this review to warn others."
      ],
    });
  
    const examples = getExampleReviews();
  
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
        case 'High': return <XCircle className="h-4 w-4" />;
        case 'Medium': return <AlertTriangle className="h-4 w-4" />;
        case 'Low': return <CheckCircle className="h-4 w-4" />;
        default: return null;
      }
    };
  
    return (
      <div className="space-y-8">
        {/* Page Header */}
        <div>
          <h1 className="text-3xl font-bold">Real-time Predictions</h1>
          <p className="text-gray-600 mt-2">Test the classification model with custom reviews</p>
        </div>
  
        {/* Custom Prediction Section */}
        <Card>
          <CardHeader>
            <CardTitle>Custom Review Prediction</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Textarea
              placeholder="Type here to analyze your review..."
              value={customText}
              onChange={(e) => setCustomText(e.target.value)}
              className="min-h-[120px]"
            />
            
            <div className="flex gap-4">
              <Button onClick={handlePredict} disabled={!customText.trim()}>
                Predict
              </Button>
              <div className="flex-1">
                <Select value={selectedLanguage} onValueChange={setSelectedLanguage}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {getLanguageOptions().map(lang => (
                      <SelectItem key={lang} value={lang}>{lang}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
  
            {/* Prediction Results */}
            {predictionResult && (
              <div className="space-y-4 pt-4 border-t">
                <h3 className="text-lg font-semibold">Prediction Results</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <Card>
                    <CardContent className="p-4 text-center">
                      <div className="text-lg font-bold">{predictionResult.category}</div>
                      <div className="text-sm text-gray-600">Category</div>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardContent className="p-4 text-center">
                      <div className="text-lg font-bold">{(predictionResult.confidence * 100).toFixed(1)}%</div>
                      <div className="text-sm text-gray-600">Confidence</div>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardContent className="p-4 text-center">
                      <div className="text-lg font-bold">{predictionResult.violations}</div>
                      <div className="text-sm text-gray-600">Violations Found</div>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardContent className="p-4 text-center">
                      <div className={`text-lg font-bold flex items-center justify-center gap-2 ${getRiskColor(predictionResult.riskLevel)}`}>
                        {getRiskIcon(predictionResult.riskLevel)}
                        {predictionResult.riskLevel}
                      </div>
                      <div className="text-sm text-gray-600">Risk Level</div>
                    </CardContent>
                  </Card>
                </div>
  
                {/* Risk Alerts */}
                {predictionResult.riskLevel === 'High' && (
                  <Alert className="border-red-200 bg-red-50">
                    <AlertTriangle className="h-4 w-4 text-red-600" />
                    <AlertDescription className="text-red-800">
                      ⚠️ <strong>High risk violations detected</strong>
                    </AlertDescription>
                  </Alert>
                )}
                
                {predictionResult.riskLevel === 'Medium' && (
                  <Alert className="border-yellow-200 bg-yellow-50">
                    <AlertTriangle className="h-4 w-4 text-yellow-600" />
                    <AlertDescription className="text-yellow-800">
                      🟡 <strong>Medium risk violations detected</strong>
                    </AlertDescription>
                  </Alert>
                )}
                
                {predictionResult.riskLevel === 'Low' && (
                  <Alert className="border-green-200 bg-green-50">
                    <CheckCircle className="h-4 w-4 text-green-600" />
                    <AlertDescription className="text-green-800">
                      ✅ <strong>No significant violations detected</strong>
                    </AlertDescription>
                  </Alert>
                )}
  
                <Alert>
                  <AlertDescription>
                    <strong>Analysis:</strong> {predictionResult.explanation}
                  </AlertDescription>
                </Alert>
  
                {/* Detailed Policy Scores */}
                <Collapsible>
                  <CollapsibleTrigger asChild>
                    <Button variant="outline" className="w-full">
                      📊 Detailed Policy Scores
                      <ChevronDown className="ml-2 h-4 w-4" />
                    </Button>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="mt-4">
                    <div className="grid grid-cols-2 gap-4">
                      {Object.entries(predictionResult.policies).map(([policy, score]) => (
                        <div key={policy} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                          <span className="text-sm font-medium">
                            {policy.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </span>
                          <Badge variant={(score as number) > 0.7 ? 'destructive' : (score as number) > 0.4 ? 'default' : 'secondary'}>
                            {((score as number) * 100).toFixed(1)}%
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              </div>
            )}
          </CardContent>
        </Card>
  
        {/* Example Reviews */}
        <Card>
          <CardHeader>
            <CardTitle>Try Example Reviews</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="relevant" className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="relevant">✅ Relevant</TabsTrigger>
                <TabsTrigger value="advertisement">📺 Advertisement</TabsTrigger>
                <TabsTrigger value="rant">🗣️ Rant</TabsTrigger>
              </TabsList>
  
              <TabsContent value="relevant" className="space-y-2">
                {examples.relevant.map((example, i) => (
                  <Button
                    key={i}
                    variant="outline"
                    className="w-full text-left h-auto p-4 whitespace-normal"
                    onClick={() => setCustomText(example)}
                  >
                    <span className="text-sm">Try Example {i + 1}: "{example.substring(0, 50)}..."</span>
                  </Button>
                ))}
              </TabsContent>
  
              <TabsContent value="advertisement" className="space-y-2">
                {examples.advertisement.map((example, i) => (
                  <Button
                    key={i}
                    variant="outline"
                    className="w-full text-left h-auto p-4 whitespace-normal"
                    onClick={() => setCustomText(example)}
                  >
                    <span className="text-sm">Try Example {i + 1}: "{example.substring(0, 50)}..."</span>
                  </Button>
                ))}
              </TabsContent>
  
              <TabsContent value="rant" className="space-y-2">
                {examples.rant.map((example, i) => (
                  <Button
                    key={i}
                    variant="outline"
                    className="w-full text-left h-auto p-4 whitespace-normal"
                    onClick={() => setCustomText(example)}
                  >
                    <span className="text-sm">Try Example {i + 1}: "{example.substring(0, 50)}..."</span>
                  </Button>
                ))}
              </TabsContent>
  
              
            </Tabs>
          </CardContent>
        </Card>
  
        {/* Model Information */}
        <Collapsible open={isModelInfoOpen} onOpenChange={setIsModelInfoOpen}>
          <Card>
            <CollapsibleTrigger asChild>
              <CardHeader className="cursor-pointer hover:bg-gray-50">
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center gap-2">
                     Model Information
                  </CardTitle>
                  <ChevronDown className={`h-4 w-4 transition-transform ${isModelInfoOpen ? 'rotate-180' : ''}`} />
                </div>
              </CardHeader>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold mb-2">BitDance Review Classification Model</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                      <div>
                        <strong>Model Type:</strong> Transformer-based Neural Network<br />
                        <strong>Training Data:</strong> 100K+ labeled restaurant reviews<br />
                        <strong>Languages Supported:</strong> 60+ languages with multilingual embeddings<br />
                        <strong>Update Frequency:</strong> Weekly retraining on new data
                      </div>
                      <div>
                        <strong>Categories:</strong><br />
                        • <strong>Relevant:</strong> Legitimate reviews from actual customers<br />
                        • <strong>Irrelevant:</strong> Reviews not related to the business<br />
                        • <strong>Advertisement:</strong> Promotional content or spam<br />
                        • <strong>Rant without Visit:</strong> Negative reviews without actual experience
                      </div>
                    </div>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">Confidence Scores:</h4>
                    <div className="text-sm space-y-1">
                      <div>• <strong>&gt;90%:</strong> High confidence, automated processing</div>
                      <div>• <strong>70-90%:</strong> Medium confidence, may need review</div>
                      <div>• <strong>&lt;70%:</strong> Low confidence, manual review recommended</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
  
        {/* Pro Tip */}
        <Alert>
          <AlertDescription>
            💡 <strong>Pro Tip:</strong> For batch processing of multiple reviews, use the API endpoint or upload a CSV file through the data ingestion pipeline.
          </AlertDescription>
        </Alert>
      </div>
    );
  };
  
  export default PredictionDashboard;


