# Advanced Annotation System: Prompt Engineering Techniques

## 🎯 **Overview**

This document explains the sophisticated prompt engineering techniques implemented in our annotation system to give you a competitive edge in the hackathon.

## 🧠 **Advanced Prompt Engineering Features**

### **1. Few-Shot Learning with Curated Examples**

Our prompts include carefully crafted examples that demonstrate:
- **Clear violation patterns** for each policy type
- **Edge cases** that might confuse simpler systems
- **Proper reasoning** in parentheses format
- **Context-aware** decision making

**Example from Advertisement Detection:**
```
Review: "Amazing pizza! Visit www.pizzapromo.com for 20% off your first order!"
Business: "Pizza Palace"
Answer: YES (Contains promotional link and discount offer)

Review: "The food was delicious and the service was great. Highly recommend!"
Business: "Tasty Restaurant"
Answer: NO (Genuine review without promotional content)
```

### **2. Context-Aware Instructions**

Each prompt includes:
- **Business context** (restaurant name)
- **Specific violation indicators** for that policy type
- **Clear output format** requirements
- **Reasoning requirements** for transparency

### **3. Rule-Based Preprocessing**

**Hybrid Approach Benefits:**
- **Speed**: Rule-based detection for obvious cases (URLs, phone numbers)
- **Accuracy**: LLM for nuanced cases requiring context understanding
- **Cost Efficiency**: Skip LLM calls when rules are confident (>90% confidence)
- **Fallback**: Use rule results if LLM fails

**Rule Categories:**
- **URL/Phone Detection**: Regex patterns for promotional contact info
- **Keyword Matching**: Promotional terms, technology mentions, weather complaints
- **Pattern Recognition**: "Never been" statements, explicit non-visitation

### **4. Confidence Scoring**

**Multi-level Confidence Assessment:**
- **Rule-based confidence**: Based on pattern strength and keyword count
- **LLM confidence**: Estimated from response quality and reasoning length
- **Combined confidence**: Weighted average (30% rule + 70% LLM)

### **5. Batch Processing with Rate Limiting**

**Optimization Features:**
- **Concurrent processing**: Up to 2 simultaneous LLM calls
- **Rate limiting**: 1-second delays between batches
- **Retry logic**: Exponential backoff for failed requests
- **Timeout handling**: 30-second timeout per request

## 🛠️ **Technical Implementation**

### **Prompt Structure**

Each prompt follows this structure:
```
[System Role] + [Few-Shot Examples] + [Context-Aware Instruction] + [Review Text] + [Business Name] + [Output Format]
```

### **Response Parsing**

**Intelligent Parsing:**
- **YES/NO extraction**: Check first 10 characters for decision
- **Reasoning extraction**: Parse content in parentheses
- **Confidence estimation**: Based on reasoning quality and length
- **Error handling**: Fallback to rule-based results

### **Data Flow**

```
Review Text → Rule-Based Detection → LLM Query (if needed) → Result Combination → Confidence Scoring → Output
```

## 🎯 **Competitive Advantages**

### **1. Sophisticated Prompt Engineering**
- **Few-shot examples** reduce hallucination and improve consistency
- **Context-aware** prompts consider business type and review content
- **Structured output** format ensures reliable parsing

### **2. Hybrid Rule+LLM Approach**
- **Cost-effective**: Skip expensive LLM calls for obvious cases
- **Robust**: Fallback mechanisms for API failures
- **Fast**: Rule-based detection for high-confidence cases

### **3. Advanced Error Handling**
- **Retry logic** with exponential backoff
- **Graceful degradation** to rule-based results
- **Comprehensive logging** for debugging and optimization

### **4. Production-Ready Features**
- **Batch processing** for scalability
- **Rate limiting** to respect API constraints
- **Confidence scoring** for decision transparency
- **Comprehensive statistics** for performance monitoring

## 📊 **Performance Metrics**

The system tracks:
- **Violation detection rates** for each policy type
- **Average confidence scores** per violation type
- **Rule vs LLM usage statistics**
- **Processing time** per review
- **API success/failure rates**

## 🔧 **Configuration Options**

### **LLM Settings**
```yaml
llm:
  provider: "huggingface"
  model: "Qwen/Qwen3-8B-Instruct"
  max_concurrency: 2
  temperature: 0.0
  max_tokens: 50
  timeout: 30
  retry_attempts: 3
  batch_size: 10
```

### **Prompt Customization**
```yaml
prompts:
  advertisement:
    system: "You are an expert content moderator..."
    few_shot_examples: true
    confidence_threshold: 0.7
```

### **Rule Configuration**
```yaml
rules:
  url_regex: true
  phone_regex: true
  never_been_patterns:
    en: ["never been", "haven't visited", "didn't go"]
  promotional_keywords:
    en: ["discount", "promo", "offer", "deal"]
```

## 🚀 **Usage Instructions**

### **1. Setup**
```bash
python setup_annotation.py
```

### **2. Run Annotation**
```bash
python -m src.orchestrator.cli run-task annotate
```

### **3. Check Results**
```bash
# View annotation results
python -c "import pandas as pd; df = pd.read_parquet('data/annotated/restaurant_reviews/annotations.parquet'); print(df.head())"
```

## 🎯 **Hackathon Strategy**

### **Day 1: Foundation**
- ✅ Implement advanced annotation system
- ✅ Test with sample data
- ✅ Validate prompt engineering effectiveness

### **Day 2: Enhancement**
- 🔧 Fine-tune prompts based on initial results
- 🔧 Add more few-shot examples for edge cases
- 🔧 Optimize rule-based patterns

### **Day 3: Optimization**
- 🔧 Analyze performance metrics
- 🔧 Adjust confidence thresholds
- 🔧 Prepare demo and documentation

## 🔍 **Advanced Techniques Used**

1. **Chain-of-Thought Prompting**: Structured reasoning in parentheses
2. **Few-Shot Learning**: Curated examples for each violation type
3. **Context Injection**: Business name and review text integration
4. **Confidence Calibration**: Multi-level confidence scoring
5. **Hybrid Architecture**: Rule-based + LLM combination
6. **Robust Error Handling**: Graceful degradation and retry logic

This implementation gives you a **significant competitive advantage** through sophisticated prompt engineering, robust error handling, and production-ready features that most teams won't have time to implement in a hackathon setting.
