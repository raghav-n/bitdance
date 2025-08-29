                
import aiohttp
import os
# Gemini API client stub
class GeminiClient:
    """Google Gemini LLM client using Generative AI API."""
    def __init__(self, config: dict):
        self.config = config
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/" + config.get('model', 'gemini-pro') + ":generateContent"
        self.api_key = config.get('gemini_api_key', os.environ.get('GEMINI_API_KEY', ''))
        self.max_tokens = config.get('max_tokens', 200)
        self.temperature = config.get('temperature', 0.0)
        self.logger = get_logger("gemini_client")

    async def query_llm(self, prompt: str, session: aiohttp.ClientSession) -> dict:
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens
            }
        }
        try:
            async with session.post(self.api_url, headers=headers, params=params, json=payload, timeout=aiohttp.ClientTimeout(total=self.config.get('timeout', 30))) as response:
                if response.status == 200:
                    result = await response.json()
                    # Gemini returns candidates[0]['content']['parts'][0]['text']
                    candidates = result.get('candidates', [])
                    if candidates:
                        text = candidates[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                        # Try to parse JSON from response
                        try:
                            json_result = json.loads(text)
                            return json_result
                        except Exception:
                            self.logger.warning(f"Failed to parse JSON from Gemini response: {text}")
                            return {"is_violation": False, "confidence": 0.5, "reasoning": "Fallback: Could not parse Gemini output", "violation_type": "fallback", "indicators": []}
                    else:
                        self.logger.warning(f"No candidates in Gemini response: {result}")
                        return {"is_violation": False, "confidence": 0.5, "reasoning": "Fallback: No candidates", "violation_type": "fallback", "indicators": []}
                else:
                    self.logger.warning(f"Gemini API error {response.status}: {await response.text()}")
                    return {"is_violation": False, "confidence": 0.5, "reasoning": "Fallback: Gemini API error", "violation_type": "fallback", "indicators": []}
        except Exception as e:
            self.logger.warning(f"Gemini API exception: {e}")
            return {"is_violation": False, "confidence": 0.5, "reasoning": f"Fallback: Gemini exception {e}", "violation_type": "fallback", "indicators": []}
"""Advanced annotation system with HuggingFace Chat Completions API integration and structured JSON output.

Implements a clean, extensible policy system for detecting violations in restaurant reviews
with rule-based preprocessing and HuggingFace Chat Completions API integration
"""

import pandas as pd
import numpy as np
import re
import json
import time
import asyncio
import aiohttp
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import torch
import logging

from ..orchestrator import task
from ..orchestrator.utils import data_dir, dataset_slug
from ..orchestrator.logging import get_logger


@dataclass
class Policy:
    """Extensible policy definition for violation detection."""
    policy_type: str
    description: str
    violation_indicators: List[str]
    examples: List[Dict[str, Any]]
    rule_patterns: List[str]
    confidence_threshold: float = 0.7


@dataclass
class AnnotationResult:
    """Structured result for each annotation type."""
    is_violation: bool
    confidence: float
    reasoning: str
    rule_triggered: Optional[str] = None
    llm_confidence: Optional[float] = None
    rule_confidence: Optional[float] = None
    indicators: List[str] = None
    processing_metadata: Dict[str, Any] = None


@dataclass
class ReviewAnnotation:
    """Complete annotation for a single review with rich metadata."""
    review_id: str
    annotations: Dict[str, AnnotationResult]
    annotation_source: str
    processing_time: float
    llm_calls_made: int
    rule_detections_used: int
    json_parsing_success: bool
    validation_status: str


class PolicyRegistry:
    """Registry for managing extensible policies."""
    
    def __init__(self):
        self.policies: Dict[str, Policy] = {}
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """Initialize default policies for the hackathon challenge."""
        
        # Advertisement Policy
        advertisement_policy = Policy(
            policy_type="advertisement",
            description="Detect promotional content, advertisements, or marketing material in restaurant reviews",
            violation_indicators=[
                "URLs, links, or website addresses",
                "Phone numbers for business promotion", 
                "Discount codes, coupons, or special offers",
                "Promotional language ('limited time', 'special deal', 'exclusive offer')",
                "Marketing calls-to-action ('visit', 'call now', 'book today')"
            ],
            examples=[
                {
                    "review": "Amazing pizza! Visit www.pizzapromo.com for 20% off your first order!",
                    "business": "Pizza Palace",
                    "is_violation": True,
                    "reasoning": "Contains promotional URL and discount offer"
                },
                {
                    "review": "The food was delicious and the service was great. Highly recommend!",
                    "business": "Tasty Restaurant", 
                    "is_violation": False,
                    "reasoning": "Genuine review about food and service without promotional content"
                },
                {
                    "review": "Best restaurant ever! Call 555-1234 for reservations and ask about our lunch specials.",
                    "business": "Fine Dining",
                    "is_violation": True,
                    "reasoning": "Contains promotional phone number and special offers"
                }
            ],
            rule_patterns=["url", "phone_number", "promotional_keywords"],
            confidence_threshold=0.7
        )
        
        # Irrelevant Content Policy
        irrelevant_policy = Policy(
            policy_type="irrelevant",
            description="Detect content that doesn't relate to the restaurant experience",
            violation_indicators=[
                "Topics unrelated to food, service, atmosphere, or dining experience",
                "Personal life updates or general complaints",
                "Reviews about other businesses or products",
                "Weather, traffic, or location complaints unrelated to the restaurant",
                "Political or social commentary"
            ],
            examples=[
                {
                    "review": "I love my new iPhone, but this place is too noisy.",
                    "business": "Quiet Cafe",
                    "is_violation": True,
                    "reasoning": "Review primarily about phone technology, not restaurant experience"
                },
                {
                    "review": "The food was excellent and the staff was friendly.",
                    "business": "Local Diner",
                    "is_violation": False,
                    "reasoning": "Review directly about restaurant food and service"
                },
                {
                    "review": "Great weather today! The traffic was terrible getting here.",
                    "business": "Seaside Restaurant",
                    "is_violation": True,
                    "reasoning": "Review about weather and traffic, not restaurant experience"
                }
            ],
            rule_patterns=["technology_mention", "weather_traffic"],
            confidence_threshold=0.7
        )
        
        # Rant Without Visit Policy
        rant_policy = Policy(
            policy_type="rant_without_visit",
            description="Detect complaints from people who likely never visited the restaurant",
            violation_indicators=[
                "Explicit statements of never visiting ('never been', 'haven't visited', 'didn't go')",
                "Complaints based on hearsay or rumors",
                "Reviews about menu prices without dining experience",
                "Complaints about location or parking without mentioning food/service",
                "Generic negative statements without specific details"
            ],
            examples=[
                {
                    "review": "Never been here, but I heard it's terrible from my friend.",
                    "business": "Local Restaurant",
                    "is_violation": True,
                    "reasoning": "Explicitly states never visited but still provides negative review"
                },
                {
                    "review": "The food was cold and the service was slow. Won't be back.",
                    "business": "Local Restaurant",
                    "is_violation": False,
                    "reasoning": "Describes specific dining experience with concrete details"
                },
                {
                    "review": "Haven't visited yet, but the menu looks expensive.",
                    "business": "Fine Dining",
                    "is_violation": True,
                    "reasoning": "States haven't visited but still complains about pricing"
                }
            ],
            rule_patterns=["explicit_non_visit"],
            confidence_threshold=0.7
        )
        
        # Register policies
        self.register_policy(advertisement_policy)
        self.register_policy(irrelevant_policy)
        self.register_policy(rant_policy)
    
    def register_policy(self, policy: Policy):
        """Register a new policy."""
        self.policies[policy.policy_type] = policy
    
    def get_policy(self, policy_type: str) -> Optional[Policy]:
        """Get a policy by type."""
        return self.policies.get(policy_type)
    
    def list_policies(self) -> List[str]:
        """List all registered policy types."""
        return list(self.policies.keys())
    
    def remove_policy(self, policy_type: str) -> bool:
        """Remove a policy by type."""
        if policy_type in self.policies:
            del self.policies[policy_type]
            return True
        return False


class JSONPromptEngineer:
    """Advanced prompt engineering with structured JSON output and extensible policies."""
    
    def __init__(self, policy_registry: PolicyRegistry):
        self.policy_registry = policy_registry
        self.logger = get_logger("json_prompt_engineer")
    
    def build_prompt_json(self, policy_type: str, review_text: str, business_name: str) -> str:
        """Build sophisticated prompt for any policy type with JSON output."""
        
        policy = self.policy_registry.get_policy(policy_type)
        if not policy:
            raise ValueError(f"Policy type '{policy_type}' not found")
        
        system_prompt = f"You are an expert content moderator specializing in detecting {policy.description}."
        
        # Build few-shot examples from policy
        few_shot_examples = self._build_few_shot_examples(policy)
        
        # Build instruction from policy
        instruction = self._build_instruction(policy, review_text, business_name)
        
        return f"{system_prompt}\n{few_shot_examples}\n{instruction}"
    
    def _build_few_shot_examples(self, policy: Policy) -> str:
        """Build few-shot examples from policy examples."""
        examples_text = "Examples:\n\n"
        
        for example in policy.examples:
            is_violation = "true" if example["is_violation"] else "false"
            examples_text += f"""Review: "{example['review']}"
            Business: "{example['business']}"
            {{
            "is_violation": {is_violation},
            "confidence": 0.9,
            "reasoning": "{example['reasoning']}",
            "violation_type": "{policy.policy_type}",
            "indicators": []
            }}
            """
        
        return examples_text
    
    def _build_instruction(self, policy, review_text, business_name):
        """Build instruction from policy indicators with strict output requirements."""
        indicators_text = "\n".join([f"- {indicator}" for indicator in policy.violation_indicators])
        instruction = f"""Analyze this restaurant review for {policy.description.lower()}.

        Review: "{review_text}"
        Business: "{business_name}"

        Consider these violation indicators:
        {indicators_text}

        Output ONLY a valid JSON object with these exact fields:
        {{
            "is_violation": boolean,
            "confidence": float (0.0-1.0),
            "reasoning": string,
            "violation_type": "{policy.policy_type}",
            "indicators": array of strings
        }}
        Do not include any Markdown formatting, code block markers, or extra text. Do not use triple backticks or the word 'json'. Only output the JSON object."""
        return instruction


class JSONResponseParser:
    """Robust JSON parsing with validation and fallback mechanisms."""
    
    def __init__(self):
        self.logger = get_logger("json_parser")
    
    def parse_llm_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Extract and validate JSON from LLM response."""
        try:
            # Find JSON in response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                
                # Validate required fields
                if self.validate_json_schema(result):
                    return self.normalize_json_output(result)
            
            return None
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.warning(f"JSON parsing error: {e}")
            return None
    
    def validate_json_schema(self, data: Dict[str, Any]) -> bool:
        """Validate JSON schema for training data quality."""
        required_fields = [
            'is_violation', 'confidence', 'reasoning', 
            'violation_type', 'indicators'
        ]
        
        # Check required fields
        if not all(field in data for field in required_fields):
            return False
        
        # Validate types
        if not isinstance(data['is_violation'], bool):
            return False
        if not isinstance(data['confidence'], (int, float)) or not 0 <= data['confidence'] <= 1:
            return False
        if not isinstance(data['reasoning'], str):
            return False
        if not isinstance(data['indicators'], list):
            return False
        
        return True
    
    def normalize_json_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize JSON output for consistent training data."""
        return {
            'is_violation': bool(data['is_violation']),
            'confidence': float(data['confidence']),
            'reasoning': str(data['reasoning']),
            'violation_type': str(data['violation_type']),
            'indicators': list(data['indicators']),
            'processing_metadata': {
                'model_used': self.config.get('model', ''),
                'prompt_version': 'v2.0',
                'timestamp': datetime.now().isoformat()
            },
            'validation_status': 'validated'
        }


class LocalLLMClient:
    """Generic local LLM client with quantization support."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("annotation")
        self.json_parser = JSONResponseParser()
        self.max_tokens = config.get('max_tokens', 200)
        self.temperature = config.get('temperature', 0.0)
        self.pipe = None
        self._load_pipeline()
            
            # Clear any existing models from memory
    def _load_pipeline(self):
        """Load transformers pipeline for image-text-to-text or text generation."""
        from transformers import pipeline
        model_name = self.config.get('model', 'google/gemma-3-4b-it')
        task_type = self.config.get('task_type', 'text-generation')
        try:
            print("Run?")
            if task_type == 'image-text-to-text':
                self.pipe = pipeline("image-text-to-text", model=model_name)
            else:
                self.pipe = pipeline("text-generation", model=model_name)
            self.logger.info(f"Loaded pipeline for {task_type} with model {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load pipeline: {e}")
            self.pipe = None
            
    
    async def query_llm(self, prompt: str, session=None, messages=None) -> Dict[str, Any]:
        """Query local model using transformers pipeline."""
        try:
            if self.pipe is None:
                self.logger.error("Pipeline not loaded.")
                return self._create_fallback_response("Pipeline not loaded")
            # If image-text-to-text, expect messages format
            if self.config.get('task_type', 'text-generation') == 'image-text-to-text' and messages is not None:
                output = self.pipe(text=messages)
                response_text = str(output)
            else:
                output = self.pipe(prompt, max_new_tokens=self.max_tokens, temperature=self.temperature)
                # output is a list of dicts with 'generated_text'
                response_text = output[0]['generated_text'] if isinstance(output, list) and 'generated_text' in output[0] else str(output)
            json_result = self.json_parser.parse_llm_response(response_text)
            if json_result:
                return json_result
            else:
                self.logger.warning(f"Failed to parse JSON from local model response: {response_text}")
                return self._create_fallback_response(response_text)
        except Exception as e:
            self.logger.error(f"Local pipeline inference error: {e}")
            return self._create_fallback_response(f"Local pipeline error: {str(e)}")
    
    def _create_fallback_response(self, error_message: str) -> Dict[str, Any]:
        """Create a fallback response when local model fails."""
        return {
            'is_violation': False,
            'confidence': 0.4,
            'reasoning': f"Local model fallback response: {error_message}",
            'violation_type': 'local_fallback',
            'indicators': [],
            'processing_metadata': {
                'model_used': 'local_model_fallback',
                'prompt_version': 'v2.0',
                'timestamp': datetime.now().isoformat()
            },
            'validation_status': 'fallback'
        }


class HuggingFaceChatCompletionsClient:
    """Generic HuggingFace Chat Completions API client for any supported model."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("hf_chat_client")
        self.api_url = "https://router.huggingface.co/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {config.get('hf_token', '')}",
            "Content-Type": "application/json"
        }
        self.model = config.get('model', 'google/gemma-3b')
        self.max_tokens = config.get('max_tokens', 200)
        self.temperature = config.get('temperature', 0.0)
        self.json_parser = JSONResponseParser()

    async def query_llm(self, prompt: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        for attempt in range(self.config.get('retry_attempts', 3)):
            try:
                async with session.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.get('timeout', 30))
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if 'choices' in result and len(result['choices']) > 0:
                            message = result['choices'][0].get('message', {})
                            content = message.get('content')
                            reasoning_content = message.get('reasoning_content')
                            if content is not None:
                                response_text = content.strip()
                            elif reasoning_content is not None:
                                response_text = reasoning_content.strip()
                                self.logger.info(f"Using reasoning_content instead of content")
                            else:
                                self.logger.warning(f"API returned null content: {result}")
                                return self._create_fallback_response("API returned null content")
                            json_result = self.json_parser.parse_llm_response(response_text)
                            if json_result:
                                return json_result
                            else:
                                self.logger.warning(f"Failed to parse JSON from response: {response_text}")
                                return self._create_fallback_response(response_text)
                        else:
                            self.logger.warning(f"Unexpected response format: {result}")
                            return self._create_fallback_response("Unexpected response format")
                    else:
                        self.logger.warning(f"API error {response.status}: {await response.text()}")
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.get('retry_attempts', 3) - 1:
                    await asyncio.sleep(2 ** attempt)
        raise Exception(f"Failed to query LLM after {self.config.get('retry_attempts', 3)} attempts")

    def _create_fallback_response(self, error_message: str) -> Dict[str, Any]:
        return {
            'is_violation': False,
            'confidence': 0.5,
            'reasoning': f"Fallback response: {error_message}",
            'violation_type': 'fallback',
            'indicators': [],
            'processing_metadata': {
                'model_used': 'hf-chat-completions-fallback',
                'prompt_version': 'v2.0',
                'timestamp': datetime.now().isoformat()
            },
            'validation_status': 'fallback'
        }


class RuleBasedDetector:
    """Enhanced rule-based preprocessing with extensible policies."""
    
    def __init__(self, policy_registry: PolicyRegistry, config: Dict[str, Any]):
        self.policy_registry = policy_registry
        self.config = config
        self.logger = get_logger("rule_detector")
        
        # Compile regex patterns
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
        self.phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        
        # Load keyword patterns from config
        self.never_been_patterns = config['rules']['never_been_patterns']['en']
        self.promotional_keywords = config['rules']['promotional_keywords']['en']
    
    def detect_violation_rules(self, text: str, policy_type: str) -> Tuple[bool, float, Optional[str], List[str]]:
        """Rule-based violation detection for any policy type."""
        
        if policy_type == "advertisement":
            return self._detect_advertisement_rules(text)
        elif policy_type == "irrelevant":
            return self._detect_irrelevant_rules(text)
        elif policy_type == "rant_without_visit":
            return self._detect_rant_rules(text)
        else:
            return False, 0.0, None, []
    
    def _detect_advertisement_rules(self, text: str) -> Tuple[bool, float, Optional[str], List[str]]:
        """Rule-based advertisement detection with indicators."""
        text_lower = text.lower()
        indicators = []
        
        # Check for URLs
        if self.url_pattern.search(text):
            indicators.append("url")
            return True, 0.95, "URL detected", indicators
        
        # Check for phone numbers
        if self.phone_pattern.search(text):
            indicators.append("phone_number")
            return True, 0.85, "Phone number detected", indicators
        
        # Check for promotional keywords
        promo_count = sum(1 for keyword in self.promotional_keywords if keyword in text_lower)
        if promo_count >= 2:
            indicators.extend(["multiple_promotional_keywords"])
            return True, 0.8, f"Multiple promotional keywords ({promo_count})", indicators
        elif promo_count == 1:
            indicators.append("promotional_keyword")
            return True, 0.6, "Promotional keyword detected", indicators
        
        return False, 0.0, None, indicators
    
    def _detect_irrelevant_rules(self, text: str) -> Tuple[bool, float, Optional[str], List[str]]:
        """Rule-based irrelevant content detection with indicators."""
        text_lower = text.lower()
        indicators = []
        
        # Check for technology/device mentions
        tech_keywords = ['iphone', 'android', 'phone', 'computer', 'laptop', 'app', 'software']
        if any(keyword in text_lower for keyword in tech_keywords):
            indicators.append("technology_mention")
            return True, 0.7, "Technology-related content", indicators
        
        # Check for weather/traffic complaints
        weather_keywords = ['weather', 'rain', 'sunny', 'cold', 'hot', 'traffic', 'parking']
        if any(keyword in text_lower for keyword in weather_keywords):
            indicators.append("weather_traffic")
            return True, 0.6, "Weather/traffic content", indicators
        
        return False, 0.0, None, indicators
    
    def _detect_rant_rules(self, text: str) -> Tuple[bool, float, Optional[str], List[str]]:
        """Rule-based rant without visit detection with indicators."""
        text_lower = text.lower()
        indicators = []
        
        # Check for explicit non-visitation statements
        for pattern in self.never_been_patterns:
            if pattern in text_lower:
                indicators.append("explicit_non_visit")
                return True, 0.9, f"Non-visitation pattern: '{pattern}'", indicators
        
        return False, 0.0, None, indicators


@task(
    name="annotate",
    inputs=lambda p: [
        f"{data_dir(p)}/raw/{dataset_slug(p)}/reviews_text_only.parquet",
        "configs/base.yaml",
    ],
    outputs=lambda p: [
        f"{data_dir(p)}/annotated/{dataset_slug(p)}/annotations.parquet",
    ],
)
def annotate(params: dict):
    """Advanced annotation system using HuggingFace Chat Completions API and extensible policy framework.
    
    Implements:
    - Extensible policy framework for easy addition/removal of policies
    - Rule-based preprocessing for quick filtering
    - Few-shot learning with JSON output examples
    - Context-aware prompt engineering
    - Structured confidence scoring and reasoning
    - HuggingFace Chat Completions API 
    - JSON validation and error handling
    - Rich metadata for training data generation
    """
    logger = get_logger("annotate")
    logger.info("Starting advanced annotation")
    
    # Load configuration
    annotate_config = params['annotate']
    dataset_slug = params['dataset']['slug']
    
    # Initialize policy registry and components
    policy_registry = PolicyRegistry()
    prompt_engineer = JSONPromptEngineer(policy_registry)
    rule_detector = RuleBasedDetector(policy_registry, annotate_config)
    
    # Choose LLM client based on provider configuration
    llm_provider = annotate_config['llm'].get('provider', 'huggingface')
    if llm_provider == 'local':
        logger.info("Using local model client with memory optimization")
        llm_client = LocalLLMClient(annotate_config['llm'])
    elif llm_provider == 'gemini':
        logger.info("Using Gemini API client")
        llm_client = GeminiClient(annotate_config['llm'])
    elif llm_provider == 'huggingface':
        logger.info("Using HuggingFace Chat Completions API client")
        llm_client = HuggingFaceChatCompletionsClient(annotate_config['llm'])
    else:
        raise ValueError(f"Unknown llm provider: {llm_provider}")
    
    # Load reviews
    input_path = f"data/raw/{dataset_slug}/reviews_text_only.parquet"
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} reviews for annotation")
    
    # Get all policy types
    policy_types = policy_registry.list_policies()
    logger.info(f"Processing {len(policy_types)} policies: {policy_types}")
    
    # Get batch control parameters
    llm_config = annotate_config['llm']
    batch_size = llm_config.get('batch_size', 5)
    max_batches = llm_config.get('max_batches', None)
    start_batch = llm_config.get('start_batch', 0)

    # Process reviews asynchronously with batch control
    results = asyncio.run(process_batch_async(
        df, prompt_engineer, rule_detector, llm_client, policy_types, logger,
        batch_size=batch_size, max_batches=max_batches, start_batch=start_batch
    ))
    
    # Convert results to DataFrame
    annotations_df = create_annotations_dataframe_extensible(results, policy_types)
    
    # Save results
    output_dir = Path(f"data/annotated/{dataset_slug}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "annotations.parquet"
    annotations_df.to_parquet(output_path, index=False)
    
    # Log summary statistics
    log_annotation_summary_extensible(annotations_df, logger, policy_types)
    
    logger.info(f"Annotation completed. Results saved to {output_path}")


async def process_batch_async(
    df: pd.DataFrame,
    prompt_engineer: JSONPromptEngineer,
    rule_detector: RuleBasedDetector,
    llm_client,
    policy_types: List[str],
    logger,
    batch_size=5,
    max_batches=None,
    start_batch=0
) -> List[ReviewAnnotation]:
    """Process reviews asynchronously with batching and batch control."""
    results = []
    total_batches = (len(df) + batch_size - 1) // batch_size
    if max_batches is not None:
        end_batch = min(start_batch + max_batches, total_batches)
    else:
        end_batch = total_batches

    async with aiohttp.ClientSession() as session:
        for batch_idx in range(start_batch, end_batch):
            i = batch_idx * batch_size
            batch = df.iloc[i:i+batch_size]
            logger.info(f"Processing batch {batch_idx + 1}/{end_batch}")

            # Process batch concurrently
            batch_tasks = [
                annotate_single_review_async(
                    row, prompt_engineer, rule_detector, llm_client, policy_types, session
                )
                for _, row in batch.iterrows()
            ]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
                    # Create fallback annotation
                    fallback_annotation = ReviewAnnotation(
                        review_id="error",
                        annotations={},
                        annotation_source="error_fallback",
                        processing_time=0.0,
                        llm_calls_made=0,
                        rule_detections_used=0,
                        json_parsing_success=False,
                        validation_status="error"
                    )
                    results.append(fallback_annotation)
                else:
                    results.append(result)

    return results


async def annotate_single_review_async(
    row: pd.Series,
    prompt_engineer: JSONPromptEngineer,
    rule_detector: RuleBasedDetector,
    llm_client,
    policy_types: List[str],
    session: aiohttp.ClientSession
) -> ReviewAnnotation:
    """Annotate a single review with all policy types using Chat Completions API."""
    
    start_time = time.time()
    review_text = row['text']
    business_name = row['place_name']
    review_id = row['review_id']
    
    llm_calls_made = 0
    rule_detections_used = 0
    json_parsing_success = True
    annotations = {}
    
    for policy_type in policy_types:
        # Rule-based preprocessing
        rule_result = rule_detector.detect_violation_rules(review_text, policy_type)
        
        # Count rule detections
        if rule_result[1] > 0.7:
            rule_detections_used += 1
        
        # LLM-based annotation (only if rule-based detection is uncertain)
        llm_result = await get_llm_annotation_async(
            prompt_engineer.build_prompt_json(policy_type, review_text, business_name),
            llm_client, rule_result, session
        )
        if llm_result:
            llm_calls_made += 1
        
        # Combine rule-based and LLM results
        annotation_result = combine_results_extensible(rule_result, llm_result, policy_type)
        annotations[policy_type] = annotation_result
    
    processing_time = time.time() - start_time
    
    return ReviewAnnotation(
        review_id=review_id,
        annotations=annotations,
        annotation_source="huggingface",
        processing_time=processing_time,
        llm_calls_made=llm_calls_made,
        rule_detections_used=rule_detections_used,
        json_parsing_success=json_parsing_success,
        validation_status="validated"
    )


async def get_llm_annotation_async(
    prompt: str,
    llm_client,
    rule_result: Tuple[bool, float, Optional[str], List[str]],
    session: aiohttp.ClientSession
) -> Optional[Dict[str, Any]]:
    """Get LLM annotation with Chat Completions API, skipping if rule-based detection is confident."""
    
    rule_is_violation, rule_confidence, rule_reason, rule_indicators = rule_result
    
    # Skip LLM if rule-based detection is very confident
    if rule_confidence > 0.9:
        return None
    
    try:
        return await llm_client.query_llm(prompt, session)
    except Exception as e:
        # Fall back to rule-based result if LLM fails
        return {
            "is_violation": rule_is_violation,
            "confidence": rule_confidence * 0.8,
            "reasoning": f"API failed, using rule: {rule_reason}",
            "violation_type": "fallback",
            "indicators": rule_indicators,
            "processing_metadata": {
                "model_used": "chat_completions_fallback",
                "prompt_version": "v2.0",
                "timestamp": datetime.now().isoformat()
            }
        }


def combine_results_extensible(
    rule_result: Tuple[bool, float, Optional[str], List[str]],
    llm_result: Optional[Dict[str, Any]],
    policy_type: str
) -> AnnotationResult:
    """Combine rule-based and LLM results intelligently with extensible policies."""
    
    rule_is_violation, rule_confidence, rule_reason, rule_indicators = rule_result
    
    if llm_result is None:
        # Use rule-based result only
        return AnnotationResult(
            is_violation=rule_is_violation,
            confidence=rule_confidence,
            reasoning=rule_reason or f"Rule-based detection for {policy_type}",
            rule_triggered=rule_reason,
            rule_confidence=rule_confidence,
            llm_confidence=None,
            indicators=rule_indicators or [],
            processing_metadata={
                "model_used": "rule_only",
                "prompt_version": "v2.0",
                "timestamp": datetime.now().isoformat()
            }
        )
    
    # Use LLM result as primary, with rule-based confidence as backup
    llm_is_violation = llm_result['is_violation']
    llm_confidence = llm_result['confidence']
    llm_reasoning = llm_result['reasoning']
    llm_indicators = llm_result.get('indicators', [])
    
    # Weighted combination (favor LLM for reasoning, rule for confidence)
    combined_confidence = (rule_confidence * 0.3 + llm_confidence * 0.7)
    
    return AnnotationResult(
        is_violation=llm_is_violation,
        confidence=combined_confidence,
        reasoning=llm_reasoning,
        rule_triggered=rule_reason if rule_confidence > 0.7 else None,
        rule_confidence=rule_confidence,
        llm_confidence=llm_confidence,
        indicators=llm_indicators,
        processing_metadata=llm_result.get('processing_metadata', {})
    )


def create_annotations_dataframe_extensible(results: List[ReviewAnnotation], policy_types: List[str]) -> pd.DataFrame:
    """Convert annotation results to DataFrame with extensible policy structure."""
    
    data = []
    for result in results:
        row_data = {
            'review_id': result.review_id,
            'annotation_source': result.annotation_source,
            'processing_time': result.processing_time,
            'llm_calls_made': result.llm_calls_made,
            'rule_detections_used': result.rule_detections_used,
            'json_parsing_success': result.json_parsing_success,
            'validation_status': result.validation_status
        }
        
        # Add annotations for each policy type
        for policy_type in policy_types:
            if policy_type in result.annotations:
                annotation = result.annotations[policy_type]
                row_data.update({
                    f'is_{policy_type}': annotation.is_violation,
                    f'{policy_type}_confidence': annotation.confidence,
                    f'{policy_type}_reasoning': annotation.reasoning,
                    f'{policy_type}_rule_triggered': annotation.rule_triggered,
                    f'{policy_type}_rule_confidence': annotation.rule_confidence,
                    f'{policy_type}_llm_confidence': annotation.llm_confidence,
                    f'{policy_type}_indicators': annotation.indicators,
                    f'{policy_type}_metadata': annotation.processing_metadata
                })
        
        data.append(row_data)
    
    return pd.DataFrame(data)


def log_annotation_summary_extensible(df: pd.DataFrame, logger, policy_types: List[str]):
    """Log comprehensive annotation statistics with extensible policy metrics."""
    
    total_reviews = len(df)
    
    logger.info("=== ADVANCED ANNOTATION SUMMARY ===")
    logger.info(f"Total reviews processed: {total_reviews}")
    logger.info(f"Policies applied: {policy_types}")
    
    # Violation counts and confidence for each policy
    for policy_type in policy_types:
        if f'is_{policy_type}' in df.columns:
            violations = df[f'is_{policy_type}'].sum()
            confidence_avg = df[f'{policy_type}_confidence'].mean()
            logger.info(f"{policy_type} violations: {violations} ({violations/total_reviews*100:.1f}%)")
            logger.info(f"{policy_type} average confidence: {confidence_avg:.3f}")
    
    # Processing statistics
    avg_llm_calls = df['llm_calls_made'].mean()
    avg_rule_detections = df['rule_detections_used'].mean()
    json_success_rate = df['json_parsing_success'].mean() * 100
    
    logger.info(f"Average LLM calls per review: {avg_llm_calls:.2f}")
    logger.info(f"Average rule detections per review: {avg_rule_detections:.2f}")
    logger.info(f"JSON parsing success rate: {json_success_rate:.1f}%")
    logger.info(f"Average processing time: {df['processing_time'].mean():.3f}s per review")
