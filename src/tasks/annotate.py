"""Annotation task: rule-based and Gemini LLM-based policy violation detection."""

import re
import pandas as pd
import time
import yaml
import json
from datetime import datetime
from typing import Dict, Any

import google.generativeai as genai

from ..orchestrator import task


# --- Policy Detector Classes ---

class PolicyDetector:
    """Base class for policy detectors."""
    def description(self):
        return ""
    def few_shot_examples(self):
        return []

class NoAdvertisementPolicy(PolicyDetector):
    def description(self):
        return "Reviews should not contain promotional content, advertisements, or links to external websites."
    def detect(self, review_text: str):
        url_pattern = r"(https?://|www\.)"
        promo_words = ["visit", "discount", "promo", "deal", "offer", "sale"]
        url_found = bool(re.search(url_pattern, review_text, re.IGNORECASE))
        promo_count = sum(word in review_text.lower() for word in promo_words)
        if url_found and promo_count > 0:
            return True, 0.99, "Contains URL and promotional word (likely advertisement)"
        if promo_count >= 2:
            return True, 0.96, "Contains multiple promotional words"
        return False, 0.7, "No strong advertisement detected"
    def few_shot_examples(self):
        return [
            {
                "review": "Best pizza! Visit www.pizzapromo.com for discounts!",
                "response": {
                    "is_violation": True,
                    "confidence": 0.98,
                    "reasoning": "Contains promotional language and a URL.",
                    "violation_type": "advertisement",
                    "indicators": ["Visit", "www.pizzapromo.com", "discounts"]
                }
            },
            {
                "review": "Try our new burger deal at www.burgerpromo.com!",
                "response": {
                    "is_violation": True,
                    "confidence": 0.97,
                    "reasoning": "Promotional offer and website link.",
                    "violation_type": "advertisement",
                    "indicators": ["deal", "www.burgerpromo.com"]
                }
            },
            {
                "review": "Great food and service, will come again.",
                "response": {
                    "is_violation": False,
                    "confidence": 0.99,
                    "reasoning": "No promotional content or links.",
                    "violation_type": "none",
                    "indicators": ["food", "service"]
                }
            }
        ]

class NoIrrelevantContentPolicy(PolicyDetector):
    def description(self):
        return "Reviews must be about the location itself, not about unrelated events, trips, or other topics."
    def detect(self, review_text: str):
        unrelated_keywords = ["holiday", "trip", "vacation", "flight", "hotel"]
        unrelated_count = sum(word in review_text.lower() for word in unrelated_keywords)
        if unrelated_count >= 2:
            return True, 0.95, "Mentions multiple unrelated topics"
        if unrelated_count == 1 and len(review_text.split()) < 15:
            return True, 0.92, "Short review about unrelated topic"
        return False, 0.7, "Content appears relevant"
    def few_shot_examples(self):
        return [
            {
                "review": "I had a wonderful vacation in Hawaii.",
                "response": {
                    "is_violation": True,
                    "confidence": 0.94,
                    "reasoning": "Mentions vacation, not focused on restaurant experience.",
                    "violation_type": "irrelevant",
                    "indicators": ["vacation", "Hawaii"]
                }
            },
            {
                "review": "My flight to Paris was delayed, so I stopped by this place.",
                "response": {
                    "is_violation": True,
                    "confidence": 0.93,
                    "reasoning": "Review is mostly about flight, not the location.",
                    "violation_type": "irrelevant",
                    "indicators": ["flight", "Paris"]
                }
            },
            {
                "review": "The pasta was delicious and the staff were friendly.",
                "response": {
                    "is_violation": False,
                    "confidence": 0.99,
                    "reasoning": "Review is relevant to the restaurant.",
                    "violation_type": "none",
                    "indicators": ["pasta", "staff"]
                }
            }
        ]

class NoRantWithoutVisitPolicy(PolicyDetector):
    def description(self):
        return "Rants or complaints must come from actual visitors. Reviews based on hearsay or without evidence of a visit are violations."
    def detect(self, review_text: str):
        rant_phrases = [
            "never been here", "haven't visited", "I heard", "someone told me", "I read"
        ]
        visit_keywords = ["ate", "ordered", "visited", "went", "tried", "service", "food"]
        rant_found = any(phrase in review_text.lower() for phrase in rant_phrases)
        visit_found = any(word in review_text.lower() for word in visit_keywords)
        if rant_found and not visit_found:
            return True, 0.98, "Rant phrase present and no evidence of visit"
        return False, 0.7, "Likely visited or not a rant"
    def few_shot_examples(self):
        return [
            {
                "review": "Never been here, but I heard it's terrible.",
                "response": {
                    "is_violation": True,
                    "confidence": 0.97,
                    "reasoning": "Reviewer admits never visiting, but makes a complaint.",
                    "violation_type": "rant_without_visit",
                    "indicators": ["Never been here", "I heard"]
                }
            },
            {
                "review": "I haven't visited, but someone told me the service is bad.",
                "response": {
                    "is_violation": True,
                    "confidence": 0.96,
                    "reasoning": "No evidence of visit, only hearsay.",
                    "violation_type": "rant_without_visit",
                    "indicators": ["haven't visited", "someone told me"]
                }
            },
            {
                "review": "I tried the pizza and loved the atmosphere.",
                "response": {
                    "is_violation": False,
                    "confidence": 0.99,
                    "reasoning": "Reviewer describes their own experience.",
                    "violation_type": "none",
                    "indicators": ["tried", "pizza", "atmosphere"]
                }
            }
        ]

POLICY_REGISTRY = {
    "NoAdvertisement": NoAdvertisementPolicy(),
    "NoIrrelevantContent": NoIrrelevantContentPolicy(),
    "NoRantWithoutVisit": NoRantWithoutVisitPolicy(),
}

def build_prompt(policy, review_row, few_shot_examples, policy_description):
    review_info = "\n".join([
        f"Business Name: {review_row.get('place_name', '')}",
        f"User Name: {review_row.get('user_name', '')}",
        f"Text: {review_row.get('text', '')}",
        f"Rating: {review_row.get('rating', '')}",
        f"Language: {review_row.get('language', '')}",
        f"Has Image: {review_row.get('has_image', '')}",
        f"Image Path: {review_row.get('image_path', '')}",
        f"Metadata: {review_row.get('metadata', '')}"
    ])
    prompt = (
        f"You are an expert Google reviewer tasked with identifying policy violations in Google location reviews.\n"
        f"Policy: {policy}\n"
        f"Policy Description: {policy_description}\n"
        "Follow these steps:\n"
        "1. Carefully read the review and reason step by step about whether it violates the policy.\n"
        "2. If you suspect the review is AI-generated, err on the side of flagging a violation for this policy.\n"
        "3. If you are unsure, explain your reasoning and err on the side of not flagging a violation unless it is AI-generated.\n"
        "4. Respond ONLY with a valid JSON object, no markdown, no extra text.\n"
        "5. The JSON must have these fields: is_violation (true/false), confidence (float)\n"
        "Constraints:\n"
        "- Only output the JSON object, no code block markers or extra commentary.\n"
        "- Do not generate extra text that is not needed.\n"
        "- Confidence must be between 0.0 and 1.0.\n"
        "- Reasoning should be concise and reference evidence from the review.\n"
        "- indicators should be a list of keywords or phrases that triggered the decision.\n"
        "- Do not use double quotes inside reasoning or indicators unless they are properly escaped.\n"
        "- Prefer single quotes for quoted phrases inside reasoning."
    )
    prompt += f"\n\nPast Examples: "
    for ex in few_shot_examples:
        prompt += f"\nReview: {ex['review']}\nResponse: {json.dumps(ex['response'])}\n"
    prompt += f"\nReview Information:\n{review_info}\nResponse: "
    return prompt

def safe_json_extract(text):
    # Remove markdown code block markers
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    # Extract first JSON object
    json_match = re.search(r'\{.*?\}', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except Exception as e:
            print("JSON decode error:", e)
            print("Raw JSON string:", json_str)
            return {"is_violation": None, "confidence": 0.0}
    else:
        return {"is_violation": None, "confidence": 0.0}

# --- Main Annotation Task ---

@task(
    name="annotate",
    inputs=lambda p: [
        "configs/base.yaml",
        "data/raw/restaurant_reviews/reviews_text_only.parquet"
    ],
    outputs=lambda p: [
        "data/annotated/restaurant_reviews/annotations.parquet",
        "data/annotated/restaurant_reviews/annotations.csv"
    ],
)
def annotate(params: dict):
    """
    Apply Gemini LLMs to label relevancy and policy violations using few-shot prompting.
    Output: original row + is_{policy} columns (1/0 for each policy).
    """
    config = params.get("annotate")
    llm_cfg = config.get("llm", {})
    api_key = llm_cfg.get("gemini_api_key")
    model_name = llm_cfg.get("gemini_model", "gemini-2.5-flash-lite")
    final_conf_threshold = llm_cfg.get("final_conf_threshold", 0.9)
    startAt = llm_cfg.get("startAt", 0)
    rate_limit = llm_cfg.get("rate_limit", 1.0)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    reviews_df = pd.read_parquet("data/raw/restaurant_reviews/reviews_text_only.parquet")

    # Prepare columns for each policy
    for policy_name, detector in POLICY_REGISTRY.items():
        col_name = f"is_{policy_name.lower().replace(' ', '_')}"
        reviews_df[col_name] = 0  # Default to 0

    # Annotate each review for each policy, starting at startAt
    count = 0
    from datetime import datetime
    print(f"Start: {datetime.now().isoformat()}")
    for idx, row in reviews_df.iloc[startAt:].iterrows():
        print(f"Row {idx}")
        print(f"Current: {datetime.now().isoformat()}")
        for policy_name, detector in POLICY_REGISTRY.items():
            few_shot = detector.few_shot_examples()
            policy_description = detector.description()
            prompt = build_prompt(policy_name, row, few_shot, policy_description)
            response = model.generate_content(prompt)
            text = response.text.strip()
            llm_result = safe_json_extract(text)

            is_violation = bool(llm_result.get("is_violation", "False"))
            confidence = float(llm_result.get("confidence", "0.0"))
            col_name = f"is_{policy_name.lower().replace(' ', '_')}"
            score = is_violation and confidence >= final_conf_threshold
            reviews_df.at[idx, col_name] = int(score)
            print(f"Policy {col_name}, Result {int(score)}")
            print(f"Rate limiting by {rate_limit} seconds...")
            time.sleep(rate_limit)
        count += 1
    print(f"End: {datetime.now().isoformat()}")
    reviews_df.to_parquet("data/annotated/restaurant_reviews/annotations.parquet")
    reviews_df.to_csv("data/annotated/restaurant_reviews/annotations.csv")