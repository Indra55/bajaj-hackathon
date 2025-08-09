"""
Response Validator Service for Insurance Query System
Validates generated responses for accuracy, completeness, and relevance
"""

import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from app.services.query_processor import QueryAnalysis

@dataclass
class ValidationResult:
    """Result of response validation."""
    is_valid: bool
    confidence_score: float
    issues: List[str]
    suggestions: List[str]
    numerical_data_found: List[str]
    key_terms_found: List[str]

class ResponseValidator:
    """Validates insurance query responses for accuracy and completeness."""
    
    def __init__(self):
        # Validation patterns
        self.patterns = {
            'numerical_data': [
                r'\b\d+(?:\.\d+)?\s*%',  # Percentages
                r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?',  # Currency
                r'₹\s*\d+(?:,\d{2,3})*(?:\.\d{2})?',  # Indian Rupees
                r'\b\d+\s*(?:days?|months?|years?)',  # Time periods
                r'\b\d+(?:\.\d+)?\s*(?:times?|x|lakh|crore)',  # Multipliers/amounts
                r'\b\d+(?:st|nd|rd|th)\b',  # Ordinals
            ],
            'definitive_answers': [
                r'\b(?:yes|no|covered|not covered|excluded|included)\b',
                r'\b(?:eligible|not eligible|applicable|not applicable)\b',
                r'\b(?:available|not available|provided|not provided)\b'
            ],
            'conditional_language': [
                r'\b(?:subject to|provided|if|when|unless|except|condition|requirement)\b',
                r'\b(?:depending on|based on|in case of|upon|after)\b',
                r'\b(?:must|shall|should|required|necessary|mandatory)\b'
            ],
            'uncertainty_indicators': [
                r'\b(?:may|might|could|possibly|potentially|likely)\b',
                r'\b(?:unclear|ambiguous|not specified|not mentioned)\b',
                r'\b(?:contact|consult|refer to|check with)\b'
            ],
            'generic_responses': [
                r'\b(?:information not found|cannot determine|unable to find)\b',
                r'\b(?:please contact|refer to policy|consult with)\b',
                r'\b(?:not clear|not specified|varies|depends)\b'
            ]
        }
        
        # Insurance-specific key terms
        self.insurance_key_terms = {
            'coverage_terms': [
                'coverage', 'benefit', 'covered', 'includes', 'eligible',
                'protection', 'insurance', 'policy', 'plan'
            ],
            'exclusion_terms': [
                'exclusion', 'excluded', 'not covered', 'limitation', 'restriction',
                'prohibited', 'denied', 'except', 'excluding'
            ],
            'financial_terms': [
                'premium', 'deductible', 'copay', 'coinsurance', 'out-of-pocket',
                'limit', 'maximum', 'minimum', 'cost', 'amount', 'fee'
            ],
            'temporal_terms': [
                'waiting period', 'grace period', 'elimination period',
                'immediate', 'annual', 'lifetime', 'temporary', 'permanent'
            ],
            'condition_terms': [
                'pre-existing', 'condition', 'requirement', 'eligibility',
                'criteria', 'qualification', 'prerequisite'
            ]
        }

    def validate_response(self, response: str, question: str, 
                         query_analysis: QueryAnalysis, 
                         context_chunks: List[str]) -> ValidationResult:
        """Comprehensive validation of insurance query response."""
        
        issues = []
        suggestions = []
        confidence_score = 1.0
        
        # Extract numerical data and key terms
        numerical_data = self._extract_numerical_data(response)
        key_terms = self._extract_key_terms(response)
        
        # 1. Check for generic/unhelpful responses
        generic_score = self._check_generic_response(response)
        if generic_score > 0.3:
            issues.append('generic_response')
            suggestions.append('Provide more specific information from the policy')
            confidence_score -= 0.3
        
        # 2. Validate numerical data presence when expected
        if query_analysis.numerical_focus and not numerical_data:
            issues.append('missing_numerical_data')
            suggestions.append('Include specific amounts, percentages, or time periods')
            confidence_score -= 0.25
        
        # 3. Check for definitive answers when appropriate
        if self._requires_definitive_answer(question) and not self._has_definitive_answer(response):
            issues.append('missing_definitive_answer')
            suggestions.append('Provide a clear yes/no or covered/not covered answer')
            confidence_score -= 0.2
        
        # 4. Validate response length and detail
        word_count = len(response.split())
        if word_count < 15:
            issues.append('too_short')
            suggestions.append('Provide more detailed explanation')
            confidence_score -= 0.15
        elif word_count > 200:
            issues.append('too_verbose')
            suggestions.append('Make the response more concise')
            confidence_score -= 0.1
        
        # 5. Check for conditional language when appropriate
        if query_analysis.query_type in ['coverage', 'waiting_period']:
            if not self._has_conditional_language(response):
                issues.append('missing_conditions')
                suggestions.append('Include relevant conditions or limitations')
                confidence_score -= 0.15
        
        # 6. Validate context relevance
        context_relevance = self._check_context_relevance(response, context_chunks)
        if context_relevance < 0.3:
            issues.append('low_context_relevance')
            suggestions.append('Ensure answer is based on provided policy context')
            confidence_score -= 0.2
        
        # 7. Check for insurance domain appropriateness
        domain_score = self._check_domain_appropriateness(response, query_analysis)
        if domain_score < 0.4:
            issues.append('inappropriate_domain_language')
            suggestions.append('Use appropriate insurance terminology')
            confidence_score -= 0.1
        
        # 8. Validate answer completeness
        completeness_score = self._check_completeness(response, question, query_analysis)
        if completeness_score < 0.5:
            issues.append('incomplete_answer')
            suggestions.append('Address all aspects of the question')
            confidence_score -= 0.15
        
        # Ensure confidence score doesn't go below 0
        confidence_score = max(0.0, confidence_score)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence_score=confidence_score,
            issues=issues,
            suggestions=suggestions,
            numerical_data_found=numerical_data,
            key_terms_found=key_terms
        )

    def _extract_numerical_data(self, text: str) -> List[str]:
        """Extract numerical data from response."""
        numerical_data = []
        for pattern in self.patterns['numerical_data']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            numerical_data.extend(matches)
        return list(set(numerical_data))

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract insurance key terms from response."""
        key_terms = []
        text_lower = text.lower()
        
        for category, terms in self.insurance_key_terms.items():
            for term in terms:
                if term in text_lower:
                    key_terms.append(term)
        
        return list(set(key_terms))

    def _check_generic_response(self, response: str) -> float:
        """Check if response is generic or unhelpful."""
        response_lower = response.lower()
        generic_count = 0
        total_patterns = 0
        
        for pattern in self.patterns['generic_responses']:
            total_patterns += 1
            if re.search(pattern, response_lower):
                generic_count += 1
        
        return generic_count / total_patterns if total_patterns > 0 else 0.0

    def _requires_definitive_answer(self, question: str) -> bool:
        """Check if question requires a definitive yes/no answer."""
        question_lower = question.lower()
        definitive_indicators = [
            r'\b(?:does|is|will|can|are)\b.*\?',
            r'\b(?:covered|eligible|included|excluded)\b',
            r'\b(?:yes or no|true or false)\b'
        ]
        
        for pattern in definitive_indicators:
            if re.search(pattern, question_lower):
                return True
        return False

    def _has_definitive_answer(self, response: str) -> bool:
        """Check if response contains definitive answer."""
        for pattern in self.patterns['definitive_answers']:
            if re.search(pattern, response, re.IGNORECASE):
                return True
        return False

    def _has_conditional_language(self, response: str) -> bool:
        """Check if response includes conditional language."""
        for pattern in self.patterns['conditional_language']:
            if re.search(pattern, response, re.IGNORECASE):
                return True
        return False

    def _check_context_relevance(self, response: str, context_chunks: List[str]) -> float:
        """Check how well the response relates to the provided context."""
        if not context_chunks:
            return 0.5  # Neutral score if no context
        
        response_words = set(response.lower().split())
        context_words = set()
        
        for chunk in context_chunks:
            context_words.update(chunk.lower().split())
        
        if not context_words:
            return 0.5
        
        # Calculate overlap
        overlap = len(response_words.intersection(context_words))
        relevance_score = overlap / len(response_words) if response_words else 0.0
        
        return min(1.0, relevance_score * 2)  # Scale up to make it more meaningful

    def _check_domain_appropriateness(self, response: str, query_analysis: QueryAnalysis) -> float:
        """Check if response uses appropriate insurance domain language."""
        response_lower = response.lower()
        
        # Count relevant insurance terms
        relevant_terms = 0
        total_terms = 0
        
        for category, terms in self.insurance_key_terms.items():
            for term in terms:
                total_terms += 1
                if term in response_lower:
                    relevant_terms += 1
        
        if total_terms == 0:
            return 0.5
        
        base_score = relevant_terms / total_terms
        
        # Boost score if query type matches response content
        query_type = query_analysis.query_type
        if query_type == 'coverage' and any(term in response_lower for term in self.insurance_key_terms['coverage_terms']):
            base_score += 0.2
        elif query_type == 'exclusion' and any(term in response_lower for term in self.insurance_key_terms['exclusion_terms']):
            base_score += 0.2
        elif query_type == 'amount' and any(term in response_lower for term in self.insurance_key_terms['financial_terms']):
            base_score += 0.2
        
        return min(1.0, base_score)

    def _check_completeness(self, response: str, question: str, query_analysis: QueryAnalysis) -> float:
        """Check if response completely addresses the question."""
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common stop words for better matching
        stop_words = {'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                     'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
                     'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        
        question_words = question_words - stop_words
        response_words = response_words - stop_words
        
        if not question_words:
            return 0.5
        
        # Calculate how many question concepts are addressed
        addressed_concepts = len(question_words.intersection(response_words))
        completeness_score = addressed_concepts / len(question_words)
        
        # Boost score if entities from query analysis are mentioned
        if query_analysis.entities:
            entity_mentions = 0
            total_entities = 0
            
            for entity_list in query_analysis.entities.values():
                for entity in entity_list:
                    total_entities += 1
                    if entity.lower() in response.lower():
                        entity_mentions += 1
            
            if total_entities > 0:
                entity_score = entity_mentions / total_entities
                completeness_score = (completeness_score + entity_score) / 2
        
        return min(1.0, completeness_score)

    def suggest_improvements(self, validation_result: ValidationResult, 
                           query_analysis: QueryAnalysis) -> List[str]:
        """Generate specific improvement suggestions based on validation results."""
        improvements = []
        
        if 'missing_numerical_data' in validation_result.issues:
            if query_analysis.query_type == 'amount':
                improvements.append("Include specific dollar amounts, percentages, or limits mentioned in the policy")
            elif query_analysis.query_type == 'waiting_period':
                improvements.append("Specify the exact waiting period in days, months, or years")
        
        if 'missing_definitive_answer' in validation_result.issues:
            improvements.append("Start with a clear 'Yes' or 'No' answer, then provide supporting details")
        
        if 'missing_conditions' in validation_result.issues:
            improvements.append("Include any conditions, limitations, or requirements that apply")
        
        if 'generic_response' in validation_result.issues:
            improvements.append("Provide specific information from the policy document rather than generic statements")
        
        if 'too_short' in validation_result.issues:
            improvements.append("Expand the answer with more details from the policy context")
        
        if 'incomplete_answer' in validation_result.issues:
            improvements.append("Address all aspects of the question, including any sub-questions or implications")
        
        return improvements

    def get_validation_summary(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """Get a summary of validation results for debugging/monitoring."""
        return {
            'is_valid': validation_result.is_valid,
            'confidence_score': round(validation_result.confidence_score, 3),
            'issue_count': len(validation_result.issues),
            'issues': validation_result.issues,
            'numerical_data_count': len(validation_result.numerical_data_found),
            'key_terms_count': len(validation_result.key_terms_found),
            'has_numerical_data': len(validation_result.numerical_data_found) > 0,
            'has_key_terms': len(validation_result.key_terms_found) > 0
        }
