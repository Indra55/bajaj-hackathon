"""
Enhanced Query Processor for Insurance Document Analysis
Handles query preprocessing, entity extraction, and query expansion
"""

import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import asyncio

@dataclass
class QueryAnalysis:
    """Structured analysis of an insurance query."""
    original_query: str
    query_type: str  # 'coverage', 'waiting_period', 'amount', 'definition', 'exclusion'
    entities: Dict[str, Any]
    expanded_queries: List[str]
    search_keywords: List[str]
    numerical_focus: bool
    priority_sections: List[str]

class EnhancedQueryProcessor:
    """Enhanced query processor with insurance domain expertise."""
    
    def __init__(self):
        # Query type patterns
        self.query_patterns = {
            'coverage': [
                r'\b(?:cover|covered|coverage|benefit|include|eligible)\b',
                r'\b(?:does.*cover|is.*covered|what.*covered)\b',
                r'\b(?:policy.*cover|insurance.*cover)\b'
            ],
            'waiting_period': [
                r'\b(?:waiting period|wait.*period|elimination period)\b',
                r'\b(?:how long.*wait|when.*eligible|after.*months)\b',
                r'\b(?:immediate.*coverage|grace period)\b'
            ],
            'amount': [
                r'\b(?:how much|amount|cost|price|premium|deductible)\b',
                r'\b(?:percentage|percent|%|limit|maximum|minimum)\b',
                r'\b(?:\$|dollar|rupee|INR|USD)\b'
            ],
            'exclusion': [
                r'\b(?:exclusion|excluded|not covered|limitation)\b',
                r'\b(?:does not cover|will not.*cover|except)\b',
                r'\b(?:restriction|prohibited|denied)\b'
            ],
            'definition': [
                r'\b(?:what is|define|definition|meaning|means)\b',
                r'\b(?:explain|clarify|interpret)\b'
            ]
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            'medical_condition': [
                r'\b(?:diabetes|cancer|heart|cardiac|mental|psychiatric|maternity|pregnancy)\b',
                r'\b(?:surgery|operation|treatment|therapy|medication|prescription)\b',
                r'\b(?:emergency|urgent|diagnostic|preventive|routine)\b'
            ],
            'body_part': [
                r'\b(?:eye|dental|vision|hearing|orthopedic|neurological)\b',
                r'\b(?:skin|bone|joint|muscle|organ)\b'
            ],
            'time_period': [
                r'\b\d+\s*(?:day|week|month|year)s?\b',
                r'\b(?:immediate|instant|annual|lifetime|temporary|permanent)\b'
            ],
            'financial': [
                r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?',
                r'\b\d+(?:\.\d+)?\s*(?:percent|%|lakh|crore)\b',
                r'\b(?:premium|deductible|copay|coinsurance|out-of-pocket)\b'
            ],
            'demographics': [
                r'\b(?:age|gender|male|female|senior|child|adult)\b',
                r'\b\d+\s*(?:year|yr)s?\s*old\b'
            ]
        }
        
        # Synonym mappings for query expansion
        self.synonym_mappings = {
            'coverage': ['benefit', 'protection', 'insurance', 'policy coverage'],
            'waiting period': ['elimination period', 'qualifying period', 'grace period'],
            'pre-existing': ['pre existing', 'preexisting', 'prior condition', 'existing condition'],
            'maternity': ['pregnancy', 'childbirth', 'prenatal', 'obstetric'],
            'mental health': ['psychiatric', 'psychological', 'behavioral health', 'counseling'],
            'prescription': ['medication', 'drug', 'pharmaceutical', 'medicine'],
            'emergency': ['urgent care', 'emergency room', 'ER', 'critical care'],
            'surgery': ['surgical procedure', 'operation', 'surgical treatment'],
            'diagnostic': ['testing', 'examination', 'screening', 'lab work'],
            'exclusion': ['not covered', 'excluded', 'limitation', 'restriction'],
            'deductible': ['out of pocket', 'copay', 'cost sharing', 'patient responsibility']
        }
        
        # Priority section mappings
        self.section_priority = {
            'coverage': ['coverage', 'definition'],
            'waiting_period': ['condition', 'coverage'],
            'amount': ['financial', 'coverage'],
            'exclusion': ['exclusion', 'condition'],
            'definition': ['definition', 'general']
        }

    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Comprehensive analysis of an insurance query."""
        
        # 1. Determine query type
        query_type = self._classify_query_type(query)
        
        # 2. Extract entities
        entities = self._extract_entities(query)
        
        # 3. Generate expanded queries
        expanded_queries = self._generate_expanded_queries(query, query_type)
        
        # 4. Extract search keywords
        search_keywords = self._extract_search_keywords(query, entities)
        
        # 5. Determine if numerical focus is needed
        numerical_focus = self._has_numerical_focus(query)
        
        # 6. Determine priority sections
        priority_sections = self.section_priority.get(query_type, ['general'])
        
        return QueryAnalysis(
            original_query=query,
            query_type=query_type,
            entities=entities,
            expanded_queries=expanded_queries,
            search_keywords=search_keywords,
            numerical_focus=numerical_focus,
            priority_sections=priority_sections
        )

    def _classify_query_type(self, query: str) -> str:
        """Classify the type of insurance query."""
        query_lower = query.lower()
        scores = {}
        
        for query_type, patterns in self.query_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower, re.IGNORECASE))
                score += matches
            scores[query_type] = score
        
        # Return the type with highest score, default to 'coverage'
        if max(scores.values()) == 0:
            return 'coverage'
        
        return max(scores, key=scores.get)

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract structured entities from the query."""
        entities = {}
        query_lower = query.lower()
        
        for entity_type, patterns in self.entity_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, query_lower, re.IGNORECASE)
                matches.extend(found)
            
            if matches:
                entities[entity_type] = list(set(matches))  # Remove duplicates
        
        return entities

    def _generate_expanded_queries(self, query: str, query_type: str) -> List[str]:
        """Generate multiple variations of the query for better retrieval."""
        expanded = [query]  # Original query
        query_lower = query.lower()
        
        # Add synonym-based expansions
        for term, synonyms in self.synonym_mappings.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded_query = query_lower.replace(term, synonym)
                    expanded.append(expanded_query)
        
        # Add query type specific expansions
        if query_type == 'coverage':
            expanded.extend([
                f"does the policy cover {query.lower()}",
                f"is {query.lower()} covered under the policy",
                f"coverage for {query.lower()}"
            ])
        elif query_type == 'waiting_period':
            expanded.extend([
                f"waiting period for {query.lower()}",
                f"how long to wait for {query.lower()}",
                f"when is {query.lower()} eligible"
            ])
        elif query_type == 'exclusion':
            expanded.extend([
                f"exclusions for {query.lower()}",
                f"what is not covered for {query.lower()}",
                f"limitations on {query.lower()}"
            ])
        
        # Remove duplicates and return unique expansions
        return list(set(expanded))

    def _extract_search_keywords(self, query: str, entities: Dict[str, List[str]]) -> List[str]:
        """Extract key search terms from query and entities."""
        keywords = []
        
        # Extract important words from query (excluding stop words)
        stop_words = {
            'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'this', 'that', 'these', 'those'
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords.extend([word for word in words if word not in stop_words and len(word) > 2])
        
        # Add entity values as keywords
        for entity_list in entities.values():
            keywords.extend(entity_list)
        
        # Add insurance-specific important terms if present
        insurance_terms = [
            'coverage', 'benefit', 'exclusion', 'limitation', 'condition',
            'premium', 'deductible', 'copay', 'waiting period', 'grace period',
            'pre-existing', 'maternity', 'emergency', 'surgery', 'prescription'
        ]
        
        query_lower = query.lower()
        for term in insurance_terms:
            if term in query_lower:
                keywords.append(term)
        
        return list(set(keywords))  # Remove duplicates

    def _has_numerical_focus(self, query: str) -> bool:
        """Determine if the query is asking for numerical information."""
        numerical_indicators = [
            r'\b(?:how much|amount|cost|price|percentage|percent|%)\b',
            r'\b(?:limit|maximum|minimum|up to|at least)\b',
            r'\b(?:\$|dollar|rupee|INR|USD)\b',
            r'\b\d+\b'  # Any number
        ]
        
        query_lower = query.lower()
        for pattern in numerical_indicators:
            if re.search(pattern, query_lower):
                return True
        
        return False

    def generate_search_variations(self, query_analysis: QueryAnalysis) -> List[Dict[str, Any]]:
        """Generate multiple search variations with different strategies."""
        variations = []
        
        # 1. Original query
        variations.append({
            'query': query_analysis.original_query,
            'strategy': 'original',
            'weight': 1.0
        })
        
        # 2. Keyword-focused search
        keyword_query = ' '.join(query_analysis.search_keywords[:5])  # Top 5 keywords
        variations.append({
            'query': keyword_query,
            'strategy': 'keywords',
            'weight': 0.8
        })
        
        # 3. Entity-focused search
        if query_analysis.entities:
            entity_terms = []
            for entity_list in query_analysis.entities.values():
                entity_terms.extend(entity_list[:2])  # Top 2 from each category
            
            if entity_terms:
                variations.append({
                    'query': ' '.join(entity_terms),
                    'strategy': 'entities',
                    'weight': 0.7
                })
        
        # 4. Expanded queries (top 2)
        for i, expanded in enumerate(query_analysis.expanded_queries[1:3]):  # Skip original
            variations.append({
                'query': expanded,
                'strategy': f'expanded_{i+1}',
                'weight': 0.6 - (i * 0.1)
            })
        
        # 5. Type-specific search
        type_specific = self._generate_type_specific_query(query_analysis)
        if type_specific:
            variations.append({
                'query': type_specific,
                'strategy': 'type_specific',
                'weight': 0.9
            })
        
        return variations

    def _generate_type_specific_query(self, query_analysis: QueryAnalysis) -> str:
        """Generate a query specific to the identified type."""
        query_type = query_analysis.query_type
        entities = query_analysis.entities
        
        if query_type == 'coverage' and 'medical_condition' in entities:
            conditions = ' '.join(entities['medical_condition'][:2])
            return f"coverage benefits for {conditions}"
        
        elif query_type == 'waiting_period' and 'medical_condition' in entities:
            conditions = ' '.join(entities['medical_condition'][:2])
            return f"waiting period elimination period for {conditions}"
        
        elif query_type == 'exclusion' and 'medical_condition' in entities:
            conditions = ' '.join(entities['medical_condition'][:2])
            return f"exclusions limitations not covered {conditions}"
        
        elif query_type == 'amount' and 'financial' in entities:
            financial_terms = ' '.join(entities['financial'][:2])
            return f"amount cost limit {financial_terms}"
        
        return ""

    def should_use_numerical_boost(self, query_analysis: QueryAnalysis) -> bool:
        """Determine if numerical data should be boosted in search results."""
        return (query_analysis.numerical_focus or 
                query_analysis.query_type in ['amount', 'waiting_period'] or
                'financial' in query_analysis.entities)

    def get_result_filtering_criteria(self, query_analysis: QueryAnalysis) -> Dict[str, Any]:
        """Get criteria for filtering and ranking search results."""
        return {
            'priority_sections': query_analysis.priority_sections,
            'required_entities': query_analysis.entities,
            'numerical_focus': query_analysis.numerical_focus,
            'query_type': query_analysis.query_type,
            'boost_keywords': query_analysis.search_keywords[:10]
        }
