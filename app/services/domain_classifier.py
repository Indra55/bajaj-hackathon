"""
Domain Classification Service for Intelligent Query-Retrieval System
Classifies documents and queries into specific domains (Insurance, Legal, HR, Compliance)
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class Domain(Enum):
    INSURANCE = "insurance"
    LEGAL = "legal"
    HR = "hr"
    COMPLIANCE = "compliance"
    GENERAL = "general"

@dataclass
class DomainClassification:
    """Result of domain classification."""
    primary_domain: Domain
    confidence: float
    secondary_domains: List[Tuple[Domain, float]]
    keywords_found: List[str]
    reasoning: str

class DomainClassifier:
    """Intelligent domain classifier for documents and queries."""
    
    def __init__(self):
        # Domain-specific keywords and patterns
        self.domain_keywords = {
            Domain.INSURANCE: {
                'primary': [
                    'insurance', 'policy', 'premium', 'coverage', 'claim', 'deductible',
                    'beneficiary', 'underwriting', 'actuarial', 'risk assessment',
                    'life insurance', 'health insurance', 'auto insurance', 'property insurance',
                    'liability', 'indemnity', 'reinsurance', 'policyholder', 'insured',
                    'waiting period', 'grace period', 'maternity', 'pre-existing',
                    'no claim discount', 'sum assured', 'rider', 'exclusion',
                    'cashless', 'network hospital', 'co-payment', 'sub-limit',
                    'bike insurance', 'vehicle insurance', 'third party', 'comprehensive',
                    'IDV', 'depreciation', 'NCB', 'add-on cover'
                ],
                'secondary': [
                    'medical', 'accident', 'disability', 'death', 'hospitalization',
                    'surgery', 'treatment', 'diagnosis', 'prescription', 'therapy',
                    'vehicle', 'automobile', 'motorcycle', 'car', 'truck', 'damage',
                    'repair', 'replacement', 'theft', 'fire', 'flood', 'natural disaster'
                ]
            },
            Domain.LEGAL: {
                'primary': [
                    'constitution', 'law', 'legal', 'court', 'judge', 'attorney', 'lawyer',
                    'litigation', 'contract', 'agreement', 'statute', 'regulation',
                    'amendment', 'article', 'section', 'clause', 'provision',
                    'fundamental rights', 'directive principles', 'judiciary',
                    'legislature', 'executive', 'parliament', 'supreme court',
                    'high court', 'magistrate', 'tribunal', 'jurisdiction',
                    'constitutional', 'legal framework', 'jurisprudence', 'precedent',
                    'civil law', 'criminal law', 'administrative law', 'tort',
                    'breach', 'liability', 'damages', 'injunction', 'writ'
                ],
                'secondary': [
                    'rights', 'duties', 'obligations', 'enforcement', 'penalty',
                    'punishment', 'fine', 'imprisonment', 'bail', 'appeal',
                    'evidence', 'witness', 'testimony', 'hearing', 'trial'
                ]
            },
            Domain.HR: {
                'primary': [
                    'employee', 'employment', 'human resources', 'HR', 'personnel',
                    'recruitment', 'hiring', 'onboarding', 'training', 'development',
                    'performance', 'appraisal', 'promotion', 'salary', 'compensation',
                    'benefits', 'leave', 'vacation', 'sick leave', 'maternity leave',
                    'resignation', 'termination', 'disciplinary', 'grievance',
                    'workplace', 'office', 'team', 'manager', 'supervisor',
                    'job description', 'role', 'responsibility', 'KPI', 'target'
                ],
                'secondary': [
                    'policy', 'procedure', 'guideline', 'handbook', 'manual',
                    'code of conduct', 'ethics', 'diversity', 'inclusion',
                    'harassment', 'discrimination', 'safety', 'health'
                ]
            },
            Domain.COMPLIANCE: {
                'primary': [
                    'compliance', 'regulatory', 'audit', 'governance', 'risk management',
                    'internal control', 'SOX', 'GDPR', 'privacy', 'data protection',
                    'anti-money laundering', 'AML', 'KYC', 'know your customer',
                    'due diligence', 'regulatory reporting', 'disclosure',
                    'whistleblower', 'ethics', 'code of conduct', 'policy violation',
                    'investigation', 'remediation', 'corrective action',
                    'regulatory body', 'regulator', 'inspection', 'examination'
                ],
                'secondary': [
                    'procedure', 'framework', 'standard', 'guideline', 'requirement',
                    'obligation', 'monitoring', 'assessment', 'evaluation',
                    'documentation', 'record keeping', 'training', 'awareness'
                ]
            }
        }
        
        # Document type patterns
        self.document_patterns = {
            Domain.INSURANCE: [
                r'insurance\s+policy', r'policy\s+document', r'coverage\s+details',
                r'premium\s+schedule', r'claim\s+form', r'benefit\s+summary',
                r'bike\s+insurance', r'vehicle\s+insurance', r'motor\s+insurance',
                r'health\s+insurance', r'life\s+insurance', r'travel\s+insurance'
            ],
            Domain.LEGAL: [
                r'constitution', r'legal\s+document', r'court\s+order',
                r'judgment', r'legal\s+opinion', r'statute', r'act\s+of',
                r'amendment', r'legal\s+framework', r'law\s+book',
                r'constitutional\s+provision', r'legal\s+precedent'
            ],
            Domain.HR: [
                r'employee\s+handbook', r'HR\s+policy', r'job\s+description',
                r'employment\s+contract', r'performance\s+review',
                r'training\s+manual', r'leave\s+policy', r'compensation\s+guide'
            ],
            Domain.COMPLIANCE: [
                r'compliance\s+manual', r'regulatory\s+guide', r'audit\s+report',
                r'risk\s+assessment', r'governance\s+framework', r'SOX\s+documentation',
                r'privacy\s+policy', r'data\s+protection', r'AML\s+policy'
            ]
        }

    def classify_document(self, text: str, filename: str = "") -> DomainClassification:
        """Classify a document into a domain."""
        text_lower = text.lower()
        filename_lower = filename.lower()
        combined_text = f"{filename_lower} {text_lower}"
        
        domain_scores = {}
        keywords_found = {}
        
        # Score based on keywords
        for domain, keyword_sets in self.domain_keywords.items():
            score = 0
            found_keywords = []
            
            # Primary keywords (higher weight)
            for keyword in keyword_sets['primary']:
                count = combined_text.count(keyword.lower())
                if count > 0:
                    score += count * 3  # Higher weight for primary keywords
                    found_keywords.append(keyword)
            
            # Secondary keywords (lower weight)
            for keyword in keyword_sets['secondary']:
                count = combined_text.count(keyword.lower())
                if count > 0:
                    score += count * 1  # Lower weight for secondary keywords
                    found_keywords.append(keyword)
            
            domain_scores[domain] = score
            keywords_found[domain] = found_keywords
        
        # Score based on document patterns
        for domain, patterns in self.document_patterns.items():
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    domain_scores[domain] += 5  # Bonus for matching document patterns
        
        # Determine primary domain
        if not any(domain_scores.values()):
            return DomainClassification(
                primary_domain=Domain.GENERAL,
                confidence=0.1,
                secondary_domains=[],
                keywords_found=[],
                reasoning="No domain-specific keywords found"
            )
        
        # Sort domains by score
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        primary_domain, primary_score = sorted_domains[0]
        
        # Calculate confidence
        total_score = sum(domain_scores.values())
        confidence = primary_score / total_score if total_score > 0 else 0
        
        # Get secondary domains
        secondary_domains = [(domain, score/total_score) for domain, score in sorted_domains[1:3] if score > 0]
        
        return DomainClassification(
            primary_domain=primary_domain,
            confidence=confidence,
            secondary_domains=secondary_domains,
            keywords_found=keywords_found[primary_domain],
            reasoning=f"Found {len(keywords_found[primary_domain])} domain-specific keywords"
        )

    def classify_query(self, query: str) -> DomainClassification:
        """Classify a query into a domain."""
        return self.classify_document(query, "")

    def is_query_domain_match(self, query_domain: Domain, document_domains: List[Domain], 
                            threshold: float = 0.3) -> bool:
        """Check if query domain matches any document domain."""
        if query_domain == Domain.GENERAL:
            return True  # General queries can search all domains
        
        return query_domain in document_domains

    def get_domain_specific_keywords(self, domain: Domain) -> List[str]:
        """Get keywords for a specific domain."""
        if domain in self.domain_keywords:
            return self.domain_keywords[domain]['primary'] + self.domain_keywords[domain]['secondary']
        return []

    def enhance_query_for_domain(self, query: str, domain: Domain) -> List[str]:
        """Enhance query with domain-specific terms."""
        enhanced_queries = [query]
        
        if domain == Domain.INSURANCE:
            # Add insurance-specific query variations
            if 'period' in query.lower():
                enhanced_queries.extend([
                    query + " insurance policy",
                    query + " coverage terms",
                    query + " policy conditions"
                ])
            if 'cover' in query.lower():
                enhanced_queries.extend([
                    query + " insurance benefits",
                    query + " policy coverage",
                    query + " claim eligibility"
                ])
        
        elif domain == Domain.LEGAL:
            # Add legal-specific query variations
            enhanced_queries.extend([
                query + " constitutional provision",
                query + " legal framework",
                query + " statutory requirement"
            ])
        
        elif domain == Domain.HR:
            # Add HR-specific query variations
            enhanced_queries.extend([
                query + " employee policy",
                query + " HR guidelines",
                query + " workplace rules"
            ])
        
        elif domain == Domain.COMPLIANCE:
            # Add compliance-specific query variations
            enhanced_queries.extend([
                query + " regulatory requirement",
                query + " compliance policy",
                query + " governance framework"
            ])
        
        return list(set(enhanced_queries))  # Remove duplicates
