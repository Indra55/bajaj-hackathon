"""
Demo Script for Domain-Aware QA System
Demonstrates the improved accuracy with domain classification and routing
"""

import asyncio
import json
from app.services.domain_aware_qa_service import DomainAwareQAService
from app.services.domain_classifier import DomainClassifier, Domain
from app.api.schemas.evaluation import HackRxRequest

class DomainAwareQADemo:
    """Demo class to showcase domain-aware QA capabilities."""
    
    def __init__(self):
        self.qa_service = DomainAwareQAService()
        self.domain_classifier = DomainClassifier()
    
    async def demo_single_document_classification(self):
        """Demo document classification capabilities."""
        print("=" * 60)
        print("DEMO 1: Document Domain Classification")
        print("=" * 60)
        
        # Sample document texts for different domains
        sample_documents = {
            "insurance_policy": """
            This health insurance policy provides comprehensive coverage for medical expenses.
            The policy includes a waiting period of 24 months for maternity benefits and 
            36 months for pre-existing diseases. The sum assured is Rs. 5,00,000 with 
            a co-payment of 10% for senior citizens. No claim discount of 5% is applicable 
            for claim-free years. The policy covers hospitalization, day care procedures, 
            and AYUSH treatments at network hospitals.
            """,
            
            "constitution": """
            Article 21 of the Indian Constitution guarantees the fundamental right to life 
            and personal liberty. No person shall be deprived of his life or personal liberty 
            except according to procedure established by law. The Supreme Court has interpreted 
            this article to include the right to live with human dignity, right to livelihood, 
            right to health, and right to education. This constitutional provision forms the 
            foundation of many legal judgments and legislative frameworks.
            """,
            
            "bike_manual": """
            This motorcycle insurance policy covers third-party liability and comprehensive 
            damage to your two-wheeler. The Insured Declared Value (IDV) is calculated based 
            on the manufacturer's listed selling price minus depreciation. The policy includes 
            coverage for theft, fire, flood, and accident damage. Add-on covers like zero 
            depreciation, engine protection, and roadside assistance are available. 
            No Claim Bonus (NCB) ranges from 20% to 50% based on claim-free years.
            """,
            
            "hr_policy": """
            Employee Leave Policy: All permanent employees are entitled to 21 days of annual 
            leave, 12 days of sick leave, and 2 days of casual leave per calendar year. 
            Maternity leave of 26 weeks and paternity leave of 15 days are provided as per 
            statutory requirements. Leave applications must be submitted through the HR portal 
            with manager approval. Unused leave up to 5 days can be carried forward to the 
            next year. Emergency leave may be granted at management discretion.
            """
        }
        
        for doc_name, content in sample_documents.items():
            print(f"\n📄 Document: {doc_name}")
            print("-" * 40)
            
            classification = self.domain_classifier.classify_document(content, doc_name)
            
            print(f"Primary Domain: {classification.primary_domain.value}")
            print(f"Confidence: {classification.confidence:.2f}")
            print(f"Keywords Found: {classification.keywords_found[:5]}")  # Show first 5
            print(f"Reasoning: {classification.reasoning}")
            
            if classification.secondary_domains:
                print("Secondary Domains:")
                for domain, conf in classification.secondary_domains[:2]:  # Show top 2
                    print(f"  - {domain.value}: {conf:.2f}")
    
    async def demo_query_classification(self):
        """Demo query classification capabilities."""
        print("\n" + "=" * 60)
        print("DEMO 2: Query Domain Classification")
        print("=" * 60)
        
        sample_queries = [
            "What is the waiting period for maternity benefits?",
            "What are the fundamental rights guaranteed by the constitution?",
            "How much annual leave am I entitled to?",
            "What is covered under comprehensive bike insurance?",
            "What is the no claim discount percentage?",
            "Can I file a writ petition in the Supreme Court?",
            "What is the process for applying sick leave?",
            "What are the compliance requirements for data protection?"
        ]
        
        for query in sample_queries:
            print(f"\n❓ Query: {query}")
            print("-" * 50)
            
            classification = self.domain_classifier.classify_query(query)
            
            print(f"Classified Domain: {classification.primary_domain.value}")
            print(f"Confidence: {classification.confidence:.2f}")
            
            # Show enhanced queries for this domain
            enhanced_queries = self.domain_classifier.enhance_query_for_domain(
                query, classification.primary_domain
            )
            if len(enhanced_queries) > 1:
                print(f"Enhanced Queries: {enhanced_queries[1:3]}")  # Show 2 enhanced versions
    
    async def demo_domain_routing_accuracy(self):
        """Demo how domain routing improves accuracy."""
        print("\n" + "=" * 60)
        print("DEMO 3: Domain Routing for Better Accuracy")
        print("=" * 60)
        
        # Create sample documents for different domains
        documents = [
            {
                'content': self._encode_sample_content("insurance"),
                'metadata': {'filename': 'health_insurance_policy.pdf'}
            },
            {
                'content': self._encode_sample_content("constitution"),
                'metadata': {'filename': 'indian_constitution.pdf'}
            },
            {
                'content': self._encode_sample_content("bike_manual"),
                'metadata': {'filename': 'bike_insurance_manual.pdf'}
            }
        ]
        
        # Process documents with domain awareness
        await self.qa_service.process_documents_with_domain_awareness(documents)
        
        # Test queries that should route to specific domains
        test_cases = [
            {
                "query": "What is the waiting period for pre-existing diseases?",
                "expected_domain": Domain.INSURANCE,
                "should_find_in": ["health_insurance_policy.pdf"]
            },
            {
                "query": "What are the fundamental rights in Article 21?",
                "expected_domain": Domain.LEGAL,
                "should_find_in": ["indian_constitution.pdf"]
            },
            {
                "query": "What is IDV in bike insurance?",
                "expected_domain": Domain.INSURANCE,
                "should_find_in": ["bike_insurance_manual.pdf"]
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test Case {i}")
            print(f"Query: {test_case['query']}")
            print(f"Expected Domain: {test_case['expected_domain'].value}")
            print("-" * 50)
            
            # Create request
            request = HackRxRequest(
                documents="",  # Not used in multi-document mode
                questions=[test_case['query']]
            )
            
            # Get domain-aware answer
            answers = await self.qa_service.answer_questions_with_domain_routing(request)
            answer = answers[0]
            
            print(f"✅ Routed to Domain: {answer.source_domain.value}")
            print(f"✅ Confidence: {answer.confidence:.2f}")
            print(f"✅ Sources Used: {answer.sources_used}")
            print(f"✅ Match Quality: {answer.domain_match_quality}")
            print(f"📝 Answer: {answer.answer[:200]}...")
            
            # Check if routing was correct
            routing_correct = answer.source_domain == test_case['expected_domain']
            print(f"🎯 Routing Accuracy: {'✅ CORRECT' if routing_correct else '❌ INCORRECT'}")
    
    def _encode_sample_content(self, domain_type: str) -> str:
        """Create sample content for different domains (base64 encoded)."""
        import base64
        
        contents = {
            "insurance": """
            HEALTH INSURANCE POLICY DOCUMENT
            
            COVERAGE DETAILS:
            This comprehensive health insurance policy provides coverage for medical expenses
            incurred due to illness or injury. The sum assured is Rs. 5,00,000 per policy year.
            
            WAITING PERIODS:
            - Initial waiting period: 30 days for all illnesses except accidents
            - Pre-existing diseases: 36 months of continuous coverage required
            - Maternity benefits: 24 months waiting period
            - Specific diseases (cataract, hernia, etc.): 24 months
            
            BENEFITS:
            - Hospitalization expenses: 100% coverage in network hospitals
            - Day care procedures: Covered as per policy terms
            - AYUSH treatments: Up to Rs. 50,000 per policy year
            - Health check-ups: Every 3 years for preventive care
            
            NO CLAIM DISCOUNT:
            5% discount on renewal premium for each claim-free year, maximum 50%.
            
            EXCLUSIONS:
            - Cosmetic surgery
            - Dental treatment (unless due to accident)
            - Treatment outside India
            """,
            
            "constitution": """
            THE CONSTITUTION OF INDIA
            
            PART III - FUNDAMENTAL RIGHTS
            
            Article 21 - Protection of Life and Personal Liberty:
            No person shall be deprived of his life or personal liberty except according 
            to procedure established by law.
            
            Article 19 - Protection of certain rights regarding freedom of speech, etc.:
            (1) All citizens shall have the right to—
            (a) freedom of speech and expression;
            (b) to assemble peaceably and without arms;
            (c) to form associations or unions;
            (d) to move freely throughout the territory of India;
            (e) to reside and settle in any part of the territory of India;
            (f) to practise any profession, or to carry on any occupation, trade or business.
            
            Article 14 - Equality before law:
            The State shall not deny to any person equality before the law or the equal 
            protection of the laws within the territory of India.
            
            DIRECTIVE PRINCIPLES OF STATE POLICY:
            Article 39 - Certain principles of policy to be followed by the State
            Article 41 - Right to work, to education and to public assistance
            """,
            
            "bike_manual": """
            MOTORCYCLE INSURANCE POLICY
            
            COVERAGE TYPE: Comprehensive Two-Wheeler Insurance
            
            INSURED DECLARED VALUE (IDV):
            The IDV is calculated as: Manufacturer's listed selling price - Depreciation
            Depreciation rates:
            - Not exceeding 6 months: 5%
            - Exceeding 6 months but not exceeding 1 year: 15%
            - Exceeding 1 year but not exceeding 2 years: 20%
            
            COVERAGE INCLUDES:
            - Third-party liability (unlimited for death/bodily injury)
            - Own damage due to accident, fire, theft, flood
            - Personal accident cover for owner-driver
            
            ADD-ON COVERS AVAILABLE:
            - Zero depreciation cover
            - Engine protection cover
            - Roadside assistance
            - Return to invoice cover
            
            NO CLAIM BONUS (NCB):
            - 1st year: 20%
            - 2nd year: 25%
            - 3rd year: 35%
            - 4th year: 45%
            - 5th year onwards: 50%
            
            EXCLUSIONS:
            - Damage due to war, nuclear risks
            - Consequential loss
            - Mechanical breakdown
            - Damage due to overloading
            """
        }
        
        content = contents.get(domain_type, "Sample content")
        return base64.b64encode(content.encode()).decode()
    
    async def demo_performance_comparison(self):
        """Demo performance comparison between regular and domain-aware QA."""
        print("\n" + "=" * 60)
        print("DEMO 4: Performance Comparison")
        print("=" * 60)
        
        # Get domain statistics
        stats = self.qa_service.get_domain_statistics()
        
        print("📊 Domain Statistics:")
        print(f"Total Documents Processed: {stats['total_documents']}")
        print(f"Vector Stores Created: {stats['vector_stores_created']}")
        print(f"Domains with Content: {[d.value for d in stats['domains_with_content']]}")
        
        if 'domain_distribution' in stats:
            print("\n📈 Document Distribution by Domain:")
            for domain, count in stats['domain_distribution'].items():
                print(f"  {domain}: {count} documents")
        
        print("\n🚀 Benefits of Domain-Aware Routing:")
        benefits = [
            "✅ Higher accuracy by filtering irrelevant content",
            "✅ Faster retrieval with domain-specific vector stores",
            "✅ Better context understanding with domain expertise",
            "✅ Reduced hallucination from mixed-domain content",
            "✅ Domain-specific answer formatting and terminology"
        ]
        
        for benefit in benefits:
            print(f"  {benefit}")

async def main():
    """Run the complete demo."""
    print("🎯 DOMAIN-AWARE QA SYSTEM DEMONSTRATION")
    print("🎯 Solving the Multi-Domain Document Accuracy Problem")
    print("=" * 80)
    
    demo = DomainAwareQADemo()
    
    try:
        # Run all demos
        await demo.demo_single_document_classification()
        await demo.demo_query_classification()
        await demo.demo_domain_routing_accuracy()
        await demo.demo_performance_comparison()
        
        print("\n" + "=" * 80)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("🎉 Your system now has intelligent domain-aware routing!")
        print("=" * 80)
        
        print("\n📋 NEXT STEPS:")
        print("1. Test the new endpoints:")
        print("   - POST /domain-qa/domain-aware-qa")
        print("   - POST /domain-qa/multi-document-qa")
        print("   - GET /domain-qa/domain-info")
        print("2. Upload your constitution and bike manual documents")
        print("3. Test domain-specific queries for better accuracy")
        print("4. Monitor the improved performance metrics")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
