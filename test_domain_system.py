"""
Quick Test Script for Domain-Aware QA System
Tests the core functionality without requiring full API setup
"""

import asyncio
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

async def test_domain_classification():
    """Test basic domain classification functionality."""
    print("🧪 Testing Domain Classification...")
    
    try:
        from app.services.domain_classifier import DomainClassifier, Domain
        
        classifier = DomainClassifier()
        
        # Test documents
        test_docs = {
            "Insurance Policy": "This health insurance policy covers medical expenses with a waiting period of 24 months for maternity benefits and 36 months for pre-existing diseases. No claim discount applies.",
            "Constitution": "Article 21 of the Indian Constitution guarantees fundamental rights to life and personal liberty. The Supreme Court has interpreted this provision extensively.",
            "HR Policy": "Employee leave policy states that all permanent employees are entitled to 21 days annual leave and 12 days sick leave per calendar year.",
            "Bike Manual": "This motorcycle insurance policy covers third-party liability and comprehensive damage. IDV is calculated based on depreciation rates."
        }
        
        print("\n📋 Document Classification Results:")
        print("-" * 50)
        
        for doc_name, content in test_docs.items():
            classification = classifier.classify_document(content, doc_name.lower())
            print(f"📄 {doc_name}")
            print(f"   Domain: {classification.primary_domain.value}")
            print(f"   Confidence: {classification.confidence:.2f}")
            print(f"   Keywords: {classification.keywords_found[:3]}")
            print()
        
        print("✅ Domain Classification Test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Domain Classification Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_query_classification():
    """Test query classification functionality."""
    print("\n🧪 Testing Query Classification...")
    
    try:
        from app.services.domain_classifier import DomainClassifier
        
        classifier = DomainClassifier()
        
        test_queries = [
            "What is the waiting period for maternity benefits?",
            "What are the fundamental rights in Article 21?",
            "How many days of annual leave am I entitled to?",
            "What is covered under comprehensive bike insurance?",
            "What is the no claim discount percentage?"
        ]
        
        print("\n❓ Query Classification Results:")
        print("-" * 50)
        
        for query in test_queries:
            classification = classifier.classify_query(query)
            print(f"❓ {query}")
            print(f"   Domain: {classification.primary_domain.value}")
            print(f"   Confidence: {classification.confidence:.2f}")
            print()
        
        print("✅ Query Classification Test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Query Classification Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_document_processing():
    """Test enhanced document processing."""
    print("\n🧪 Testing Enhanced Document Processing...")
    
    try:
        from app.services.enhanced_document_processor import EnhancedDocumentProcessor
        import base64
        
        processor = EnhancedDocumentProcessor()
        
        # Create a sample document
        sample_content = """
        HEALTH INSURANCE POLICY
        
        This comprehensive health insurance policy provides coverage for medical expenses.
        
        WAITING PERIODS:
        - Pre-existing diseases: 36 months
        - Maternity benefits: 24 months
        - Specific diseases: 24 months
        
        BENEFITS:
        - Sum assured: Rs. 5,00,000
        - No claim discount: 5% per claim-free year
        - Network hospitals: Cashless treatment
        """
        
        # Encode content
        encoded_content = base64.b64encode(sample_content.encode()).decode()
        
        document = {
            'content': encoded_content,
            'metadata': {'filename': 'health_insurance.pdf'}
        }
        
        # Process document
        processed_doc = processor.process_document_with_domain(document)
        
        print(f"📄 Processed Document: {processed_doc.filename}")
        print(f"   Domain: {processed_doc.domain_classification.primary_domain.value}")
        print(f"   Confidence: {processed_doc.domain_classification.confidence:.2f}")
        print(f"   Chunks: {len(processed_doc.chunks)}")
        print(f"   Length: {processed_doc.total_length} characters")
        
        print("✅ Document Processing Test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Document Processing Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing Module Imports...")
    
    modules_to_test = [
        'app.services.domain_classifier',
        'app.services.enhanced_document_processor', 
        'app.services.domain_aware_qa_service',
        'app.api.endpoints.domain_aware_qa'
    ]
    
    failed_imports = []
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {module_name}")
        except Exception as e:
            print(f"❌ {module_name}: {e}")
            failed_imports.append(module_name)
    
    if failed_imports:
        print(f"\n❌ Import Test FAILED: {len(failed_imports)} modules failed")
        return False
    else:
        print("\n✅ All Module Imports PASSED")
        return True

async def main():
    """Run all tests."""
    print("🎯 DOMAIN-AWARE QA SYSTEM - QUICK TESTS")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Domain Classification", test_domain_classification),
        ("Query Classification", test_query_classification),
        ("Document Processing", test_document_processing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your domain-aware system is ready!")
        print("\n📋 Next Steps:")
        print("1. Start the server: python -m app.main")
        print("2. Test the API endpoints at http://localhost:8000/docs")
        print("3. Try the demo: python demo_domain_aware_qa.py")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())
