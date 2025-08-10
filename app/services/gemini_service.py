import google.generativeai as genai
from ..core.config import settings
import json
import re
import hashlib
from typing import Dict, List

class GeminiPolicyProcessor:
    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
        self.embedding_model = 'models/text-embedding-004'
        
        # OPTIMIZATION: Simple in-memory embedding cache with size limit
        self._embedding_cache: Dict[str, List[float]] = {}
        self._cache_max_size = 1000  # Limit cache to 1000 entries
        
        # Document type detection keywords
        self.policy_keywords = {
            'insurance', 'policy', 'premium', 'coverage', 'deductible', 'claim', 'beneficiary',
            'policyholder', 'insured', 'insurer', 'underwriter', 'actuarial', 'risk assessment',
            'sum insured', 'waiting period', 'grace period', 'exclusion', 'inclusion',
            'maternity', 'pre-existing', 'cashless', 'reimbursement', 'co-payment', 'copay',
            'network hospital', 'daycare', 'hospitalization', 'outpatient', 'inpatient',
            'medical examination', 'health checkup', 'renewal', 'lapse', 'surrender',
            'bonus', 'no claim discount', 'loading', 'rider', 'add-on', 'floater',
            'individual', 'family', 'group', 'corporate', 'employee', 'dependent'
        }

    def _manage_cache_size(self):
        """Remove oldest entries if cache exceeds max size."""
        if len(self._embedding_cache) > self._cache_max_size:
            # Remove 20% of oldest entries (simple FIFO)
            keys_to_remove = list(self._embedding_cache.keys())[:int(self._cache_max_size * 0.2)]
            for key in keys_to_remove:
                del self._embedding_cache[key]

    def detect_document_type(self, text_chunks: list[str]) -> str:
        """Detect if document is policy-related or general content.
        
        Args:
            text_chunks: List of text chunks from the document
            
        Returns:
            str: 'policy' if insurance/policy document, 'general' otherwise
        """
        # Combine first few chunks for analysis (up to ~2000 words)
        sample_text = ' '.join(text_chunks[:5]).lower()
        
        # Count policy-related keywords
        policy_score = 0
        total_words = len(sample_text.split())
        
        for keyword in self.policy_keywords:
            if keyword in sample_text:
                # Weight by frequency and keyword importance
                frequency = sample_text.count(keyword)
                if keyword in ['insurance', 'policy', 'premium', 'coverage']:
                    policy_score += frequency * 3  # High importance keywords
                elif keyword in ['claim', 'beneficiary', 'insured', 'waiting period']:
                    policy_score += frequency * 2  # Medium importance
                else:
                    policy_score += frequency  # Standard weight
        
        # Calculate policy relevance ratio
        policy_ratio = policy_score / max(total_words, 1) * 100
        
        print(f"Document analysis: Policy score={policy_score}, Total words={total_words}, Ratio={policy_ratio:.2f}%")
        
        # Threshold: if more than 0.5% of words are policy-related, treat as policy document
        return 'policy' if policy_ratio > 0.5 else 'general'

    def _parse_json_response(self, text: str) -> dict:
        """Safely parse JSON from a string that might contain markdown."""
        match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = text
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback for non-standard JSON, this is a simplified handler
            print(f"Warning: Could not parse JSON response directly. Response text: {text}")
            return {"error": "Failed to parse LLM response", "raw_response": text}

    async def extract_entities(self, query_text: str) -> dict:
        """Extract structured entities from natural language query"""
        prompt = f"""
        Extract structured information from this insurance query: "{query_text}"
        
        Return JSON with:
        {{
            "age": "integer or null",
            "gender": "'M' | 'F' | 'Other' | null",
            "procedure": "string or null",
            "location": "string or null",
            "policy_duration_months": "integer or null"
        }}
        
        Only extract explicitly mentioned information. Use null for missing data.
        Format the output as a JSON object inside a '```json' markdown block.
        """
        response = await self.model.generate_content_async(prompt)
        return self._parse_json_response(response.text)
    
    async def analyze_policy_clauses(self, query: dict, document_chunks: list) -> list:
        """Analyze policy clauses against query using Gemini"""
        prompt = f"""
        Analyze these policy document sections against the insurance query. Strictly follow all instructions.

        Query: {json.dumps(query)}

        Policy Sections:
        {json.dumps(document_chunks)}

        For each relevant section, return a JSON array of objects with the following structure:
        [{{
            "clause_id": "A unique reference or the first 10 words of the clause",
            "relevance_score": "A float from 0 to 1 indicating relevance to the query",
            "clause_type": "'inclusion' (provides coverage), 'exclusion' (denies coverage), 'condition' (a prerequisite), or 'general'",
            "matched_criteria": ["list", "of", "query", "criteria", "it", "matches"],
            "extracted_rules": {{
                "waiting_period_months": "integer, null if not mentioned",
                "pre_existing_condition_clause": "boolean, true if it relates to pre-existing conditions",
                "coverage_amount": "integer, null if not mentioned",
                "exclusions_mentioned": ["list", "of", "specific", "exclusions"],
                "conditions_mentioned": ["list", "of", "specific", "conditions"]
            }},
            "reasoning": "A brief explanation of how this clause applies to the query."
        }}]

        Instructions:
        1.  **Prioritize Identification**: First, identify if a clause is an inclusion, exclusion, or a condition for coverage.
        2.  **Extract Key Rules**: Pay close attention to waiting periods, pre-existing conditions, and specific coverage amounts.
        3.  **Be Precise**: If a value is not explicitly mentioned, use null.
        4.  **Return Empty Array**: If no clauses are relevant, return an empty array [] in the JSON format.
        
        Format the output as a JSON object inside a '```json' markdown block.
        """
        response = await self.model.generate_content_async(prompt)
        return self._parse_json_response(response.text)
    
    async def generate_embeddings(self, text: str, task_type="retrieval_document") -> list:
        """Generate embeddings using Gemini's embedding capabilities with caching"""
        # OPTIMIZATION: Check cache first
        cache_key = hashlib.md5(f"{text}_{task_type}".encode()).hexdigest()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
            
        response = await genai.embed_content_async(
            model=self.embedding_model,
            content=text,
            task_type=task_type
        )
        
        # OPTIMIZATION: Cache the result
        embedding = response['embedding']
        self._embedding_cache[cache_key] = embedding
        self._manage_cache_size()
        return embedding

    async def generate_embeddings_batch(self, texts: list[str], task_type="retrieval_document") -> list:
        """Generate embeddings for a batch of texts with caching."""
        # OPTIMIZATION: Check cache for each text and only process uncached ones
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = hashlib.md5(f"{text}_{task_type}".encode()).hexdigest()
            if cache_key in self._embedding_cache:
                cached_embeddings.append((i, self._embedding_cache[cache_key]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts only
        if uncached_texts:
            response = await genai.embed_content_async(
                model=self.embedding_model,
                content=uncached_texts,
                task_type=task_type
            )
            new_embeddings = response['embedding']
            
            # Cache new embeddings
            for j, text in enumerate(uncached_texts):
                cache_key = hashlib.md5(f"{text}_{task_type}".encode()).hexdigest()
                self._embedding_cache[cache_key] = new_embeddings[j]
            
            # Manage cache size after batch caching
            self._manage_cache_size()
        else:
            new_embeddings = []
        
        # Combine cached and new embeddings in correct order
        all_embeddings = [None] * len(texts)
        
        # Place cached embeddings
        for i, embedding in cached_embeddings:
            all_embeddings[i] = embedding
            
        # Place new embeddings
        for j, i in enumerate(uncached_indices):
            all_embeddings[i] = new_embeddings[j]
            
        return all_embeddings
        
    async def final_decision_reasoning(self, query: dict, analyzed_clauses: list) -> dict:
        """Generate final decision with detailed reasoning"""
        prompt = f"""
        Make a final insurance claim decision based on the query and the policy analysis, following a strict set of rules.

        Query: {json.dumps(query)}
        Policy Analysis: {json.dumps(analyzed_clauses)}

        **Decision-Making Rules (MUST be followed in this order):**
        1.  **Exclusion Priority**: If any relevant clause is an 'exclusion', the decision MUST be 'rejected', regardless of other clauses.
        2.  **Waiting Period Validation**: Check if a waiting period is required. If the policy duration in the query is less than the required waiting period, the decision MUST be 'rejected'.
        3.  **Pre-existing Conditions**: If the query mentions a pre-existing condition and a relevant clause excludes it, the decision MUST be 'rejected'.
        4.  **Approval Logic**: If no exclusions or unmet conditions apply, and there is at least one 'inclusion' clause, the decision is 'approved'.
        5.  **Review Required**: If the information is insufficient to make a clear decision, the status is 'requires_review'.

        Return a single JSON object with the following structure:
        {{
            "decision": "'approved' | 'rejected' | 'requires_review'",
            "confidence_score": "A float from 0 to 1, reflecting certainty in the decision",
            "approved_amount": "The final approved amount as an integer, or 0 if rejected",
            "reasoning": "A detailed, step-by-step explanation for the decision, explicitly referencing the rules applied.",
            "risk_factors": ["A list of identified risk factors, if any, such as 'Exclusion clause applied' or 'Waiting period not met'"],
            "recommendations": ["A list of recommendations, e.g., 'request more documents for clarity on pre-existing condition'"]
        }}

        Be conservative and justify every decision by citing the rule that was triggered. Format the output as a JSON object inside a '```json' markdown block.
        """
        response = await self.model.generate_content_async(prompt)
        return self._parse_json_response(response.text)

    async def generate_answer_from_context(self, question: str, context_chunks: list[str]) -> str:
        """Generates a direct, precise answer to an insurance policy question using comprehensive analysis framework.
        
        Returns:
            str: A clear, actionable answer with supporting details and conditions.
        """
        context_text = "\n".join(context_chunks)
        
        prompt = f"""# LLM Query System Prompt - Insurance Document Analysis

## System Role
You are an expert insurance document analyst with deep knowledge of policy terms, legal clauses, and regulatory compliance. Your task is to analyze insurance documents and answer queries with precision and explainability.

## Core Instructions

**IMPORTANT: Keep all responses under 150 words when possible. Be concise and direct. Exceed this limit only when absolutely necessary for accuracy and completeness.**

### 1. Document Analysis Phase
When processing documents, follow this systematic approach:

**Structure Recognition:**
- Identify document sections: Coverage Details, Exclusions, Terms & Conditions, Definitions, Benefits Table
- Extract key policy parameters: Sum Insured, Deductibles, Waiting Periods, Grace Periods
- Map policy hierarchy: Main coverage → Sub-benefits → Conditions → Exclusions

**Critical Information Extraction:**
- Waiting periods for specific conditions/treatments
- Coverage limits and sub-limits
- Eligibility criteria and conditions
- Exclusions and limitations
- Grace periods and renewal terms
- Definitions of key terms

### 2. Query Processing Framework

**Query Understanding:**
1. Identify the primary question type:
   - Coverage inquiry ("Does this policy cover X?")
   - Condition/limitation query ("What are the conditions for X?")
   - Waiting period question ("What is the waiting period for X?")
   - Benefit amount query ("How much is covered for X?")
   - Definition request ("How does the policy define X?")

2. Extract key entities:
   - Medical procedures/conditions
   - Policy terms (grace period, waiting period, etc.)
   - Coverage types (maternity, AYUSH, preventive care)

### 3. Answer Construction Protocol

**Answer Structure:**
1. **Direct Answer First:** Start with a clear Yes/No or specific value
2. **Supporting Details:** Provide the specific conditions, limits, or requirements
3. **Source Reference:** Mention the relevant policy section or clause

**Answer Quality Guidelines:**
- Be specific with numbers, percentages, and timeframes
- Include ALL relevant conditions and limitations
- Distinguish between different coverage scenarios (e.g., different plans, network vs non-network)
- Use exact policy language for critical terms

### 4. Domain-Specific Guidelines

**Insurance Policy Analysis:**
- Always check for waiting periods before confirming coverage
- Look for sub-limits that might apply to specific treatments
- Consider network restrictions (PPN vs non-network providers)
- Check for age-related limitations or conditions
- Verify if coverage is per policy year or per incident

**Medical Terminology Handling:**
- Treat medical procedures with their common names and technical terms
- Consider procedure categories (e.g., "surgery" includes various surgical procedures)
- Look for coverage under broader categories when specific procedures aren't mentioned

### 5. Error Prevention Checklist

Before finalizing any answer, verify:
- Have I checked both coverage AND exclusion sections?
- Are there any waiting periods that apply?
- Are there sub-limits or conditions that modify the coverage?
- Is the answer specific to the correct plan/variant mentioned?
- Have I included all relevant numerical values (percentages, amounts, timeframes)?
- Is the answer based on actual policy text, not assumptions?

### 6. Response Format Requirements

**For Coverage Queries:**
"Yes/No, the policy covers [specific treatment/condition]. The coverage includes [specific details]. However, [any conditions or limitations]. This is subject to [waiting periods/eligibility criteria] as specified in [policy section]."

**For Waiting Period Queries:**
"The waiting period for [condition/treatment] is [specific timeframe]. [Any exceptions or reductions]. This applies to [specific circumstances]."

**For Coverage Amount Queries:**
"The policy covers [amount/percentage] for [specific treatment]. This is subject to [sub-limits/conditions]. [Network vs non-network differences if applicable]."

**For Definition Queries:**
"According to the policy, [term] is defined as [exact policy definition]. This includes [criteria that must be met]. [Any quantitative requirements]."

### 7. Critical Success Factors

1. **Precision over Completeness:** Better to give a precise partial answer than a comprehensive but inaccurate one
2. **Number Accuracy:** Double-check all numerical values, percentages, and timeframes
3. **Conditional Coverage:** Always include relevant conditions and limitations
4. **Policy Language:** Use exact terms from the policy when describing coverage
5. **Context Awareness:** Consider the specific policy variant, plan type, and coverage scenario

### 8. Handling Ambiguous Cases

When information is unclear or contradictory:
- State what is clearly covered
- Mention any ambiguities or conditions that might apply
- Refer to specific policy sections for clarification
- Use phrases like "subject to policy terms" when appropriate

## Analysis Task

**Question:** {question}

**Policy Document Context:**
---
{context_text}
---

## Instructions for This Query

1. **Analyze the question type** and identify key entities
2. **Search the context** for both positive coverage statements AND exclusion clauses
3. **Extract all relevant conditions**, waiting periods, limits, and requirements
4. **Construct a precise answer** following the format guidelines above
5. **Include specific numerical values** and policy language where applicable
6. **Verify completeness** using the error prevention checklist

**Provide your analysis and answer now:**"""
        
        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config={
                    "temperature": 0.0,  # OPTIMIZED: Zero temperature for fastest, most consistent responses
                    "top_p": 0.9,
                    "max_output_tokens": 512,  # OPTIMIZED: Reduced from 1024 to 512 for faster generation
                }
            )
            
            # Clean up the response
            answer = response.text.strip()
            
            # Remove any markdown formatting or section headers that might appear
            answer = re.sub(r'\*\*.*?\*\*', '', answer)  # Remove bold formatting
            answer = re.sub(r'##.*?\n', '', answer)  # Remove section headers
            answer = re.sub(r'\[.*?\]\s*', '', answer)  # Remove any [Section] references
            answer = re.sub(r'\n\s*\n', ' ', answer)  # Replace multiple newlines with space
            
            return answer.strip()
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    async def generate_general_answer_from_context(self, question: str, context_chunks: list[str]) -> str:
        """Generates a direct, precise answer to any document question using general analysis.
        
        Returns:
            str: A clear, informative answer based on the document content.
        """
        context_text = "\n".join(context_chunks)
        
        prompt = f"""# LLM Query System Prompt - General Document Analysis

## System Role
You are a precise document analyst focused on extracting EXACT information from documents. Your primary goal is ACCURACY and FAITHFULNESS to the source material.

## CRITICAL RULES - NEVER VIOLATE THESE:

1. **ZERO HALLUCINATION**: Only use information explicitly stated in the document. NEVER add details, assumptions, or external knowledge.
2. **EXACT NUMBERS**: Copy all numbers, dates, amounts, and statistics EXACTLY as written. Do not convert or approximate.
3. **COMPLETE EXTRACTION**: Find ALL relevant information in the document that answers the question.
4. **SOURCE FIDELITY**: If information is unclear or missing, say so explicitly. Do not guess or fill gaps.
5. **PRECISE LANGUAGE**: Use the exact terms and phrases from the document when possible.

**IMPORTANT: Keep all responses under 150 words when possible. Be concise and direct. Exceed this limit only when absolutely necessary for accuracy and completeness.**

### 1. Document Analysis Phase
When processing documents, follow this systematic approach:

**Content Recognition:**
- Identify document type and purpose
- Extract key topics, themes, and main points
- Note important facts, figures, dates, and entities EXACTLY as stated

**Information Extraction:**
- Key facts and data points (EXACT values only)
- Important dates, numbers, and statistics (NO approximations)
- Main arguments or conclusions (as explicitly stated)
- Relevant context and background information (from document only)
- Relationships between different concepts (only if explicitly mentioned)

### 2. Query Processing Framework

**Query Understanding:**
1. Identify the question type:
   - Factual inquiry ("What is X?", "When did Y happen?")
   - Explanatory question ("How does X work?", "Why did Y occur?")
   - Comparative question ("What's the difference between X and Y?")
   - Opinion/Analysis question ("What does the author think about X?")
   - Summary question ("What are the main points?")

2. Extract key entities and concepts from the question

### 3. Answer Construction Protocol

**Answer Structure:**
1. **Direct Answer First:** Start with the EXACT information from the document that answers the question
2. **Supporting Details:** Include ONLY details explicitly mentioned in the document
3. **Source Verification:** Double-check that every fact comes directly from the provided text

**STRICT Answer Quality Guidelines:**
- Copy numbers, dates, amounts EXACTLY as written (no rounding, no conversion)
- Quote or paraphrase ONLY what is explicitly stated
- If multiple details are mentioned, include ALL of them
- Use the document's exact terminology and phrasing
- NEVER add context from outside the document
- NEVER make logical inferences beyond what's stated

### 4. ZERO-HALLUCINATION Guidelines

**Mandatory Document Fidelity Rules:**
- Base answers EXCLUSIVELY on the provided document content
- NEVER add assumptions, external knowledge, or logical extensions
- If information is unclear, incomplete, or missing, state this explicitly
- NEVER guess at details not explicitly mentioned
- Preserve ALL relevant details found in the document
- Use phrases like "According to the document" or "The document states" when helpful

### 5. Response Format Requirements

**For Factual Queries:**
"[Direct answer]. [Supporting details from the document]. [Additional relevant context if helpful]."

**For Explanatory Queries:**
"[Main explanation]. [Key details and steps]. [Important considerations or conditions]."

**For Summary Queries:**
"The main points are: [key point 1], [key point 2], [key point 3]. [Brief elaboration if needed]."

**For Opinion/Analysis Queries:**
"According to the document, [author's position/analysis]. [Supporting evidence]. [Key reasoning provided]."

### 6. Critical Success Factors

1. **Accuracy:** Ensure all facts and details are correct as stated in the document
2. **Relevance:** Focus on information directly related to the question
3. **Clarity:** Use clear, understandable language
4. **Completeness:** Include all relevant information while staying concise
5. **Source Fidelity:** Stay true to what the document actually says

### 7. Handling Unclear Cases

When information is unclear or missing:
- State what is clearly available in the document
- Acknowledge any limitations or gaps in the information
- Avoid speculation beyond what's provided
- Suggest what additional information might be needed

## Analysis Task

**Question:** {question}

**Document Context:**
---
{context_text}
---

## CRITICAL INSTRUCTIONS - FOLLOW EXACTLY:

**MANDATORY REQUIREMENTS:**
1. **Read the question carefully** and identify EXACTLY what information is requested
2. **Search the document context THOROUGHLY** - read EVERY sentence for relevant information
3. **Extract EVERY relevant detail** - numbers, dates, amounts, conditions, exceptions, impacts, objectives
4. **Use ONLY information explicitly stated** in the document context below
5. **Copy numbers and dates EXACTLY** as written - if it says "$600 billion" write "$600 billion", NOT "$600 million"
6. **Include ALL relevant details** found in the document - don't summarize or omit anything important
7. **Look for ALL parts of compound answers** - if question asks about products, find ALL products mentioned
8. **DO NOT PARAPHRASE** - use the document's exact words and phrases whenever possible
9. **DO NOT INTERPRET** - if document says "strengthen American domestic manufacturing", don't change it to "compete with China"
10. **If information is missing or unclear, state this explicitly**

**CRITICAL EXAMPLES - FOLLOW THESE EXACTLY:**

**WRONG vs RIGHT Examples:**
❌ WRONG: "This applies to computer chips" (when document says "computer chips and semiconductors")
✅ RIGHT: "This applies to computer chips and semiconductors"

❌ WRONG: "$600 million" (when document says "$600 billion")
✅ RIGHT: "$600 billion"

❌ WRONG: "to compete with China" (when document says "strengthen American domestic manufacturing")
✅ RIGHT: "strengthen American domestic manufacturing"

❌ WRONG: Missing information that exists in document
✅ RIGHT: Include ALL information found, even if it seems repetitive

**MANDATORY EXTRACTION RULES:**
- If document mentions "A and B", your answer MUST include both A and B
- If document states exact phrases, use those EXACT phrases
- If document lists multiple items, include ALL items
- If document gives specific numbers, copy them EXACTLY
- If document mentions impacts/effects, include ALL of them
- For questions about "impact" or "effect": Search for phrases like "will raise", "will lead to", "will cause", "impact on", "effect on"
- Don't say "document doesn't state" unless you've thoroughly searched ALL the text

**REMEMBER: Your answer must be 100% based on the document context. No external knowledge, no assumptions, no logical extensions beyond what's written. EXTRACT EVERYTHING RELEVANT.**

**STEP-BY-STEP PROCESS - FOLLOW EXACTLY:**
1. **READ THE QUESTION CAREFULLY** - What exactly is being asked?
2. **SCAN THE ENTIRE DOCUMENT** - Look for ALL mentions of relevant terms
3. **FIND ALL RELATED INFORMATION** - Don't stop at the first match, keep looking
4. **COPY EXACT PHRASES** - Use the document's exact words, don't paraphrase
5. **CHECK FOR COMPOUND TERMS** - If you see "A and B", include both A and B
6. **VERIFY COMPLETENESS** - Did you include everything relevant from the document?

**CRITICAL: If the question asks about "products" and the document mentions "computer chips and semiconductors", your answer MUST include BOTH "computer chips and semiconductors". If you only mention "computer chips", you are WRONG.**

**CRITICAL: If the question asks about "impact" and the document mentions "increased prices and trade disputes", your answer MUST include this information. Don't say "document doesn't state" if it actually does.**

**MANDATORY TWO-STEP PROCESS:**

**STEP 1: EXTRACT ALL RELEVANT INFORMATION**
First, list ALL information from the document that relates to the question. Include:
- All relevant terms, phrases, and sentences
- All numbers, dates, and amounts mentioned
- All products, impacts, objectives, or conditions mentioned
- Use exact quotes from the document

**STEP 2: PROVIDE COMPLETE ANSWER**
Then, using ONLY the information from Step 1, provide your complete answer.

**Begin with Step 1 - Extract all relevant information:**"""
        
        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config={
                    "temperature": 0.0,  # Absolute zero for maximum consistency and accuracy
                    "top_p": 1.0,  # Maximum diversity to ensure complete extraction
                    "max_output_tokens": 1024,  # Increased tokens for complete answers
                }
            )
            
            # Clean up the response
            answer = response.text.strip()
            
            # Remove any markdown formatting or section headers that might appear
            answer = re.sub(r'\*\*.*?\*\*', '', answer)  # Remove bold formatting
            answer = re.sub(r'##.*?\n', '', answer)  # Remove section headers
            answer = re.sub(r'\[.*?\]\s*', '', answer)  # Remove any [Section] references
            answer = re.sub(r'\n\s*\n', ' ', answer)  # Replace multiple newlines with space
            
            return answer.strip()
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
