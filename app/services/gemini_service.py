import google.generativeai as genai
from ..core.config import settings
import json
import re

class GeminiPolicyProcessor:
    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
        self.embedding_model = 'models/text-embedding-004'

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
        """Generate embeddings using Gemini's embedding capabilities"""
        response = await genai.embed_content_async(
            model=self.embedding_model,
            content=text,
            task_type=task_type
        )
        return response['embedding']

    async def generate_embeddings_batch(self, texts: list[str], task_type="retrieval_document") -> list:
        """Generate embeddings for a batch of texts for improved efficiency."""
        response = await genai.embed_content_async(
            model=self.embedding_model,
            content=texts,
            task_type=task_type
        )
        return response['embedding']
        
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
                    "temperature": 0.1,  # Lower temperature for more consistent, precise answers
                    "top_p": 0.9,
                    "max_output_tokens": 1024,  # Increased for detailed analysis
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
