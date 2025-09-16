import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List,  Any, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
from abc import ABC, abstractmethod
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('code_consolidation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CodeCategory(BaseModel):
    """A category of consolidated codes"""
    category_name: str = Field(description="Descriptive name for the category")
    codes: List[str] = Field(description="List of codes that belong to this category")


class CodeMapping(BaseModel):
    """Mapping of original code to final category"""
    original_code: str = Field(description="Original code name")
    final_category: str = Field(description="Final category name this code was assigned to")
    reasoning: str = Field(description="Brief reason for this categorization")


class ConsolidationResultWithMapping(BaseModel):
    """Result model for code consolidation with explicit merge tracking"""
    categories: List[CodeCategory] = Field(description="List of code categories with their constituent codes")
    code_mappings: List[CodeMapping] = Field(description="Explicit mapping of each original code to its final category")


class ConceptualCategory(BaseModel):
    category_name: str = Field(description="High-level conceptual category name")
    description: str = Field(description="Brief description of what this category represents")
    subcategories: List[str] = Field(description="List of technical subcategories that belong to this conceptual category")


class ConceptualConsolidationResult(BaseModel):
    """Result model for conceptual consolidation"""
    conceptual_categories: List[ConceptualCategory] = Field(description="List of high-level conceptual categories")


class CodeInstance:
    """Individual code instance with metadata"""
    def __init__(self, code: str, issue_key: str, source_text: str = "", source_location: str = ""):
        self.code = code
        self.issue_key = issue_key
        self.source_text = source_text
        self.source_location = source_location


@dataclass
class ConsolidationConfig:
    """Configuration for code consolidation"""
    model_name: str = "gpt-4o"
    temperature: float = 0
    max_tokens: int = 10000
    batch_size: int = 40
    input_file: str = ""
    research_question: str = ""
    output_dir: str = ""
    openai_api_key: str = ""


class BaseCodeExtractor(ABC):
    """Abstract base class for extracting codes from different research questions"""
    
    @abstractmethod
    def extract_codes(self, results_data: Dict[str, Any]) -> List[CodeInstance]:
        """Extract code instances from analysis results"""
        pass
    
    @abstractmethod
    def get_consolidation_prompt_template(self, dataset_type: str) -> str:
        """Get the consolidation prompt template for this research question"""
        pass
    
    @property
    @abstractmethod
    def research_question_name(self) -> str:
        """Name of the research question"""
        pass


class RoleSequenceCodeExtractor(BaseCodeExtractor):
    """Code extractor for role sequence analysis"""
    
    @property
    def research_question_name(self) -> str:
        return "role_sequence"
    
    def extract_codes(self, results_data: Dict[str, Any]) -> List[CodeInstance]:
        """Extract back-and-forth codes from role sequence results"""
        code_instances = []
        results = results_data.get('results', [])
        
        for result in results:
            if 'error' in result:
                continue
                
            issue_key = result.get('issue_key', 'unknown')
            codes = result.get('back_and_forth_codes', [])
            
            for code_obj in codes:
                if isinstance(code_obj, dict):
                    code = code_obj.get('code', '')
                    source_text = code_obj.get('source_text', '')
                    source_location = code_obj.get('source_location', '')
                elif hasattr(code_obj, 'code'):  # Pydantic object
                    code = code_obj.code
                    source_text = getattr(code_obj, 'source_text', '')
                    source_location = getattr(code_obj, 'source_location', '')
                else:
                    code = str(code_obj)
                    source_text = ''
                    source_location = ''
                
                if code:
                    code_instances.append(CodeInstance(
                        code=code,
                        issue_key=issue_key,
                        source_text=source_text,
                        source_location=source_location
                    ))
        
        return code_instances
    
    def get_consolidation_prompt_template(self, dataset_type: str) -> str:
        """Get consolidation prompt for role sequence codes with explicit mapping requirements"""
        if dataset_type == "a11y":
            return """
    ROLE SEQUENCE CONSOLIDATION - ACCESSIBILITY ISSUES

    I have a list of codes from analyzing accessibility issue role reversions (when control goes back to previous roles due to problems). Please consolidate these codes into logical categories.

    IMPORTANT: Each issue can have MULTIPLE factors causing role reversions. The same issue may contribute to several categories if it has multiple underlying problems.

    CONSOLIDATION GUIDELINES:
    - Group codes by the TYPE of accessibility problem that caused role reversions
    - Create meaningful category names that capture the essence of the grouped codes
    - Focus on accessibility-specific groupings when possible
    - Consider both technical aspects and user impact
    - **USE THE PROVIDED CONTEXT EXAMPLES** to better understand what each code represents
    - Look for semantic similarity in codes (e.g., "BEHAT_FAILURES_ON_401_AND_402" and "BEHAT_FAILURE_DETECTED" should be grouped together as they both relate to Behat test failures)
    - **PROVIDE EXPLICIT MAPPING**: For each original code, specify which final category it belongs to

    {codes_section}

    REQUIRED OUTPUT FORMAT:
    You must provide both categories AND explicit code mappings. For each original code listed above, specify exactly which category it was assigned to and why.

    Example format expected:
    {{
    "categories": [
        {{
        "category_name": "Testing and Verification Problems",
        "codes": ["BEHAT_FAILURES_ON_401_AND_402", "BEHAT_FAILURE_DETECTED", "TEST_EXECUTION_ERROR"]
        }}
    ]
    }}
    """
        else:
            return """
    ROLE SEQUENCE CONSOLIDATION - NON-ACCESSIBILITY ISSUES

    I have a list of codes from analyzing software development issue role reversions (when control goes back to previous roles due to problems). Please consolidate these codes into logical categories.

    IMPORTANT: Each issue can have MULTIPLE factors causing role reversions. The same issue may contribute to several categories if it has multiple underlying problems.

    CONSOLIDATION GUIDELINES:
    - Group codes by the TYPE of software development problem that caused role reversions
    - Create meaningful category names that capture development workflow issues
    - Focus on technical aspects and development processes
    - **USE THE PROVIDED CONTEXT EXAMPLES** to better understand what each code represents
    - Look for semantic similarity in codes (e.g., "BEHAT_FAILURES_ON_401_AND_402" and "BEHAT_FAILURE_DETECTED" should be grouped together as they both relate to Behat test failures)
    - **PROVIDE EXPLICIT MAPPING**: For each original code, specify which final category it belongs to

    {codes_section}

    REQUIRED OUTPUT FORMAT:
    You must provide both categories AND explicit code mappings. For each original code listed above, specify exactly which category it was assigned to and why.

    Example format expected:
    {{
    "categories": [
        {{
        "category_name": "Testing and Verification Problems",
        "codes": ["BEHAT_FAILURES_ON_401_AND_402", "BEHAT_FAILURE_DETECTED", "TEST_EXECUTION_ERROR"]
        }}
    ]
    }}
    """


class ParticipantInfluenceCodeExtractor(BaseCodeExtractor):
    """Code extractor for participant influence analysis"""

    @property
    def research_question_name(self) -> str:
        return "participant_influence"

    def extract_codes(self, results_data: Dict[str, Any]) -> List[CodeInstance]:
        """Extract participant influence codes from results"""
        code_instances = []
        results = results_data.get('results', [])

        for result in results:
            if 'error' in result:
                continue

            issue_key = result.get('issue_key', 'unknown')
            codes = result.get('participant_influence_codes', [])  # Adjust field name based on your data structure

            for code_obj in codes:
                if isinstance(code_obj, dict):
                    code = code_obj.get('code', '')
                    source_text = code_obj.get('source_text', '')
                    source_location = code_obj.get('source_location', '')
                elif hasattr(code_obj, 'code'):  # Pydantic object
                    code = code_obj.code
                    source_text = getattr(code_obj, 'source_text', '')
                    source_location = getattr(code_obj, 'source_location', '')
                else:
                    code = str(code_obj)
                    source_text = ''
                    source_location = ''

                if code:
                    code_instances.append(CodeInstance(
                        code=code,
                        issue_key=issue_key,
                        source_text=source_text,
                        source_location=source_location
                    ))

        return code_instances

    def get_consolidation_prompt_template(self, dataset_type: str) -> str:
        """Get consolidation prompt for participant influence codes"""
        if dataset_type == "a11y":
            return """
    PARTICIPANT INFLUENCE CONSOLIDATION - ACCESSIBILITY ISSUES

    I have a list of codes from analyzing how participants other than designated developers influence accessibility issue development. Please consolidate these codes into logical categories.

    CONSOLIDATION GUIDELINES:
    - Group codes by the TYPE of participant influence observed
    - Consider different types of participants: community users, other developers, testers, reviewers, etc.
    - Focus on HOW the influence manifests (feedback, suggestions, code contributions, testing, etc.)
    - Create meaningful category names that capture participant influence patterns
    - **USE THE PROVIDED CONTEXT EXAMPLES** to better understand what each code represents
    - **PROVIDE EXPLICIT MAPPING**: For each original code, specify which final category it belongs to

    {codes_section}

    REQUIRED OUTPUT FORMAT:
    You must provide both categories AND explicit code mappings.
    """
        else:
            return """
    PARTICIPANT INFLUENCE CONSOLIDATION - NON-ACCESSIBILITY ISSUES

    I have a list of codes from analyzing how participants other than designated developers influence software development issue resolution. Please consolidate these codes into logical categories.

    CONSOLIDATION GUIDELINES:
    - Group codes by the TYPE of participant influence observed
    - Consider different types of participants: community users, other developers, maintainers, reviewers, etc.
    - Focus on HOW the influence manifests (feedback, code review, suggestions, testing, etc.)
    - Create meaningful category names that capture participant influence patterns
    - **USE THE PROVIDED CONTEXT EXAMPLES** to better understand what each code represents
    - **PROVIDE EXPLICIT MAPPING**: For each original code, specify which final category it belongs to

    {codes_section}

    REQUIRED OUTPUT FORMAT:
    You must provide both categories AND explicit code mappings.
    """


class SolutionDevelopmentCodeExtractor(BaseCodeExtractor):
    """Code extractor for solution development analysis"""

    @property
    def research_question_name(self) -> str:
        return "solution_development"

    def extract_codes(self, results_data: Dict[str, Any]) -> List[CodeInstance]:
        """Extract solution development codes from results"""
        code_instances = []
        results = results_data.get('results', [])

        for result in results:
            if 'error' in result:
                continue

            issue_key = result.get('issue_key', 'unknown')
            codes = result.get('solution_development_codes', [])  # Adjust field name based on your data structure

            for code_obj in codes:
                if isinstance(code_obj, dict):
                    code = code_obj.get('code', '')
                    source_text = code_obj.get('source_text', '')
                    source_location = code_obj.get('source_location', '')
                elif hasattr(code_obj, 'code'):  # Pydantic object
                    code = code_obj.code
                    source_text = getattr(code_obj, 'source_text', '')
                    source_location = getattr(code_obj, 'source_location', '')
                else:
                    code = str(code_obj)
                    source_text = ''
                    source_location = ''

                if code:
                    code_instances.append(CodeInstance(
                        code=code,
                        issue_key=issue_key,
                        source_text=source_text,
                        source_location=source_location
                    ))

        return code_instances

    def get_consolidation_prompt_template(self, dataset_type: str) -> str:
        """Get consolidation prompt for solution development codes"""
        if dataset_type == "a11y":
            return """
    SOLUTION DEVELOPMENT CONSOLIDATION - ACCESSIBILITY ISSUES

    I have a list of codes from analyzing how assignees arrive at their first proposed solution for accessibility issues. Please consolidate these codes into logical categories.

    CONSOLIDATION GUIDELINES:
    - Group codes by the APPROACH or METHOD used to develop solutions
    - Consider different solution development patterns: research-based, trial-and-error, reuse existing patterns, community input, etc.
    - Focus on the PROCESS of solution development rather than the solution itself
    - Create meaningful category names that capture solution development approaches
    - **USE THE PROVIDED CONTEXT EXAMPLES** to better understand what each code represents
    - **PROVIDE EXPLICIT MAPPING**: For each original code, specify which final category it belongs to

    {codes_section}

    REQUIRED OUTPUT FORMAT:
    You must provide both categories AND explicit code mappings.
    """
        else:
            return """
    SOLUTION DEVELOPMENT CONSOLIDATION - NON-ACCESSIBILITY ISSUES

    I have a list of codes from analyzing how assignees arrive at their first proposed solution for software development issues. Please consolidate these codes into logical categories.

    CONSOLIDATION GUIDELINES:
    - Group codes by the APPROACH or METHOD used to develop solutions
    - Consider different solution development patterns: debugging, research, reusing code, consulting documentation, etc.
    - Focus on the PROCESS of solution development rather than the solution content
    - Create meaningful category names that capture solution development approaches
    - **USE THE PROVIDED CONTEXT EXAMPLES** to better understand what each code represents
    - **PROVIDE EXPLICIT MAPPING**: For each original code, specify which final category it belongs to

    {codes_section}

    REQUIRED OUTPUT FORMAT:
    You must provide both categories AND explicit code mappings.
    """


class CodeConsolidator:
    """Main class for consolidating codes with batch processing and detailed tracking"""
    
    def __init__(self, config: ConsolidationConfig):
        self.config = config
        
        # Get API key from config or environment
        api_key = config.openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file or pass it in config.")
        
        self.llm = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=api_key
        )
        
        # Initialize code extractor based on research question
        self.code_extractor = self._get_code_extractor(config.research_question)
        
        # Track batch processing results
        self.batch_results = []
        self.intermediate_categories = {}
        self.all_code_instances = []
    
    def merge_categories_across_batches(self, dataset_type: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """Merge categories from all batches into final consolidated categories with mapping tracking"""
        logger.info("Merging categories across all batches...")
        
        # Collect all categories and their codes
        all_batch_categories = {}
        all_code_mappings = {}
        
        for batch_result in self.batch_results:
            # Collect categories
            for category_name, codes in batch_result["categories"].items():
                if category_name not in all_batch_categories:
                    all_batch_categories[category_name] = []
                all_batch_categories[category_name].extend(codes)
            
            # Collect code mappings
            batch_mappings = batch_result.get("code_mappings", {})
            all_code_mappings.update(batch_mappings)
        
        # Remove duplicates within each category
        for category_name in all_batch_categories:
            all_batch_categories[category_name] = list(set(all_batch_categories[category_name]))
        
        # Create prompt to merge similar categories
        merge_prompt = self._create_merge_prompt_with_mappings(all_batch_categories, all_code_mappings, dataset_type)
        
        try:
            parser = PydanticOutputParser(pydantic_object=ConsolidationResultWithMapping)
            system_message = SystemMessage(content="You are an expert at merging and organizing qualitative research categories. Merge similar categories and provide updated code mappings.")
            human_message = HumanMessage(content=f"{merge_prompt}\n\n{parser.get_format_instructions()}")
            
            with get_openai_callback() as cb:
                start_time = time.time()
                response = self.llm.invoke([system_message, human_message])
                processing_time = time.time() - start_time
            
            result = parser.parse(response.content)
            
            # Convert to dictionary format
            final_categories = {}
            for category in result.categories:
                final_categories[category.category_name] = category.codes
            
            # Extract final code mappings
            final_code_mappings = {}
            for mapping in result.code_mappings:
                final_code_mappings[mapping.original_code] = mapping.final_category
            
            logger.info(f"Final merge complete: {len(final_categories)} categories, ${cb.total_cost:.4f}")
            
            return final_categories, final_code_mappings
            
        except Exception as e:
            logger.error(f"Error merging categories: {e}")
            return all_batch_categories, all_code_mappings
    
    def _create_merge_prompt_with_mappings(self, batch_categories: Dict[str, List[str]], code_mappings: Dict[str, str], dataset_type: str) -> str:
        """Create prompt for merging categories with existing code mappings"""
        categories_text = ""
        for category_name, codes in batch_categories.items():
            categories_text += f"\n**{category_name}** ({len(codes)} codes)\n"
            for code in codes:
                categories_text += f"- {code}\n"
        
        mappings_text = ""
        for original_code, category in code_mappings.items():
            mappings_text += f"- {original_code} → {category}\n"
        
        rq_type = "accessibility" if dataset_type == "a11y" else "non-accessibility"
        
        return f"""
    FINAL CATEGORY MERGE WITH MAPPING TRACKING - {rq_type.upper()} {self.config.research_question.upper()} ANALYSIS

    I have categories from multiple batches that need to be merged into final consolidated categories, along with existing code mappings that must be updated.

    CURRENT BATCH CATEGORIES:
    {categories_text}

    CURRENT CODE MAPPINGS:
    {mappings_text}

    MERGE TASK:
    1. Merge similar categories into final consolidated categories
    2. Update the code mappings to reflect the new final category names
    3. Ensure every original code is mapped to exactly one final category
    4. Provide reasoning for any mapping changes

    MERGE GUIDELINES:
    - Merge categories that represent the same or very similar concepts
    - Keep categories separate if they represent distinct aspects
    - Ensure all codes end up in exactly one final category
    - Create clear, descriptive final category names
    - Update all code mappings to use the final category names

    Your response must include both the final categories AND updated code mappings for every original code.
    """
    
    
    def validate_code_consolidation_with_explicit_mappings(self, final_categories: Dict[str, List[str]], code_mappings: Dict[str, str]) -> Dict[str, Any]:
        """Validate code consolidation using explicit mappings with detailed tracking"""
        logger.info("Validating code consolidation with explicit mapping tracking...")
        
        # Get all original codes
        original_codes = set(instance.code for instance in self.all_code_instances)
        
        # Get codes from final categories
        consolidated_codes = set()
        for codes in final_categories.values():
            consolidated_codes.update(codes)
        
        # Get codes from mappings
        mapped_codes = set(code_mappings.keys())
        
        # Detailed analysis
        exactly_preserved = original_codes & consolidated_codes  # Codes that appear unchanged
        mapped_but_not_in_categories = mapped_codes - consolidated_codes  # Mapping says it's there but it's not
        in_categories_but_not_mapped = consolidated_codes - mapped_codes  # In categories but no mapping
        truly_missing = original_codes - consolidated_codes - mapped_codes  # Completely missing
        
        # For each original code, determine what happened
        code_status = {}
        for original_code in original_codes:
            if original_code in exactly_preserved:
                code_status[original_code] = "preserved_exactly"
            elif original_code in mapped_codes:
                mapped_to = code_mappings[original_code]
                if original_code in consolidated_codes:
                    code_status[original_code] = f"preserved_in_category_{mapped_to}"
                else:
                    code_status[original_code] = f"mapped_to_{mapped_to}_but_missing_from_categories"
            else:
                # Check if it appears in any category (might be unmapped merge)
                found_in_category = None
                for cat_name, codes in final_categories.items():
                    if original_code in codes:
                        found_in_category = cat_name
                        break
                
                if found_in_category:
                    code_status[original_code] = f"unmapped_but_found_in_{found_in_category}"
                else:
                    code_status[original_code] = "truly_missing"
        
        validation_result = {
            "validation_passed": len(truly_missing) == 0 and len(mapped_but_not_in_categories) == 0,
            "total_original_codes": len(original_codes),
            "total_consolidated_codes": len(consolidated_codes),
            "total_mapped_codes": len(mapped_codes),
            
            # Detailed status for each code
            "code_status": code_status,
            
            # Summary counts
            "exactly_preserved_count": len(exactly_preserved),
            "truly_missing_count": len(truly_missing),
            "mapping_errors_count": len(mapped_but_not_in_categories),
            "unmapped_but_present_count": len(in_categories_but_not_mapped),
            
            # Lists for investigation
            "truly_missing_codes": list(truly_missing),
            "mapping_errors": list(mapped_but_not_in_categories),
            "unmapped_but_present": list(in_categories_but_not_mapped),
            
            # Coverage percentage
            "coverage_percentage": ((len(original_codes) - len(truly_missing)) / len(original_codes)) * 100 if original_codes else 100
        }
        
        # Detailed logging
        logger.info(f"=== ROLLING CONSOLIDATION VALIDATION ===")
        logger.info(f"Original codes: {len(original_codes)}")
        logger.info(f"Final consolidated codes: {len(consolidated_codes)}")
        logger.info(f"Exactly preserved: {len(exactly_preserved)}")
        logger.info(f"Truly missing: {len(truly_missing)}")
        logger.info(f"Coverage: {validation_result['coverage_percentage']:.1f}%")
        
        if truly_missing:
            logger.error(f"TRULY MISSING CODES ({len(truly_missing)}):")
            for code in list(truly_missing)[:5]:
                logger.error(f"  {code}")
            if len(truly_missing) > 5:
                logger.error(f"  ... and {len(truly_missing) - 5} more")
        
        if mapped_but_not_in_categories:
            logger.error(f"MAPPING ERRORS ({len(mapped_but_not_in_categories)}):")
            for code in list(mapped_but_not_in_categories)[:5]:
                mapped_to = code_mappings.get(code, "unknown")
                logger.error(f" {code} → mapped to '{mapped_to}' but not found in categories")
        
        if validation_result["validation_passed"]:
            logger.info("VALIDATION PASSED: All codes accounted for")
        else:
            logger.error("VALIDATION FAILED: Some codes missing or mapping errors")
        
        return validation_result

    def validate_semantic_categorization(self, final_categories: Dict[str, List[str]],
                                         code_mappings: Dict[str, str],
                                         dataset_type: str,
                                         sample_size: int = 20) -> Dict[str, Any]:
        """Validate that codes are semantically categorized correctly using LLM review"""

        logger.info(f"Performing semantic validation with sample size: {sample_size}")

        # Select random sample of codes for validation
        import random
        all_original_codes = list(code_mappings.keys())
        sample_codes = random.sample(all_original_codes, min(sample_size, len(all_original_codes)))

        # Create validation prompt with sampled codes and their contexts
        validation_prompt = self._create_semantic_validation_prompt(
            sample_codes, final_categories, code_mappings, dataset_type
        )

        try:
            # Define validation result model
            class SemanticValidationResult(BaseModel):
                correctly_categorized: List[str] = Field(description="Codes that are correctly categorized")
                incorrectly_categorized: List[Dict[str, str]] = Field(
                    description="Codes with incorrect categorization: [{'code': 'X', 'current_category': 'Y', 'suggested_category': 'Z', 'reason': 'explanation'}]"
                )
                overall_quality_score: int = Field(description="Overall categorization quality from 1-10")
                quality_assessment: str = Field(description="Brief assessment of categorization quality")

            parser = PydanticOutputParser(pydantic_object=SemanticValidationResult)
            system_message = SystemMessage(
                content="You are an expert qualitative researcher evaluating categorization quality.")
            human_message = HumanMessage(content=f"{validation_prompt}\n\n{parser.get_format_instructions()}")

            with get_openai_callback() as cb:
                response = self.llm.invoke([system_message, human_message])

            result = parser.parse(response.content)

            semantic_validation = {
                "sample_size": len(sample_codes),
                "correctly_categorized": result.correctly_categorized,
                "incorrectly_categorized": result.incorrectly_categorized,
                "correct_count": len(result.correctly_categorized),
                "incorrect_count": len(result.incorrectly_categorized),
                "accuracy_percentage": (len(result.correctly_categorized) / len(sample_codes)) * 100,
                "overall_quality_score": result.overall_quality_score,
                "quality_assessment": result.quality_assessment,
                "validation_cost": cb.total_cost
            }

            logger.info(f"Semantic validation complete:")
            logger.info(f"  - Sample size: {len(sample_codes)}")
            logger.info(f"  - Correctly categorized: {len(result.correctly_categorized)}")
            logger.info(f"  - Incorrectly categorized: {len(result.incorrectly_categorized)}")
            logger.info(f"  - Accuracy: {semantic_validation['accuracy_percentage']:.1f}%")
            logger.info(f"  - Quality score: {result.overall_quality_score}/10")

            return semantic_validation

        except Exception as e:
            logger.error(f"Error in semantic validation: {e}")
            return {"error": str(e), "sample_size": len(sample_codes)}

    def _create_semantic_validation_prompt(self, sample_codes: List[str],
                                           final_categories: Dict[str, List[str]],
                                           code_mappings: Dict[str, str],
                                           dataset_type: str) -> str:
        """Create prompt for semantic validation with research question-specific context"""

        # Get context for sample codes
        code_contexts = {}
        for instance in self.all_code_instances:
            if instance.code in sample_codes:
                if instance.code not in code_contexts:
                    code_contexts[instance.code] = []
                if instance.source_text:
                    code_contexts[instance.code].append(instance.source_text)

        # Format sample codes with their current categorization and context
        codes_text = ""
        for code in sample_codes:
            current_category = code_mappings.get(code, "Unknown")
            codes_text += f"\n**Code**: {code}\n"
            codes_text += f"**Current Category**: {current_category}\n"

            if code in code_contexts:
                codes_text += f"**Context Examples**:\n"
                for i, context in enumerate(code_contexts[code][:2], 1):
                    codes_text += f"  {i}. {context}\n"
            codes_text += "\n"

        # Format available categories
        categories_text = ""
        for cat_name, codes in final_categories.items():
            categories_text += f"\n**{cat_name}**: {len(codes)} codes\n"

        rq_type = "accessibility" if dataset_type == "a11y" else "non-accessibility"

        # Research question-specific prompts
        if self.config.research_question == "role_sequence":
            return f"""
    SEMANTIC CATEGORIZATION VALIDATION - {rq_type.upper()} ROLE SEQUENCE ANALYSIS

    Your task is to evaluate whether codes about role reversions (when control goes back to previous roles due to problems) have been categorized correctly based on their meaning and context.

    AVAILABLE CATEGORIES:
    {categories_text}

    CODES TO VALIDATE:
    {codes_text}

    EVALUATION CRITERIA FOR ROLE SEQUENCE CODES:
    1. Does the code represent a clear cause of role reversion?
    2. Is the categorization consistent with the TYPE of problem that caused the role reversion?
    3. Are similar technical problems (e.g., different types of test failures) grouped together logically?
    4. Does the category name accurately represent the underlying cause of role reversions?

    Focus on whether codes representing similar CAUSES of role reversions are grouped together, and whether the category accurately captures the nature of the problem that forced the workflow backward.
    """

        elif self.config.research_question == "participant_influence":
            return f"""
    SEMANTIC CATEGORIZATION VALIDATION - {rq_type.upper()} PARTICIPANT INFLUENCE ANALYSIS

    Your task is to evaluate whether codes about how non-designated participants influence issue development have been categorized correctly based on their meaning and context.

    AVAILABLE CATEGORIES:
    {categories_text}

    CODES TO VALIDATE:
    {codes_text}

    EVALUATION CRITERIA FOR PARTICIPANT INFLUENCE CODES:
    1. Does the code clearly represent a type of participant influence?
    2. Are codes grouped by WHO is influencing (community users, other developers, reviewers) or HOW they influence (feedback, code contributions, testing)?
    3. Is the influence mechanism (direct/indirect, formal/informal) consistently categorized?
    4. Are similar types of participant contributions grouped logically?

    Focus on whether codes representing similar TYPES of participant influence or similar INFLUENCE MECHANISMS are grouped together appropriately.
    """

        elif self.config.research_question == "solution_development":
            return f"""
    SEMANTIC CATEGORIZATION VALIDATION - {rq_type.upper()} SOLUTION DEVELOPMENT ANALYSIS

    Your task is to evaluate whether codes about how assignees arrive at their first proposed solution have been categorized correctly based on their meaning and context.

    AVAILABLE CATEGORIES:
    {categories_text}

    CODES TO VALIDATE:
    {codes_text}

    EVALUATION CRITERIA FOR SOLUTION DEVELOPMENT CODES:
    1. Does the code represent a clear approach or method for developing solutions?
    2. Are codes grouped by PROCESS TYPE (research-based, trial-and-error, reuse existing) or INFORMATION SOURCE (documentation, community input, experimentation)?
    3. Is the categorization focused on the APPROACH rather than the solution content?
    4. Are similar solution development strategies grouped logically?

    Focus on whether codes representing similar SOLUTION DEVELOPMENT APPROACHES or similar INFORMATION GATHERING METHODS are grouped together appropriately.
    """

        else:
            # Fallback for any new research questions
            context_descriptions = {
                "role_sequence": "codes about when control goes back to previous roles due to problems",
                "participant_influence": "codes about how non-designated participants influence issue development",
                "solution_development": "codes about how assignees arrive at their first proposed solution"
            }

            context_description = context_descriptions.get(self.config.research_question, "codes from the analysis")

            return f"""
    SEMANTIC CATEGORIZATION VALIDATION - {rq_type.upper()} {self.config.research_question.upper()} ANALYSIS

    Your task is to evaluate whether {context_description} have been categorized correctly based on their meaning and context.

    AVAILABLE CATEGORIES:
    {categories_text}

    CODES TO VALIDATE:
    {codes_text}

    EVALUATION CRITERIA:
    1. Does the code's meaning align with its assigned category?
    2. Would the code fit better in a different existing category?
    3. Is the categorization consistent with similar codes?
    4. Does the category name accurately represent the grouped concepts?

    For each code, determine if it's correctly categorized. If not, suggest the most appropriate existing category and explain why.

    Also provide an overall quality assessment of the categorization scheme.
    """

    def _get_code_extractor(self, research_question: str) -> BaseCodeExtractor:
        """Get appropriate code extractor for research question"""
        extractors = {
            'role_sequence': RoleSequenceCodeExtractor(),
            'participant_influence': ParticipantInfluenceCodeExtractor(),
            'solution_development': SolutionDevelopmentCodeExtractor(),
        }

        if research_question not in extractors:
            raise ValueError(
                f"Unsupported research question: {research_question}. Supported: {list(extractors.keys())}")

        return extractors[research_question]
    
    def extract_codes(self, results_data: Dict[str, Any]) -> List[CodeInstance]:
        """Extract codes using the appropriate extractor"""
        return self.code_extractor.extract_codes(results_data)
    
    def create_batch_consolidation_prompt(self, batch_codes: List[str], dataset_type: str, batch_num: int, total_batches: int) -> str:
        """Create consolidation prompt for a batch of codes with source text context"""
        # Create a mapping from codes to their source texts and frequencies
        code_info = defaultdict(lambda: {"count": 0, "source_texts": []})
        
        # Extract codes and their contexts from all_code_instances for this batch
        for code_instance in self.all_code_instances:
            if code_instance.code in batch_codes:
                code_info[code_instance.code]["count"] += 1
                # Add source_text if it exists and isn't already in the list
                if (code_instance.source_text and 
                    code_instance.source_text not in code_info[code_instance.code]["source_texts"]):
                    code_info[code_instance.code]["source_texts"].append(code_instance.source_text)
        
        # Format codes with context - show up to 2 examples with full source text
        codes_text = ""
        for code, info in sorted(code_info.items(), key=lambda x: x[1]["count"], reverse=True):
            codes_text += f"- {code} (appears {info['count']} times)\n"
            if info["source_texts"]:
                # Show up to 2 context examples with full length
                for i, source_text in enumerate(info["source_texts"][:2], 1):
                    codes_text += f"  Context {i}: {source_text}\n"
                # If there are more than 2, indicate how many more
                if len(info["source_texts"]) > 2:
                    codes_text += f"  ... and {len(info['source_texts']) - 2} more context examples\n"
        
        batch_header = f"\nBATCH {batch_num} of {total_batches} - Processing {len(code_info)} unique codes from {len(batch_codes)} total instances:\n"
        
        template = self.code_extractor.get_consolidation_prompt_template(dataset_type)
        return template.format(codes_section=batch_header + codes_text)
    
    def consolidate_batch_simplified(self, batch_codes: List[str], dataset_type: str, batch_num: int, total_batches: int) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """Simplified consolidation without explicit mappings in LLM response"""
        logger.info(f"Consolidating batch {batch_num}/{total_batches} ({len(batch_codes)} codes)...")
        
        try:
            consolidation_prompt = self.create_batch_consolidation_prompt(batch_codes, dataset_type, batch_num, total_batches)
            
            # Use simplified format (no explicit mappings)
            parser = PydanticOutputParser(pydantic_object=ConsolidationResultWithMapping)  # Original simple format
            system_message = SystemMessage(content="You are an expert at organizing and categorizing qualitative research codes. Provide clear, logical categorizations based on the codes provided.")
            human_message = HumanMessage(content=f"{consolidation_prompt}\n\n{parser.get_format_instructions()}")
            
            with get_openai_callback() as cb:
                start_time = time.time()
                response = self.llm.invoke([system_message, human_message])
                processing_time = time.time() - start_time
            
            result = parser.parse(response.content)
            
            # Convert to dictionary format
            batch_categories = {}
            for category in result.categories:
                batch_categories[category.category_name] = category.codes
            
            # Infer mappings from category membership
            batch_code_mappings = {}
            for category_name, codes in batch_categories.items():
                for code in codes:
                    if code in batch_codes:  # Ensure it's from this batch
                        batch_code_mappings[code] = category_name
            
            # Validate all batch codes are mapped
            unmapped_codes = set(batch_codes) - set(batch_code_mappings.keys())
            if unmapped_codes:
                logger.warning(f"Unmapped codes in batch {batch_num}: {unmapped_codes}")
                # Add to uncategorized
                if "Uncategorized Issues" not in batch_categories:
                    batch_categories["Uncategorized Issues"] = []
                for unmapped_code in unmapped_codes:
                    batch_categories["Uncategorized Issues"].append(unmapped_code)
                    batch_code_mappings[unmapped_code] = "Uncategorized Issues"
            
            # Store batch metadata
            batch_metadata = {
                "batch_number": batch_num,
                "total_codes": len(batch_codes),
                "unique_codes": len(set(batch_codes)),
                "categories_created": len(batch_categories),
                "mapped_codes": len(batch_code_mappings),
                "processing_time_seconds": round(processing_time, 2),
                "tokens_used": cb.total_tokens,
                "cost_estimate": cb.total_cost
            }
            
            self.batch_results.append({
                "metadata": batch_metadata,
                "categories": batch_categories,
                "code_mappings": batch_code_mappings,
                "codes_processed": list(set(batch_codes))
            })
            
            logger.info(f"Batch {batch_num} complete: {len(batch_categories)} categories, {len(batch_code_mappings)} mappings, ${cb.total_cost:.4f}")
            
            return batch_categories, batch_code_mappings
            
        except Exception as e:
            logger.error(f"Error consolidating batch {batch_num}: {e}")
            return {}, {}
    
    def consolidate_batch(self, batch_codes: List[str], dataset_type: str, batch_num: int, total_batches: int) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """Consolidate a single batch of codes with explicit merge tracking"""
        logger.info(f"Consolidating batch {batch_num}/{total_batches} ({len(batch_codes)} codes)...")
        
        try:
            consolidation_prompt = self.create_batch_consolidation_prompt(batch_codes, dataset_type, batch_num, total_batches)
            
            parser = PydanticOutputParser(pydantic_object=ConsolidationResultWithMapping)
            system_message = SystemMessage(content="You are an expert at organizing and categorizing qualitative research codes. Provide clear, logical categorizations AND explicit mappings for each original code.")
            human_message = HumanMessage(content=f"{consolidation_prompt}\n\n{parser.get_format_instructions()}")
            
            with get_openai_callback() as cb:
                start_time = time.time()
                response = self.llm.invoke([system_message, human_message])
                processing_time = time.time() - start_time
            
            result = parser.parse(response.content)
            
            # Convert to dictionary format
            batch_categories = {}
            for category in result.categories:
                batch_categories[category.category_name] = category.codes
            
            # Extract code mappings
            batch_code_mappings = {}
            for mapping in result.code_mappings:
                batch_code_mappings[mapping.original_code] = mapping.final_category
            
            # Store batch metadata with mappings
            batch_metadata = {
                "batch_number": batch_num,
                "total_codes": len(batch_codes),
                "unique_codes": len(set(batch_codes)),
                "categories_created": len(batch_categories),
                "processing_time_seconds": round(processing_time, 2),
                "tokens_used": cb.total_tokens,
                "cost_estimate": cb.total_cost
            }
            
            self.batch_results.append({
                "metadata": batch_metadata,
                "categories": batch_categories,
                "code_mappings": batch_code_mappings,  # Add mappings to batch results
                "codes_processed": list(set(batch_codes))
            })
            
            logger.info(f"Batch {batch_num} complete: {len(batch_categories)} categories, ${cb.total_cost:.4f}")
            
            return batch_categories, batch_code_mappings
            
        except Exception as e:
            logger.error(f"Error consolidating batch {batch_num}: {e}")
            return {}, {}
    
    def _create_merge_prompt(self, batch_categories: Dict[str, List[str]], dataset_type: str) -> str:
        """Create prompt for merging categories across batches with enhanced semantic matching"""
        categories_text = ""
        for category_name, codes in batch_categories.items():
            categories_text += f"\n**{category_name}** ({len(codes)} codes)\n"
            for code in codes:
                categories_text += f"- {code}\n"
        
        rq_type = "accessibility" if dataset_type == "a11y" else "non-accessibility"
        
        return f"""
    FINAL CATEGORY MERGE - {rq_type.upper()} {self.config.research_question.upper()} ANALYSIS

    I have categories from multiple batches that need to be merged into final consolidated categories. Some categories may be similar and should be merged, while others should remain separate.

    MERGE GUIDELINES:
    - Merge categories that represent the same or very similar concepts
    - Keep categories separate if they represent distinct aspects
    - Ensure all codes end up in exactly one final category
    - Create clear, descriptive category names
    - Maintain logical groupings based on the type of issues/contributions

    ENHANCED SEMANTIC MATCHING RULES:
    - Look for codes that share common keywords or concepts (e.g., "BEHAT_FAILURES_ON_401_AND_402" and "BEHAT_FAILURE_DETECTED" both relate to Behat testing failures)
    - Group codes with similar technical terms (e.g., all database-related codes, all UI-related codes)
    - Consider abbreviations and variations (e.g., "DB_ERROR" and "DATABASE_CONNECTION_ISSUE")
    - Merge codes that represent the same underlying issue with different wording
    - Pay attention to semantic similarity even when exact wording differs

    EXAMPLES OF CODES THAT SHOULD BE MERGED:
    - BEHAT_FAILURES_ON_401_AND_402, BEHAT_FAILURE_DETECTED → Testing and Verification Problems
    - DB_CONNECTION_ERROR, DATABASE_TIMEOUT → Database Issues

    BATCH CATEGORIES TO MERGE:
    {categories_text}

    Your task is to:
    1. Identify categories that should be merged (similar concepts)
    2. Look for codes within different categories that actually belong together based on semantic similarity
    3. Create final category names that clearly represent the grouped codes
    4. Ensure no codes are lost in the merge process
    5. Organize codes logically within each final category

    Format your response as:
    **Final Category Name**
    - Code 1
    - Code 2
    - Code 3

    **Next Final Category Name**
    - Code 4
    - Code 5
    etc.

    Only provide the final categories and code lists - no additional explanations needed.
    """
    
    def create_detailed_final_result(self, final_categories: Dict[str, List[str]]) -> Dict[str, Any]:
        """Create detailed final result with category -> codes -> issue keys mapping"""
        logger.info("Creating detailed final result with issue key mappings...")
        
        # Create mapping from code to issue keys
        code_to_issues = defaultdict(list)
        for code_instance in self.all_code_instances:
            code_to_issues[code_instance.code].append(code_instance.issue_key)
        
        # Build final structure: category -> codes -> issue keys
        detailed_result = {
            "categories": {},
            "metadata": {
                "total_categories": len(final_categories),
                "total_unique_codes": sum(len(codes) for codes in final_categories.values()),
                "total_code_instances": len(self.all_code_instances),
                "total_unique_issues": len(set(instance.issue_key for instance in self.all_code_instances)),
                "research_question": self.config.research_question,
                "dataset_type": self._detect_dataset_type(),
                "model_used": self.config.model_name,
                "batch_size": self.config.batch_size,
                "total_batches": len(self.batch_results),
                "consolidation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        for category_name, codes in final_categories.items():
            detailed_result["categories"][category_name] = {}
            for code in codes:
                issue_keys = list(set(code_to_issues.get(code, [])))  # Remove duplicates
                detailed_result["categories"][category_name][code] = issue_keys
        
        return detailed_result
    
    def create_conceptual_categorization(self, technical_categories: Dict[str, List[str]], dataset_type: str) -> Dict[str, Any]:
        """Create high-level conceptual categories from technical categories"""
        logger.info("Creating conceptual categorization for academic analysis...")
        
        # Debug: Check the structure of technical_categories
        logger.info(f"Technical categories structure debug:")
        for cat_name, cat_data in list(technical_categories.items())[:2]:  # Show first 2 for debugging
            logger.info(f"  {cat_name}: {type(cat_data)} with {len(cat_data) if hasattr(cat_data, '__len__') else 'unknown'} items")
            if isinstance(cat_data, dict):
                sample_items = list(cat_data.items())[:2]
                logger.info(f"    Sample items: {sample_items}")
            elif isinstance(cat_data, list):
                logger.info(f"    Sample items: {cat_data[:3]}")
        
        try:
            conceptual_prompt = self._create_conceptual_prompt(technical_categories, dataset_type)
            
            parser = PydanticOutputParser(pydantic_object=ConceptualConsolidationResult)
            system_message = SystemMessage(content="You are an expert researcher specializing in accessibility and software development. Create high-level conceptual categories suitable for academic analysis and discussion.")
            human_message = HumanMessage(content=f"{conceptual_prompt}\n\n{parser.get_format_instructions()}")
            
            with get_openai_callback() as cb:
                start_time = time.time()
                response = self.llm.invoke([system_message, human_message])
                processing_time = time.time() - start_time
            
            result = parser.parse(response.content)
            
            # Create mapping to ensure each technical category belongs to exactly ONE conceptual category
            technical_to_conceptual_mapping = {}
            conceptual_structure = {}
            
            # First pass: Create conceptual structure and build exclusive mapping
            for concept in result.conceptual_categories:
                conceptual_structure[concept.category_name] = {
                    "description": concept.description,
                    "technical_subcategories": [],
                    "codes": {},
                    "total_issues": 0
                }
                
                # Assign technical subcategories exclusively to this conceptual category
                for subcategory in concept.subcategories:
                    if subcategory in technical_categories:
                        if subcategory not in technical_to_conceptual_mapping:
                            # First time seeing this technical category - assign it
                            technical_to_conceptual_mapping[subcategory] = concept.category_name
                            conceptual_structure[concept.category_name]["technical_subcategories"].append(subcategory)
                            logger.info(f"Mapped '{subcategory}' → '{concept.category_name}'")
                        else:
                            # Technical category already assigned to another conceptual category
                            existing_assignment = technical_to_conceptual_mapping[subcategory]
                            logger.warning(f"Technical category '{subcategory}' already assigned to '{existing_assignment}', skipping assignment to '{concept.category_name}'")
                    else:
                        logger.warning(f"Technical subcategory '{subcategory}' not found in technical categories")
            
            # Second pass: Map codes based on the exclusive technical-to-conceptual mapping
            for technical_category, conceptual_category in technical_to_conceptual_mapping.items():
                subcategory_data = technical_categories[technical_category]
                
                if isinstance(subcategory_data, dict):
                    # Expected format: {code_name: [issue1, issue2, ...]}
                    for code_name, issue_list in subcategory_data.items():
                        conceptual_structure[conceptual_category]["codes"][code_name] = issue_list
                
                elif isinstance(subcategory_data, list):
                    # Handle case where technical_categories[subcategory] is just a list of codes
                    logger.info(f"Technical category '{technical_category}' has list format, mapping codes to issues...")
                    
                    # Create a mapping from codes to issues using self.all_code_instances
                    code_to_issues_map = defaultdict(list)
                    for code_instance in self.all_code_instances:
                        code_to_issues_map[code_instance.code].append(code_instance.issue_key)
                    
                    # Map each code in the list to its issues
                    for code_name in subcategory_data:
                        issue_list = list(set(code_to_issues_map.get(code_name, [])))  # Remove duplicates
                        if issue_list:  # Only add if we found issues for this code
                            conceptual_structure[conceptual_category]["codes"][code_name] = issue_list
                        else:
                            logger.warning(f"No issues found for code '{code_name}' in technical category '{technical_category}'")
                
                else:
                    logger.warning(f"Unexpected format for technical category '{technical_category}': {type(subcategory_data)}")
            
            # Third pass: Count total issues per conceptual category (now without duplication)
            for concept_name, concept_data in conceptual_structure.items():
                total_issues = set()
                for code, issues in concept_data["codes"].items():
                    total_issues.update(issues)
                concept_data["total_issues"] = len(total_issues)
                
                # Debug: Log the results
                logger.info(f"Conceptual category '{concept_name}': {len(concept_data['codes'])} codes, {concept_data['total_issues']} issues")
            
            # Validation: Check for unmapped technical categories
            unmapped_categories = set(technical_categories.keys()) - set(technical_to_conceptual_mapping.keys())
            if unmapped_categories:
                logger.warning(f"Unmapped technical categories: {unmapped_categories}")
                # Create a catch-all category for unmapped technical categories
                if unmapped_categories:
                    catch_all_name = "Other Issues" if dataset_type == "a11y" else "Other Development Issues"
                    conceptual_structure[catch_all_name] = {
                        "description": "Technical categories that could not be mapped to specific conceptual themes.",
                        "technical_subcategories": list(unmapped_categories),
                        "codes": {},
                        "total_issues": 0
                    }
                    
                    # Map unmapped categories to catch-all
                    for unmapped_cat in unmapped_categories:
                        technical_to_conceptual_mapping[unmapped_cat] = catch_all_name
                        subcategory_data = technical_categories[unmapped_cat]
                        
                        if isinstance(subcategory_data, dict):
                            for code_name, issue_list in subcategory_data.items():
                                conceptual_structure[catch_all_name]["codes"][code_name] = issue_list
                        elif isinstance(subcategory_data, list):
                            code_to_issues_map = defaultdict(list)
                            for code_instance in self.all_code_instances:
                                code_to_issues_map[code_instance.code].append(code_instance.issue_key)
                            
                            for code_name in subcategory_data:
                                issue_list = list(set(code_to_issues_map.get(code_name, [])))
                                if issue_list:
                                    conceptual_structure[catch_all_name]["codes"][code_name] = issue_list
                    
                    # Count issues for catch-all category
                    total_issues = set()
                    for code, issues in conceptual_structure[catch_all_name]["codes"].items():
                        total_issues.update(issues)
                    conceptual_structure[catch_all_name]["total_issues"] = len(total_issues)
                    
                    logger.info(f"Created catch-all category '{catch_all_name}' with {len(unmapped_categories)} technical categories")
            
            # Final validation: Check total issue count
            total_conceptual_issues = sum(cat["total_issues"] for cat in conceptual_structure.values())
            total_technical_issues = len(set(instance.issue_key for instance in self.all_code_instances))
            logger.info(f"Conceptual categorization validation:")
            logger.info(f"  - Total technical categories: {len(technical_categories)}")
            logger.info(f"  - Mapped technical categories: {len(technical_to_conceptual_mapping)}")
            logger.info(f"  - Total issues in conceptual analysis: {total_conceptual_issues}")
            logger.info(f"  - Total unique issues in dataset: {total_technical_issues}")
            logger.info(f"  - Issue count match: {'✓' if total_conceptual_issues == total_technical_issues else '✗'}")
            
            logger.info(f"Conceptual categorization complete: {len(conceptual_structure)} high-level categories")
            
            return {
                "conceptual_categories": conceptual_structure,
                "technical_mapping": technical_to_conceptual_mapping,
                "metadata": {
                    "processing_time_seconds": round(processing_time, 2),
                    "tokens_used": cb.total_tokens,
                    "cost_estimate": cb.total_cost,
                    "conceptual_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "mapping_validation": {
                        "total_technical_categories": len(technical_categories),
                        "mapped_technical_categories": len(technical_to_conceptual_mapping),
                        "total_conceptual_issues": total_conceptual_issues,
                        "total_unique_issues": total_technical_issues,
                        "issue_count_match": total_conceptual_issues == total_technical_issues
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating conceptual categorization: {e}")
            return {"conceptual_categories": {}, "metadata": {"error": str(e)}}
    
    def _create_conceptual_prompt(self, technical_categories: Dict[str, List[str]], dataset_type: str) -> str:
        """Create prompt for conceptual categorization"""
        
        # Format technical categories for the prompt
        technical_summary = ""
        for category_name, codes in technical_categories.items():
            technical_summary += f"\n**{category_name}** ({len(codes)} codes)\n"
        
        if dataset_type == "a11y":
            return f"""
    CONCEPTUAL CATEGORIZATION FOR ACCESSIBILITY RESEARCH

    I have technical categories from accessibility issue analysis that need to be organized into high-level conceptual categories suitable for academic research and paper discussion.

    CURRENT TECHNICAL CATEGORIES:
    {technical_summary}

    TASK: Create high-level conceptual categories that group these technical categories based on UNDERLYING CAUSES and BROADER THEMES suitable for academic discussion about accessibility in software development.

    INSTRUCTIONS:
    - You may use the suggested categories above IF they fit the data well
    - You are encouraged to create NEW categories if they better represent the underlying patterns in the technical categories
    - You may modify the suggested category names to better fit your analysis
    - Focus on creating categories that help understand WHY accessibility issues occur and what they reveal about the development process
    - Ensure each technical category is assigned to exactly one conceptual category
    - Create categories that are meaningful for academic discussion and research

    For each conceptual category, provide:
    1. category_name: Clear, descriptive name suitable for academic discussion
    2. description: Brief explanation of what this category represents and why it's important
    3. subcategories: List of technical category names that belong to this conceptual category

    Your goal is to create the most meaningful and insightful categorization based on the actual data, not to force-fit the suggested framework.
    """
        else:
            return f"""
    CONCEPTUAL CATEGORIZATION FOR SOFTWARE DEVELOPMENT RESEARCH

    I have technical categories from software development issue analysis that need to be organized into high-level conceptual categories suitable for academic research and paper discussion.

    CURRENT TECHNICAL CATEGORIES:
    {technical_summary}

    TASK: Create high-level conceptual categories that group these technical categories based on UNDERLYING CAUSES and BROADER THEMES suitable for academic discussion about software development processes.

    SUGGESTED CONCEPTUAL FRAMEWORK (use if applicable, but feel free to create your own categories):

    **Development Process and Workflow Issues**
    - Categories related to testing, integration, code review, and development workflow problems
    - Issues with testing procedures, verification processes, documentation
    - Problems in the software development lifecycle and process management

    **Code Quality and Standards Issues**
    - Categories related to coding standards, code quality, documentation, and maintainability
    - Issues with code consistency, style guidelines, best practices
    - Problems with code organization, naming conventions, and technical debt

    **Technical Integration and Compatibility Issues**
    - Categories related to system integration, third-party libraries, version conflicts
    - Issues with compatibility across different systems, browsers, or platforms
    - Problems with technical dependencies and system interoperability

    **User Interface and User Experience Issues**
    - Categories related to UI design, layout, visual elements, and user interaction
    - Issues with interface design, responsiveness, and user experience
    - Problems with form design, navigation, and content presentation

    **Testing and Quality Assurance Issues**
    - Categories related to test failures, testing procedures, and quality assurance
    - Issues with automated testing, manual testing, and verification processes
    - Problems with test coverage, test reliability, and quality control

    INSTRUCTIONS:
    - You may use the suggested categories above IF they fit the data well
    - You are encouraged to create NEW categories if they better represent the underlying patterns in the technical categories
    - You may modify the suggested category names to better fit your analysis
    - Focus on creating categories that help understand the UNDERLYING CAUSES of development issues
    - Ensure each technical category is assigned to exactly one conceptual category
    - Create categories that are meaningful for academic discussion and research

    For each conceptual category, provide:
    1. category_name: Clear, descriptive name suitable for academic discussion
    2. description: Brief explanation of what this category represents and its significance in software development
    3. subcategories: List of technical category names that belong to this conceptual category

    Your goal is to create the most meaningful and insightful categorization based on the actual data, not to force-fit the suggested framework.
    """
    
    def merge_with_existing(self, existing_categories: Dict[str, List[str]], existing_mappings: Dict[str, str],
                        new_categories: Dict[str, List[str]], new_mappings: Dict[str, str],
                        dataset_type: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """Merge new batch results with existing consolidated results"""
        
        # Combine categories - look for similar ones first
        merged_categories = existing_categories.copy()
        
        # Track category name changes for mapping updates
        category_mapping_changes = {}  # old_name -> new_name
        
        for new_cat_name, new_codes in new_categories.items():
            # Look for similar existing category
            similar_existing = self.find_similar_category(new_cat_name, existing_categories.keys())
            
            if similar_existing and len(merged_categories[similar_existing]) < 50:  # Don't merge into overly large categories
                # Merge with existing similar category
                merged_categories[similar_existing].extend(new_codes)
                merged_categories[similar_existing] = list(set(merged_categories[similar_existing]))  # Remove duplicates
                category_mapping_changes[new_cat_name] = similar_existing
                logger.info(f"Merged '{new_cat_name}' into existing '{similar_existing}'")
            else:
                # Add as new category
                merged_categories[new_cat_name] = new_codes
                # No mapping change needed
        
        # Update ALL mappings based on final category structure
        updated_mappings = {}
        
        # Update existing mappings
        for code, old_category in existing_mappings.items():
            # Find where this code actually ended up
            final_category = self.find_code_final_category(code, merged_categories)
            if final_category:
                updated_mappings[code] = final_category
            else:
                logger.warning(f"Code {code} from existing mappings not found in any final category")
                updated_mappings[code] = old_category  # Keep old mapping as fallback
        
        # Update new mappings
        for code, old_category in new_mappings.items():
            # Check if the category was renamed due to merging
            if old_category in category_mapping_changes:
                final_category = category_mapping_changes[old_category]
            else:
                # Find where this code actually ended up
                final_category = self.find_code_final_category(code, merged_categories)
            
            if final_category:
                updated_mappings[code] = final_category
            else:
                logger.warning(f"Code {code} from new mappings not found in any final category")
                updated_mappings[code] = old_category  # Keep old mapping as fallback
        
        return merged_categories, updated_mappings

    def find_similar_category(self, category_name: str, existing_categories: List[str]) -> Optional[str]:
        """Find existing category similar to new category"""
        from difflib import SequenceMatcher
        
        # Normalize category names for comparison
        normalized_new = category_name.lower().replace(" and ", " ").replace("&", "")
        
        best_match = None
        best_score = 0
        
        for existing_cat in existing_categories:
            normalized_existing = existing_cat.lower().replace(" and ", " ").replace("&", "")
            
            # Check similarity
            similarity = SequenceMatcher(None, normalized_new, normalized_existing).ratio()
            
            # Check for keyword overlap
            new_keywords = set(normalized_new.split())
            existing_keywords = set(normalized_existing.split())
            keyword_overlap = len(new_keywords & existing_keywords) / max(len(new_keywords), len(existing_keywords))
            
            # Combined score
            combined_score = (similarity * 0.7) + (keyword_overlap * 0.3)
            
            if combined_score > best_score and combined_score > 0.6:  # Threshold
                best_score = combined_score
                best_match = existing_cat
        
        return best_match
    
    def intermediate_consolidation(self, categories: Dict[str, List[str]], mappings: Dict[str, str], dataset_type: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """Perform intermediate consolidation when too many categories accumulate"""
        
        if len(categories) <= 15:
            return categories, mappings
        
        logger.info(f"Performing intermediate consolidation of {len(categories)} categories...")
        
        # Use LLM to merge similar categories
        merge_prompt = self._create_intermediate_merge_prompt(categories, dataset_type)
        
        try:
            parser = PydanticOutputParser(pydantic_object=ConsolidationResultWithMapping)
            system_message = SystemMessage(content="You are an expert at merging similar categories. Consolidate the provided categories into fewer, more general categories while preserving all codes.")
            human_message = HumanMessage(content=f"{merge_prompt}\n\n{parser.get_format_instructions()}")
            
            with get_openai_callback() as cb:
                start_time = time.time()
                response = self.llm.invoke([system_message, human_message])
                processing_time = time.time() - start_time
            
            result = parser.parse(response.content)
            
            # Convert to dictionary format
            consolidated_categories = {}
            for category in result.categories:
                consolidated_categories[category.category_name] = category.codes
            
            # REBUILD mappings from scratch based on where codes actually ended up
            updated_mappings = {}
            for original_code in mappings.keys():
                # Find which new category this code ended up in
                new_category = self.find_code_final_category(original_code, consolidated_categories)
                if new_category:
                    updated_mappings[original_code] = new_category
                else:
                    logger.error(f"Code {original_code} lost during intermediate consolidation!")
                    # Try to find it in any category (maybe there was a spelling variation)
                    found = False
                    for cat_name, codes in consolidated_categories.items():
                        if any(original_code.lower() in code.lower() or code.lower() in original_code.lower() for code in codes):
                            logger.warning(f"Found similar code for {original_code} in category {cat_name}")
                            updated_mappings[original_code] = cat_name
                            found = True
                            break
                    
                    if not found:
                        # Last resort: keep old mapping
                        updated_mappings[original_code] = mappings[original_code]
            
            logger.info(f"Intermediate consolidation: {len(categories)} → {len(consolidated_categories)} categories")
            logger.info(f"Updated {len(updated_mappings)} code mappings")
            
            return consolidated_categories, updated_mappings
            
        except Exception as e:
            logger.error(f"Error in intermediate consolidation: {e}")
            return categories, mappings

    def find_code_final_category(self, code: str, categories: Dict[str, List[str]]) -> Optional[str]:
        """Find which category a code belongs to"""
        for category_name, codes in categories.items():
            if code in codes:
                return category_name
        return None

    def _create_intermediate_merge_prompt(self, categories: Dict[str, List[str]], dataset_type: str) -> str:
        """Create prompt for intermediate consolidation"""
        categories_text = ""
        total_codes = 0
        for category_name, codes in categories.items():
            categories_text += f"\n**{category_name}** ({len(codes)} codes)\n"
            # Show first few codes as examples
            example_codes = codes[:3]
            for code in example_codes:
                categories_text += f"- {code}\n"
            if len(codes) > 3:
                categories_text += f"... and {len(codes) - 3} more codes\n"
            total_codes += len(codes)
        
        rq_type = "accessibility" if dataset_type == "a11y" else "non-accessibility"
        
        return f"""
    INTERMEDIATE CONSOLIDATION - {rq_type.upper()} ISSUES

    I have {len(categories)} categories with {total_codes} total codes that need to be consolidated into fewer, more manageable categories (target: 10-15 categories).

    CURRENT CATEGORIES:
    {categories_text}

    CONSOLIDATION TASK:
    1. Merge categories that represent similar concepts
    2. Create broader, more general category names
    3. Ensure ALL codes are preserved in the final result
    4. Aim for 10-15 final categories

    MERGE GUIDELINES:
    - Look for categories that could be combined under broader themes
    - Preserve all codes - no code should be lost
    - Create clear, descriptive category names that encompass the merged concepts
    - Group by underlying problem types rather than specific technical details

    Target: Reduce to approximately 10-15 well-organized categories.
    """
    
    def consolidate_with_batching(self, code_instances: List[CodeInstance], dataset_type: str) -> Dict[str, Any]:
        """Rolling consolidation method with merge tracking"""
        if not code_instances:
            logger.warning("No code instances to consolidate")
            return {"categories": {}, "metadata": {"total_codes": 0}}
        
        self.all_code_instances = code_instances
        
        # Extract just the codes for processing
        all_codes = [instance.code for instance in code_instances]
        unique_codes = list(set(all_codes))
        
        logger.info(f"Starting rolling consolidation of {len(all_codes)} code instances ({len(unique_codes)} unique codes)")
        logger.info(f"Batch size: {self.config.batch_size}")
        
        # Create batches based on issue count to distribute codes evenly
        issue_to_codes = defaultdict(list)
        for instance in code_instances:
            issue_to_codes[instance.issue_key].append(instance.code)
        
        # Group issues into batches
        issues = list(issue_to_codes.keys())
        total_batches = (len(issues) + self.config.batch_size - 1) // self.config.batch_size
        
        logger.info(f"Processing {len(issues)} issues in {total_batches} batches of ~{self.config.batch_size} issues each")
        
        # Initialize consolidated results
        consolidated_categories = {}
        consolidated_mappings = {}
        
        # Process each batch with rolling merge
        for batch_num in range(total_batches):
            start_idx = batch_num * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(issues))
            batch_issues = issues[start_idx:end_idx]
            
            # Collect all codes from issues in this batch
            batch_codes = []
            for issue_key in batch_issues:
                batch_codes.extend(issue_to_codes[issue_key])
            
            logger.info(f"Batch {batch_num + 1}: Issues {start_idx + 1}-{end_idx} ({len(batch_issues)} issues, {len(batch_codes)} codes)")
            
            # Process current batch
            batch_categories, batch_mappings = self.consolidate_batch_simplified(batch_codes, dataset_type, batch_num + 1, total_batches)
            
            # Save intermediate result
            self._save_intermediate_result(batch_num + 1, batch_categories, batch_issues, batch_codes)
            
            # Rolling merge with existing consolidated results
            if not consolidated_categories:
                # First batch - use as-is
                consolidated_categories = batch_categories
                consolidated_mappings = batch_mappings
                logger.info(f"First batch: {len(consolidated_categories)} initial categories")
            else:
                # Merge with existing results
                logger.info(f"Before merge: {len(consolidated_categories)} existing + {len(batch_categories)} new categories")
                consolidated_categories, consolidated_mappings = self.merge_with_existing(
                    consolidated_categories, consolidated_mappings,
                    batch_categories, batch_mappings,
                    dataset_type
                )
                logger.info(f"After merge: {len(consolidated_categories)} total categories")
            
            # Prevent category explosion - consolidate if too many
            if len(consolidated_categories) > 25:  # Threshold
                logger.info(f"Too many categories ({len(consolidated_categories)}), performing intermediate consolidation...")
                consolidated_categories, consolidated_mappings = self.intermediate_consolidation(
                    consolidated_categories, consolidated_mappings, dataset_type
                )
                logger.info(f"After intermediate consolidation: {len(consolidated_categories)} categories")
            
            # Small delay between batches
            if batch_num < total_batches - 1:
                time.sleep(1)
        
        # VALIDATION: Ensure all codes are preserved with detailed tracking
        validation_result = self.validate_code_consolidation_with_explicit_mappings(consolidated_categories, consolidated_mappings)
        # RESOLVE MAPPING ERRORS: Give error codes another chance
        if validation_result["mapping_errors"]:
            logger.info(f"Found {len(validation_result['mapping_errors'])} mapping errors, attempting resolution...")
            consolidated_categories, consolidated_mappings = self.resolve_mapping_errors(
                consolidated_categories, 
                consolidated_mappings, 
                validation_result["mapping_errors"], 
                dataset_type
            )
            
            # Re-validate after error resolution
            logger.info("Re-validating after mapping error resolution...")
            validation_result = self.validate_code_consolidation_with_explicit_mappings(consolidated_categories, consolidated_mappings)
            
            if validation_result["mapping_errors"]:
                logger.warning(f"Still have {len(validation_result['mapping_errors'])} mapping errors after resolution")
            else:
                logger.info("All mapping errors resolved successfully!")

        # SEMANTIC VALIDATION: Verify categorization quality
        logger.info("Performing semantic validation of categorization quality...")
        semantic_validation = self.validate_semantic_categorization(
            consolidated_categories,
            consolidated_mappings,
            dataset_type,
            sample_size=25
        )

        # Log semantic validation results
        if semantic_validation.get("accuracy_percentage"):
            logger.info(f"Semantic validation results:")
            logger.info(f"  - Accuracy: {semantic_validation['accuracy_percentage']:.1f}%")
            logger.info(f"  - Quality score: {semantic_validation['overall_quality_score']}/10")
            if semantic_validation["incorrectly_categorized"]:
                logger.warning(f"  - {len(semantic_validation['incorrectly_categorized'])} codes may be miscategorized")
        else:
            logger.warning("Semantic validation failed or returned incomplete results")
        
        # Create detailed result
        detailed_result = self.create_detailed_final_result(consolidated_categories)
        
        # Add conceptual categorization layer
        logger.info("Creating conceptual categorization for academic analysis...")
        conceptual_result = self.create_conceptual_categorization(consolidated_categories, dataset_type)
        
        # Combine all results
        final_result = {
            "conceptual_analysis": conceptual_result["conceptual_categories"],
            "technical_categories": detailed_result["categories"],
            "metadata": detailed_result["metadata"],
            "validation": validation_result,
            "batch_processing": {
                "total_batches": len(self.batch_results),
                "batch_results": self.batch_results,
                "total_cost": sum(batch["metadata"]["cost_estimate"] for batch in self.batch_results) + conceptual_result["metadata"].get("cost_estimate", 0),
                "consolidation_method": "rolling_merge"
            },
            "conceptual_metadata": conceptual_result["metadata"]
        }
        
        return final_result
    
    def _save_intermediate_result(self, batch_num: int, categories: Dict[str, List[str]], issues: List[str], codes: List[str]):
        """Save intermediate batch result"""
        intermediate_result = {
            "batch_number": batch_num,
            "issues_processed": issues,
            "codes_in_batch": codes,
            "categories": categories,
            "merge_info": {
                f"Batch {batch_num} merged codes": f"{len(set(codes))} unique codes from {len(issues)} issues into {len(categories)} categories"
            }
        }
        
        # Save to intermediate results directory
        intermediate_dir = Path(self.config.output_dir) / "intermediate_results"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        intermediate_file = intermediate_dir / f"batch_{batch_num:03d}_result.json"
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(intermediate_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Intermediate result saved: {intermediate_file}")

    def resolve_mapping_errors(self, final_categories: Dict[str, List[str]], code_mappings: Dict[str, str], 
                          mapping_errors: List[str], dataset_type: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """Give mapping error codes another chance to be properly categorized"""
        
        if not mapping_errors:
            return final_categories, code_mappings
        
        logger.info(f"Resolving {len(mapping_errors)} mapping error codes...")
        
        # Create a prompt to categorize the error codes
        error_codes_context = self._create_error_codes_context(mapping_errors)
        categories_summary = self._create_categories_summary(final_categories)
        
        resolution_prompt = f"""
    MAPPING ERROR RESOLUTION

    I have {len(mapping_errors)} codes that had mapping errors during consolidation. Please categorize these codes into the existing final categories.

    EXISTING FINAL CATEGORIES:
    {categories_summary}

    CODES TO CATEGORIZE:
    {error_codes_context}

    TASK: For each error code, assign it to the most appropriate existing category from the list above. If absolutely necessary, you can create 1-2 new categories, but prefer to use existing ones.

    IMPORTANT: Every code must be assigned to exactly one category.
    """

        try:
            parser = PydanticOutputParser(pydantic_object=ConsolidationResultWithMapping)
            system_message = SystemMessage(content="You are an expert at categorizing codes. Assign the provided codes to the most appropriate categories.")
            human_message = HumanMessage(content=f"{resolution_prompt}\n\n{parser.get_format_instructions()}")
            
            with get_openai_callback() as cb:
                start_time = time.time()
                response = self.llm.invoke([system_message, human_message])
                processing_time = time.time() - start_time
            
            result = parser.parse(response.content)
            
            # Update final categories with resolved codes
            updated_categories = final_categories.copy()
            updated_mappings = code_mappings.copy()
            
            for category in result.categories:
                category_name = category.category_name
                codes = category.codes
                
                # If it's a new category, add it
                if category_name not in updated_categories:
                    updated_categories[category_name] = []
                
                # Add codes to the category and update mappings
                for code in codes:
                    if code in mapping_errors:  # Only process error codes
                        # Add to category if not already there
                        if code not in updated_categories[category_name]:
                            updated_categories[category_name].append(code)
                        
                        # Update mapping
                        updated_mappings[code] = category_name
                        
                        logger.info(f"Resolved: {code} → {category_name}")
            
            # Verify all error codes were resolved
            unresolved = set(mapping_errors) - set(updated_mappings.keys())
            if unresolved:
                logger.warning(f"Still unresolved codes: {unresolved}")
                # Add to a catch-all category
                catch_all = "Unresolved Issues"
                if catch_all not in updated_categories:
                    updated_categories[catch_all] = []
                
                for code in unresolved:
                    updated_categories[catch_all].append(code)
                    updated_mappings[code] = catch_all
            
            logger.info(f"Mapping error resolution complete: ${cb.total_cost:.4f}")
            logger.info(f"Resolved {len(mapping_errors)} codes, created/updated {len([c for c in result.categories])} categories")
            
            return updated_categories, updated_mappings
            
        except Exception as e:
            logger.error(f"Error resolving mapping errors: {e}")
            
            # Fallback: add all error codes to a catch-all category
            updated_categories = final_categories.copy()
            updated_mappings = code_mappings.copy()
            
            fallback_category = "Mapping Error Recovery"
            updated_categories[fallback_category] = mapping_errors.copy()
            
            for code in mapping_errors:
                updated_mappings[code] = fallback_category
            
            logger.info(f"Fallback: Added {len(mapping_errors)} codes to '{fallback_category}' category")
            
            return updated_categories, updated_mappings

    def _create_error_codes_context(self, error_codes: List[str]) -> str:
        """Create context for error codes including their source text"""
        codes_text = ""
        
        # Get context for each error code
        code_contexts = defaultdict(list)
        for instance in self.all_code_instances:
            if instance.code in error_codes and instance.source_text:
                code_contexts[instance.code].append(instance.source_text)
        
        for code in error_codes:
            codes_text += f"- {code}\n"
            if code in code_contexts:
                # Show up to 2 context examples
                for i, context in enumerate(code_contexts[code][:2], 1):
                    codes_text += f"  Context {i}: {context}\n"
                if len(code_contexts[code]) > 2:
                    codes_text += f"  ... and {len(code_contexts[code]) - 2} more contexts\n"
        
        return codes_text

    def _create_categories_summary(self, categories: Dict[str, List[str]]) -> str:
        """Create a summary of existing categories"""
        summary = ""
        for category_name, codes in categories.items():
            summary += f"\n**{category_name}** ({len(codes)} codes)\n"
            # Show a few example codes
            example_codes = codes[:3]
            for code in example_codes:
                summary += f"  - {code}\n"
            if len(codes) > 3:
                summary += f"  ... and {len(codes) - 3} more codes\n"
        
        return summary
    
    def _detect_dataset_type(self) -> str:
        """Detect dataset type from input file path or config"""
        if "non_a11y" in self.config.input_file.lower() or "non_a11y" in self.config.output_dir.lower():
            return "non_a11y"
        else:
            return "a11y"
    
    def save_final_results(self, detailed_result: Dict[str, Any]):
        """Save final consolidated results"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Main result file
        dataset_suffix = "_non_a11y" if detailed_result["metadata"]["dataset_type"] == "non_a11y" else ""
        output_file = output_dir / f"consolidated_{self.config.research_question}{dataset_suffix}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Final consolidated results saved: {output_file}")
        
        # Also create a summary file for easier reading
        summary_file = output_dir / f"consolidation_summary_{self.config.research_question}{dataset_suffix}.txt"
        self._create_summary_file(detailed_result, summary_file)
    
    def _create_summary_file(self, detailed_result: Dict[str, Any], summary_file: Path):
        """Create human-readable summary file with both conceptual and technical categories"""
        with open(summary_file, 'w', encoding='utf-8') as f:
            metadata = detailed_result["metadata"]
            f.write(f"CODE CONSOLIDATION SUMMARY\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Research Question: {metadata['research_question']}\n")
            f.write(f"Dataset Type: {metadata['dataset_type']}\n")
            f.write(f"Total Technical Categories: {metadata['total_categories']}\n")
            f.write(f"Total Unique Codes: {metadata['total_unique_codes']}\n")
            f.write(f"Total Code Instances: {metadata['total_code_instances']}\n")
            f.write(f"Total Issues: {metadata['total_unique_issues']}\n")
            f.write(f"Batch Size: {metadata['batch_size']}\n")
            f.write(f"Total Batches: {metadata['total_batches']}\n")
            f.write(f"Model Used: {metadata['model_used']}\n")
            f.write(f"Timestamp: {metadata['consolidation_timestamp']}\n\n")
            
            # Conceptual categories for academic analysis
            f.write("CONCEPTUAL CATEGORIES (FOR ACADEMIC ANALYSIS)\n")
            f.write("=" * 50 + "\n\n")
            
            conceptual_categories = detailed_result.get("conceptual_analysis", {})
            for concept_name, concept_data in conceptual_categories.items():
                f.write(f"{concept_name} ({concept_data['total_issues']} issues)\n")
                f.write("-" * len(concept_name) + "\n")
                f.write(f"Description: {concept_data['description']}\n")
                f.write(f"Technical Subcategories: {', '.join(concept_data['technical_subcategories'])}\n")
                
                # Show top codes in this conceptual category
                codes_in_concept = concept_data['codes']
                if codes_in_concept:
                    f.write(f"Top Codes ({len(codes_in_concept)} total):\n")
                    # Sort by number of issues
                    sorted_codes = sorted(codes_in_concept.items(), key=lambda x: len(x[1]), reverse=True)
                    for code, issues in sorted_codes[:5]:  # Show top 5
                        f.write(f"  • {code} ({len(issues)} issues)\n")
                    if len(sorted_codes) > 5:
                        f.write(f"  ... and {len(sorted_codes) - 5} more codes\n")
                f.write("\n")
            
            # Technical categories breakdown
            f.write("\nTECHNICAL CATEGORIES (DETAILED BREAKDOWN)\n")
            f.write("=" * 50 + "\n\n")
            
            technical_categories = detailed_result.get("technical_categories", {})
            for category_name, codes_dict in technical_categories.items():
                f.write(f"{category_name}\n")
                f.write("-" * len(category_name) + "\n")
                
                for code, issue_keys in codes_dict.items():
                    f.write(f"  • {code} ({len(issue_keys)} issues)\n")
                    f.write(f"    Issues: {', '.join(issue_keys[:3])}")
                    if len(issue_keys) > 3:
                        f.write(f" ... and {len(issue_keys) - 3} more")
                    f.write("\n")
                f.write("\n")
        
        logger.info(f"Enhanced summary file created: {summary_file}")
        
        # Also create a conceptual-only summary for easier academic discussion
        conceptual_summary_file = summary_file.parent / f"conceptual_summary_{self.config.research_question}{'_non_a11y' if detailed_result['metadata']['dataset_type'] == 'non_a11y' else ''}.txt"
        self._create_conceptual_summary_file(detailed_result, conceptual_summary_file)
    
    def _create_conceptual_summary_file(self, detailed_result: Dict[str, Any], summary_file: Path):
        """Create a summary file focused only on conceptual categories for academic use"""
        with open(summary_file, 'w', encoding='utf-8') as f:
            metadata = detailed_result["metadata"]
            f.write(f"CONCEPTUAL ANALYSIS SUMMARY - FOR ACADEMIC DISCUSSION\n")
            f.write(f"=" * 60 + "\n\n")
            f.write(f"Research Question: {metadata['research_question']}\n")
            f.write(f"Dataset: {'Accessibility Issues' if metadata['dataset_type'] == 'a11y' else 'Non-Accessibility Issues'}\n")
            f.write(f"Total Issues Analyzed: {metadata['total_unique_issues']}\n")
            f.write(f"Analysis Date: {metadata['consolidation_timestamp']}\n\n")
            
            conceptual_categories = detailed_result.get("conceptual_analysis", {})
            
            # Summary statistics
            f.write("OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"High-level Categories: {len(conceptual_categories)}\n")
            total_conceptual_issues = sum(cat['total_issues'] for cat in conceptual_categories.values())
            f.write(f"Total Issues Categorized: {total_conceptual_issues}\n\n")
            
            # Detailed conceptual breakdown
            f.write("CONCEPTUAL CATEGORIES\n")
            f.write("-" * 30 + "\n\n")
            
            # Sort by number of issues
            sorted_concepts = sorted(conceptual_categories.items(), 
                                   key=lambda x: x[1]['total_issues'], reverse=True)
            
            for concept_name, concept_data in sorted_concepts:
                total_issues = concept_data['total_issues']
                percentage = (total_issues / metadata['total_unique_issues']) * 100 if metadata['total_unique_issues'] > 0 else 0
                
                f.write(f"{concept_name}\n")
                f.write("=" * len(concept_name) + "\n")
                f.write(f"Issues: {total_issues} ({percentage:.1f}% of total)\n")
                f.write(f"Description: {concept_data['description']}\n\n")
                
                f.write("Technical Subcategories Included:\n")
                for subcategory in concept_data['technical_subcategories']:
                    f.write(f"  • {subcategory}\n")
                
                f.write(f"\nMost Frequent Issues in this Category:\n")
                codes_in_concept = concept_data['codes']
                if codes_in_concept:
                    sorted_codes = sorted(codes_in_concept.items(), 
                                        key=lambda x: len(x[1]), reverse=True)
                    for code, issues in sorted_codes[:3]:  # Show top 3
                        f.write(f"  • {code} ({len(issues)} issues)\n")
                
                f.write("\n" + "-" * 40 + "\n\n")
        
        logger.info(f"Conceptual summary for academic use created: {summary_file}")


def load_environment():
    """Load environment variables from .env file"""
    env_file = Path('.env')
    if not env_file.exists():
        env_file = Path('../.env')
    
    if env_file.exists():
        load_dotenv(env_file)
        logger.info(f"Loaded environment variables from {env_file}")
    else:
        logger.warning("No .env file found. Make sure OPENAI_API_KEY is set in environment.")


def parse_arguments() -> ConsolidationConfig:
    """Parse command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Consolidate codes from research question analysis with batch processing',
        epilog="""
Examples:
  python consolidate_codes.py --dataset a11y --rq role_sequence
  python consolidate_codes.py --dataset non_a11y --rq role_sequence --model gpt-4o --batch-size 30
  python consolidate_codes.py --input Data/analysis_results/role_sequence_results.json --rq role_sequence
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input specification - either dataset or direct file path
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--dataset', choices=['a11y', 'non_a11y'], 
                           help='Dataset type (uses standard file paths)')
    input_group.add_argument('--input', dest='input_file', 
                           help='Direct path to analysis results JSON file')

    parser.add_argument('--rq', '--research-question', dest='research_question', required=True,
                        choices=['role_sequence', 'participant_influence', 'solution_development'],
                        help='Research question type')
    parser.add_argument('--model', default='gpt-4o', 
                       choices=['gpt-4o', 'gpt-4', 'gpt-3.5-turbo'],
                       help='OpenAI model to use (default: gpt-4o)')
    parser.add_argument('--batch-size', type=int, default=40, 
                       help='Number of issues per batch (default: 40)')
    parser.add_argument('--output-dir', help='Output directory (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Determine input file based on dataset or direct input
    if args.dataset:
        if args.dataset == 'a11y':
            args.input_file = "Data/analysis_results/role_sequence_results.json"
        elif args.dataset == 'non_a11y':
            args.input_file = "Data/analysis_results_non_a11y/non_a11y_role_sequence_results.json"
    
    # Auto-generate output directory if not specified
    if not args.output_dir:
        if args.dataset:
            if args.dataset == 'a11y':
                args.output_dir = "Data/analysis_results/consolidated_results"
            else:
                args.output_dir = "Data/analysis_results_non_a11y/consolidated_results"
        else:
            # Use input file path to determine output
            input_path = Path(args.input_file)
            if "non_a11y" in str(input_path):
                args.output_dir = str(input_path.parent / "consolidated_results")
            else:
                args.output_dir = str(input_path.parent / "consolidated_results")
    
    return ConsolidationConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        input_file=args.input_file,
        research_question=args.research_question,
        output_dir=args.output_dir
    )


def main():
    """Main function"""
    load_environment()
    
    try:
        config = parse_arguments()
        
        # Load the analysis results
        logger.info(f"Loading analysis results from: {config.input_file}")
        
        if not Path(config.input_file).exists():
            logger.error(f"Input file not found: {config.input_file}")
            return
        
        with open(config.input_file, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        # Initialize consolidator
        consolidator = CodeConsolidator(config)
        
        # Extract code instances
        code_instances = consolidator.extract_codes(results_data)
        
        if not code_instances:
            logger.warning("No codes found in the results")
            return
        
        logger.info(f"Extracted {len(code_instances)} code instances from {len(results_data.get('results', []))} results")
        
        # Show initial statistics
        all_codes = [instance.code for instance in code_instances]
        unique_codes = set(all_codes)
        unique_issues = set(instance.issue_key for instance in code_instances)
        
        logger.info(f"Statistics:")
        logger.info(f"  - {len(all_codes)} total code instances")
        logger.info(f"  - {len(unique_codes)} unique codes")
        logger.info(f"  - {len(unique_issues)} unique issues")
        logger.info(f"  - {config.batch_size} issues per batch")
        
        # Estimate cost
        estimated_batches = (len(unique_issues) + config.batch_size - 1) // config.batch_size
        estimated_cost = estimated_batches * 0.05  # Rough estimate
        logger.info(f"  - ~{estimated_batches} batches estimated")
        logger.info(f"  - Estimated cost: ~${estimated_cost:.2f}")
        
        # Confirm before proceeding
        confirm = input(f"\nProceed with consolidation? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes']:
            logger.info("Consolidation cancelled")
            return
        
        # Detect dataset type
        dataset_type = consolidator._detect_dataset_type()
        logger.info(f"Dataset type: {dataset_type}")
        
        # Consolidate with batching
        detailed_result = consolidator.consolidate_with_batching(code_instances, dataset_type)
        
        # Save results
        consolidator.save_final_results(detailed_result)
        
        # Print final summary
        metadata = detailed_result["metadata"]
        total_cost = detailed_result["batch_processing"]["total_cost"]
        
        logger.info(f"\n" + "="*60)
        logger.info("CONSOLIDATION COMPLETE!")
        logger.info(f"="*60)
        logger.info(f"Final Categories: {metadata['total_categories']}")
        logger.info(f"Total Codes Consolidated: {metadata['total_unique_codes']}")
        logger.info(f"Issues Analyzed: {metadata['total_unique_issues']}")
        logger.info(f"Batches Processed: {metadata['total_batches']}")
        logger.info(f"Total Cost: ${total_cost:.4f}")
        logger.info(f"Results saved to: {config.output_dir}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        sys.exit(1)