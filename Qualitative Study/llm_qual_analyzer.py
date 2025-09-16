import json
import time
import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, Set
from datetime import datetime
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Add this import for .env file support
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('moodle_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage 
from langchain_community.callbacks.manager import get_openai_callback 
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class BackAndForthCode(BaseModel):
    """Individual back-and-forth code with source evidence"""
    code: str = Field(description="The accessibility-specific code assigned")
    source_text: str = Field(description="The original text excerpt that supports this code assignment")
    source_location: str = Field(description="Location of evidence (e.g., 'comment_3', 'description')")

# Data Models for Analysis Results
class RoleSequenceResult(BaseModel):
    """Result model for role sequence analysis"""
    issue_key: str
    complete_role_sequence: List[str] = Field(description="Complete chronological sequence of roles")
    sequence_explanation: str = Field(description="Brief explanation of the workflow progression")
    back_and_forth_codes: List[BackAndForthCode] = Field(description="Low-level accessibility-specific codes with source evidence for why roles went back to previous roles. Empty list if default sequence.")


class ValidationResult(BaseModel):
    """Result model for validation analysis"""
    original_sequence: List[str] = Field(description="Original sequence from first analysis")
    validated_sequence: List[str] = Field(description="Sequence after validation review")
    original_codes: List[BackAndForthCode] = Field(description="Original back-and-forth codes from first analysis")
    validated_codes: List[BackAndForthCode] = Field(description="Back-and-forth codes after validation review")
    original_explanation: str = Field(description="Original sequence explanation from first analysis")
    validated_explanation: str = Field(description="Sequence explanation after validation review")
    validation_confidence: str = Field(description="High/Medium/Low confidence in validation")
    sequence_changes: List[str] = Field(description="List of changes made during validation")
    validation_explanation: str = Field(description="Explanation of validation decision")
    agreement_status: str = Field(description="Agrees/Disagrees/Partially_Agrees with original")


class ParticipantInfluenceCode(BaseModel):
    """Individual participant influence code with source evidence"""
    code: str = Field(description="Low-level qualitative code describing participant influence")
    source_text: str = Field(description="The original text excerpt that supports this code assignment")
    source_location: str = Field(description="Location of evidence (e.g., 'comment_3', 'comment_7')")

class ParticipantInfluenceResult(BaseModel):
    """Result model for Participant Influence analysis"""
    issue_key: str
    participant_influence_codes: List[ParticipantInfluenceCode] = Field(description="Low-level codes describing participant influence on solution/patch development", default=[])

class ParticipantValidationResult(BaseModel):
    """Result model for Participant Influence validation"""
    original_codes: List[ParticipantInfluenceCode] = Field(description="Original codes from first analysis")
    validated_codes: List[ParticipantInfluenceCode] = Field(description="Codes after validation review")
    validation_changes: List[str] = Field(description="List of changes made during validation")
    validation_explanation: str = Field(description="Explanation of validation decision")
    agreement_status: str = Field(description="Agrees/Disagrees/Partially_Agrees with original")


class AnalysisMetadata(BaseModel):
    """Metadata for analysis results"""
    research_question: str
    model_used: str
    analysis_timestamp: str
    processing_time_seconds: float
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None

class SolutionApproach(BaseModel):
    """Individual solution approach with open coding focused on methodology"""
    solution_codes: List[str] = Field(description="Open codes describing HOW the assignee approached the solution development (methodology, reasoning patterns, analysis approaches). Examples: ['ROOT_CAUSE_VERSUS_SYMPTOM_ANALYSIS', 'WCAG_SUCCESS_CRITERIA_SYSTEMATIC_APPLICATION', 'MULTIPLE_SOLUTION_OPTIONS_EVALUATION'] or ['NOT_OBVIOUS'] when methodology cannot be determined")
    approach_description: str = Field(description="Detailed description of the assignee's problem-solving methodology and reasoning process (not technical implementation details)")
    source_text: str = Field(description="The original text excerpt from assignee comment that shows the methodology")
    source_location: str = Field(description="Location of evidence (e.g., 'comment_3')")
    technical_details: str = Field(description="Specific technical details, tools, standards mentioned in the source (but codes focus on methodology)")

class SolutionDevelopmentResult(BaseModel):
    """Result model for solution development analysis with open coding focused on methodology"""
    issue_key: str
    solution_approaches: List[SolutionApproach] = Field(description="List of solution development methodologies and reasoning approaches with open codes. Empty list if no assignee solution development found", default=[])
    development_summary: str = Field(description="Brief summary of the assignee's overall solution development methodology and problem-solving approach")

class SolutionValidationResult(BaseModel):
    """Result model for solution development validation with open coding"""
    original_approaches: List[SolutionApproach] = Field(description="Original approaches from first analysis")
    validated_approaches: List[SolutionApproach] = Field(description="Approaches after validation review")
    validation_changes: List[str] = Field(description="List of changes made during validation")
    validation_explanation: str = Field(description="Explanation of validation decision")
    agreement_status: str = Field(description="Agrees/Disagrees/Partially_Agrees with original")



@dataclass
class AnalysisConfig:
    """Configuration for analysis"""
    model_name: str = "gpt-4o"
    temperature: float = 0
    max_tokens: int = 5000
    delay_seconds: float = 0.5
    max_issues: Optional[int] = None
    openai_api_key: Optional[str] = None
    continue_analysis: bool = False
    
    # Research question selection
    research_questions: List[str] = None  # NEW: List of research questions to run
    
    # Validation configuration
    enable_validation: bool = False
    validation_model_name: str = "gpt-4o"
    validate_only_non_default: bool = True
    validation_delay_seconds: float = 0.5
    
    @classmethod
    def from_user_input(cls) -> 'AnalysisConfig':
        """Create configuration with user input for number of issues and research questions"""
        
        # Ask about research questions to run
        logger.info("Available Research Questions:")
        logger.info("1. Role Sequence Analysis")
        logger.info("2. WCAG 2.2 Categorization") 
        logger.info("3. Testing & Verification")
        logger.info("4. All Research Questions")
        
        while True:
            rq_choice = input("Select research questions to run (1-3): ").strip()
            
            if rq_choice == "1":
                research_questions = ["role_sequence"]
                logger.info("Selected: Role Sequence Analysis only")
                break
            elif rq_choice == "2":
                research_questions = ["wcag_categorization"]
                logger.info("Selected: WCAG 2.2 Categorization only")
                break
            elif rq_choice == "3":
                research_questions = ["testing_verification"]
                logger.info("Selected: Testing & Verification only")
                break
            elif rq_choice == "4":
                research_questions = ["role_sequence", "wcag_categorization", "testing_verification"]
                logger.info("Selected: All Research Questions")
            else:
                logger.warning("Please enter 1, 2, 3, or 4")
                continue
        
        # Ask about continue vs start over
        logger.info("\nAnalysis Mode:")
        logger.info("1. Start Over (analyze all issues from scratch)")
        logger.info("2. Continue (skip already analyzed issues and append new results)")
        
        while True:
            mode_choice = input("Select mode (1-2): ").strip()
            
            if mode_choice == "1":
                continue_analysis = False
                logger.info("Selected: Start Over mode")
                break
            elif mode_choice == "2":
                continue_analysis = True
                logger.info("Selected: Continue mode")
                break
            else:
                logger.warning("Please enter 1 or 2")
                continue
        
        # Get number of issues from user
        while True:
            try:
                user_input = input("\nHow many issues would you like to analyze? (Enter number or 'all'): ").strip().lower()
                
                if user_input == 'all':
                    max_issues = None
                    logger.info("Will analyze ALL issues")
                    break
                else:
                    max_issues = int(user_input)
                    if max_issues <= 0:
                        logger.warning("Please enter a positive number or 'all'")
                        continue
                    logger.info(f"Will analyze {max_issues} issues")
                    break
            except ValueError:
                logger.warning("Please enter a valid number or 'all'")
                continue
        
        # Get model choice
        logger.info("Available models:")
        logger.info("1. gpt-4o (recommended, more accurate)")
        logger.info("2. gpt-4 (good balance)")
        logger.info("3. gpt-3.5-turbo (faster, cheaper)")
        
        while True:
            model_choice = input("Select model (1-3) or press Enter for default (gpt-4o): ").strip()
            
            if model_choice == "" or model_choice == "1":
                model_name = "gpt-4o"
                break
            elif model_choice == "2":
                model_name = "gpt-4"
                break
            elif model_choice == "3":
                model_name = "gpt-3.5-turbo"
                break
            else:
                logger.warning("Please enter 1, 2, 3, or press Enter for default")
                continue
        
        logger.info(f"Selected model: {model_name}")
        
        # Ask about validation
        logger.info("\nValidation Options:")
        logger.info("1. No validation (faster, cheaper)")
        logger.info("2. Validate non-default sequences only (recommended)")
        logger.info("3. Validate all sequences (thorough, more expensive)")
        
        enable_validation = False
        validate_only_non_default = True
        validation_model_name = model_name  # Default to same model
        
        while True:
            validation_choice = input("Select validation option (1-3) or press Enter for default (1): ").strip()
            
            if validation_choice == "" or validation_choice == "1":
                enable_validation = False
                logger.info("Selected: No validation")
                break
            elif validation_choice == "2":
                enable_validation = True
                validate_only_non_default = True
                logger.info("Selected: Validate non-default sequences only")
                break
            elif validation_choice == "3":
                enable_validation = True
                validate_only_non_default = False
                logger.info("Selected: Validate all sequences")
                break
            else:
                logger.warning("Please enter 1, 2, 3, or press Enter for default")
                continue
        
        # If validation enabled, ask about validation model
        if enable_validation:
            logger.info(f"\nValidation Model (current analysis model: {model_name}):")
            logger.info("1. Same as analysis model")
            logger.info("2. gpt-4o")
            logger.info("3. gpt-4")
            logger.info("4. gpt-3.5-turbo")
            
            while True:
                val_model_choice = input("Select validation model (1-4) or press Enter for same: ").strip()
                
                if val_model_choice == "" or val_model_choice == "1":
                    validation_model_name = model_name
                    break
                elif val_model_choice == "2":
                    validation_model_name = "gpt-4o"
                    break
                elif val_model_choice == "3":
                    validation_model_name = "gpt-4"
                    break
                elif val_model_choice == "4":
                    validation_model_name = "gpt-3.5-turbo"
                    break
                else:
                    logger.warning("Please enter 1, 2, 3, 4, or press Enter for same")
                    continue
            
            logger.info(f"Selected validation model: {validation_model_name}")
        
        return cls(
            model_name=model_name,
            max_issues=max_issues,
            temperature=0,
            max_tokens=800,
            delay_seconds=1.0,
            continue_analysis=continue_analysis,
            research_questions=research_questions,
            enable_validation=enable_validation,
            validation_model_name=validation_model_name,
            validate_only_non_default=validate_only_non_default,
            validation_delay_seconds=0.5
        )

class WCAGViolation(BaseModel):
    """Individual WCAG violation details - SIMPLIFIED"""
    sc_name: str = Field(description="Complete SC info with SC prefix: 'SC X.X.X Name (Level)' (e.g., 'SC 2.1.1 Keyboard (A)')")
    evidence_source: str = Field(description="Source with original text evidence (e.g., 'description: Users cannot navigate to activities via keyboard', 'comment_2: Screen reader announces incorrect roles')")
    violation_reason: str = Field(description="Specific reason why this constitutes a violation")

class WCAGCategoryResult(BaseModel):
    """Result model for WCAG 2.2 new SC categorization analysis - SIMPLIFIED"""
    issue_key: str
    violated_sc: List[WCAGViolation] = Field(description="List of WCAG 2.2 SCs violated", default=[])

class WCAGValidationResult(BaseModel):
    """Result model for WCAG categorization validation - WITH EVIDENCE TEXT"""
    original_violations: List[WCAGViolation] = Field(description="Original violations from first analysis")
    validated_violations: List[WCAGViolation] = Field(description="Violations after validation review")
    validation_changes: List[str] = Field(description="List of changes made during validation")
    validation_explanation: str = Field(description="Explanation of validation decision")
    agreement_status: str = Field(description="Agrees/Disagrees/Partially_Agrees with original")

class TestingMethod(BaseModel):
    """Individual testing method details"""
    method_type: str = Field(description="Type: 'automated' or 'manual'")
    tool_or_technique: str = Field(description="Tool name (for automated) or technique used (for manual, e.g., 'screen_reader', 'keyboard_navigation', 'mouse_interaction')")
    purpose: str = Field(description="Purpose: 'testing' or 'verification'")
    target_issue: str = Field(description="The specific accessibility issue being tested/verified")
    evidence_source: str = Field(description="Source with original text evidence (e.g., 'description: ran automated tests', 'comment_3: tested with NVDA screen reader')")
    details: str = Field(description="Additional context about the testing/verification method")

class TestingVerificationResult(BaseModel):
    """Result model for Testing & Verification analysis"""
    issue_key: str
    testing_methods: List[TestingMethod] = Field(description="List of testing and verification methods identified", default=[])
    summary: str = Field(description="Brief summary of testing approach used in this issue")

class TestingValidationResult(BaseModel):
    """Result model for Testing & Verification validation"""
    original_methods: List[TestingMethod] = Field(description="Original methods from first analysis")
    validated_methods: List[TestingMethod] = Field(description="Methods after validation review")
    validation_changes: List[str] = Field(description="List of changes made during validation")
    validation_explanation: str = Field(description="Explanation of validation decision")
    agreement_status: str = Field(description="Agrees/Disagrees/Partially_Agrees with original")

class BaseResearchQuestion(ABC):
    """Abstract base class for research questions"""
    
    @property
    @abstractmethod
    def question_id(self) -> str:
        """Unique identifier for this research question"""
        pass
    
    @property
    @abstractmethod
    def question_description(self) -> str:
        """Human-readable description of the research question"""
        pass
    
    @property
    @abstractmethod
    def result_model(self) -> Type[BaseModel]:
        """Pydantic model for the analysis result"""
        pass
    
    @abstractmethod
    def create_system_prompt(self) -> str:
        """Create the system prompt for this research question"""
        pass
    
    @abstractmethod
    def create_analysis_prompt(self, issue_data: Dict[str, Any]) -> str:
        """Create the analysis prompt for a specific issue"""
        pass
    
    @abstractmethod
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for this research question"""
        pass

class SolutionDevelopmentQuestion(BaseResearchQuestion):
    """Research Question: How do assignees come up with solutions for accessibility issues"""
    
    @property
    def question_id(self) -> str:
        return "solution_development"
    
    @property
    def question_description(self) -> str:
        return "How do assignees initially develop and articulate technical solutions for accessibility issues?"
    
    @property
    def result_model(self) -> Type[BaseModel]:
        return SolutionDevelopmentResult
    
    def create_system_prompt(self) -> str:
        return """You are a software engineering research expert analyzing initial solution development METHODOLOGIES in accessibility issue resolution using open coding methodology.

        Your task is to identify and assign specific, descriptive open codes to HOW assignees initially approach, develop, and articulate accessibility solutions by analyzing their reasoning processes, problem-solving methodologies, and decision-making patterns.

        **CRITICAL FOCUS: METHODOLOGY ANALYSIS, NOT TECHNICAL IMPLEMENTATION**
        
        **CORE RESEARCH OBJECTIVE:**
        Understand HOW accessibility professionals approach problem-solving, NOT what specific technical changes they make.
        
        **ANALYSIS SCOPE:**
        - INITIAL solution development methodology (original problem-solving approaches)
        - EXCLUDE subsequent refinements (responses to feedback, covered by Role Sequence RQ)
        - Focus on reasoning patterns, decision-making processes, and methodological approaches
        - Capture authentic accessibility thinking and professional practice patterns


        **OPEN CODING PRINCIPLES FOR METHODOLOGY:**

        **1. PROCESS-ORIENTED CODES:**
        - Focus on HOW they approach problems, not WHAT they implement
        - Capture reasoning patterns and decision-making processes
        - Identify systematic vs. intuitive problem-solving approaches
        - Code the methodology, not the technical outcome

        **2. ACCESSIBILITY-SPECIFIC METHODOLOGY:**
        - Emphasize accessibility-specific problem-solving approaches
        - Capture assistive technology consideration patterns
        - Code standards application methodologies
        - Focus on inclusive design thinking processes

        **3. AUTHENTIC PRACTICE PATTERNS:**
        - Ground codes in actual evidence from assignee reasoning
        - Avoid inferring beyond what's explicitly described
        - Capture the authentic thought processes of accessibility professionals
        - Code the actual problem-solving methodology demonstrated

        **4. NOT_OBVIOUS RECOGNITION:**
        - When methodology cannot be determined from evidence, use ["NOT_OBVIOUS"]
        - Don't force codes when reasoning patterns aren't clear
        - Acknowledge when technical implementation is described without methodology
        - Maintain analytical integrity by recognizing limitations

        **GUIDELINES:**
        - Focus only on assignee comments (exclude other roles)
        - Extract INITIAL methodology before external input influences
        - Create open codes that capture authentic accessibility solution development approaches
        - Use "NOT_OBVIOUS" when problem-solving methodology cannot be determined
        - EXCLUDE technical implementation details - focus on reasoning and approach
        - Capture professional accessibility practice patterns, not code changes

        **CRITICAL DISTINCTION:**
        - INCLUDE: "I analyzed WCAG guidelines to understand bypass mechanisms and evaluated skip links versus landmarks"
        - EXCLUDE: "I added aria-label to the button and changed the tabindex value"
        - INCLUDE: "I researched screen reader compatibility patterns and tested with multiple assistive technologies"
        - EXCLUDE: "I modified the CSS color from gray-500 to gray-600"

        Respond only with valid JSON matching the specified schema, focusing on methodology analysis."""
    
    def create_analysis_prompt(self, issue_data: Dict[str, Any]) -> str:
        # Extract ALL assignee comments (no filtering)
        assignee_comments = []
        for i, comment in enumerate(issue_data.get('comments', []), 1):
            roles = comment.get('role', [])
            if isinstance(roles, str):
                roles = [roles]
            
            if 'assignee' in roles:
                assignee_comments.append(f"comment_{i}: ASSIGNEE ({comment['author']}): {comment['body']}")
        
        assignee_comments_text = "\n\n".join(assignee_comments) if assignee_comments else "No assignee comments found"
        
        return f"""
        **INITIAL ACCESSIBILITY SOLUTION DEVELOPMENT ANALYSIS - FOCUS ON "HOW"**

        Analyze HOW the assignee approaches and develops their accessibility solution using open coding methodology.

        **ISSUE DETAILS:**
        - Key: {issue_data.get('issue_key', 'N/A')}
        - Title: {issue_data.get('title', 'N/A')}
        - Priority: {issue_data.get('priority', 'N/A')}

        **ALL ASSIGNEE COMMENTS (Complete Text for Analysis):**
        {assignee_comments_text}

        **CRITICAL ANALYSIS FOCUS: HOW NOT WHAT**
        
        **PRIMARY FOCUS: METHODOLOGY AND APPROACH (NOT TECHNICAL CONTENT)**
        - Focus on HOW the assignee approaches the accessibility problem-solving process
        - Analyze the METHODOLOGY, not the specific technical changes made
        - Capture the REASONING PATTERNS and DECISION-MAKING PROCESS
        - Identify HOW they gather information, apply guidelines, consider users
        - IGNORE the specific technical implementation details (what they did)

        **OPEN CODING FOR "HOW" - METHODOLOGY FOCUS:**

        **AVOID THESE "WHAT" CODES (TECHNICAL IMPLEMENTATION):**
        - "COLOR_CONTRAST_ADJUSTMENT"  (this is what was changed)
        - "ARIA_LABEL_ADDITION"  (this is what was implemented)
        - "TABINDEX_MODIFICATION"  (this is what was modified)
        - "CSS_SELECTOR_CHANGE"  (this is what was altered)

        **CRITICAL DETECTION RULE: NOT OBVIOUS OPTION**
        
        **When methodology is unclear, use:**
        - If the assignee comment doesn't reveal HOW they approached the problem
        - If only technical implementation details are provided without methodology
        - If the reasoning process cannot be determined from the available evidence
        - If the comment is too brief to understand the approach

        **NOT_OBVIOUS handling:**
        - If the solution development methodology cannot be clearly determined from assignee comments
        - Create approach with solution_codes: ["NOT_OBVIOUS"]
        - approach_description: "The assignee's solution development methodology cannot be clearly determined from available evidence"
        - Still include source_text and technical_details if implementation details are mentioned

        **OUTPUT REQUIREMENTS:**
        
        For each initial accessibility solution development approach:
        - **solution_codes**: 1-3 codes focusing on HOW/methodology (or ["NOT_OBVIOUS"] if unclear)
        - **approach_description**: Describe the METHODOLOGY and REASONING PROCESS (not technical implementation)
        - **source_text**: Complete relevant excerpt from assignee comment
        - **source_location**: Exact comment location
        - **technical_details**: Implementation details mentioned (but codes should focus on methodology)

        **VALIDATION QUESTIONS FOR EACH CODE:**
        1. Does this code describe HOW the assignee approached the problem?
        2. Does this code focus on methodology rather than implementation?
        3. Could I understand the problem-solving approach from this code?
        4. Does this code help analyze accessibility solution development patterns?

        **IMPORTANT REMINDERS:**
        - Focus on METHODOLOGY and APPROACH, not technical implementation
        - Use "NOT_OBVIOUS" when methodology cannot be determined
        - Capture HOW they think about accessibility, not WHAT they implemented
        - Look for reasoning patterns, not just technical changes
        - Analyze the problem-solving process, not the code changes

        Analyze the assignee comments for INITIAL accessibility solution development METHODOLOGY and provide your findings in the specified JSON format.
        """
    
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            return {"error": "No successful analyses to summarize"}
        
        # Collect all approaches
        all_approaches = []
        issues_with_approaches = 0
        issues_without_approaches = 0
        
        for result in successful_results:
            approaches = result.get('solution_approaches', [])
            if approaches:
                issues_with_approaches += 1
                # Convert Pydantic objects to dictionaries and extract approach details
                for approach_obj in approaches:
                    if hasattr(approach_obj, 'model_dump'):
                        approach_dict = approach_obj.model_dump()
                        all_approaches.append(approach_dict)
                    elif hasattr(approach_obj, 'dict'):
                        approach_dict = approach_obj.dict()
                        all_approaches.append(approach_dict)
                    elif isinstance(approach_obj, dict):
                        all_approaches.append(approach_obj)
                    else:
                        all_approaches.append({'approach_type': str(approach_obj)})
            else:
                issues_without_approaches += 1
        
        # Analyze approach distributions
        approach_types = Counter()
        reasoning_patterns = Counter()
        
        for approach in all_approaches:
            approach_type = approach.get('approach_type', 'Unknown')
            reasoning_pattern = approach.get('reasoning_pattern', 'Unknown')
            
            approach_types[approach_type] += 1
            reasoning_patterns[reasoning_pattern] += 1
        
        # Validation statistics
        validation_stats = {
            "total_validations_performed": len([r for r in successful_results if r.get('solution_validation', {}).get('performed')]),
            "validation_agreements": len([r for r in successful_results if r.get('solution_validation', {}).get('agreement_status') == 'Agrees']),
            "validation_disagreements": len([r for r in successful_results if r.get('solution_validation', {}).get('agreement_status') == 'Disagrees']),
            "validation_partial_agreements": len([r for r in successful_results if r.get('solution_validation', {}).get('agreement_status') == 'Partially_Agrees']),
            "approaches_changed_by_validation": len([r for r in successful_results if r.get('original_approaches') != r.get('solution_approaches')])
        }
        
        return {
            "research_question": self.question_description,
            "total_issues_analyzed": len(results),
            "successful_analyses": len(successful_results),
            "failed_analyses": len(results) - len(successful_results),
            
            "solution_development_overview": {
                "issues_with_solution_approaches": issues_with_approaches,
                "issues_without_solution_approaches": issues_without_approaches,
                "total_approaches_found": len(all_approaches),
                "percentage_issues_with_approaches": round(issues_with_approaches / len(successful_results) * 100, 1) if successful_results else 0
            },
            
            "approach_type_distribution": [
                {
                    "approach_type": approach_type,
                    "count": count,
                    "percentage": round(count / len(all_approaches) * 100, 1) if all_approaches else 0
                }
                for approach_type, count in approach_types.most_common()
            ],
            
            "reasoning_pattern_distribution": [
                {
                    "reasoning_pattern": pattern,
                    "count": count,
                    "percentage": round(count / len(all_approaches) * 100, 1) if all_approaches else 0
                }
                for pattern, count in reasoning_patterns.most_common()
            ],
            
            "solution_development_statistics": {
                "average_approaches_per_issue": round(len(all_approaches) / len(successful_results), 2) if successful_results else 0,
                "max_approaches_in_single_issue": max([len(r.get('solution_approaches', [])) for r in successful_results]) if successful_results else 0,
                "unique_approach_types": len(approach_types),
                "unique_reasoning_patterns": len(reasoning_patterns)
            },
            
            "validation_statistics": validation_stats
        }


class NonA11ySolutionDevelopmentQuestion(BaseResearchQuestion):
    """Research Question: How do assignees come up with solutions for non-accessibility issues"""
    
    @property
    def question_id(self) -> str:
        return "non_a11y_solution_development"
    
    @property
    def question_description(self) -> str:
        return "How do assignees initially develop and articulate technical solutions for non-accessibility issues?"
    
    @property
    def result_model(self) -> Type[BaseModel]:
        return SolutionDevelopmentResult
    
    def create_system_prompt(self) -> str:
        return """You are a software engineering research expert analyzing initial solution development processes in issue resolution.

        Your task is to identify and categorize how assignees initially approach, develop, and articulate technical solutions by analyzing their complete comments, with specific focus on their FIRST/ORIGINAL solution development before any external feedback or refinement.

        **CRITICAL FOCUS: INITIAL SOLUTION DEVELOPMENT ONLY**
        This analysis focuses on the universal aspects of initial solution development while being sensitive to domain-specific considerations and methodologies. You must distinguish between:
        - INITIAL solution development (original problem-solving and solution design)
        - SUBSEQUENT refinements (responses to feedback, covered by Role Sequence RQ)

        Guidelines:
        - Focus only on assignee comments (exclude reporter/peer_reviewer/integrator/tester/participant/bot roles)
        - Extract INITIAL thought processes and reasoning patterns before external input
        - Identify original technical approaches and methodologies
        - Capture first references to standards, tools, and best practices
        - Note initial constraint considerations and impact analysis
        - EXCLUDE subsequent modifications, refinements, or responses to reviewer feedback

        Respond only with valid JSON matching the specified schema."""
    
    def create_analysis_prompt(self, issue_data: Dict[str, Any]) -> str:
        # Extract ALL assignee comments (no filtering)
        assignee_comments = []
        for i, comment in enumerate(issue_data.get('comments', []), 1):
            roles = comment.get('role', [])
            if isinstance(roles, str):
                roles = [roles]
            
            if 'assignee' in roles:
                assignee_comments.append(f"comment_{i}: ASSIGNEE ({comment['author']}): {comment['body']}")
        
        assignee_comments_text = "\n\n".join(assignee_comments) if assignee_comments else "No assignee comments found"
        
        return f"""
        **INITIAL SOLUTION DEVELOPMENT ANALYSIS**

        Analyze how the assignee initially develops and articulates their technical solution approach for this issue.

        **ISSUE DETAILS:**
        - Key: {issue_data.get('issue_key', 'N/A')}
        - Title: {issue_data.get('title', 'N/A')}
        - Priority: {issue_data.get('priority', 'N/A')}

        **ALL ASSIGNEE COMMENTS (Complete Text for Analysis):**
        {assignee_comments_text}

        **CRITICAL ANALYSIS FOCUS:**
        
        **PRIMARY FOCUS: INITIAL SOLUTION DEVELOPMENT ONLY**
        - Focus specifically on the assignee's FIRST/ORIGINAL solution approach and reasoning
        - Analyze how they initially conceptualize and design their solution BEFORE any external feedback
        - Capture the original problem-solving methodology and technical thinking
        - IGNORE subsequent modifications, refinements, or responses to reviewer feedback
        - IGNORE solutions that are clearly responses to problems found by others (peer reviewers, integrators, testers)

        **IDENTIFICATION GUIDELINES:**
        
        **INCLUDE (Initial Solution Development):**
        - First substantial technical analysis and problem diagnosis
        - Original solution design and implementation strategy  
        - Initial reasoning patterns and decision-making processes
        - First application of standards, tools, or best practices
        - Original testing/validation approach conception
        - Initial constraint consideration and trade-off analysis

        **EXCLUDE (Subsequent Refinements - covered by Role Sequence RQ):**
        - Solutions developed in response to peer reviewer feedback
        - Fixes implemented after integration problems are found
        - Changes made after testing failures are reported
        - Refinements based on participant suggestions
        - Any solution modifications that are clearly reactive to external input

        **DETECTION STRATEGY:**
        Look through all assignee comments chronologically and identify:
        1. **First Substantive Solution**: The first comment where the assignee articulates a meaningful technical approach (may not be comment #1)
        2. **Original Reasoning**: Evidence of independent problem-solving before external feedback
        3. **Initial Technical Decisions**: First choices about implementation, standards, tools, etc.

        **EXAMPLES TO HELP DISTINGUISH:**

        **INITIAL SOLUTION (INCLUDE):**
        - "I analyzed the database queries and found an N+1 problem, so I'll implement eager loading to optimize performance"
        - "The issue is in the authentication middleware - I'll add proper session validation and CSRF protection"
        - "I identified a race condition in the caching layer, so I'll implement proper locking mechanisms"

        **SUBSEQUENT REFINEMENT (EXCLUDE):**
        - "The peer reviewer found security issues with my approach, so I'm changing the implementation"
        - "Integration failed because of API conflicts, so I'm modifying the endpoint structure"  
        - "Unit tests are failing due to my changes, so I need to adjust the validation logic"
        - "Based on code review feedback, I'm now using a different design pattern"

        **EVIDENCE EXTRACTION REQUIREMENTS:**

        For each approach identified:
        - **approach_description**: Detailed description of the specific approach used
        - **source_text**: Complete relevant excerpt from assignee comment (no length limit)
        - **source_location**: Exact comment location (e.g., "comment_3")
        - **technical_details**: Specific tools, methods, standards, calculations mentioned
        - **reasoning_pattern**: The underlying problem-solving methodology observed

        **COMPREHENSIVE DETECTION RULES:**

        1. **Capture Complete Initial Solution Processes**: Don't just identify what was done, but HOW it was initially approached
        2. **Include Initial Reasoning Articulation**: Extract explanations of why certain initial approaches were chosen
        3. **Note Initial Technical Sophistication**: Capture depth of initial technical analysis and solution design
        4. **Focus on Original Solutions**: If solutions are refined later, capture only the initial conceptualization
        5. **Include Initial Constraint Consideration**: Note how assignees initially handle limitations and requirements
        6. **Extract Initial Standards Application**: Capture how standards first influence solution development

        **OUTPUT REQUIREMENTS:**
        - solution_approaches: Focus ONLY on approaches from initial solution development
        - development_summary: Characterize the assignee's INITIAL solution methodology
        - If no clear initial solution development is found, note this in the summary
        - Do NOT include approaches that are clearly responses to external feedback

        **IMPORTANT:** Remember that subsequent solution refinements and back-and-forth processes are analyzed in the Role Sequence research question. This analysis should capture the pure, original solution development process only.

        Analyze the assignee comments for INITIAL solution development and provide your findings in the specified JSON format.
        """
    
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Same implementation as accessibility version since they use the same framework
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            return {"error": "No successful analyses to summarize"}
        
        # Collect all approaches
        all_approaches = []
        issues_with_approaches = 0
        issues_without_approaches = 0
        
        for result in successful_results:
            approaches = result.get('solution_approaches', [])
            if approaches:
                issues_with_approaches += 1
                # Convert Pydantic objects to dictionaries and extract approach details
                for approach_obj in approaches:
                    if hasattr(approach_obj, 'model_dump'):
                        approach_dict = approach_obj.model_dump()
                        all_approaches.append(approach_dict)
                    elif hasattr(approach_obj, 'dict'):
                        approach_dict = approach_obj.dict()
                        all_approaches.append(approach_dict)
                    elif isinstance(approach_obj, dict):
                        all_approaches.append(approach_obj)
                    else:
                        all_approaches.append({'approach_type': str(approach_obj)})
            else:
                issues_without_approaches += 1
        
        # Analyze approach distributions
        approach_types = Counter()
        reasoning_patterns = Counter()
        
        for approach in all_approaches:
            approach_type = approach.get('approach_type', 'Unknown')
            reasoning_pattern = approach.get('reasoning_pattern', 'Unknown')
            
            approach_types[approach_type] += 1
            reasoning_patterns[reasoning_pattern] += 1
        
        # Validation statistics
        validation_stats = {
            "total_validations_performed": len([r for r in successful_results if r.get('solution_validation', {}).get('performed')]),
            "validation_agreements": len([r for r in successful_results if r.get('solution_validation', {}).get('agreement_status') == 'Agrees']),
            "validation_disagreements": len([r for r in successful_results if r.get('solution_validation', {}).get('agreement_status') == 'Disagrees']),
            "validation_partial_agreements": len([r for r in successful_results if r.get('solution_validation', {}).get('agreement_status') == 'Partially_Agrees']),
            "approaches_changed_by_validation": len([r for r in successful_results if r.get('original_approaches') != r.get('solution_approaches')])
        }
        
        return {
            "research_question": self.question_description,
            "total_issues_analyzed": len(results),
            "successful_analyses": len(successful_results),
            "failed_analyses": len(results) - len(successful_results),
            
            "solution_development_overview": {
                "issues_with_solution_approaches": issues_with_approaches,
                "issues_without_solution_approaches": issues_without_approaches,
                "total_approaches_found": len(all_approaches),
                "percentage_issues_with_approaches": round(issues_with_approaches / len(successful_results) * 100, 1) if successful_results else 0
            },
            
            "approach_type_distribution": [
                {
                    "approach_type": approach_type,
                    "count": count,
                    "percentage": round(count / len(all_approaches) * 100, 1) if all_approaches else 0
                }
                for approach_type, count in approach_types.most_common()
            ],
            
            "reasoning_pattern_distribution": [
                {
                    "reasoning_pattern": pattern,
                    "count": count,
                    "percentage": round(count / len(all_approaches) * 100, 1) if all_approaches else 0
                }
                for pattern, count in reasoning_patterns.most_common()
            ],
            
            "solution_development_statistics": {
                "average_approaches_per_issue": round(len(all_approaches) / len(successful_results), 2) if successful_results else 0,
                "max_approaches_in_single_issue": max([len(r.get('solution_approaches', [])) for r in successful_results]) if successful_results else 0,
                "unique_approach_types": len(approach_types),
                "unique_reasoning_patterns": len(reasoning_patterns)
            },
            
            "validation_statistics": validation_stats
        }


class RoleSequenceQuestion(BaseResearchQuestion):
    """Research Question 1: Complete role sequence analysis"""
    
    @property
    def question_id(self) -> str:
        return "role_sequence"
    
    @property
    def question_description(self) -> str:
        return "What is the complete role sequence for accessibility issues, including back-and-forth patterns?"
    
    @property
    def result_model(self) -> Type[BaseModel]:
        return RoleSequenceResult
    
    def create_system_prompt(self) -> str:
        return """You are a software engineering researcher analyzing role sequences in issue workflows. 
        
    Key considerations:
    - A person can have multiple roles throughout an issue's lifecycle
    - Focus on chronological analysis of comments and timeline events to identify complete role progressions
    - If no issues are found, the role sequence should be reporter, assignee, peer_reviewer, integrator, tester
    - Include all meaningful role transitions, especially when control goes back to previous roles due to problems
    - Exclude bot roles from the sequence
    - Assign specific codes for back-and-forth patterns when they occur
        
    Respond only with valid JSON matching the specified schema."""
    
    def create_analysis_prompt(self, issue_data: Dict[str, Any]) -> str:
        # Format comments chronologically
        comments_text = ""
        for i, comment in enumerate(issue_data.get('comments', []), 1):
            # comments_text += f"{i}. {comment['role']} ({comment['author']}): {comment['body'][:300]}{'...' if len(comment['body']) > 300 else ''}\n"
            comments_text += f"{i}. {comment['role']} ({comment['author']}): {comment['body']}\n"
        
        # Format timeline events
        timeline_text = ""
        # Fixed version - handle None values properly
        for i, event in enumerate(issue_data.get('timeline_events', []), 1):
            from_value = event.get('from_value') or ''  # Convert None to empty string
            to_value = event.get('to_value') or ''      # Convert None to empty string
            
            from_value_truncated = from_value[:100] + ('...' if len(from_value) > 100 else '')
            to_value_truncated = to_value[:100] + ('...' if len(to_value) > 100 else '')
            
            timeline_text += f"{i}. {event.get('timestamp', 'N/A')} - {event.get('author', 'N/A')}: {event.get('field', 'N/A')} changed from '{from_value_truncated}' to '{to_value_truncated}'\n"
            
        return f"""
        Analyze the complete role sequence in this Moodle accessibility issue by examining the chronological comments and changelog events.

        **ISSUE DETAILS:**
        - Key: {issue_data.get('issue_key', 'N/A')}
        - Title: {issue_data.get('title', 'N/A')}
        - Priority: {issue_data.get('priority', 'N/A')}

        **CHRONOLOGICAL COMMENTS:**
        {comments_text}

        **TIMELINE EVENTS (field changes):**
        {timeline_text}

        **RESEARCH QUESTION:**
        What is the complete role sequence for this issue, including when control goes back to previous roles? However, a role as a bot should not be included in the sequence.

        **INSTRUCTIONS:**
        1. **Focus on Comments and Changelog**: Analyze the chronological flow of comments and timeline events
        2. **Track Every Role Transition**: List every time control/responsibility shifts from one role to another
        3. **Include Back-and-Forth**: When someone sends work back to a previous role, include that transition
        4. **NEVER Include Participants in Role Sequence**: Participants should never appear in the role sequence itself
        5. **Assign Back-and-Forth Codes**: When control returns to previous roles due to problems, assign specific codes

        **CORE WORKFLOW ROLES TO TRACK:**
        - reporter
        - assignee  
        - peer_reviewer
        - integrator
        - tester
        - (other formal workflow roles, but NOT participant)

        **BACK-AND-FORTH CODE GENERATION:** When roles return to previous roles due to problems, create specific, descriptive codes:

        **Code Creation Guidelines:**
        - Create **descriptive, specific codes** that clearly indicate the accessibility problem found
        - Use **UPPER_CASE_WITH_UNDERSCORES** format for consistency
        - Focus on **what the actual problem is** based on the evidence (be specific, not generic)
        - **When appropriate**, reference relevant accessibility standards or guidelines
        - Keep codes **concise but clear** - they should be meaningful for analysis
        - Be **consistent** - similar problems should get similar codes
        - **Ignore bot comments** - Do not create codes based on the authors of CiBoT, noreply, or automated system comments

        **Examples of Good Codes** (create others as needed based on the actual problems you find):
        
        **Be specific about the actual problem:**
        - SCREEN_READER_CONTEXT_MISSING, KEYBOARD_NAVIGATION_BROKEN, FOCUS_INDICATOR_INVISIBLE

        **EXAMPLES of "goes back to previous roles":**
        **IMPORTANT**: Only consider it "going back to previous roles" when there is a problem or issue that requires returning to fix something:
        - assignee → peer_reviewer → assignee (reviewer found issues, sent back to assignee to fix)
        - assignee → integrator → assignee (integration failed, sent back to assignee)
        - peer_reviewer → assignee → peer_reviewer (fixes made, sent back for re-review)

        **NOT considered "going back"**: Normal forward progression through roles, even if the same person appears multiple times in different contexts.

        **CODE ASSIGNMENT WITH SOURCE EVIDENCE:**
        - **Default Sequence**: If the sequence is exactly [reporter, assignee, peer_reviewer, integrator, tester] with NO back-and-forth, assign back_and_forth_codes = []
        - **Back-and-Forth Present**: For each instance where control returns to a previous role due to a problem, create a code that:
          - `code`: Describes the specific problem that caused the return (be descriptive and specific)
          - `source_text`: The exact text excerpt that shows the problem (max 200 characters)
          - `source_location`: Where the evidence comes from (e.g., "comment_3", "description", "timeline_5")
        - **Multiple Returns**: If there are multiple back-and-forth instances, assign multiple code objects in chronological order
        - **Evidence Requirements**: Always include the actual text that supports the code assignment
        - **Ignore Bot Comments**: Do NOT assign codes based on comments from CiBoT, noreply, or other automated systems

        **TASK:**
        Examine the comments and timeline events chronologically. For each significant action, determine:
        1. Who is taking action (which role)? **Skip CiBoT, noreply, and automated systems**
        2. Does this action represent a meaningful transition in control/responsibility?
        3. If someone returns to a previous role, is it because of a problem that needs fixing?
        4. What specific problem caused the return? (create an appropriate code based on the evidence)
        5. What is the next role in the sequence?

        **Code Creation Approach:**
        - Read the evidence carefully and identify the specific problem mentioned
        - Create a code that clearly describes what the problem is
        - Be specific but not overly complex
        - Use accessibility standards references when they're relevant
        - Focus on making the code useful for analysis

        **OUTPUT REQUIREMENTS:**
        - complete_role_sequence: Complete chronological sequence of roles
        - sequence_explanation: Brief explanation of the workflow progression including back-and-forth triggers
        - back_and_forth_codes: List of code objects with evidence for each back-and-forth instance (empty list if default sequence)

        Output your analysis in the specified JSON format.
        """
    
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            return {"error": "No successful analyses to summarize"}
        
        # Collect all sequences (use validated sequences if available)
        all_sequences = []
        all_codes = []
        for result in successful_results:
            # Use validated sequence if available, otherwise original
            sequence = result.get('complete_role_sequence', [])
            codes = result.get('back_and_forth_codes', [])
            if sequence:
                all_sequences.append(tuple(sequence))
            # Extract just the code strings for counting, handling both dict and Pydantic objects
            for code_obj in codes:
                if hasattr(code_obj, 'model_dump'):  # Pydantic v2
                    all_codes.append(code_obj.model_dump().get('code', ''))
                elif hasattr(code_obj, 'dict'):  # Pydantic v1
                    all_codes.append(code_obj.dict().get('code', ''))
                elif isinstance(code_obj, dict):
                    all_codes.append(code_obj.get('code', ''))
                else:
                    all_codes.append(str(code_obj))  # fallback for string codes
        
        # Count sequence patterns
        sequence_counts = Counter(all_sequences)
        
        # Count back-and-forth codes
        code_counts = Counter(all_codes)
        
        # Count issues with and without back-and-forth (handling Pydantic objects)
        issues_with_backforth = 0
        for result in successful_results:
            codes = result.get('back_and_forth_codes', [])
            if codes and len(codes) > 0:
                issues_with_backforth += 1
        issues_without_backforth = len(successful_results) - issues_with_backforth
        
        # Validation statistics
        validation_stats = {
            "total_validations_performed": len([r for r in successful_results if r.get('validation', {}).get('performed')]),
            "validation_agreements": len([r for r in successful_results if r.get('validation', {}).get('agreement_status') == 'Agrees']),
            "validation_disagreements": len([r for r in successful_results if r.get('validation', {}).get('agreement_status') == 'Disagrees']),
            "validation_partial_agreements": len([r for r in successful_results if r.get('validation', {}).get('agreement_status') == 'Partially_Agrees']),
            "sequences_changed_by_validation": len([r for r in successful_results if r.get('original_sequence') != r.get('complete_role_sequence')])
        }
        
        return {
            "research_question": self.question_description,
            "total_issues_analyzed": len(results),
            "successful_analyses": len(successful_results),
            "failed_analyses": len(results) - len(successful_results),
            
            "most_common_role_sequences": [
                {
                    "sequence": list(seq),
                    "count": count,
                    "percentage": round(count / len(successful_results) * 100, 1)
                }
                for seq, count in sequence_counts.most_common(15)
            ],
            
            "back_and_forth_analysis": {
                "issues_with_back_and_forth": issues_with_backforth,
                "issues_without_back_and_forth": issues_without_backforth,
                "percentage_with_back_and_forth": round(issues_with_backforth / len(successful_results) * 100, 1) if successful_results else 0,
                "most_common_codes": [
                    {
                        "code": code,
                        "count": count,
                        "percentage": round(count / len(all_codes) * 100, 1) if all_codes else 0
                    }
                    for code, count in code_counts.most_common(10)
                ],
                "total_back_and_forth_instances": len(all_codes)
            },
            
            "sequence_statistics": {
                "total_unique_sequences": len(sequence_counts),
                "average_sequence_length": round(sum(len(seq) for seq in all_sequences) / len(all_sequences), 2) if all_sequences else 0,
                "shortest_sequence": min(len(seq) for seq in all_sequences) if all_sequences else 0,
                "longest_sequence": max(len(seq) for seq in all_sequences) if all_sequences else 0
            },
            
            "validation_statistics": validation_stats
        }

class NonA11yRoleSequenceQuestion(BaseResearchQuestion):
    """Research Question 1: Complete role sequence analysis for non-accessibility issues"""
    
    @property
    def question_id(self) -> str:
        return "non_a11y_role_sequence"
    
    @property
    def question_description(self) -> str:
        return "What is the complete role sequence for non-accessibility issues, including back-and-forth patterns?"
    
    @property
    def result_model(self) -> Type[BaseModel]:
        return RoleSequenceResult  # Reuse the same data model
    
    def create_system_prompt(self) -> str:
        return """You are a software engineering researcher analyzing role sequences in issue workflows. 
        
    Key considerations:
    - A person can have multiple roles throughout an issue's lifecycle
    - Focus on chronological analysis of comments and timeline events to identify complete role progressions
    - If no issues are found, the role sequence should be reporter, assignee, peer_reviewer, integrator, tester
    - Include all meaningful role transitions, especially when control goes back to previous roles due to problems
    - Exclude bot roles from the sequence (CiBoT, noreply, and automated systems). The typical comment from these roles is "The integrator needs more information or changes from your patch in order to progress this issue." Ignore these comments.
    - Assign specific codes for back-and-forth patterns when they occur, not generic codes like "PEER_REVIEW_FEEDBACK"
        
    Respond only with valid JSON matching the specified schema."""
    
    def create_analysis_prompt(self, issue_data: Dict[str, Any]) -> str:
        # Format comments chronologically
        comments_text = ""
        for i, comment in enumerate(issue_data.get('comments', []), 1):
            # comments_text += f"{i}. {comment['role']} ({comment['author']}): {comment['body'][:300]}{'...' if len(comment['body']) > 300 else ''}\n"
            comments_text += f"{i}. {comment['role']} ({comment['author']}): {comment['body']}\n"
        
        # Format timeline events
        timeline_text = ""
        # Fixed version - handle None values properly
        for i, event in enumerate(issue_data.get('timeline_events', []), 1):
            from_value = event.get('from_value') or ''  # Convert None to empty string
            to_value = event.get('to_value') or ''      # Convert None to empty string
            
            from_value_truncated = from_value[:100] + ('...' if len(from_value) > 100 else '')
            to_value_truncated = to_value[:100] + ('...' if len(to_value) > 100 else '')
            
            timeline_text += f"{i}. {event.get('timestamp', 'N/A')} - {event.get('author', 'N/A')}: {event.get('field', 'N/A')} changed from '{from_value_truncated}' to '{to_value_truncated}'\n"
        
        return f"""
        Analyze the complete role sequence in this Moodle issue by examining the chronological comments and changelog events.

        **ISSUE DETAILS:**
        - Key: {issue_data.get('issue_key', 'N/A')}
        - Title: {issue_data.get('title', 'N/A')}
        - Priority: {issue_data.get('priority', 'N/A')}

        **CHRONOLOGICAL COMMENTS:**
        {comments_text}

        **TIMELINE EVENTS (field changes):**
        {timeline_text}

        **RESEARCH QUESTION:**
        What is the complete role sequence for this issue, including when control goes back to previous roles? However, a role as a bot should not be included in the sequence.

        **INSTRUCTIONS:**
        1. **Focus on Comments and Changelog**: Analyze the chronological flow of comments and timeline events
        2. **Track Every Role Transition**: List every time control/responsibility shifts from one role to another
        3. **Include Back-and-Forth**: When someone sends work back to a previous role, include that transition
        4. **NEVER Include Participants in Role Sequence**: Participants should never appear in the role sequence itself
        5. **Assign Back-and-Forth Codes**: When control returns to previous roles due to problems, assign specific codes

        **CORE WORKFLOW ROLES TO TRACK:**
        - reporter
        - assignee  
        - peer_reviewer
        - integrator
        - tester
        - (other formal workflow roles, but NOT participant)

        **BACK-AND-FORTH CODE GENERATION:** When roles return to previous roles due to problems, create specific, descriptive codes:

        **Code Creation Guidelines:**
        - Create **descriptive, specific codes** that clearly indicate the problem found
        - Use **UPPER_CASE_WITH_UNDERSCORES** format for consistency
        - Focus on **what the actual problem is** based on the evidence (be specific, not generic).
        - Keep codes **concise but clear** - they should be meaningful for analysis
        - Be **consistent** - similar problems should get similar codes
        - **Ignore bot comments** - Do not create codes based on the authors of CiBoT, noreply, or automated system comments

        **Examples of Good Codes** (create others as needed based on the actual problems you find):
        
        **Be specific about the actual problem:**
        - CODE_QUALITY_STANDARDS, INTEGRATION_CONFLICTS, UNIT_TEST_FAILURES, MERGE_CONFLICTS
        
        **Examples of bad codes:**
        - PEER_REVIEW_FEEDBACK
        - DEVELOPMENT_ISSUES
        - TECHNICAL_ISSUES
        - DEVELOPMENT_WORKFLOW_ISSUES

        **EXAMPLES of "goes back to previous roles":**
        **IMPORTANT**: Only consider it "going back to previous roles" when there is a problem or issue that requires returning to fix something:
        - assignee → peer_reviewer → assignee (reviewer found issues, sent back to assignee to fix)
        - assignee → integrator → assignee (integration failed, sent back to assignee)
        - peer_reviewer → assignee → peer_reviewer (fixes made, sent back for re-review)

        **NOT considered "going back"**: Normal forward progression through roles, even if the same person appears multiple times in different contexts.

        **CODE ASSIGNMENT WITH SOURCE EVIDENCE:**
        - **Default Sequence**: If the sequence is exactly [reporter, assignee, peer_reviewer, integrator, tester] with NO back-and-forth, assign back_and_forth_codes = []
        - **Back-and-Forth Present**: For each instance where control returns to a previous role due to a problem, create a code that:
          - `code`: Describes the specific problem that caused the return (be descriptive and specific)
          - `source_text`: The exact text excerpt that shows the problem (max 200 characters)
          - `source_location`: Where the evidence comes from (e.g., "comment_3", "description", "timeline_5")
        - **Multiple Returns**: If there are multiple back-and-forth instances, assign multiple code objects in chronological order
        - **Evidence Requirements**: Always include the actual text that supports the code assignment
        - **Ignore Bot Comments**: Do NOT assign codes based on comments from CiBoT, noreply, or other automated systems

        **HANDLING PARTICIPANT INFLUENCE:**
        If a participant identifies critical problems that cause work to return to a previous role, include this in the back-and-forth codes:
        - Example code: "PARTICIPANT_REPORTED_REGRESSION" with source evidence showing participant feedback
        - The participant still doesn't appear in the role sequence, but their influence is captured in the codes

        **TASK:**
        Examine the comments and timeline events chronologically. For each significant action, determine:
        1. Who is taking action (which role)? **Skip CiBoT, noreply, and automated systems**
        2. Does this action represent a meaningful transition in control/responsibility?
        3. If someone returns to a previous role, is it because of a problem that needs fixing?
        4. What specific problem caused the return? (create an appropriate code based on the evidence)
        5. What is the next role in the sequence?

        **Code Creation Approach:**
        - Read the evidence carefully and identify the specific problem mentioned
        - Create a code that clearly describes what the problem is
        - Be specific but not overly complex
        - Focus on making the code useful for analysis of general software development issues

        **OUTPUT REQUIREMENTS:**
        - complete_role_sequence: Complete chronological sequence of roles
        - sequence_explanation: Brief explanation of the workflow progression including back-and-forth triggers
        - back_and_forth_codes: List of code objects with evidence for each back-and-forth instance (empty list if default sequence)

        Output your analysis in the specified JSON format.
        """
    
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            return {"error": "No successful analyses to summarize"}
        
        # Collect all sequences (use validated sequences if available)
        all_sequences = []
        all_codes = []
        for result in successful_results:
            # Use validated sequence if available, otherwise original
            sequence = result.get('complete_role_sequence', [])
            codes = result.get('back_and_forth_codes', [])
            if sequence:
                all_sequences.append(tuple(sequence))
            # Extract just the code strings for counting, handling both dict and Pydantic objects
            for code_obj in codes:
                if hasattr(code_obj, 'model_dump'):  # Pydantic v2
                    all_codes.append(code_obj.model_dump().get('code', ''))
                elif hasattr(code_obj, 'dict'):  # Pydantic v1
                    all_codes.append(code_obj.dict().get('code', ''))
                elif isinstance(code_obj, dict):
                    all_codes.append(code_obj.get('code', ''))
                else:
                    all_codes.append(str(code_obj))  # fallback for string codes
        
        # Count sequence patterns
        sequence_counts = Counter(all_sequences)
        
        # Count back-and-forth codes
        code_counts = Counter(all_codes)
        
        # Count issues with and without back-and-forth (handling Pydantic objects)
        issues_with_backforth = 0
        for result in successful_results:
            codes = result.get('back_and_forth_codes', [])
            if codes and len(codes) > 0:
                issues_with_backforth += 1
        issues_without_backforth = len(successful_results) - issues_with_backforth
        
        # Validation statistics
        validation_stats = {
            "total_validations_performed": len([r for r in successful_results if r.get('validation', {}).get('performed')]),
            "validation_agreements": len([r for r in successful_results if r.get('validation', {}).get('agreement_status') == 'Agrees']),
            "validation_disagreements": len([r for r in successful_results if r.get('validation', {}).get('agreement_status') == 'Disagrees']),
            "validation_partial_agreements": len([r for r in successful_results if r.get('validation', {}).get('agreement_status') == 'Partially_Agrees']),
            "sequences_changed_by_validation": len([r for r in successful_results if r.get('original_sequence') != r.get('complete_role_sequence')])
        }
        
        return {
            "research_question": self.question_description,
            "total_issues_analyzed": len(results),
            "successful_analyses": len(successful_results),
            "failed_analyses": len(results) - len(successful_results),
            
            "most_common_role_sequences": [
                {
                    "sequence": list(seq),
                    "count": count,
                    "percentage": round(count / len(successful_results) * 100, 1)
                }
                for seq, count in sequence_counts.most_common(15)
            ],
            
            "back_and_forth_analysis": {
                "issues_with_back_and_forth": issues_with_backforth,
                "issues_without_back_and_forth": issues_without_backforth,
                "percentage_with_back_and_forth": round(issues_with_backforth / len(successful_results) * 100, 1) if successful_results else 0,
                "most_common_codes": [
                    {
                        "code": code,
                        "count": count,
                        "percentage": round(count / len(all_codes) * 100, 1) if all_codes else 0
                    }
                    for code, count in code_counts.most_common(10)
                ],
                "total_back_and_forth_instances": len(all_codes)
            },
            
            "sequence_statistics": {
                "total_unique_sequences": len(sequence_counts),
                "average_sequence_length": round(sum(len(seq) for seq in all_sequences) / len(all_sequences), 2) if all_sequences else 0,
                "shortest_sequence": min(len(seq) for seq in all_sequences) if all_sequences else 0,
                "longest_sequence": max(len(seq) for seq in all_sequences) if all_sequences else 0
            },
            
            "validation_statistics": validation_stats
        }

class NonA11yTestingVerificationQuestion(BaseResearchQuestion):
    """Research Question 3: Testing & Verification methods analysis for non-accessibility issues"""
    
    @property
    def question_id(self) -> str:
        return "non_a11y_testing_verification"
    
    @property
    def question_description(self) -> str:
        return "What testing and verification methods do Moodle developers use to validate non-accessibility issues and fixes?"
    
    @property
    def result_model(self) -> Type[BaseModel]:
        return TestingVerificationResult  # Reuse the same data model
    
    def create_system_prompt(self) -> str:
        return """You are a software testing expert analyzing testing and verification methods in issue workflows.

                Your task is to identify how Moodle developers test reported bugs and verify that fixes address the problems.

                Key areas to analyze:
                - **Automated Testing**: Unit tests, integration tests, functional tests, regression tests, CI/CD pipeline tests
                - **Manual Testing**: User interface testing, browser testing, cross-platform testing, performance testing
                - **Testing vs Verification**: Testing = checking if bug exists; Verification = confirming fix works
                - **Testing Tools**

                Guidelines:
                - Analyze issue descriptions, comments, and test instructions for testing evidence
                - Identify specific testing tools, techniques, and methods mentioned
                - Distinguish between testing (bug validation) and verification (fix validation)
                - Include evidence sources with original text excerpts
                - Focus on software testing approaches and development tools
                - Note both successful and failed testing attempts
                - Include manual testing procedures and automated test suites

                Respond only with valid JSON matching the specified schema."""
    
    def create_analysis_prompt(self, issue_data: Dict[str, Any]) -> str:
        # Format comments for analysis
        comments_text = ""
        for i, comment in enumerate(issue_data.get('comments', []), 1):
            #comments_text += f"comment_{i}: {comment['role']} ({comment['author']}): {comment['body'][:600]}{'...' if len(comment['body']) > 600 else ''}\n"
            comments_text += f"comment_{i}: {comment['role']} ({comment['author']}): {comment['body']}\n"
        
        # Format test instructions if available
        test_instructions_text = ""
        if issue_data.get('test_instructions'):
            # test_instructions_text = f"{issue_data.get('test_instructions', '')[:800]}{'...' if len(issue_data.get('test_instructions', '')) > 800 else ''}"
            test_instructions_text = f"{issue_data.get('test_instructions', '')}"
        
        return f"""
            **TESTING & VERIFICATION METHODS ANALYSIS**

            Analyze this Moodle issue to identify testing and verification methods used by developers.

            **ISSUE DETAILS:**
            - Key: {issue_data.get('issue_key', 'N/A')}
            - Priority: {issue_data.get('priority', 'N/A')}
            - Labels: {', '.join(issue_data.get('labels', []))}

            **TITLE:**
            {issue_data.get('title', 'N/A')}

            **DESCRIPTION:**
            {issue_data.get('description', 'N/A')}

            **TEST INSTRUCTIONS:**
            {test_instructions_text if test_instructions_text else 'No test instructions provided'}

            **COMMENTS:**
            {comments_text}

            **ANALYSIS INSTRUCTIONS:**

            1. **Identify Testing Methods**: Look for evidence of testing to validate reported bugs (primarily in description and comments)
            2. **Identify Verification Methods**: Look for evidence of verifying that fixes work (primarily in test_instructions and comments)
            3. **Categorize Method Types**: Distinguish between automated tools and manual techniques
            4. **Extract Evidence**: Include specific text excerpts that indicate testing/verification

            **FIELD-SPECIFIC ANALYSIS:**
            - **description & comments**: Look for methods used to TEST/validate the original bug
            - **test_instructions**: Look for methods used to VERIFY/confirm that the patch/fix works
            - **comments**: May contain both testing (early comments) and verification (later comments after patches)

            **PURPOSE CLASSIFICATION:**
            - **testing**: Validating if the reported bug actually exists (usually found in description/early comments)
            - **verification**: Confirming that a proposed/implemented fix resolves the issue (usually found in test_instructions/later comments)

            **FIELD GUIDANCE:**
            - **test_instructions**: Almost always contains VERIFICATION methods (how to test the patch)
            - **description**: Usually contains TESTING methods (how the bug was discovered/validated)
            - **comments**: Can contain both - early comments often show TESTING, later comments show VERIFICATION

            **EVIDENCE SOURCING WITH ORIGINAL TEXT:**
            - **title**: "title: [excerpt from title]"
            - **description**: "description: [excerpt from description]" (usually TESTING methods)
            - **test_instructions**: "test_instructions: [excerpt from test instructions]" (usually VERIFICATION methods)
            - **comment_X**: "comment_X: [excerpt from comment]" (can be TESTING or VERIFICATION depending on timing)

            **OUTPUT REQUIREMENTS:**
            For each testing/verification method found:
            - method_type: "automated" or "manual"
            - tool_or_technique: Specific tool name or technique used
            - purpose: "testing" or "verification"
            - target_issue: What specific issue is being tested/verified
            - evidence_source: Source with original text excerpt
            - details: Additional context about the method

            **DEDUPLICATION RULES:**
            - If the same tool/technique is used multiple times for the same purpose and target issue, report only the FIRST instance
            - If the same tool/technique is used for different purposes (testing vs verification) or different target issues, report each instance separately
            - Example: If a tool is mentioned 3 times for testing database functionality, report only the first mention
            - Example: If a tool is used for testing database issues AND verifying fixes, report both instances
            
            **EXAMPLES:**
            - method_type: "automated", tool_or_technique: "PHPUnit", purpose: "verification", target_issue: "database query failure", evidence_source: "comment_5: PHPUnit tests now pass for the updated query"
            - method_type: "manual", tool_or_technique: "browser_testing", purpose: "testing", target_issue: "UI rendering", evidence_source: "description: tested in Chrome and Firefox, layout breaks in both browsers"
            - method_type: "manual", tool_or_technique: "step_by_step_testing", purpose: "verification", target_issue: "user workflow", evidence_source: "test_instructions: Navigate to course page, click enroll button, verify enrollment is recorded"
            - method_type: "automated", tool_or_technique: "Behat", purpose: "testing", target_issue: "form submission", evidence_source: "comment_2: Behat tests show form validation is failing"

            **IMPORTANT:**
            - Only report testing/verification methods you can clearly justify from the evidence
            - Include actual text excerpts in evidence_source
            - If no testing/verification methods are mentioned, return empty testing_methods list
            - Be specific about tools and techniques mentioned
            - Distinguish between different testing frameworks, browsers, etc.
            - Focus on software testing, not just general issue resolution
            - Remove redundancy: If the same tool/technique is used multiple times for the same purpose and target issue, report only the FIRST instance

            Analyze this issue and provide your findings in the specified JSON format.
            """
    
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            return {"error": "No successful analyses to summarize"}
        
        # Collect all testing methods
        all_methods = []
        issues_with_testing = 0
        issues_without_testing = 0
        
        for result in successful_results:
            methods = result.get('testing_methods', [])
            if methods:
                issues_with_testing += 1
                # Convert Pydantic objects to dictionaries
                for method in methods:
                    if hasattr(method, 'model_dump'):
                        all_methods.append(method.model_dump())
                    elif hasattr(method, 'dict'):
                        all_methods.append(method.dict())
                    else:
                        all_methods.append(method)
            else:
                issues_without_testing += 1
        
        # Analyze method distributions
        method_types = Counter()
        tools_techniques = Counter()
        purposes = Counter()
        evidence_sources = Counter()
        
        automated_tools = Counter()
        manual_techniques = Counter()
        
        for method in all_methods:
            method_type = method.get('method_type', 'Unknown')
            tool_technique = method.get('tool_or_technique', 'Unknown')
            purpose = method.get('purpose', 'Unknown')
            evidence_source = method.get('evidence_source', 'Unknown')
            
            method_types[method_type] += 1
            tools_techniques[tool_technique] += 1
            purposes[purpose] += 1
            evidence_sources[evidence_source] += 1
            
            if method_type == 'automated':
                automated_tools[tool_technique] += 1
            elif method_type == 'manual':
                manual_techniques[tool_technique] += 1
        
        # Validation statistics
        validation_stats = {
            "total_validations_performed": len([r for r in successful_results if r.get('testing_validation', {}).get('performed')]),
            "validation_agreements": len([r for r in successful_results if r.get('testing_validation', {}).get('agreement_status') == 'Agrees']),
            "validation_disagreements": len([r for r in successful_results if r.get('testing_validation', {}).get('agreement_status') == 'Disagrees']),
            "validation_partial_agreements": len([r for r in successful_results if r.get('testing_validation', {}).get('agreement_status') == 'Partially_Agrees']),
            "methods_changed_by_validation": len([r for r in successful_results if r.get('original_methods') != r.get('testing_methods')])
        }
        
        return {
            "research_question": self.question_description,
            "total_issues_analyzed": len(results),
            "successful_analyses": len(successful_results),
            "failed_analyses": len(results) - len(successful_results),
            
            "testing_overview": {
                "issues_with_testing_methods": issues_with_testing,
                "issues_without_testing_methods": issues_without_testing,
                "total_methods_found": len(all_methods),
                "percentage_issues_with_testing": round(issues_with_testing / len(successful_results) * 100, 1) if successful_results else 0
            },
            
            "method_type_distribution": [
                {
                    "type": method_type,
                    "count": count,
                    "percentage": round(count / len(all_methods) * 100, 1) if all_methods else 0
                }
                for method_type, count in method_types.most_common()
            ],
            
            "most_used_tools_techniques": [
                {
                    "tool_technique": tool,
                    "count": count,
                    "percentage": round(count / len(all_methods) * 100, 1) if all_methods else 0
                }
                for tool, count in tools_techniques.most_common(15)
            ],
            
            "purpose_distribution": [
                {
                    "purpose": purpose,
                    "count": count,
                    "percentage": round(count / len(all_methods) * 100, 1) if all_methods else 0
                }
                for purpose, count in purposes.most_common()
            ],
            
            "automated_tools": [
                {
                    "tool": tool,
                    "count": count,
                    "percentage": round(count / len([m for m in all_methods if m.get('method_type') == 'automated']) * 100, 1) if [m for m in all_methods if m.get('method_type') == 'automated'] else 0
                }
                for tool, count in automated_tools.most_common(10)
            ],
            
            "manual_techniques": [
                {
                    "technique": technique,
                    "count": count,
                    "percentage": round(count / len([m for m in all_methods if m.get('method_type') == 'manual']) * 100, 1) if [m for m in all_methods if m.get('method_type') == 'manual'] else 0
                }
                for technique, count in manual_techniques.most_common(10)
            ],
            
            "evidence_source_distribution": [
                {
                    "source": source,
                    "count": count,
                    "percentage": round(count / len(all_methods) * 100, 1) if all_methods else 0
                }
                for source, count in evidence_sources.most_common()
            ],
            
            "testing_statistics": {
                "average_methods_per_issue": round(len(all_methods) / len(successful_results), 2) if successful_results else 0,
                "max_methods_in_single_issue": max([len(r.get('testing_methods', [])) for r in successful_results]) if successful_results else 0,
                "automated_vs_manual_ratio": round(method_types.get('automated', 0) / max(method_types.get('manual', 1), 1), 2),
                "testing_vs_verification_ratio": round(purposes.get('testing', 0) / max(purposes.get('verification', 1), 1), 2)
            },
            
            "validation_statistics": validation_stats
        }


class ParticipantInfluenceQuestion(BaseResearchQuestion):
    """Research Question: Participant Influence on Accessibility Solution Development"""
    
    @property
    def question_id(self) -> str:
        return "participant_influence"
    
    @property
    def question_description(self) -> str:
        return "How do participants influence accessibility solution/patch development through suggestions and feedback?"
    
    @property
    def result_model(self) -> Type[BaseModel]:
        return ParticipantInfluenceResult
    
    def create_system_prompt(self) -> str:
        return """You are a software development research expert analyzing how participants influence solution development in Moodle issues.

        Your task is to identify specific ways participants contribute to solutions through suggestions, feedback, testing, and guidance, and assign low-level qualitative codes to these influences.

        Key focus areas:
        - **Technical Suggestions**: Specific improvements proposed by participants
        - **Problem Identification**: Participants identifying bugs, issues, or barriers
        - **Solution Guidance**: Participants providing direction on best practices or approaches
        - **Testing Feedback**: Participants testing features and providing feedback
        - **Standards Reference**: Participants citing standards, guidelines, or compliance requirements
        - **Process Improvement**: Participants suggesting workflow or methodology improvements

        Guidelines:
        - Focus only on participants (NOT reporter/assignee/peer_reviewer/integrator/tester/bot roles)
        - Exclude bot comments or automated comments (author: CiBoT or noreply)
        - Create specific, descriptive codes that capture participant influence
        - Extract evidence showing how participant input influenced solution development
        - Include evidence sources with original text excerpts
        - Consider all types of contributions that influenced the development process

        Respond only with valid JSON matching the specified schema."""
    
    def create_analysis_prompt(self, issue_data: Dict[str, Any]) -> str:
        # Format ALL comments to identify participants vs core roles
        all_comments_text = ""
        participant_comments = []
        
        for i, comment in enumerate(issue_data.get('comments', []), 1):
            # Get role information
            roles = comment.get('role', [])
            if isinstance(roles, str):
                roles = [roles]
            
            author = comment['author']
            body = comment['body']
            
            # Identify core workflow roles
            core_roles = {'reporter', 'assignee', 'peer_reviewer', 'integrator', 'tester', 'CiBoT', 'noreply'}
            is_participant = not any(role in core_roles for role in roles)
            
            comment_text = f"comment_{i}: {roles} ({author}): {body[:1000]}{'...' if len(body) > 1000 else ''}\n"
            all_comments_text += comment_text
            
            # Mark participants for focused analysis
            if is_participant or 'participant' in roles:
                # participant_comments.append(f"comment_{i}: PARTICIPANT ({author}): {body[:1500]}{'...' if len(body) > 1500 else ''}")
                participant_comments.append(f"comment_{i}: PARTICIPANT ({author}): {body}")
        
        participant_comments_text = "\n".join(participant_comments) if participant_comments else "No participant comments found"
        
        return f"""
            **PARTICIPANT INFLUENCE ON ACCESSIBILITY SOLUTION DEVELOPMENT ANALYSIS**

            Analyze how participants influence accessibility solution development by assigning low-level qualitative codes to their contributions.

            **ISSUE DETAILS:**
            - Key: {issue_data.get('issue_key', 'N/A')}
            - Priority: {issue_data.get('priority', 'N/A')}
            - Labels: {', '.join(issue_data.get('labels', []))}

            **TITLE:**
            {issue_data.get('title', 'N/A')}

            **DESCRIPTION:**
            {issue_data.get('description', 'N/A')[:800]}{'...' if len(issue_data.get('description', '')) > 800 else ''}

            **WHO ARE PARTICIPANTS:**
            Participants are users who are NOT in core development workflow roles:
            - **CORE ROLES (NOT participants)**: reporter, assignee, peer_reviewer, integrator, tester, bot
            - **PARTICIPANTS**: Anyone whose role is NOT one of the above core roles
            - **Examples**: Users with roles like "participant", "community", "user", or any other non-core role
            - **Focus**: Look for comments from users who are NOT reporter/assignee/peer_reviewer/integrator/tester/bot

            **PARTICIPANT COMMENTS (Focus Analysis Here):**
            {participant_comments_text}

            **ALL COMMENTS (For Context and Participant Identification):**
            {all_comments_text}

            **ANALYSIS INSTRUCTIONS:**

            1. **Identify All Participants**: Anyone whose role is NOT reporter/assignee/peer_reviewer/integrator/tester/bot
            2. **Exclude bot comments or automated comments (author: CiBoT or noreply)**
            2. **Focus on Accessibility**: Prioritize accessibility-related contributions but include general technical issues that affect accessibility solutions
            3. **Comprehensive Influence Detection**: Look for ALL types of participant contributions that influenced accessibility solution development
            4. **Generate Specific Codes**: Create descriptive codes for each type of influence found
            5. **Include All Evidence**: Don't miss subtle influences like test failures, problem reports, or guidance
           

            Generate codes that are **specific, descriptive, and meaningful** by following these principles:

            **1. SPECIFICITY PRINCIPLE:**
            - Capture the EXACT nature of the participant's contribution
            - Include the specific accessibility aspect being addressed
            - Avoid generic terms like "barrier," "issue," or "guidance"
            - Be precise about what was identified, suggested, or tested

            **2. TECHNICAL PRECISION:**
            - Include specific assistive technologies when mentioned (NVDA, JAWS, screen readers, etc.)
            - Reference specific accessibility standards when cited (WCAG criteria, ARIA attributes, etc.)
            - Mention specific UI elements or interactions when relevant
            - Include testing contexts when provided (browsers, devices, assistive technologies, etc.)

            **3. ACTION-ORIENTED CODING:**
            - Start with the TYPE of contribution (REPORTED, SUGGESTED, IDENTIFIED, RECOMMENDED, etc.)
            - Follow with the SPECIFIC accessibility aspect
            - End with the CONTEXT or SOLUTION when applicable

            **4. EVIDENCE-BASED CODING:**
            - Code should directly reflect what's in the source text
            - Don't infer beyond what's explicitly stated
            - Capture the participant's actual contribution, not assumed impact

            **5. GRANULAR DIFFERENTIATION:**
            - Different accessibility problems should get different codes
            - Same participant making different contributions should get different codes
            - Similar contributions from different participants can share codes

            **CODE CONSTRUCTION GUIDANCE:**

            **Format Pattern:** ACTION_SPECIFIC_ACCESSIBILITY_ASPECT_CONTEXT

            **Action Words:** REPORTED, SUGGESTED, IDENTIFIED, RECOMMENDED, PROPOSED, TESTED, HIGHLIGHTED, DISCOVERED, etc.

            **Accessibility Aspects:** Be specific about what accessibility area:
            - Instead of "accessibility" → use "screen_reader_announcement," "keyboard_navigation," "color_contrast," etc.
            - Instead of "barrier" → use "focus_trap_missing," "aria_label_absent," "tab_order_broken," etc.
            - Instead of "issue" → use the actual problem "announcement_failure," "visibility_loss," "navigation_block," etc.

            **Context/Solution:** When relevant, include:
            - Specific assistive technology tested
            - Specific WCAG criteria referenced  
            - Specific UI component affected
            - Specific accessibility solution suggested

            **QUALITY CHECKLIST:**
            Before finalizing a code, ask:
            1. Does this code tell me exactly what the participant contributed?
            2. Would someone reading just this code understand the specific accessibility aspect?
            3. Is this code distinct from other codes I might generate?
            4. Does this code reflect what's actually in the evidence text?
            5. Could this code be more specific without being overly verbose?

            **LENGTH GUIDANCE:**
            - Aim for codes that are descriptive but not excessively long
            - 3-6 words typically optimal
            - Prioritize clarity over brevity
            - Use underscores to separate concepts clearly

            **CODE ASSIGNMENT WITH SOURCE EVIDENCE:**

            For each participant contribution that influenced accessibility solution development:
            - `code`: Apply the principles above to create a specific, descriptive code
            - `source_text`: Exact text excerpt showing the contribution (max 200 characters)
            - `source_location`: Comment location (e.g., "comment_3", "comment_7")

            **EVIDENCE REQUIREMENTS:**
            - Include actual text excerpts in source_text
            - Focus on substantial contributions that demonstrably influenced accessibility solution development
            - Prioritize accessibility-related contributions
            - Include general technical contributions (like Behat test failures) that affect accessibility solutions

            **IMPORTANT DETECTION RULES:**

            1. **Don't Miss Test Failures**: If participants report testing failures (including Behat, automation), create specific codes
            2. **Include Accessibility Problem Reports**: Any accessibility problem identified by participants gets a specific code
            3. **Capture All Accessibility Suggestions**: Even small accessibility suggestions that influenced the solution should be coded
            4. **Standards References**: When participants cite WCAG, ARIA, etc., create codes reflecting the specific standard
            5. **Assistive Technology Feedback**: When participants provide AT testing feedback, create specific codes
            6. **Cross-Platform Accessibility Issues**: When participants test accessibility on different platforms/ATs

            **EXCLUSION CRITERIA:**
            - Comments from reporter, assignee, peer_reviewer, integrator, tester, bot roles
            - General comments that don't influence accessibility solution development
            - Purely conversational or irrelevant comments

            **DEDUPLICATION RULES:**
            - If same participant makes same suggestion multiple times, report only first instance
            - If different participants make similar contributions, report each separately
            - Focus on substantial contributions that demonstrably influenced accessibility solution development

            **OUTPUT REQUIREMENTS:**
            - For each participant influence: specific code following the principles above, supporting evidence text, source location
            - If no participant influences found, return empty participant_influence_codes list
            - Be comprehensive - don't miss any type of participant contribution to accessibility solutions

            Analyze this issue comprehensively and provide your findings in the specified JSON format.
            """
    
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            return {"error": "No successful analyses to summarize"}
        
        # Collect all codes
        all_codes = []
        issues_with_codes = 0
        issues_without_codes = 0
        
        for result in successful_results:
            codes = result.get('participant_influence_codes', [])
            if codes:
                issues_with_codes += 1
                # Convert Pydantic objects to dictionaries and extract code strings
                for code_obj in codes:
                    if hasattr(code_obj, 'model_dump'):
                        code_dict = code_obj.model_dump()
                        all_codes.append(code_dict.get('code', ''))
                    elif hasattr(code_obj, 'dict'):
                        code_dict = code_obj.dict()
                        all_codes.append(code_dict.get('code', ''))
                    elif isinstance(code_obj, dict):
                        all_codes.append(code_obj.get('code', ''))
                    else:
                        all_codes.append(str(code_obj))
            else:
                issues_without_codes += 1
        
        # Count code frequencies
        code_counts = Counter(all_codes)
        
        # Validation statistics
        validation_stats = {
            "total_validations_performed": len([r for r in successful_results if r.get('participant_validation', {}).get('performed')]),
            "validation_agreements": len([r for r in successful_results if r.get('participant_validation', {}).get('agreement_status') == 'Agrees']),
            "validation_disagreements": len([r for r in successful_results if r.get('participant_validation', {}).get('agreement_status') == 'Disagrees']),
            "validation_partial_agreements": len([r for r in successful_results if r.get('participant_validation', {}).get('agreement_status') == 'Partially_Agrees']),
            "codes_changed_by_validation": len([r for r in successful_results if r.get('original_codes') != r.get('participant_influence_codes')])
        }
        
        return {
            "research_question": self.question_description,
            "total_issues_analyzed": len(results),
            "successful_analyses": len(successful_results),
            "failed_analyses": len(results) - len(successful_results),
            
            "participant_influence_overview": {
                "issues_with_participant_influences": issues_with_codes,
                "issues_without_participant_influences": issues_without_codes,
                "total_influence_codes_found": len(all_codes),
                "percentage_issues_with_influences": round(issues_with_codes / len(successful_results) * 100, 1) if successful_results else 0
            },
            
            "most_common_influence_codes": [
                {
                    "code": code,
                    "count": count,
                    "percentage": round(count / len(all_codes) * 100, 1) if all_codes else 0
                }
                for code, count in code_counts.most_common(15)
            ],
            
            "participant_statistics": {
                "average_codes_per_issue": round(len(all_codes) / len(successful_results), 2) if successful_results else 0,
                "max_codes_in_single_issue": max([len(r.get('participant_influence_codes', [])) for r in successful_results]) if successful_results else 0,
                "unique_influence_codes": len(code_counts)
            },
            
            "validation_statistics": validation_stats
        }
        
class NonA11yParticipantInfluenceQuestion(BaseResearchQuestion):
    """Research Question: Participant Influence on Non-Accessibility Solution Development"""
    
    @property
    def question_id(self) -> str:
        return "non_a11y_participant_influence"
    
    @property
    def question_description(self) -> str:
        return "How do participants influence non-accessibility solution/patch development through suggestions and feedback?"
    
    @property
    def result_model(self) -> Type[BaseModel]:
        return ParticipantInfluenceResult  # Reuse the same data model
    
    def create_system_prompt(self) -> str:
        return """You are a software development research expert analyzing how participants influence solution development in Moodle non-accessibility issues.

        Your task is to identify specific ways participants contribute to solutions through suggestions, feedback, testing, and guidance, and assign low-level qualitative codes to these influences.

        Key focus areas:
        - **Technical Suggestions**: Specific improvements proposed by participants for general software development
        - **Problem Identification**: Participants identifying bugs, issues, or barriers in general functionality
        - **Solution Guidance**: Participants providing direction on best practices or approaches for software development
        - **Testing Feedback**: Participants testing features and providing general testing feedback
        - **Standards Reference**: Participants citing coding standards, development guidelines, or compliance requirements
        - **Process Improvement**: Participants suggesting workflow or methodology improvements

        Guidelines:
        - Focus only on participants (NOT reporter/assignee/peer_reviewer/integrator/tester/bot roles)
        - Exclude bot comments or automated comments (author: CiBoT or noreply)
        - Create specific, descriptive codes that capture participant influence on general software development
        - Extract evidence showing how participant input influenced solution development
        - Include evidence sources with original text excerpts
        - Consider all types of contributions that influenced the development process

        Respond only with valid JSON matching the specified schema."""
    
    def create_analysis_prompt(self, issue_data: Dict[str, Any]) -> str:
        # Format ALL comments to identify participants vs core roles
        all_comments_text = ""
        participant_comments = []
        
        for i, comment in enumerate(issue_data.get('comments', []), 1):
            # Get role information
            roles = comment.get('role', [])
            if isinstance(roles, str):
                roles = [roles]
            
            author = comment['author']
            body = comment['body']
            
            # Identify core workflow roles
            core_roles = {'reporter', 'assignee', 'peer_reviewer', 'integrator', 'tester', 'CiBoT', 'noreply'}
            is_participant = not any(role in core_roles for role in roles)
            
            comment_text = f"comment_{i}: {roles} ({author}): {body[:800]}{'...' if len(body) > 800 else ''}\n"
            all_comments_text += comment_text
            
            # Mark participants for focused analysis
            if is_participant or 'participant' in roles:
                # participant_comments.append(f"comment_{i}: PARTICIPANT ({author}): {body[:1500]}{'...' if len(body) > 1500 else ''}")
                participant_comments.append(f"comment_{i}: PARTICIPANT ({author}): {body}")
        
        participant_comments_text = "\n".join(participant_comments) if participant_comments else "No participant comments found"
        
        return f"""
            **PARTICIPANT INFLUENCE ON SOFTWARE SOLUTION DEVELOPMENT ANALYSIS**

            Analyze how participants influence software solution development by assigning low-level qualitative codes to their contributions.

            **ISSUE DETAILS:**
            - Key: {issue_data.get('issue_key', 'N/A')}
            - Priority: {issue_data.get('priority', 'N/A')}
            - Labels: {', '.join(issue_data.get('labels', []))}

            **TITLE:**
            {issue_data.get('title', 'N/A')}

            **DESCRIPTION:**
            {issue_data.get('description', 'N/A')[:800]}{'...' if len(issue_data.get('description', '')) > 800 else ''}

            **WHO ARE PARTICIPANTS:**
            Participants are users who are NOT in core development workflow roles:
            - **CORE ROLES (NOT participants)**: reporter, assignee, peer_reviewer, integrator, tester, bot
            - **PARTICIPANTS**: Anyone whose role is NOT one of the above core roles
            - **Examples**: Users with roles like "participant", "community", "user", or any other non-core role
            - **Focus**: Look for comments from users who are NOT reporter/assignee/peer_reviewer/integrator/tester/bot

            **PARTICIPANT COMMENTS (Focus Analysis Here):**
            {participant_comments_text}

            **ALL COMMENTS (For Context and Participant Identification):**
            {all_comments_text}

            **ANALYSIS INSTRUCTIONS:**

            1. **Identify All Participants**: Anyone whose role is NOT reporter/assignee/peer_reviewer/integrator/tester/bot
            2. **Exclude bot comments**: Skip comments from CiBoT, noreply, or automated systems
            3. **Focus on Solution Development Influence**: Prioritize contributions that influenced how the software solution was developed
            4. **Generate Specific Codes**: Create descriptive codes for each type of influence found
            5. **Include All Evidence**: Capture all substantial contributions that affected solution development


            **CODE ASSIGNMENT WITH SOURCE EVIDENCE:**

            For each participant contribution that influenced solution development:
            - `code`: Apply enhanced principles above to create specific, impact-focused codes
            - `source_text`: Exact text excerpt showing the contribution (max 200 characters)
            - `source_location`: Comment location (e.g., "comment_3", "comment_7")

            **ENHANCED QUALITY CHECKLIST:**
            Before finalizing a code, verify:
            1. Does this code show clear influence on solution development (not just general feedback)?
            2. Is the code specific enough to understand the exact contribution and its impact?
            3. Does the code indicate both the TYPE of contribution and the AREA affected?
            4. Would this code be meaningful for analyzing how participants shape solutions?
            5. Is the supporting evidence sufficient and clearly related to solution development?

            **IMPORTANT DETECTION RULES:**

            1. **Prioritize Solution-Shaping Contributions**: Focus on contributions that demonstrably influenced how the solution was developed, implemented, or designed
            2. **Capture Implementation Influence**: Include suggestions that led to specific implementation decisions or approaches
            3. **Include Requirement Clarifications**: Code contributions that helped clarify or expand requirements
            4. **Don't Miss Technical Guidance**: Include contributions that provided technical direction or best practices
            5. **Include Problem Analysis**: Code problem identification that influenced solution design

            **EXCLUSION CRITERIA:**
            - Comments from reporter, assignee, peer_reviewer, integrator, tester, bot roles
            - General appreciation or acknowledgment comments without solution influence
            - Purely conversational comments without development impact
            - Comments that don't demonstrate influence on solution development

            **OUTPUT REQUIREMENTS:**
            - For each participant influence: specific, impact-focused code following enhanced principles, supporting evidence text, source location
            - If no participant influences found, return empty participant_influence_codes list
            - Be comprehensive but selective - include all substantial solution-shaping contributions

            Analyze this issue comprehensively and provide your findings in the specified JSON format.
            """
    
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            return {"error": "No successful analyses to summarize"}
        
        # Collect all codes
        all_codes = []
        issues_with_codes = 0
        issues_without_codes = 0
        
        for result in successful_results:
            codes = result.get('participant_influence_codes', [])
            if codes:
                issues_with_codes += 1
                # Convert Pydantic objects to dictionaries and extract code strings
                for code_obj in codes:
                    if hasattr(code_obj, 'model_dump'):
                        code_dict = code_obj.model_dump()
                        all_codes.append(code_dict.get('code', ''))
                    elif hasattr(code_obj, 'dict'):
                        code_dict = code_obj.dict()
                        all_codes.append(code_dict.get('code', ''))
                    elif isinstance(code_obj, dict):
                        all_codes.append(code_obj.get('code', ''))
                    else:
                        all_codes.append(str(code_obj))
            else:
                issues_without_codes += 1
        
        # Count code frequencies
        code_counts = Counter(all_codes)
        
        # Validation statistics
        validation_stats = {
            "total_validations_performed": len([r for r in successful_results if r.get('participant_validation', {}).get('performed')]),
            "validation_agreements": len([r for r in successful_results if r.get('participant_validation', {}).get('agreement_status') == 'Agrees']),
            "validation_disagreements": len([r for r in successful_results if r.get('participant_validation', {}).get('agreement_status') == 'Disagrees']),
            "validation_partial_agreements": len([r for r in successful_results if r.get('participant_validation', {}).get('agreement_status') == 'Partially_Agrees']),
            "codes_changed_by_validation": len([r for r in successful_results if r.get('original_codes') != r.get('participant_influence_codes')])
        }
        
        return {
            "research_question": self.question_description,
            "total_issues_analyzed": len(results),
            "successful_analyses": len(successful_results),
            "failed_analyses": len(results) - len(successful_results),
            
            "participant_influence_overview": {
                "issues_with_participant_influences": issues_with_codes,
                "issues_without_participant_influences": issues_without_codes,
                "total_influence_codes_found": len(all_codes),
                "percentage_issues_with_influences": round(issues_with_codes / len(successful_results) * 100, 1) if successful_results else 0
            },
            
            "most_common_influence_codes": [
                {
                    "code": code,
                    "count": count,
                    "percentage": round(count / len(all_codes) * 100, 1) if all_codes else 0
                }
                for code, count in code_counts.most_common(15)
            ],
            
            "participant_statistics": {
                "average_codes_per_issue": round(len(all_codes) / len(successful_results), 2) if successful_results else 0,
                "max_codes_in_single_issue": max([len(r.get('participant_influence_codes', [])) for r in successful_results]) if successful_results else 0,
                "unique_influence_codes": len(code_counts)
            },
            
            "validation_statistics": validation_stats
        }

class WCAGCategoryQuestion(BaseResearchQuestion):
    """Research Question 2: WCAG 2.2 SC violation categorization"""
    
    @property
    def question_id(self) -> str:
        return "wcag_categorization"
    
    @property
    def question_description(self) -> str:
        return "What WCAG 2.2 Success Criteria are violated in accessibility issues?"
    
    @property
    def result_model(self) -> Type[BaseModel]:
        return WCAGCategoryResult
    
    def create_system_prompt(self) -> str:
        return """You are an accessibility expert specializing in WCAG 2.2 Success Criteria analysis.

Your task is to analyze Moodle accessibility issues and identify which WCAG 2.2 Success Criteria are violated.

Key expertise areas:
- Complete knowledge of ALL WCAG 2.2 Success Criteria (A, AA, AAA levels)
- Understanding of the four principles: Perceivable, Operable, Understandable, Robust
- Experience with common accessibility patterns in web applications
- Ability to map technical implementation issues to specific WCAG violations

Guidelines:
- Analyze for violations of ANY WCAG 2.2 Success Criteria
- Use evidence from issue title, description, and user comments
- Specify the exact source of evidence (title, description, comment_X)
- Provide specific reasons why each violation occurs
- If multiple instances of the same SC violation exist, report only the FIRST instance
- Be conservative - only assign violations you can clearly justify
- Consider the context of educational/learning management systems
- Note: The system will automatically add higher-level violations for these criteria when lower levels are violated:
  * Contrast (Minimum) → Contrast (Enhanced)
  * Link Purpose (In Context) → Link Purpose (Link Only)
  * Target Size (Enhanced) → Target Size (Minimum)
  * Accessible Authentication (Minimum) → Accessible Authentication (Enhanced)
  * Images of Text → Images of Text (No Exception)

Respond only with valid JSON matching the specified schema."""
    
    def create_analysis_prompt(self, issue_data: Dict[str, Any]) -> str:
        # Format comments for analysis with clear numbering
        comments_text = ""
        for i, comment in enumerate(issue_data.get('comments', []), 1):
            comments_text += f"comment_{i}: {comment['role']} ({comment['author']}): {comment['body'][:500]}{'...' if len(comment['body']) > 500 else ''}\n"
        
        return f"""
    **WCAG 2.2 SUCCESS CRITERIA VIOLATION ANALYSIS**

    Analyze this Moodle accessibility issue to identify violations of ANY WCAG 2.2 Success Criteria.

    **ISSUE DETAILS:**
    - Key: {issue_data.get('issue_key', 'N/A')}
    - Priority: {issue_data.get('priority', 'N/A')}
    - Labels: {', '.join(issue_data.get('labels', []))}

    **TITLE:**
    {issue_data.get('title', 'N/A')}

    **DESCRIPTION:**
    {issue_data.get('description', 'N/A')}

    **COMMENTS:**
    {comments_text}

    **ANALYSIS INSTRUCTIONS:**

    1. **Comprehensive Analysis**: Examine for violations of ANY WCAG 2.2 Success Criteria (all levels A, AA, AAA)
    2. **Evidence Sourcing**: Identify specific evidence from title, description, or comments AND include the actual text
    3. **Violation Reasoning**: Explain clearly why each issue constitutes a violation
    4. **Deduplication**: If the same SC is violated multiple times, report only the FIRST instance
    5. **Source Attribution**: Specify source AND include relevant excerpt from the original text

    **EVIDENCE SOURCING FORMAT WITH ORIGINAL TEXT:**
    - **title**: "title: [excerpt from title text]" (e.g., "title: Button not accessible via keyboard")
    - **description**: "description: [excerpt from description text]" (e.g., "description: Users cannot navigate to activities via keyboard due to incorrect tabindex")
    - **comment_X**: "comment_X: [excerpt from comment text]" (e.g., "comment_2: Screen reader announces incorrect role for navigation element")

    **OUTPUT REQUIREMENTS:**
    For each WCAG 2.2 SC violation found:
    - sc_name: Include "SC" prefix, number, official name, and level in format "SC X.X.X Name (Level)" (e.g., "SC 2.1.1 Keyboard (A)", "SC 2.4.11 Focus Not Obscured (Minimum) (AA)")
    - evidence_source: Specify source WITH original text excerpt from the issue (e.g., "description: tabindex is not set correctly, making it impossible to navigate to the activities via keyboard")
    - violation_reason: Explain specifically why this violates the SC

    **IMPORTANT RULES:**
    - Analyze for violations of ALL WCAG 2.2 Success Criteria (not just new ones)
    - If the same SC is violated multiple times, report only the FIRST instance
    - Be specific about evidence sources and include actual text excerpts
    - Only report violations you can clearly justify from the available evidence
    - If no WCAG 2.2 SC violations are found, return an empty violated_sc list
    - Always use "SC" prefix in sc_name field
    - Always include original text in evidence_source field
    - DO NOT worry about hierarchical violations - the system will automatically add higher-level violations

    **EXAMPLES OF PROPER evidence_source FORMAT:**
    - "title: Course navigation buttons not keyboard accessible"
    - "description: The tabindex is not set correctly, making it impossible to navigate to the activities via keyboard"
    - "comment_1: Screen reader users cannot access the dropdown menu options"
    - "comment_3: Focus indicator is barely visible with only 1px border"

    Analyze this issue and provide your categorization in the specified JSON format.
    """
    def _get_hierarchical_violations(self, base_violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add higher-level violations when lower-level criteria are violated.
        Only includes criteria where the name is the same but level is different.
        """
        
        # Define WCAG hierarchical relationships (lower → higher levels)
        # Only including criteria with same name, different levels
        wcag_hierarchies = {
            # Contrast: Minimum (AA) → Enhanced (AAA)
            "1.4.3": ["1.4.6"],  
            
            # Link Purpose: In Context (A) → Link Only (AAA)  
            "2.4.4": ["2.4.9"],  
            
            # Target Size: Enhanced (AAA) → Minimum (AA) 
            # Note: In WCAG 2.2, "Minimum" is actually less strict than "Enhanced"
            "2.5.5": ["2.5.8"],  
            
            # Authentication: Minimum (AA) → Enhanced (AAA)
            "3.3.8": ["3.3.9"],  
            
            # Images of Text: (AA) → No Exception (AAA)
            "1.4.5": ["1.4.9"],  
        }
        
        # SC number to full name mapping
        sc_details = {
            "1.4.3": "SC 1.4.3 Contrast (Minimum) (AA)",
            "1.4.6": "SC 1.4.6 Contrast (Enhanced) (AAA)",
            "2.4.4": "SC 2.4.4 Link Purpose (In Context) (A)", 
            "2.4.9": "SC 2.4.9 Link Purpose (Link Only) (AAA)",
            "2.5.5": "SC 2.5.5 Target Size (Enhanced) (AAA)",
            "2.5.8": "SC 2.5.8 Target Size (Minimum) (AA)",
            "3.3.8": "SC 3.3.8 Accessible Authentication (Minimum) (AA)",
            "3.3.9": "SC 3.3.9 Accessible Authentication (Enhanced) (AAA)",
            "1.4.5": "SC 1.4.5 Images of Text (AA)",
            "1.4.9": "SC 1.4.9 Images of Text (No Exception) (AAA)"
        }
        
        # Convert base violations to dictionaries if they're Pydantic objects
        all_violations = []
        for violation in base_violations:
            if hasattr(violation, 'model_dump'):  # Pydantic object
                all_violations.append(violation.model_dump())
            else:  # Already a dictionary
                all_violations.append(violation)
        
        existing_sc_numbers = set()
        
        # Extract SC numbers from existing violations
        for violation in all_violations:
            sc_name = violation.get('sc_name', '')
            # Extract number from "SC X.X.X Name (Level)" format
            if sc_name.startswith('SC '):
                sc_number = sc_name.split(' ')[1]  # Get "X.X.X" part
                existing_sc_numbers.add(sc_number)
        
        # Check for hierarchical violations to add
        for violation in all_violations[:]:  # Use slice to avoid modifying list during iteration
            sc_name = violation.get('sc_name', '')
            if sc_name.startswith('SC '):
                sc_number = sc_name.split(' ')[1]  # Get "X.X.X" part
                
                # Check if this violation should trigger higher-level violations
                if sc_number in wcag_hierarchies:
                    higher_scs = wcag_hierarchies[sc_number]
                    
                    for higher_sc in higher_scs:
                        # Only add if not already present
                        if higher_sc not in existing_sc_numbers:
                            higher_violation = {
                                "sc_name": sc_details.get(higher_sc, f"SC {higher_sc}"),
                                "evidence_source": f"hierarchical: {violation.get('evidence_source', '')}",
                                "violation_reason": f"Higher-level violation automatically triggered by {sc_name}. {violation.get('violation_reason', '')}"
                            }
                            all_violations.append(higher_violation)
                            existing_sc_numbers.add(higher_sc)
        
        return all_violations
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            return {"error": "No successful analyses to summarize"}
        
        # Collect all violations from all issues
        all_violations = []
        issues_with_violations = 0
        issues_without_violations = 0
        
        for result in successful_results:
            violations = result.get('violated_sc', [])
            if violations:
                issues_with_violations += 1
                # Convert any Pydantic objects to dictionaries
                for violation in violations:
                    if hasattr(violation, 'model_dump'):  # Pydantic object
                        all_violations.append(violation.model_dump())
                    elif hasattr(violation, 'dict'):  # Older Pydantic versions
                        all_violations.append(violation.dict())
                    else:  # Already a dictionary
                        all_violations.append(violation)
            else:
                issues_without_violations += 1
        
        # Count violations by SC - now all are guaranteed to be dictionaries
        sc_counts = Counter()
        evidence_sources = Counter()
        
        for violation in all_violations:
            sc_name = violation.get('sc_name', 'Unknown')
            evidence_source = violation.get('evidence_source', 'Unknown')
            sc_counts[sc_name] += 1
            evidence_sources[evidence_source] += 1
        
        # Validation statistics
        validation_stats = {
            "total_validations_performed": len([r for r in successful_results if r.get('wcag_validation', {}).get('performed')]),
            "validation_agreements": len([r for r in successful_results if r.get('wcag_validation', {}).get('agreement_status') == 'Agrees']),
            "validation_disagreements": len([r for r in successful_results if r.get('wcag_validation', {}).get('agreement_status') == 'Disagrees']),
            "validation_partial_agreements": len([r for r in successful_results if r.get('wcag_validation', {}).get('agreement_status') == 'Partially_Agrees']),
            "violations_changed_by_validation": len([r for r in successful_results if r.get('original_violations') != r.get('violated_sc')])
        }
        
        return {
            "research_question": self.question_description,
            "total_issues_analyzed": len(results),
            "successful_analyses": len(successful_results),
            "failed_analyses": len(results) - len(successful_results),
            
            "violation_overview": {
                "issues_with_wcag_violations": issues_with_violations,
                "issues_without_wcag_violations": issues_without_violations,
                "total_violations_found": len(all_violations),
                "percentage_issues_with_violations": round(issues_with_violations / len(successful_results) * 100, 1) if successful_results else 0
            },
            
            "most_violated_success_criteria": [
                {
                    "success_criterion": sc,
                    "count": count,
                    "percentage": round(count / len(all_violations) * 100, 1) if all_violations else 0
                }
                for sc, count in sc_counts.most_common(10)
            ],
            
            "evidence_source_distribution": [
                {
                    "source": source,
                    "count": count,
                    "percentage": round(count / len(all_violations) * 100, 1) if all_violations else 0
                }
                for source, count in evidence_sources.most_common()
            ],
            
            "wcag_statistics": {
                "average_violations_per_issue": round(len(all_violations) / len(successful_results), 2) if successful_results else 0,
                "max_violations_in_single_issue": max([len(r.get('violated_sc', [])) for r in successful_results]) if successful_results else 0
            },
            
            "validation_statistics": validation_stats
        }

class TestingVerificationQuestion(BaseResearchQuestion):
    """Research Question 3: Testing & Verification methods analysis"""
    
    @property
    def question_id(self) -> str:
        return "testing_verification"
    
    @property
    def question_description(self) -> str:
        return "What testing and verification methods do Moodle developers use to validate accessibility issues and fixes?"
    
    @property
    def result_model(self) -> Type[BaseModel]:
        return TestingVerificationResult
    
    def create_system_prompt(self) -> str:
        return """You are a software testing expert analyzing testing and verification methods in accessibility issue workflows.

                Your task is to identify how Moodle developers test reported accessibility bugs and verify that fixes address the problems.

                Key areas to analyze:
                - **Automated Accessibility Testing**: Tools like axe-core, Pa11y, WAVE, Lighthouse accessibility audits. Tools like bot or CI/CD tools should be excluded as they are not accessibility testing tools.
                - **Manual Accessibility Testing**: Screen readers (NVDA, JAWS, VoiceOver), keyboard-only navigation, switch navigation, voice control, high contrast testing
                - **Testing vs Verification**: Testing = checking if accessibility bug exists; Verification = confirming accessibility fix works
                - **Assistive Technology (AT)**: Specific screen readers, voice control software, switch navigation, magnification tools

                Guidelines:
                - Analyze issue descriptions, comments, and test instructions for accessibility testing evidence
                - Identify specific accessibility tools, techniques, and methods mentioned
                - Distinguish between testing (accessibility bug validation) and verification (accessibility fix validation)
                - Include evidence sources with original text excerpts
                - Focus on accessibility-specific testing approaches and assistive technology
                - Note both successful and failed accessibility testing attempts
                - Only include tools/methods specifically related to accessibility testing

                Respond only with valid JSON matching the specified schema."""
    
    def create_analysis_prompt(self, issue_data: Dict[str, Any]) -> str:
        # Format comments for analysis
        comments_text = ""
        for i, comment in enumerate(issue_data.get('comments', []), 1):
            # comments_text += f"comment_{i}: {comment['role']} ({comment['author']}): {comment['body'][:600]}{'...' if len(comment['body']) > 600 else ''}\n"
            comments_text += f"comment_{i}: {comment['role']} ({comment['author']}): {comment['body']}\n"
        
        # Format test instructions if available
        test_instructions_text = ""
        if issue_data.get('test_instructions'):
            # test_instructions_text = f"{issue_data.get('test_instructions', '')[:800]}{'...' if len(issue_data.get('test_instructions', '')) > 800 else ''}"
            test_instructions_text = f"{issue_data.get('test_instructions', '')}"
        
        return f"""
            **TESTING & VERIFICATION METHODS ANALYSIS**

            Analyze this Moodle accessibility issue to identify testing and verification methods used by developers.

            **ISSUE DETAILS:**
            - Key: {issue_data.get('issue_key', 'N/A')}
            - Priority: {issue_data.get('priority', 'N/A')}
            - Labels: {', '.join(issue_data.get('labels', []))}

            **TITLE:**
            {issue_data.get('title', 'N/A')}

            **DESCRIPTION:**
            {issue_data.get('description', 'N/A')[:1000]}{'...' if len(issue_data.get('description', '')) > 1000 else ''}

            **TEST INSTRUCTIONS:**
            {test_instructions_text if test_instructions_text else 'No test instructions provided'}

            **COMMENTS:**
            {comments_text}

            **ANALYSIS INSTRUCTIONS:**

            1. **Identify Testing Methods**: Look for evidence of testing to validate reported bugs (primarily in description and comments)
            2. **Identify Verification Methods**: Look for evidence of verifying that fixes work (primarily in test_instructions and comments)
            3. **Categorize Method Types**: Distinguish between automated tools and manual techniques
            4. **Extract Evidence**: Include specific text excerpts that indicate testing/verification

            **FIELD-SPECIFIC ANALYSIS:**
            - **description & comments**: Look for methods used to TEST/validate the original accessibility bug
            - **test_instructions**: Look for methods used to VERIFY/confirm that the patch/fix works
            - **comments**: May contain both testing (early comments) and verification (later comments after patches)

            **EXAMPLE OF METHOD TYPES TO IDENTIFY:**
            
            **IT IS JUST A OUTLINE, DO NOT COPY THE EXAMPLES OR LIMIT YOUR ANALYSIS TO THEM ONLY**

            **Automated Accessibility Testing:**
            - Accessibility scanners: axe-core, axe-webdriver, Pa11y, WAVE, Lighthouse accessibility audit
            - Browser extensions: axe DevTools, WAVE browser extension, Accessibility Insights
            - Automated screen reader testing: Guidepup, @guidepup/virtual-screen-reader
            - Color contrast analyzers: Colour Contrast Analyser (CCA), WebAIM contrast checker

            **Manual Accessibility Testing:**
            - Screen readers: NVDA, JAWS, VoiceOver, ORCA, TalkBack, Dragon NaturallySpeaking
            - Keyboard-only navigation testing
            - Switch navigation and alternative input devices
            - Voice control testing (Dragon, Voice Control)
            - High contrast and zoom testing
            - Mobile accessibility testing with TalkBack/VoiceOver
            - Color blindness simulation testing
            - Focus management and tab order testing

            **PURPOSE CLASSIFICATION:**
            - **testing**: Validating if the reported accessibility bug actually exists (usually found in description/early comments)
            - **verification**: Confirming that a proposed/implemented fix resolves the issue (usually found in test_instructions/later comments)

            **FIELD GUIDANCE:**
            - **test_instructions**: Almost always contains VERIFICATION methods (how to test the patch)
            - **description**: Usually contains TESTING methods (how the bug was discovered/validated)
            - **comments**: Can contain both - early comments often show TESTING, later comments show VERIFICATION

            **EVIDENCE SOURCING WITH ORIGINAL TEXT:**
            - **title**: "title: [excerpt from title]"
            - **description**: "description: [excerpt from description]" (usually TESTING methods)
            - **test_instructions**: "test_instructions: [excerpt from test instructions]" (usually VERIFICATION methods)
            - **comment_X**: "comment_X: [excerpt from comment]" (can be TESTING or VERIFICATION depending on timing)

            **OUTPUT REQUIREMENTS:**
            For each testing/verification method found:
            - method_type: "automated" or "manual"
            - tool_or_technique: Specific tool name or technique used
            - purpose: "testing" or "verification"
            - target_issue: What specific accessibility issue is being tested/verified
            - evidence_source: Source with original text excerpt
            - details: Additional context about the method

            **DEDUPLICATION RULES:**
            - If the same tool/technique is used multiple times for the same purpose and target issue, report only the FIRST instance
            - If the same tool/technique is used for different purposes (testing vs verification) or different target issues, report each instance separately
            - Example: If NVDA is mentioned 3 times for testing screen reader announcements, report only the first mention
            - Example: If NVDA is used for testing announcements AND verifying fixes, report both instances
            
            **EXAMPLES:**
            - method_type: "automated", tool_or_technique: "axe-core", purpose: "verification", target_issue: "keyboard navigation failure", evidence_source: "comment_5: axe tests now pass for keyboard accessibility"
            - method_type: "manual", tool_or_technique: "NVDA", purpose: "testing", target_issue: "screen reader announcements", evidence_source: "description: tested with NVDA and focus is not announced"
            - method_type: "manual", tool_or_technique: "keyboard_navigation", purpose: "verification", target_issue: "tab order", evidence_source: "test_instructions: Press Tab key to navigate through all form elements and verify focus is visible"
            - method_type: "automated", tool_or_technique: "Pa11y", purpose: "testing", target_issue: "color contrast", evidence_source: "comment_2: Pa11y reports contrast ratio below 4.5:1"

            **IMPORTANT:**
            - Only report accessibility-related testing methods you can clearly justify from the evidence
            - Include actual text excerpts in evidence_source
            - If no accessibility testing/verification methods are mentioned, return empty testing_methods list
            - Be specific about accessibility tools and techniques mentioned
            - Distinguish between different screen readers, accessibility scanners, etc.
            - Focus on accessibility testing, not general software testing
            - Remove redundancy: If the same tool/technique is used multiple times for the same purpose and target issue, report only the FIRST instance

            Analyze this issue and provide your findings in the specified JSON format.
            """
    
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            return {"error": "No successful analyses to summarize"}
        
        # Collect all testing methods
        all_methods = []
        issues_with_testing = 0
        issues_without_testing = 0
        
        for result in successful_results:
            methods = result.get('testing_methods', [])
            if methods:
                issues_with_testing += 1
                # Convert Pydantic objects to dictionaries
                for method in methods:
                    if hasattr(method, 'model_dump'):
                        all_methods.append(method.model_dump())
                    elif hasattr(method, 'dict'):
                        all_methods.append(method.dict())
                    else:
                        all_methods.append(method)
            else:
                issues_without_testing += 1
        
        # Analyze method distributions
        method_types = Counter()
        tools_techniques = Counter()
        purposes = Counter()
        evidence_sources = Counter()
        
        automated_tools = Counter()
        manual_techniques = Counter()
        
        for method in all_methods:
            method_type = method.get('method_type', 'Unknown')
            tool_technique = method.get('tool_or_technique', 'Unknown')
            purpose = method.get('purpose', 'Unknown')
            evidence_source = method.get('evidence_source', 'Unknown')
            
            method_types[method_type] += 1
            tools_techniques[tool_technique] += 1
            purposes[purpose] += 1
            evidence_sources[evidence_source] += 1
            
            if method_type == 'automated':
                automated_tools[tool_technique] += 1
            elif method_type == 'manual':
                manual_techniques[tool_technique] += 1
        
        # Validation statistics
        validation_stats = {
            "total_validations_performed": len([r for r in successful_results if r.get('testing_validation', {}).get('performed')]),
            "validation_agreements": len([r for r in successful_results if r.get('testing_validation', {}).get('agreement_status') == 'Agrees']),
            "validation_disagreements": len([r for r in successful_results if r.get('testing_validation', {}).get('agreement_status') == 'Disagrees']),
            "validation_partial_agreements": len([r for r in successful_results if r.get('testing_validation', {}).get('agreement_status') == 'Partially_Agrees']),
            "methods_changed_by_validation": len([r for r in successful_results if r.get('original_methods') != r.get('testing_methods')])
        }
        
        return {
            "research_question": self.question_description,
            "total_issues_analyzed": len(results),
            "successful_analyses": len(successful_results),
            "failed_analyses": len(results) - len(successful_results),
            
            "testing_overview": {
                "issues_with_testing_methods": issues_with_testing,
                "issues_without_testing_methods": issues_without_testing,
                "total_methods_found": len(all_methods),
                "percentage_issues_with_testing": round(issues_with_testing / len(successful_results) * 100, 1) if successful_results else 0
            },
            
            "method_type_distribution": [
                {
                    "type": method_type,
                    "count": count,
                    "percentage": round(count / len(all_methods) * 100, 1) if all_methods else 0
                }
                for method_type, count in method_types.most_common()
            ],
            
            "most_used_tools_techniques": [
                {
                    "tool_technique": tool,
                    "count": count,
                    "percentage": round(count / len(all_methods) * 100, 1) if all_methods else 0
                }
                for tool, count in tools_techniques.most_common(15)
            ],
            
            "purpose_distribution": [
                {
                    "purpose": purpose,
                    "count": count,
                    "percentage": round(count / len(all_methods) * 100, 1) if all_methods else 0
                }
                for purpose, count in purposes.most_common()
            ],
            
            "automated_tools": [
                {
                    "tool": tool,
                    "count": count,
                    "percentage": round(count / len([m for m in all_methods if m.get('method_type') == 'automated']) * 100, 1) if [m for m in all_methods if m.get('method_type') == 'automated'] else 0
                }
                for tool, count in automated_tools.most_common(10)
            ],
            
            "manual_techniques": [
                {
                    "technique": technique,
                    "count": count,
                    "percentage": round(count / len([m for m in all_methods if m.get('method_type') == 'manual']) * 100, 1) if [m for m in all_methods if m.get('method_type') == 'manual'] else 0
                }
                for technique, count in manual_techniques.most_common(10)
            ],
            
            "evidence_source_distribution": [
                {
                    "source": source,
                    "count": count,
                    "percentage": round(count / len(all_methods) * 100, 1) if all_methods else 0
                }
                for source, count in evidence_sources.most_common()
            ],
            
            "testing_statistics": {
                "average_methods_per_issue": round(len(all_methods) / len(successful_results), 2) if successful_results else 0,
                "max_methods_in_single_issue": max([len(r.get('testing_methods', [])) for r in successful_results]) if successful_results else 0,
                "automated_vs_manual_ratio": round(method_types.get('automated', 0) / max(method_types.get('manual', 1), 1), 2),
                "testing_vs_verification_ratio": round(purposes.get('testing', 0) / max(purposes.get('verification', 1), 1), 2)
            },
            
            "validation_statistics": validation_stats
        }

class MoodleIssueAnalyzer:
    """Main analyzer class that orchestrates different research questions"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        
        # Get API key from config or environment
        api_key = config.openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file or pass it in config.")
        
        self.llm = ChatOpenAI(
            model=config.model_name,  # Updated parameter name from model_name to model
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=api_key  # Updated parameter name from openai_api_key to api_key
        )
        
        # NEW: Create separate LLM instance for validation if different model
        if config.enable_validation:
            self.validation_llm = ChatOpenAI(
                model=config.validation_model_name,
                temperature=0,  # Use deterministic temperature for validation
                max_tokens=config.max_tokens,
                api_key=api_key
            )
        
        self.research_questions: Dict[str, BaseResearchQuestion] = {}
        
        # Track existing results for continue mode
        self.existing_results: Dict[str, Dict[str, Any]] = {}
        self.output_dir = Path("Data/analysis_results")
    
    def _ensure_dict_format(self, data):
        """Convert Pydantic objects to dictionaries recursively"""
        if hasattr(data, 'model_dump'):  # Pydantic v2
            # Force conversion of nested objects
            return data.model_dump(mode='python', by_alias=False)
        elif hasattr(data, 'dict'):  # Pydantic v1
            # Force conversion of nested objects
            return data.dict(by_alias=False)
        elif isinstance(data, list):
            return [self._ensure_dict_format(item) for item in data]
        elif isinstance(data, dict):
            return {key: self._ensure_dict_format(value) for key, value in data.items()}
        else:
            return data
    
    def _create_wcag_validation_system_prompt(self) -> str:
        """Create system prompt for WCAG validation"""
        return """You are a senior accessibility consultant performing quality assurance on WCAG 2.2 Success Criteria analysis.

    Your task is to review and validate another expert's categorization of WCAG 2.2 SC violations in Moodle accessibility issues.

    Key responsibilities:
    - Verify the accuracy of WCAG 2.2 SC identification
    - Check if violations are correctly mapped to the appropriate SCs
    - Validate evidence sources and violation reasons
    - Ensure no duplicate violations are reported
    - Confirm conformance levels and categories are correct
    - Assess if the confidence level is appropriate

    For reference, WCAG 2.2 changes:
    NEW: 2.4.11, 2.4.12, 2.4.13, 2.5.7, 2.5.8, 3.2.6, 3.3.7, 3.3.8, 3.3.9

    **IMPORTANT: Understanding Hierarchical WCAG Violations**
    The analysis system automatically adds higher-level Success Criteria when lower-level ones are violated. This is CORRECT behavior for these specific pairs:

    - SC 1.4.3 Contrast (Minimum) (AA) → automatically adds SC 1.4.6 Contrast (Enhanced) (AAA)
    - SC 2.4.4 Link Purpose (In Context) (A) → automatically adds SC 2.4.9 Link Purpose (Link Only) (AAA)  
    - SC 2.5.5 Target Size (Enhanced) (AAA) → automatically adds SC 2.5.8 Target Size (Minimum) (AA)
    - SC 3.3.8 Accessible Authentication (Minimum) (AA) → automatically adds SC 3.3.9 Accessible Authentication (Enhanced) (AAA)
    - SC 1.4.5 Images of Text (AA) → automatically adds SC 1.4.9 Images of Text (No Exception) (AAA)

    **When you see evidence_source starting with "hierarchical:"** - this indicates an automatically added higher-level violation. These are VALID and should generally be kept unless the original lower-level violation is incorrect.

    For reference, WCAG 2.2 changes:
    NEW: 2.4.11, 2.4.12, 2.4.13, 2.5.7, 2.5.8, 3.2.6, 3.3.7, 3.3.8, 3.3.9

    Be thorough but fair. Only suggest changes if there are clear errors in WCAG 2.2 SC mapping.
    Respond only with valid JSON matching the specified schema."""
    
    def needs_wcag_validation(self, analysis_result: Dict[str, Any]) -> bool:
        """Check if WCAG result needs validation - simplified check"""
        if not self.config.enable_validation:
            return False
            
        if 'error' in analysis_result:
            return False  # Don't validate error results
        
        violated_sc = analysis_result.get('violated_sc', [])
        
        if self.config.validate_only_non_default:
            # Validate when violations are found
            return len(violated_sc) > 0
        
        return True  # Validate all if not restricted
    
    def _create_wcag_validation_prompt(self, original_result: Dict[str, Any], issue_data: Dict[str, Any]) -> str:
        """Create simplified validation prompt for WCAG analysis"""
        # Format the original analysis for review
        original_violations = original_result.get('violated_sc', [])
        
        violations_summary = ""
        if original_violations:
            for i, violation in enumerate(original_violations, 1):
                violations_summary += f"  {i}. {violation.get('sc_name', '')}\n"
                violations_summary += f"     Source: {violation.get('evidence_source', '')}\n"
                violations_summary += f"     Reason: {violation.get('violation_reason', '')}\n\n"
        else:
            violations_summary = "  No WCAG 2.2 SC violations identified\n"
        
        # Reuse issue formatting with clear source labels
        comments_text = ""
        for i, comment in enumerate(issue_data.get('comments', []), 1):
            comments_text += f"comment_{i}: {comment['role']} ({comment['author']}): {comment['body'][:300]}{'...' if len(comment['body']) > 300 else ''}\n"
        
        return f"""
**WCAG 2.2 SC VALIDATION TASK**: Review the analysis below for accuracy.

**ORIGINAL ANALYSIS TO VALIDATE:**
- Issue Key: {issue_data.get('issue_key', 'N/A')}

**Identified Violations:**
{violations_summary}

**ISSUE DATA FOR VERIFICATION:**
**title:** {issue_data.get('title', 'N/A')}

**description:**
{issue_data.get('description', 'N/A')[:800]}{'...' if len(issue_data.get('description', '')) > 800 else ''}

**Comments:**
{comments_text}

**VALIDATION CHECKLIST:**
1. **SC Accuracy**: Are the identified WCAG 2.2 SCs correctly applied?
2. **Evidence Verification**: Do the evidence sources support the violations?
3. **Violation Reasons**: Are the violation explanations accurate and specific?
4. **No Duplicates**: Are there any duplicate violations that should be removed?
5. **Completeness**: Are there any missed WCAG 2.2 SC violations?

**VALIDATION DECISION CRITERIA:**
- **Agrees**: Original analysis is accurate and complete
- **Partially_Agrees**: Minor issues or improvements needed
- **Disagrees**: Major errors in SC identification or missing violations

**OUTPUT REQUIREMENTS:**
- original_violations: Copy the original violations list exactly
- validated_violations: Your final recommended violations list
- validation_changes: List specific changes made (or ["None"] if no changes)
- validation_explanation: Brief explanation of your validation decision
- agreement_status: Agrees/Partially_Agrees/Disagrees with original analysis

Provide your validation analysis in the specified JSON format.
"""

    def validate_wcag_categorization(self, original_result: Dict[str, Any], issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a WCAG categorization analysis with a second LLM"""
        try:
            validation_prompt = self._create_wcag_validation_prompt(original_result, issue_data)
            
            parser = PydanticOutputParser(pydantic_object=WCAGValidationResult)
            system_message = SystemMessage(content=self._create_wcag_validation_system_prompt())
            human_message = HumanMessage(content=f"{validation_prompt}\n\n{parser.get_format_instructions()}")
            
            with get_openai_callback() as cb:
                start_time = time.time()
                response = self.validation_llm.invoke([system_message, human_message])
                processing_time = time.time() - start_time
            
            validation_result = parser.parse(response.content)
            
            # Merge validation with original result
            return self._merge_wcag_validation_result(original_result, validation_result, cb, processing_time)
            
        except Exception as e:
            logger.error(f"WCAG validation error for issue {issue_data.get('issue_key', 'unknown')}: {e}")
            # Return original result with validation error noted
            original_result['wcag_validation_error'] = str(e)
            return original_result

    def _merge_wcag_validation_result(self, original_result: Dict[str, Any], 
                                validation_result: WCAGValidationResult,
                                callback_info, processing_time: float) -> Dict[str, Any]:
        """Merge WCAG validation results with original analysis - simplified"""
        merged_result = original_result.copy()
        
        # Add simplified validation metadata
        merged_result['wcag_validation'] = {
            'performed': True,
            'agreement_status': validation_result.agreement_status,
            'validation_changes': validation_result.validation_changes,
            'validation_explanation': validation_result.validation_explanation
        }
        
        # Update violations if validation suggests changes
        if validation_result.agreement_status in ['Disagrees', 'Partially_Agrees']:
            merged_result['original_violations'] = original_result.get('violated_sc', [])
            # Convert Pydantic objects to dictionaries
            validated_violations = []
            for violation in validation_result.validated_violations:
                if hasattr(violation, 'model_dump'):
                    validated_violations.append(violation.model_dump())
                else:
                    validated_violations.append(violation)
            merged_result['violated_sc'] = validated_violations
        
        return merged_result
        
    def register_research_question(self, question: BaseResearchQuestion):
        """Register a new research question for analysis"""
        self.research_questions[question.question_id] = question
        logger.info(f"Registered research question: {question.question_id} - {question.question_description}")
    
    def _load_existing_results(self, question_id: str) -> Set[str]:
        """Load existing results and return set of analyzed issue keys"""
        results_file = self.output_dir / f"{question_id}_results.json"
        analyzed_keys = set()
        
        if not results_file.exists():
            logger.info(f"No existing results file found for {question_id}")
            return analyzed_keys
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            # Extract issue keys from existing results
            existing_results = existing_data.get('results', [])
            for result in existing_results:
                if 'issue_key' in result:
                    analyzed_keys.add(result['issue_key'])
            
            # Store existing results for merging later
            self.existing_results[question_id] = existing_data
            
            logger.info(f"Loaded {len(analyzed_keys)} existing results for {question_id}")
            return analyzed_keys
            
        except Exception as e:
            logger.error(f"Error loading existing results for {question_id}: {e}")
            return analyzed_keys
    
    def _filter_unanalyzed_issues(self, issues: List[Dict[str, Any]], question_id: str) -> List[Dict[str, Any]]:
        """Filter out already analyzed issues if in continue mode"""
        if not self.config.continue_analysis:
            return issues
        
        analyzed_keys = self._load_existing_results(question_id)
        
        if not analyzed_keys:
            logger.info(f"No existing analysis found for {question_id}, analyzing all issues")
            return issues
        
        # Filter out already analyzed issues
        unanalyzed_issues = [
            issue for issue in issues 
            if issue.get('issue_key') not in analyzed_keys
        ]
        
        skipped_count = len(issues) - len(unanalyzed_issues)
        logger.info(f"Continue mode: Skipping {skipped_count} already analyzed issues")
        logger.info(f"Remaining to analyze: {len(unanalyzed_issues)} issues")
        
        return unanalyzed_issues
    
    def _merge_with_existing_results(self, new_results: List[Dict[str, Any]], question_id: str) -> List[Dict[str, Any]]:
        """Merge new results with existing results if in continue mode"""
        if not self.config.continue_analysis or question_id not in self.existing_results:
            return new_results
        
        existing_data = self.existing_results[question_id]
        existing_results = existing_data.get('results', [])
        
        # Combine existing and new results
        merged_results = existing_results + new_results
        
        logger.info(f"Merged results: {len(existing_results)} existing + {len(new_results)} new = {len(merged_results)} total")
        
        return merged_results
    
    # NEW: Validation methods
    def needs_validation(self, analysis_result: Dict[str, Any]) -> bool:
        """Check if result needs validation based on configuration"""
        if not self.config.enable_validation:
            return False
            
        if 'error' in analysis_result:
            return False  # Don't validate error results
            
        default_sequence = ["reporter", "assignee", "peer_reviewer", "integrator", "tester"]
        actual_sequence = analysis_result.get('complete_role_sequence', [])
        
        if self.config.validate_only_non_default:
            return actual_sequence != default_sequence
        
        return True  # Validate all if not restricted to non-default
    
    def needs_testing_validation(self, analysis_result: Dict[str, Any]) -> bool:
        """Check if testing result needs validation"""
        if not self.config.enable_validation:
            return False
            
        if 'error' in analysis_result:
            return False  # Don't validate error results
        
        testing_methods = analysis_result.get('testing_methods', [])
        
        if self.config.validate_only_non_default:
            # Validate when testing methods are found
            return len(testing_methods) > 0
        
        return True  # Validate all if not restricted
    
# Modified validation methods in MoodleIssueAnalyzer class
    def _create_validation_system_prompt(self) -> str:
        """Create system prompt for validation"""
        return """You are a senior software engineering researcher performing quality assurance on role sequence analysis.

    Your task is to review and validate another AI's analysis of role sequences in Moodle accessibility issues.

    Key responsibilities:
    - Verify the chronological accuracy of the role sequence
    - Check if all meaningful role transitions are captured
    - Identify any missed back-and-forth patterns
    - Ensure proper exclusion of bot roles and automated systems (CiBoT, noreply)
    - Validate that participant influences are correctly handled (noted but not included in sequence)
    - Verify that back-and-forth codes accurately represent the problems found in the evidence
    - Assess if codes are descriptive, specific, and well-supported by evidence
    - Check that codes are appropriate for the type of problem described
    - Ensure evidence sources are legitimate (not from bots)

    Focus on code quality: Are the codes specific and descriptive? Do they accurately capture what the problem actually is? Is the supporting evidence sufficient and from legitimate sources?

    Be critical but fair. Only suggest changes if there are clear errors or omissions.
    Respond only with valid JSON matching the specified schema."""
    
    def _create_validation_prompt(self, original_result: Dict[str, Any], issue_data: Dict[str, Any]) -> str:
        """Create validation prompt for a specific analysis"""
        # Format the original analysis for review
        original_sequence = original_result.get('complete_role_sequence', [])
        original_codes = original_result.get('back_and_forth_codes', [])
        
        # Reuse the same issue formatting from the original analysis
        comments_text = ""
        for i, comment in enumerate(issue_data.get('comments', []), 1):
            comments_text += f"{i}. {comment['role']} ({comment['author']}): {comment['body'][:300]}{'...' if len(comment['body']) > 300 else ''}\n"
        
        timeline_text = ""
        for i, event in enumerate(issue_data.get('timeline_events', []), 1):
            timeline_text += f"{i}. {event.get('timestamp', 'N/A')} - {event.get('author', 'N/A')}: {event.get('field', 'N/A')} changed from '{event.get('from_value', '')}' to '{event.get('to_value', '')}'\n"
        
        return f"""
    VALIDATION TASK: Review the role sequence analysis below for accuracy and completeness.

    **ORIGINAL ANALYSIS TO VALIDATE:**
    - Issue Key: {issue_data.get('issue_key', 'N/A')}
    - Analyzed Sequence: {' → '.join(original_sequence)}
    - Back-and-Forth Codes: {original_codes}
    - Sequence Explanation: {original_result.get('sequence_explanation', '')}

    **RAW DATA FOR VERIFICATION:**
    **CHRONOLOGICAL COMMENTS:**
    {comments_text}

    **TIMELINE EVENTS:**
    {timeline_text}

    **VALIDATION INSTRUCTIONS:**
    1. **Verify Chronological Accuracy**: Check if the sequence matches the actual chronological flow
    2. **Check for Missing Transitions**: Are there any role transitions the original analysis missed?
    3. **Validate Back-and-Forth Patterns**: Are returns to previous roles correctly identified?
    4. **Validate Code Quality**: Are the back-and-forth codes descriptive, specific, and well-supported by evidence?
    5. **Check Code Consistency**: Are similar problems coded consistently?
    6. **Confirm Bot Exclusion**: Ensure no bot roles are included in the sequence
    7. **Review Participant Handling**: Verify participants aren't in sequence but their influence is coded
    8. **Assess Evidence Quality**: Is the source text sufficient to support each code assignment?

    **AVAILABLE BACK-AND-FORTH CODES:**
    The original analysis should create descriptive codes based on the actual problems found. Codes should be:
    - **Specific and descriptive** (e.g., "SCREEN_READER_FOCUS" not "SR_ISSUE")
    - **Consistent in naming** (similar problems should get similar codes)
    - **Well-supported by evidence** (source text should clearly support the code)
    - **Appropriately detailed** (not too broad, not too narrow)

    **Code Quality Assessment:**
    - Do the codes accurately represent the problems described in the evidence?
    - Are the codes specific enough to be meaningful for analysis?
    - Is the evidence sufficient to support each code assignment?
    - Are similar problems coded consistently?
    - Are accessibility issues distinguished from general development issues?

    **DEFAULT REFERENCE SEQUENCE:** reporter → assignee → peer_reviewer → integrator → tester (codes = [])

    **VALIDATION CRITERIA:**
    - If original analysis is correct: Set agreement_status="Agrees" and keep original sequence, explanation, and codes
    - If minor issues found: Set agreement_status="Partially_Agrees" and suggest improvements  
    - If major errors found: Set agreement_status="Disagrees" and provide corrected sequence, explanation, and codes

    **OUTPUT REQUIREMENTS:**
    - original_sequence: Copy the original sequence exactly
    - validated_sequence: Your final recommended sequence (may be same as original)
    - original_codes: Copy the original codes exactly
    - validated_codes: Your final recommended codes (may be same as original)
    - original_explanation: Copy the original explanation exactly
    - validated_explanation: Your final recommended explanation (may be same as original)
    - validation_confidence: High/Medium/Low based on data clarity
    - sequence_changes: List specific changes made (or ["None"] if no changes)
    - validation_explanation: Brief explanation of your validation decision
    - agreement_status: Agrees/Partially_Agrees/Disagrees with original analysis
    """
    
    def validate_role_sequence(self, original_result: Dict[str, Any], issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a role sequence analysis with a second LLM"""
        try:
            validation_prompt = self._create_validation_prompt(original_result, issue_data)
            
            parser = PydanticOutputParser(pydantic_object=ValidationResult)
            system_message = SystemMessage(content=self._create_validation_system_prompt())
            human_message = HumanMessage(content=f"{validation_prompt}\n\n{parser.get_format_instructions()}")
            
            with get_openai_callback() as cb:
                start_time = time.time()
                response = self.validation_llm.invoke([system_message, human_message])
                processing_time = time.time() - start_time
            
            validation_result = parser.parse(response.content)
            
            # Merge validation with original result
            return self._merge_validation_result(original_result, validation_result, cb, processing_time)
            
        except Exception as e:
            logger.error(f"Validation error for issue {issue_data.get('issue_key', 'unknown')}: {e}")
            # Return original result with validation error noted
            original_result['validation_error'] = str(e)
            return original_result
    
    def _merge_validation_result(self, original_result: Dict[str, Any], 
                            validation_result: ValidationResult,
                            callback_info, processing_time: float) -> Dict[str, Any]:
        """Merge validation results with original analysis"""
        merged_result = original_result.copy()
        
        # Add validation metadata
        merged_result['validation'] = {
            'performed': True,
            'validation_model': self.config.validation_model_name,
            'agreement_status': validation_result.agreement_status,
            'validation_confidence': validation_result.validation_confidence,
            'sequence_changes': validation_result.sequence_changes,
            'validation_explanation': validation_result.validation_explanation,
            'processing_time_seconds': round(processing_time, 2),
            'tokens_used': callback_info.total_tokens,
            'cost_estimate': callback_info.total_cost
        }
        
        # Update sequence and codes if validation suggests changes
        if validation_result.agreement_status in ['Disagrees', 'Partially_Agrees']:
            merged_result['original_sequence'] = original_result.get('complete_role_sequence', [])
            merged_result['original_codes'] = original_result.get('back_and_forth_codes', [])
            merged_result['original_explanation'] = original_result.get('sequence_explanation', '')
            merged_result['complete_role_sequence'] = validation_result.validated_sequence
            # Convert Pydantic objects to dictionaries here
            merged_result['back_and_forth_codes'] = self._ensure_dict_format(validation_result.validated_codes)
            merged_result['sequence_explanation'] = validation_result.validated_explanation
        
        return merged_result
    
    def analyze_single_issue(self, issue_data: Dict[str, Any], question_id: str) -> Dict[str, Any]:
        """Enhanced analyze_single_issue method with hierarchical WCAG violations"""
        if question_id not in self.research_questions:
            raise ValueError(f"Research question '{question_id}' not registered")
        
        question = self.research_questions[question_id]
        
        try:
            # Create parser for structured output
            parser = PydanticOutputParser(pydantic_object=question.result_model)
            
            # Create messages
            system_message = SystemMessage(content=question.create_system_prompt())
            analysis_prompt = question.create_analysis_prompt(issue_data)
            human_message = HumanMessage(content=f"{analysis_prompt}\n\n{parser.get_format_instructions()}")
            
            # Track token usage and cost
            with get_openai_callback() as cb:
                start_time = time.time()
                response = self.llm.invoke([system_message, human_message])
                processing_time = time.time() - start_time
            
            # Parse the response and convert to dictionary format immediately
            result = parser.parse(response.content)
            analysis_result = self._ensure_dict_format(result)
            
            # Apply hierarchical violations for WCAG categorization (a11y only)
            if question_id == "wcag_categorization" and analysis_result.get('violated_sc'):
                original_violations = analysis_result['violated_sc']
                hierarchical_violations = question._get_hierarchical_violations(original_violations)
                analysis_result['violated_sc'] = hierarchical_violations
                
                if len(hierarchical_violations) > len(original_violations):
                    added_count = len(hierarchical_violations) - len(original_violations)
                    logger.debug(f"Added {added_count} hierarchical violations for {issue_data.get('issue_key', 'unknown')}")
            
            # Add validation step based on question type and dataset
            if question_id == "role_sequence" and self.needs_validation(analysis_result):
                logger.debug(f"Validating role sequence for {issue_data.get('issue_key', 'unknown')}")
                analysis_result = self.validate_role_sequence(analysis_result, issue_data)
                time.sleep(self.config.validation_delay_seconds)
                
            elif question_id == "non_a11y_role_sequence" and self.needs_validation(analysis_result):
                logger.debug(f"Validating non-a11y role sequence for {issue_data.get('issue_key', 'unknown')}")
                analysis_result = self.validate_non_a11y_role_sequence(analysis_result, issue_data)
                time.sleep(self.config.validation_delay_seconds)
                
            elif question_id == "wcag_categorization" and self.needs_wcag_validation(analysis_result):
                logger.debug(f"Validating WCAG categorization for {issue_data.get('issue_key', 'unknown')}")
                analysis_result = self.validate_wcag_categorization(analysis_result, issue_data)
                time.sleep(self.config.validation_delay_seconds)
                
            elif question_id == "testing_verification" and self.needs_testing_validation(analysis_result):
                logger.debug(f"Validating testing methods for {issue_data.get('issue_key', 'unknown')}")
                analysis_result = self.validate_testing_verification(analysis_result, issue_data)
                time.sleep(self.config.validation_delay_seconds)
                
            elif question_id == "non_a11y_testing_verification" and self.needs_testing_validation(analysis_result):
                logger.debug(f"Validating non-a11y testing methods for {issue_data.get('issue_key', 'unknown')}")
                analysis_result = self.validate_non_a11y_testing_verification(analysis_result, issue_data)
                time.sleep(self.config.validation_delay_seconds)
            elif question_id == "participant_influence" and self.needs_participant_validation(analysis_result):
                logger.debug(f"Validating participant influence for {issue_data.get('issue_key', 'unknown')}")
                analysis_result = self.validate_participant_influence(analysis_result, issue_data)
                time.sleep(self.config.validation_delay_seconds)
            elif question_id == "non_a11y_participant_influence" and self.needs_participant_validation(analysis_result):
                logger.debug(f"Validating non-a11y participant influence for {issue_data.get('issue_key', 'unknown')}")
                analysis_result = self.validate_non_a11y_participant_influence(analysis_result, issue_data)
                time.sleep(self.config.validation_delay_seconds)
            elif question_id == "solution_development" and self.needs_solution_validation(analysis_result):
                logger.debug(f"Validating solution development for {issue_data.get('issue_key', 'unknown')}")
                analysis_result = self.validate_solution_development(analysis_result, issue_data)
                time.sleep(self.config.validation_delay_seconds)

            elif question_id == "non_a11y_solution_development" and self.needs_solution_validation(analysis_result):
                logger.debug(f"Validating non-a11y solution development for {issue_data.get('issue_key', 'unknown')}")
                analysis_result = self.validate_non_a11y_solution_development(analysis_result, issue_data)
                time.sleep(self.config.validation_delay_seconds)
            return analysis_result
            
        except Exception as e:
            logger.error(f"Analysis error for issue {issue_data.get('issue_key', 'unknown')}: {e}")
            return self._create_error_result(issue_data, question_id, str(e))
    
    def _create_error_result(self, issue_data: Dict[str, Any], question_id: str, error_message: str) -> Dict[str, Any]:
        """Create simplified error result structure"""
        return {
            "issue_key": issue_data.get('issue_key', 'unknown'),
            "error": error_message
        }
    
    def analyze_batch(self, issues: List[Dict[str, Any]], question_id: str) -> List[Dict[str, Any]]:
        """Analyze multiple issues for a specific research question"""
        if question_id not in self.research_questions:
            raise ValueError(f"Research question '{question_id}' not registered")
        
        # Filter out already analyzed issues if in continue mode
        issues_to_analyze = self._filter_unanalyzed_issues(issues, question_id)
        
        if self.config.max_issues:
            issues_to_analyze = issues_to_analyze[:self.config.max_issues]
        
        # If no issues to analyze (all already done), return empty list
        if not issues_to_analyze:
            logger.info(f"No new issues to analyze for {question_id}")
            return []
        
        results = []
        total_issues = len(issues_to_analyze)
        total_cost = 0.0
        validation_count = 0
        
        logger.info(f"Starting batch analysis:")
        logger.info(f"  Research Question: {self.research_questions[question_id].question_description}")
        logger.info(f"  Issues to analyze: {total_issues}")
        logger.info(f"  Model: {self.config.model_name}")
        if self.config.enable_validation:
            logger.info(f"  Validation: Enabled ({self.config.validation_model_name})")
            logger.info(f"  Validation scope: {'Non-default sequences only' if self.config.validate_only_non_default else 'All sequences'}")
        else:
            logger.info(f"  Validation: Disabled")
        logger.info(f"  Rate limiting: {self.config.delay_seconds} seconds between requests")
        if self.config.continue_analysis:
            logger.info(f"  Continue mode: Appending to existing results")
        
        for i, issue in enumerate(issues_to_analyze):
            logger.debug(f"Analyzing issue {i+1}/{total_issues}: {issue.get('issue_key', 'unknown')}")
            
            result = self.analyze_single_issue(issue, question_id)
            results.append(result)
            
            # Track costs and validation count
            if 'metadata' in result and result['metadata'].get('cost_estimate'):
                total_cost += result['metadata']['cost_estimate']
            
            if result.get('validation', {}).get('performed'):
                validation_count += 1
                # Add validation cost
                if result.get('validation', {}).get('cost_estimate'):
                    total_cost += result['validation']['cost_estimate']
            
            # Rate limiting
            if i < total_issues - 1:
                time.sleep(self.config.delay_seconds)
            
            # Progress update
            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i+1}/{total_issues} issues, {validation_count} validations (Est. cost: ${total_cost:.4f})")
        
        logger.info(f"Batch analysis complete!")
        logger.info(f"  Processed: {len(results)} new issues")
        logger.info(f"  Validations performed: {validation_count}")
        logger.info(f"  Estimated cost for new analyses: ${total_cost:.4f}")
        
        # Merge with existing results if in continue mode
        merged_results = self._merge_with_existing_results(results, question_id)
        
        return merged_results
    
    def analyze_all_questions(self, issues: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze issues for all registered research questions"""
        all_results = {}
        
        for question_id in self.research_questions:
            logger.info(f"{'='*60}")
            logger.info(f"ANALYZING: {question_id}")
            logger.info(f"{'='*60}")
            
            results = self.analyze_batch(issues, question_id)
            all_results[question_id] = results
        
        return all_results
    
    def generate_summaries(self, all_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """Generate summary reports for all research questions"""
        summaries = {}
        
        for question_id, results in all_results.items():
            question = self.research_questions[question_id]
            summaries[question_id] = question.generate_summary(results)
        
        return summaries
    
    def save_results(self, all_results: Dict[str, List[Dict[str, Any]]], 
                    summaries: Dict[str, Dict[str, Any]], output_dir: str):
        """Save all analysis results and summaries"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed results for each research question
        for question_id, results in all_results.items():
            results_file = output_path / f"{question_id}_results.json"
            
            # Calculate metadata for all results (including existing ones in continue mode)
            successful_analyses = len([r for r in results if 'error' not in r])
            failed_analyses = len([r for r in results if 'error' in r])
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "research_question": self.research_questions[question_id].question_description,
                    "analysis_metadata": {
                        "total_issues": len(results),
                        "successful_analyses": successful_analyses,
                        "failed_analyses": failed_analyses,
                        "model_used": self.config.model_name,
                        "validation_enabled": self.config.enable_validation,
                        "validation_model_used": self.config.validation_model_name if self.config.enable_validation else None,
                        "analysis_date": datetime.now().isoformat(),
                        "continue_mode": self.config.continue_analysis
                    },
                    "results": results
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved detailed results: {results_file}")
        
        # Save summaries
        summary_file = output_path / "analysis_summaries.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "analysis_overview": {
                    "research_questions_analyzed": list(summaries.keys()),
                    "analysis_date": datetime.now().isoformat(),
                    "model_used": self.config.model_name,
                    "validation_enabled": self.config.enable_validation,
                    "validation_model_used": self.config.validation_model_name if self.config.enable_validation else None,
                    "continue_mode": self.config.continue_analysis
                },
                "summaries": summaries
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved summaries: {summary_file}")
        
    def _create_testing_validation_system_prompt(self) -> str:
        """Create system prompt for testing & verification validation"""
        return """You are a senior software testing expert performing quality assurance on testing and verification method analysis.

                Your task is to review and validate another expert's identification of testing and verification methods in Moodle accessibility issues.

                Key responsibilities:
                - Verify accurate identification of automated accessibility testing tools
                - Check correct classification of manual accessibility testing techniques
                - Validate distinction between testing (accessibility bug validation) and verification (accessibility fix confirmation)
                - Ensure evidence sources support the identified methods
                - Confirm tool names and technique descriptions are accurate
                - Assess completeness of method identification
                - Check for proper deduplication (same tool/technique for same purpose should appear only once)

                **Common Automated Accessibility Tools to Recognize:**
                - Accessibility scanners: axe-core, axe-webdriver, Pa11y, WAVE, Lighthouse accessibility
                - Browser extensions: axe DevTools, WAVE extension, Accessibility Insights
                - CI/CD accessibility: axe-playwright, axe-puppeteer, pa11y-ci, axe-selenium
                - Automated AT testing: Guidepup, virtual screen reader tools
                - Contrast analyzers: Colour Contrast Analyser, WebAIM contrast checker

                **Common Manual Accessibility Techniques to Recognize:**
                - Screen readers: NVDA, JAWS, VoiceOver, ORCA, TalkBack, Dragon
                - Navigation methods: keyboard-only, switch navigation, voice control
                - Accessibility testing: focus management, tab order, ARIA testing
                - Visual testing: high contrast, zoom, color blindness simulation
                - Mobile AT: TalkBack, VoiceOver mobile testing

                Be thorough but fair. Only suggest changes if there are clear errors in method identification or classification.
                Respond only with valid JSON matching the specified schema."""

    def _create_testing_validation_prompt(self, original_result: Dict[str, Any], issue_data: Dict[str, Any]) -> str:
        """Create validation prompt for testing & verification analysis"""
        # Format the original analysis for review
        original_methods = original_result.get('testing_methods', [])
        
        methods_summary = ""
        if original_methods:
            for i, method in enumerate(original_methods, 1):
                methods_summary += f"  {i}. {method.get('method_type', '')} - {method.get('tool_or_technique', '')}\n"
                methods_summary += f"     Purpose: {method.get('purpose', '')}\n"
                methods_summary += f"     Target: {method.get('target_issue', '')}\n"
                methods_summary += f"     Source: {method.get('evidence_source', '')}\n"
                methods_summary += f"     Details: {method.get('details', '')}\n\n"
        else:
            methods_summary = "  No testing/verification methods identified\n"
        
        # Format test instructions if available
        test_instructions_text = ""
        if issue_data.get('test_instructions'):
            test_instructions_text = f"{issue_data.get('test_instructions', '')[:600]}{'...' if len(issue_data.get('test_instructions', '')) > 600 else ''}"
        
        # Reuse issue formatting
        comments_text = ""
        for i, comment in enumerate(issue_data.get('comments', []), 1):
            comments_text += f"comment_{i}: {comment['role']} ({comment['author']}): {comment['body'][:400]}{'...' if len(comment['body']) > 400 else ''}\n"
        
        return f"""
                **TESTING & VERIFICATION VALIDATION TASK**: Review the analysis below for accuracy.

                **ORIGINAL ANALYSIS TO VALIDATE:**
                - Issue Key: {issue_data.get('issue_key', 'N/A')}

                **Identified Methods:**
                {methods_summary}

                **ISSUE DATA FOR VERIFICATION:**
                **title:** {issue_data.get('title', 'N/A')}

                **description:**
                {issue_data.get('description', 'N/A')[:800]}{'...' if len(issue_data.get('description', '')) > 800 else ''}

                **test_instructions:**
                {test_instructions_text if test_instructions_text else 'No test instructions provided'}

                **Comments:**
                {comments_text}

                **VALIDATION CHECKLIST:**
                1. **Method Classification**: Are automated/manual categorizations correct?
                2. **Tool/Technique Accuracy**: Are tool names and techniques correctly identified?
                3. **Purpose Classification**: Is testing vs verification distinction accurate?
                4. **Evidence Verification**: Do the evidence sources support the identified methods?
                5. **Completeness**: Are there any missed testing/verification methods?
                6. **Target Issue Accuracy**: Are the target issues correctly identified?
                7. **Deduplication**: Are there redundant entries that should be removed?
                
                **FIELD CONTEXT FOR VALIDATION:**
                - **test_instructions**: Should primarily contain VERIFICATION methods (testing patches/fixes)
                - **description**: Should primarily contain TESTING methods (validating original bugs)  
                - **comments**: Can contain both, depending on when they were posted (early = testing, later = verification)

                **VALIDATION DECISION CRITERIA:**
                - **Agrees**: Original analysis is accurate and complete
                - **Partially_Agrees**: Minor issues or improvements needed  
                - **Disagrees**: Major errors in method identification or classification

                **OUTPUT REQUIREMENTS:**
                - original_methods: Copy the original methods list exactly
                - validated_methods: Your final recommended methods list
                - validation_changes: List specific changes made (or ["None"] if no changes)
                - validation_explanation: Brief explanation of your validation decision
                - agreement_status: Agrees/Partially_Agrees/Disagrees with original analysis

                Provide your validation analysis in the specified JSON format.
                """

    def validate_testing_verification(self, original_result: Dict[str, Any], issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a testing & verification analysis with a second LLM"""
        try:
            validation_prompt = self._create_testing_validation_prompt(original_result, issue_data)
            
            parser = PydanticOutputParser(pydantic_object=TestingValidationResult)
            system_message = SystemMessage(content=self._create_testing_validation_system_prompt())
            human_message = HumanMessage(content=f"{validation_prompt}\n\n{parser.get_format_instructions()}")
            
            with get_openai_callback() as cb:
                start_time = time.time()
                response = self.validation_llm.invoke([system_message, human_message])
                processing_time = time.time() - start_time
            
            validation_result = parser.parse(response.content)
            
            # Merge validation with original result
            return self._merge_testing_validation_result(original_result, validation_result, cb, processing_time)
            
        except Exception as e:
            logger.error(f"Testing validation error for issue {issue_data.get('issue_key', 'unknown')}: {e}")
            # Return original result with validation error noted
            original_result['testing_validation_error'] = str(e)
            return original_result

    def _merge_testing_validation_result(self, original_result: Dict[str, Any], 
                            validation_result: TestingValidationResult,
                            callback_info, processing_time: float) -> Dict[str, Any]:
        """Merge testing validation results with original analysis"""
        merged_result = original_result.copy()
        
        # Add validation metadata
        merged_result['testing_validation'] = {
            'performed': True,
            'agreement_status': validation_result.agreement_status,
            'validation_changes': validation_result.validation_changes,
            'validation_explanation': validation_result.validation_explanation
        }
        
        # Update methods if validation suggests changes
        if validation_result.agreement_status in ['Disagrees', 'Partially_Agrees']:
            merged_result['original_methods'] = original_result.get('testing_methods', [])
            # Convert Pydantic objects to dictionaries
            validated_methods = []
            for method in validation_result.validated_methods:
                if hasattr(method, 'model_dump'):
                    validated_methods.append(method.model_dump())
                else:
                    validated_methods.append(method)
            merged_result['testing_methods'] = validated_methods
        
        return merged_result
    
    def _create_non_a11y_role_validation_system_prompt(self) -> str:
        """Create system prompt for non-a11y role sequence validation"""
        return """You are a senior software engineering researcher performing quality assurance on role sequence analysis for general software development issues.

        Your task is to review and validate another AI's analysis of role sequences in Moodle non-accessibility issues.

        Key responsibilities:
        - Verify the chronological accuracy of the role sequence
        - Check if all meaningful role transitions are captured
        - Identify any missed back-and-forth patterns
        - Ensure proper exclusion of bot roles and automated systems (CiBoT, noreply). The typical comment from these roles is "The integrator needs more information or changes from your patch in order to progress this issue." Ignore these comments.
        - Validate that participant influences are correctly handled (noted but not included in sequence)
        - Verify that back-and-forth codes accurately represent the problems found in the evidence, it has to be specific and descriptive!!!
        - Assess if codes are descriptive, specific, and well-supported by evidence
        - Check that codes are appropriate for general software development issues
        - Ensure evidence sources are legitimate (not from bots)

        Focus on code quality: Are the codes specific and descriptive? Do they accurately capture what the problem actually is? Is the supporting evidence sufficient and from legitimate sources?

        Be critical but fair. Only suggest changes if there are clear errors or omissions.
        Respond only with valid JSON matching the specified schema."""
    
    def _create_non_a11y_role_validation_prompt(self, original_result: Dict[str, Any], issue_data: Dict[str, Any]) -> str:
        """Create validation prompt for non-a11y role sequence analysis"""
        # Format the original analysis for review
        original_sequence = original_result.get('complete_role_sequence', [])
        original_codes = original_result.get('back_and_forth_codes', [])
        
        # Reuse the same issue formatting from the original analysis
        comments_text = ""
        for i, comment in enumerate(issue_data.get('comments', []), 1):
            comments_text += f"{i}. {comment['role']} ({comment['author']}): {comment['body'][:300]}{'...' if len(comment['body']) > 300 else ''}\n"
        
        timeline_text = ""
        for i, event in enumerate(issue_data.get('timeline_events', []), 1):
            timeline_text += f"{i}. {event.get('timestamp', 'N/A')} - {event.get('author', 'N/A')}: {event.get('field', 'N/A')} changed from '{event.get('from_value', '')}' to '{event.get('to_value', '')}'\n"
        
        return f"""
    VALIDATION TASK: Review the role sequence analysis below for accuracy and completeness.

    **ORIGINAL ANALYSIS TO VALIDATE:**
    - Issue Key: {issue_data.get('issue_key', 'N/A')}
    - Analyzed Sequence: {' → '.join(original_sequence)}
    - Back-and-Forth Codes: {original_codes}
    - Sequence Explanation: {original_result.get('sequence_explanation', '')}

    **RAW DATA FOR VERIFICATION:**
    **CHRONOLOGICAL COMMENTS:**
    {comments_text}

    **TIMELINE EVENTS:**
    {timeline_text}

    **VALIDATION INSTRUCTIONS:**
    1. **Verify Chronological Accuracy**: Check if the sequence matches the actual chronological flow
    2. **Check for Missing Transitions**: Are there any role transitions the original analysis missed?
    3. **Validate Back-and-Forth Patterns**: Are returns to previous roles correctly identified?
    4. **Validate Code Quality**: Are the back-and-forth codes descriptive, specific, and well-supported by evidence?
    5. **Check Code Consistency**: Are similar problems coded consistently?
    6. **Confirm Bot Exclusion**: Ensure no bot roles are included in the sequence
    7. **Review Participant Handling**: Verify participants aren't in sequence but their influence is coded
    8. **Assess Evidence Quality**: Is the source text sufficient to support each code assignment?

    **AVAILABLE BACK-AND-FORTH CODES FOR NON-ACCESSIBILITY ISSUES:**
    The original analysis should create descriptive codes based on the actual problems found. Codes should be:
    - **Specific and descriptive** (e.g., "UNIT_TEST_FAILURES" not "TEST_ISSUE")
    - **Consistent in naming** (similar problems should get similar codes)
    - **Well-supported by evidence** (source text should clearly support the code)
    - **Appropriately detailed** (not too broad, not too narrow)

    **Code Quality Assessment:**
    - Do the codes accurately represent the problems described in the evidence?
    - Are the codes specific enough to be meaningful for analysis?
    - Is the evidence sufficient to support each code assignment?
    - Are similar problems coded consistently?
    - Are general development issues distinguished appropriately?

    **Examples of Good Non-A11y Codes:**
    - CODE_QUALITY_STANDARDS, INTEGRATION_CONFLICTS, UNIT_TEST_FAILURES, MERGE_CONFLICTS
    - PERFORMANCE_ISSUES, DOCUMENTATION_MISSING, SECURITY_CONCERNS, UI_LAYOUT_PROBLEMS
    - DATABASE_SCHEMA_ERRORS, API_ENDPOINT_FAILURES, VALIDATION_LOGIC_INCORRECT
    - BROWSER_COMPATIBILITY_ISSUES, RESPONSIVE_DESIGN_PROBLEMS, LOCALIZATION_ERRORS

    **DEFAULT REFERENCE SEQUENCE:** reporter → assignee → peer_reviewer → integrator → tester (codes = [])

    **VALIDATION CRITERIA:**
    - If original analysis is correct: Set agreement_status="Agrees" and keep original sequence, explanation, and codes
    - If minor issues found: Set agreement_status="Partially_Agrees" and suggest improvements  
    - If major errors found: Set agreement_status="Disagrees" and provide corrected sequence, explanation, and codes

    **OUTPUT REQUIREMENTS:**
    - original_sequence: Copy the original sequence exactly
    - validated_sequence: Your final recommended sequence (may be same as original)
    - original_codes: Copy the original codes exactly
    - validated_codes: Your final recommended codes (may be same as original)
    - original_explanation: Copy the original explanation exactly
    - validated_explanation: Your final recommended explanation (may be same as original)
    - validation_confidence: High/Medium/Low based on data clarity
    - sequence_changes: List specific changes made (or ["None"] if no changes)
    - validation_explanation: Brief explanation of your validation decision
    - agreement_status: Agrees/Partially_Agrees/Disagrees with original analysis
    """
    
    def _create_non_a11y_testing_validation_system_prompt(self) -> str:
        """Create system prompt for non-a11y testing & verification validation"""
        return """You are a senior software testing expert performing quality assurance on testing and verification method analysis for general software development issues.

Your task is to review and validate another expert's identification of testing and verification methods in Moodle non-accessibility issues.

Key responsibilities:
- Verify accurate identification of automated testing tools and frameworks
- Check correct classification of manual testing techniques
- Validate distinction between testing (bug validation) and verification (fix confirmation)
- Ensure evidence sources support the identified methods
- Confirm tool names and technique descriptions are accurate
- Assess completeness of method identification
- Check for proper deduplication (same tool/technique for same purpose should appear only once)

**Common Testing Tools and Techniques to Recognize (include but not limited to):**
- Unit testing: PHPUnit, Jest, Mocha, Jasmine, QUnit
- Integration testing: Behat, Selenium, Cypress, Playwright, Puppeteer
- CI/CD testing: GitHub Actions, Travis CI, Jenkins
- Manual testing: browser testing, UI testing, step-by-step workflows
- Database testing: SQL queries, data validation, migration tests
- API testing: REST API tests, Postman, curl commands
- Performance testing: load testing, profiling, resource monitoring

Be thorough but fair. Only suggest changes if there are clear errors in method identification or classification.
Respond only with valid JSON matching the specified schema."""
    
    def _create_non_a11y_testing_validation_prompt(self, original_result: Dict[str, Any], issue_data: Dict[str, Any]) -> str:
        """Create validation prompt for non-a11y testing & verification analysis"""
        # Format the original analysis for review
        original_methods = original_result.get('testing_methods', [])
        
        methods_summary = ""
        if original_methods:
            for i, method in enumerate(original_methods, 1):
                methods_summary += f"  {i}. {method.get('method_type', '')} - {method.get('tool_or_technique', '')}\n"
                methods_summary += f"     Purpose: {method.get('purpose', '')}\n"
                methods_summary += f"     Target: {method.get('target_issue', '')}\n"
                methods_summary += f"     Source: {method.get('evidence_source', '')}\n"
                methods_summary += f"     Details: {method.get('details', '')}\n\n"
        else:
            methods_summary = "  No testing/verification methods identified\n"
        
        # Format test instructions if available
        test_instructions_text = ""
        if issue_data.get('test_instructions'):
            test_instructions_text = f"{issue_data.get('test_instructions', '')[:600]}{'...' if len(issue_data.get('test_instructions', '')) > 600 else ''}"
        
        # Reuse issue formatting
        comments_text = ""
        for i, comment in enumerate(issue_data.get('comments', []), 1):
            comments_text += f"comment_{i}: {comment['role']} ({comment['author']}): {comment['body'][:400]}{'...' if len(comment['body']) > 400 else ''}\n"
        
        return f"""
**TESTING & VERIFICATION VALIDATION TASK**: Review the analysis below for accuracy.

**ORIGINAL ANALYSIS TO VALIDATE:**
- Issue Key: {issue_data.get('issue_key', 'N/A')}

**Identified Methods:**
{methods_summary}

**ISSUE DATA FOR VERIFICATION:**
**title:** {issue_data.get('title', 'N/A')}

**description:**
{issue_data.get('description', 'N/A')[:800]}{'...' if len(issue_data.get('description', '')) > 800 else ''}

**test_instructions:**
{test_instructions_text if test_instructions_text else 'No test instructions provided'}

**Comments:**
{comments_text}

**VALIDATION CHECKLIST:**
1. **Method Classification**: Are automated/manual categorizations correct?
2. **Tool/Technique Accuracy**: Are tool names and techniques correctly identified?
3. **Purpose Classification**: Is testing vs verification distinction accurate?
4. **Evidence Verification**: Do the evidence sources support the identified methods?
5. **Completeness**: Are there any missed testing/verification methods?
6. **Target Issue Accuracy**: Are the target issues correctly identified?
7. **Deduplication**: Are there redundant entries that should be removed?

**FIELD CONTEXT FOR VALIDATION:**
- **test_instructions**: Should primarily contain VERIFICATION methods (testing patches/fixes)
- **description**: Should primarily contain TESTING methods (validating original bugs)  
- **comments**: Can contain both, depending on when they were posted (early = testing, later = verification)

**COMMON TESTING METHODS FOR NON-ACCESSIBILITY ISSUES (include but not limited to):**
- **Automated**: PHPUnit, Behat, Selenium, Jest, CI/CD tests, database tests, API tests
- **Manual**: Browser testing, UI testing, step-by-step workflows, cross-platform testing

**VALIDATION DECISION CRITERIA:**
- **Agrees**: Original analysis is accurate and complete
- **Partially_Agrees**: Minor issues or improvements needed  
- **Disagrees**: Major errors in method identification or classification

**OUTPUT REQUIREMENTS:**
- original_methods: Copy the original methods list exactly
- validated_methods: Your final recommended methods list
- validation_changes: List specific changes made (or ["None"] if no changes)
- validation_explanation: Brief explanation of your validation decision
- agreement_status: Agrees/Partially_Agrees/Disagrees with original analysis

Provide your validation analysis in the specified JSON format.
"""
    
    def validate_non_a11y_role_sequence(self, original_result: Dict[str, Any], issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a non-a11y role sequence analysis with a second LLM"""
        try:
            validation_prompt = self._create_non_a11y_role_validation_prompt(original_result, issue_data)
            
            parser = PydanticOutputParser(pydantic_object=ValidationResult)
            system_message = SystemMessage(content=self._create_non_a11y_role_validation_system_prompt())
            human_message = HumanMessage(content=f"{validation_prompt}\n\n{parser.get_format_instructions()}")
            
            with get_openai_callback() as cb:
                start_time = time.time()
                response = self.validation_llm.invoke([system_message, human_message])
                processing_time = time.time() - start_time
            
            validation_result = parser.parse(response.content)
            
            # Merge validation with original result (reuse existing merge method)
            return self._merge_validation_result(original_result, validation_result, cb, processing_time)
            
        except Exception as e:
            logger.error(f"Non-A11y role validation error for issue {issue_data.get('issue_key', 'unknown')}: {e}")
            original_result['validation_error'] = str(e)
            return original_result
    
    def validate_non_a11y_testing_verification(self, original_result: Dict[str, Any], issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a non-a11y testing & verification analysis with a second LLM"""
        try:
            validation_prompt = self._create_non_a11y_testing_validation_prompt(original_result, issue_data)
            
            parser = PydanticOutputParser(pydantic_object=TestingValidationResult)
            system_message = SystemMessage(content=self._create_non_a11y_testing_validation_system_prompt())
            human_message = HumanMessage(content=f"{validation_prompt}\n\n{parser.get_format_instructions()}")
            
            with get_openai_callback() as cb:
                start_time = time.time()
                response = self.validation_llm.invoke([system_message, human_message])
                processing_time = time.time() - start_time
            
            validation_result = parser.parse(response.content)
            
            # Merge validation with original result (reuse existing merge method)
            return self._merge_testing_validation_result(original_result, validation_result, cb, processing_time)
            
        except Exception as e:
            logger.error(f"Non-A11y testing validation error for issue {issue_data.get('issue_key', 'unknown')}: {e}")
            original_result['testing_validation_error'] = str(e)
            return original_result
    
    def needs_participant_validation(self, analysis_result: Dict[str, Any]) -> bool:
        """Check if participant influence result needs validation"""
        if not self.config.enable_validation:
            return False
            
        if 'error' in analysis_result:
            return False  # Don't validate error results
        
        participant_codes = analysis_result.get('participant_influence_codes', [])
        
        if self.config.validate_only_non_default:
            # Validate when participant influence codes are found
            return len(participant_codes) > 0
    
        return True  # Validate all if not restricted

    def _create_participant_validation_system_prompt(self) -> str:
        """Create system prompt for participant influence validation"""
        return """You are a senior accessibility research expert performing quality assurance on participant influence analysis.

        Your task is to review and validate another expert's analysis of how participants influence accessibility solution development in Moodle issues.

        Key responsibilities:
        - Verify accurate identification of participant contributions (NOT reporter/assignee/peer_reviewer/integrator/tester/bot)
        - Exclude bot comments or automated comments (author: CiBoT or noreply)
        - Check that codes are specific, descriptive, and well-supported by evidence
        - Prioritize accessibility-related contributions while including general technical issues that affect accessibility solutions
        - Ensure evidence sources support the assigned codes
        - Assess completeness of participant influence identification
        - Check for proper focus on accessibility solution/patch development influence

        **Code Quality Criteria:**
        - Codes should be specific and descriptive (e.g., "REPORTED_SCREEN_READER_FOCUS_ISSUE" not "REPORTED_ISSUE")
        - Codes should capture actual influence on accessibility solution development
        - Evidence should clearly support the assigned code
        - Focus on accessibility-related contributions, but include general technical contributions (like test failures) that affect accessibility solutions

        Be thorough but fair. Only suggest changes if there are clear errors in participant influence identification or code assignment.
        Respond only with valid JSON matching the specified schema."""

    def _create_participant_validation_prompt(self, original_result: Dict[str, Any], issue_data: Dict[str, Any]) -> str:
                # Format the original analysis for review
        original_codes = original_result.get('participant_influence_codes', [])
        
        codes_summary = ""
        if original_codes:
            for i, code_obj in enumerate(original_codes, 1):
                if isinstance(code_obj, dict):
                    code = code_obj.get('code', '')
                    source_text = code_obj.get('source_text', '')
                    source_location = code_obj.get('source_location', '')
                else:
                    # Handle Pydantic objects
                    code = getattr(code_obj, 'code', '')
                    source_text = getattr(code_obj, 'source_text', '')
                    source_location = getattr(code_obj, 'source_location', '')
                
                codes_summary += f"  {i}. {code}\n"
                codes_summary += f"     Source: {source_text}\n"
                codes_summary += f"     Location: {source_location}\n\n"
        else:
            codes_summary = "  No participant influence codes identified\n"
        
        # Format participant comments for verification
        participant_comments = []
        all_comments_with_roles = []
        
        for i, comment in enumerate(issue_data.get('comments', []), 1):
            roles = comment.get('role', [])
            if isinstance(roles, str):
                roles = [roles]
            
            author = comment['author']
            body = comment['body']
            
            # Add all comments with role information for context
            all_comments_with_roles.append(f"comment_{i}: {roles} ({author}): {body[:350]}{'...' if len(body) > 350 else ''}")
            
            # Identify participants
            core_roles = {'reporter', 'assignee', 'peer_reviewer', 'integrator', 'tester', 'bot'}
            is_participant = not any(role in core_roles for role in roles)
            
            if is_participant or 'participant' in roles:
                # participant_comments.append(f"comment_{i}: PARTICIPANT ({author}): {body[:400]}{'...' if len(body) > 400 else ''}")
                participant_comments.append(f"comment_{i}: PARTICIPANT ({author}): {body}")
        
        participant_comments_text = "\n".join(participant_comments) if participant_comments else "No participant comments found"
        all_comments_text = "\n".join(all_comments_with_roles)
        
        return f"""
        **PARTICIPANT INFLUENCE VALIDATION TASK**: Review the analysis below for accuracy and completeness.

        **ORIGINAL ANALYSIS TO VALIDATE:**
        - Issue Key: {issue_data.get('issue_key', 'N/A')}

        **Identified Participant Influence Codes:**
        {codes_summary}

        **PARTICIPANT COMMENTS FOR VERIFICATION:**
        {participant_comments_text}

        **ALL COMMENTS WITH ROLES (For Context):**
        {all_comments_text}

        **VALIDATION CRITERIA:**

        **1. PARTICIPANT IDENTIFICATION ACCURACY:**
        - Verify that coded influences come from actual participants (NOT reporter/assignee/peer_reviewer/integrator/tester/bot)
        - Check that all participants with accessibility-related contributions were identified
        - Confirm no core development roles were incorrectly classified as participants

        **2. CODE QUALITY ASSESSMENT:**
        Apply these quality standards to each code:

        **Specificity Check:**
        - Is the code specific enough to understand the exact contribution?
        - Does it avoid generic terms like "barrier," "issue," or "guidance"?
        - Does it capture the precise accessibility aspect being addressed?

        **Technical Precision:**
        - Are specific assistive technologies mentioned when relevant?
        - Are specific accessibility standards referenced when cited?
        - Are specific UI components or interactions included when mentioned?

        **Evidence Alignment:**
        - Does the code accurately reflect what's stated in the source text?
        - Is the code supported by the actual evidence provided?
        - Does the code avoid inference beyond what's explicitly stated?

        **Granular Differentiation:**
        - Are different accessibility contributions assigned appropriately different codes?
        - Do similar contributions from different participants share appropriate codes?
        - Is each code distinct and meaningful?

        **3. COMPLETENESS VERIFICATION:**
        - Are there any missed participant contributions to accessibility solution development?
        - Were all types of influence detected (accessibility suggestions, testing, problem identification, etc.)?
        - Are there participant comments about accessibility that weren't coded?
        - Are there general technical contributions (like test failures) that affect accessibility solutions?

        **4. ACCESSIBILITY FOCUS VALIDATION:**
        - Do codes focus on contributions that influenced accessibility solution development?
        - Are accessibility-related contributions prioritized?
        - Are general technical contributions (like Behat failures) that affect accessibility solutions included?

        **CODE IMPROVEMENT GUIDELINES:**

        **If codes are too generic, make them more specific by:**
        - Adding the specific accessibility aspect (screen_reader, keyboard_navigation, color_contrast, etc.)
        - Including the specific problem or solution mentioned
        - Adding context when provided (specific AT, WCAG criteria, UI component)

        **If codes are missing, add them following:**
        - ACTION_SPECIFIC_ACCESSIBILITY_ASPECT_CONTEXT pattern
        - Evidence-based coding (only what's explicitly stated)
        - Meaningful differentiation from existing codes

        **If codes are inaccurate, correct them by:**
        - Better alignment with source text evidence
        - More precise capture of the actual accessibility contribution
        - Appropriate specificity level (not too broad, not too narrow)

        **IMPORTANT NOTE:**
        Focus primarily on accessibility-related participant contributions. Also include general technical contributions (like test failures, integration issues) that affect accessibility solution development.

        Provide your validation analysis in the specified JSON format.
        """

    def validate_participant_influence(self, original_result: Dict[str, Any], issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a participant influence analysis with a second LLM"""
        try:
            validation_prompt = self._create_participant_validation_prompt(original_result, issue_data)
            
            parser = PydanticOutputParser(pydantic_object=ParticipantValidationResult)
            system_message = SystemMessage(content=self._create_participant_validation_system_prompt())
            human_message = HumanMessage(content=f"{validation_prompt}\n\n{parser.get_format_instructions()}")
            
            with get_openai_callback() as cb:
                start_time = time.time()
                response = self.validation_llm.invoke([system_message, human_message])
                processing_time = time.time() - start_time
            
            validation_result = parser.parse(response.content)
            
            # Merge validation with original result
            return self._merge_participant_validation_result(original_result, validation_result, cb, processing_time)
            
        except Exception as e:
            logger.error(f"Participant influence validation error for issue {issue_data.get('issue_key', 'unknown')}: {e}")
            original_result['participant_validation_error'] = str(e)
            return original_result

    def _merge_participant_validation_result(self, original_result: Dict[str, Any], 
                            validation_result: ParticipantValidationResult,
                            callback_info, processing_time: float) -> Dict[str, Any]:
        """Merge participant validation results with original analysis"""
        merged_result = original_result.copy()
        
        # Add validation metadata
        merged_result['participant_validation'] = {
            'performed': True,
            'agreement_status': validation_result.agreement_status,
            'validation_changes': validation_result.validation_changes,
            'validation_explanation': validation_result.validation_explanation
        }
        
        # Update codes if validation suggests changes
        if validation_result.agreement_status in ['Disagrees', 'Partially_Agrees']:
            merged_result['original_codes'] = original_result.get('participant_influence_codes', [])
            # Convert Pydantic objects to dictionaries
            validated_codes = []
            for code in validation_result.validated_codes:
                if hasattr(code, 'model_dump'):
                    validated_codes.append(code.model_dump())
                else:
                    validated_codes.append(code)
            merged_result['participant_influence_codes'] = validated_codes
        
        return merged_result
    
    def _create_non_a11y_participant_validation_system_prompt(self) -> str:
        """Create system prompt for non-a11y participant influence validation"""
        return """You are a senior software development research expert performing quality assurance on participant influence analysis for general software development issues.

        Your task is to review and validate another expert's analysis of how participants influence general software solution development in Moodle issues.

        Key responsibilities:
        - Verify accurate identification of participant contributions (NOT reporter/assignee/peer_reviewer/integrator/tester/bot)
        - Exclude bot comments or automated comments (author: CiBoT or noreply)
        - Check that codes are specific, descriptive, and well-supported by evidence
        - Prioritize general software development contributions while including technical issues that affect solution development
        - Ensure evidence sources support the assigned codes
        - Assess completeness of participant influence identification
        - Check for proper focus on general software solution/patch development influence

        **Code Quality Criteria:**
        - Codes should be specific and descriptive (e.g., "REPORTED_DATABASE_QUERY_FAILURE" not "REPORTED_ISSUE")
        - Codes should capture actual influence on software solution development
        - Evidence should clearly support the assigned code
        - Focus on general software development contributions, including technical contributions that affect solution development

        Be thorough but fair. Only suggest changes if there are clear errors in participant influence identification or code assignment.
        Respond only with valid JSON matching the specified schema."""
    
    
    def _create_non_a11y_participant_validation_prompt(self, original_result: Dict[str, Any], issue_data: Dict[str, Any]) -> str:
        """Create validation prompt for non-a11y participant influence analysis"""
        # Format the original analysis for review
        original_codes = original_result.get('participant_influence_codes', [])
        
        codes_summary = ""
        if original_codes:
            for i, code_obj in enumerate(original_codes, 1):
                if isinstance(code_obj, dict):
                    code = code_obj.get('code', '')
                    source_text = code_obj.get('source_text', '')
                    source_location = code_obj.get('source_location', '')
                else:
                    # Handle Pydantic objects
                    code = getattr(code_obj, 'code', '')
                    source_text = getattr(code_obj, 'source_text', '')
                    source_location = getattr(code_obj, 'source_location', '')
                
                codes_summary += f"  {i}. {code}\n"
                codes_summary += f"     Source: {source_text}\n"
                codes_summary += f"     Location: {source_location}\n\n"
        else:
            codes_summary = "  No participant influence codes identified\n"
        
        # Format participant comments for verification
        participant_comments = []
        all_comments_with_roles = []
        
        for i, comment in enumerate(issue_data.get('comments', []), 1):
            roles = comment.get('role', [])
            if isinstance(roles, str):
                roles = [roles]
            
            author = comment['author']
            body = comment['body']
            
            # Add all comments with role information for context
            all_comments_with_roles.append(f"comment_{i}: {roles} ({author}): {body[:350]}{'...' if len(body) > 350 else ''}")
            
            # Identify participants
            core_roles = {'reporter', 'assignee', 'peer_reviewer', 'integrator', 'tester', 'bot'}
            is_participant = not any(role in core_roles for role in roles)
            
            if is_participant or 'participant' in roles:
                # participant_comments.append(f"comment_{i}: PARTICIPANT ({author}): {body[:400]}{'...' if len(body) > 400 else ''}")
                participant_comments.append(f"comment_{i}: PARTICIPANT ({author}): {body}")
        
        participant_comments_text = "\n".join(participant_comments) if participant_comments else "No participant comments found"
        all_comments_text = "\n".join(all_comments_with_roles)
        
        return f"""
        **NON-A11Y PARTICIPANT INFLUENCE VALIDATION TASK**: Review the analysis below for accuracy and completeness.

        **ORIGINAL ANALYSIS TO VALIDATE:**
        - Issue Key: {issue_data.get('issue_key', 'N/A')}

        **Identified Participant Influence Codes:**
        {codes_summary}

        **PARTICIPANT COMMENTS FOR VERIFICATION:**
        {participant_comments_text}

        **ALL COMMENTS WITH ROLES (For Context):**
        {all_comments_text}

        **VALIDATION INSTRUCTIONS:**

        1. **PARTICIPANT IDENTIFICATION ACCURACY:**
        - Verify that coded influences come from actual participants (NOT reporter/assignee/peer_reviewer/integrator/tester/bot)
        - Check that all participants with software development contributions were identified
        - Confirm no core development roles were incorrectly classified as participants

        2. **CODE QUALITY ASSESSMENT FOR SOFTWARE DEVELOPMENT:**
        - Are codes specific to software development aspects (database, UI, API, testing, etc.)?
        - Do codes avoid generic terms and capture precise technical contributions?
        - Are development technologies and frameworks mentioned when relevant?
        - Do codes reflect actual influence on solution development?

        3. **COMPLETENESS VERIFICATION:**
        - Are there any missed participant contributions to software solution development?
        - Were all types of technical influence detected (code suggestions, testing, problem identification, etc.)?
        - Are there participant comments about software development that weren't coded?

        4. **EVIDENCE QUALITY:**
        - Does each code have sufficient supporting evidence?
        - Are the source locations accurate?
        - Does the evidence clearly support the assigned code?

        **VALIDATION DECISION CRITERIA:**
        - **Agrees**: Original analysis is accurate and complete
        - **Partially_Agrees**: Minor issues or improvements needed  
        - **Disagrees**: Major errors in participant influence identification or code assignment

        **REQUIRED OUTPUT FORMAT:**
        You must provide ALL of the following fields:

        1. **original_codes**: Copy the exact original codes list from above
        2. **validated_codes**: Your final recommended codes list (may be same as original if no changes needed)
        3. **validation_changes**: List of specific changes made (use ["None"] if no changes)
        4. **validation_explanation**: Brief explanation of your validation decision (always required)
        5. **agreement_status**: Must be exactly one of: "Agrees", "Partially_Agrees", or "Disagrees"

        **EXAMPLES:**

        If analysis is correct:
        - validation_changes: ["None"]
        - validation_explanation: "The original analysis accurately identified all participant influences with appropriate codes and evidence."
        - agreement_status: "Agrees"

        If minor improvements needed:
        - validation_changes: ["Made code XYZ more specific", "Added missing participant contribution from comment_5"]
        - validation_explanation: "Original analysis was mostly accurate but needed minor improvements to code specificity."
        - agreement_status: "Partially_Agrees"

        If major errors found:
        - validation_changes: ["Removed incorrect codes", "Added missed participant influences", "Corrected evidence sources"]
        - validation_explanation: "Original analysis missed several participant contributions and had incorrect code assignments."
        - agreement_status: "Disagrees"

        **CRITICAL**: Your response must include all 5 required fields (original_codes, validated_codes, validation_changes, validation_explanation, agreement_status) in valid JSON format.

        Provide your validation analysis in the specified JSON format with ALL required fields.
        """
    
    def validate_non_a11y_participant_influence(self, original_result: Dict[str, Any], issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a non-a11y participant influence analysis with a second LLM"""
        try:
            validation_prompt = self._create_non_a11y_participant_validation_prompt(original_result, issue_data)
            
            parser = PydanticOutputParser(pydantic_object=ParticipantValidationResult)
            system_message = SystemMessage(content=self._create_non_a11y_participant_validation_system_prompt())
            human_message = HumanMessage(content=f"{validation_prompt}\n\n{parser.get_format_instructions()}")
            
            with get_openai_callback() as cb:
                start_time = time.time()
                response = self.validation_llm.invoke([system_message, human_message])
                processing_time = time.time() - start_time
            
            validation_result = parser.parse(response.content)
            
            # Merge validation with original result (reuse existing merge method)
            return self._merge_participant_validation_result(original_result, validation_result, cb, processing_time)
            
        except Exception as e:
            logger.error(f"Non-A11y participant influence validation error for issue {issue_data.get('issue_key', 'unknown')}: {e}")
            original_result['participant_validation_error'] = str(e)
            return original_result
    
    def needs_solution_validation(self, analysis_result: Dict[str, Any]) -> bool:
        """Check if solution development result needs validation"""
        if not self.config.enable_validation:
            return False
            
        if 'error' in analysis_result:
            return False
        
        solution_approaches = analysis_result.get('solution_approaches', [])
        
        if self.config.validate_only_non_default:
            return len(solution_approaches) > 0
        
        return True

    def _create_solution_validation_system_prompt(self) -> str:
        """Create system prompt for solution development validation with open coding"""
        return """You are a senior accessibility research expert performing quality assurance on initial accessibility solution development analysis using open coding methodology.

        Your task is to review and validate another expert's open coding analysis of how assignees initially develop accessibility solutions.

        **CRITICAL VALIDATION FOCUS:**
        Ensure the analysis focused ONLY on initial accessibility solution development and used appropriate open coding:
        - Verify codes represent FIRST/ORIGINAL accessibility solution articulation, not refinements
        - Check that reactive solutions (responses to feedback) were properly excluded
        - Confirm focus on independent accessibility problem-solving before external input
        - Validate that subsequent modifications covered by Role Sequence RQ were not included
        - Assess quality and specificity of open codes for accessibility solution development

        **OPEN CODING VALIDATION CRITERIA:**
        - Are codes specific and descriptive of accessibility solution approaches?
        - Do codes capture authentic accessibility thinking processes?
        - Are codes grounded in the actual evidence from assignee comments?
        - Do codes reflect accessibility-specific considerations (AT, WCAG, user impact)?
        - Are codes appropriately technical and action-oriented?

        Key responsibilities:
        - Verify accurate identification of INITIAL accessibility solution development approaches
        - Check that open codes appropriately capture accessibility solution thinking
        - Ensure evidence sources support the assigned codes and represent initial solutions
        - Assess completeness of INITIAL accessibility solution development identification
        - Validate that accessibility details and reasoning are accurately captured
        - Verify clear separation from subsequent refinements handled by Role Sequence RQ

        **SCOPE VALIDATION:**
        - INCLUDE: Original accessibility analysis, initial solution design, first accessibility decisions
        - EXCLUDE: Responses to peer review, fixes after integration problems, changes after test failures

        Be thorough but fair. Only suggest changes if there are clear errors in initial accessibility solution development identification or inappropriate open coding.
        Respond only with valid JSON matching the specified schema."""

    def _create_solution_validation_prompt(self, original_result: Dict[str, Any], issue_data: Dict[str, Any]) -> str:
        """Create validation prompt for solution development analysis with open coding"""
        # Format the original analysis for review
        original_approaches = original_result.get('solution_approaches', [])
        
        approaches_summary = ""
        if original_approaches:
            for i, approach_obj in enumerate(original_approaches, 1):
                if isinstance(approach_obj, dict):
                    solution_codes = approach_obj.get('solution_codes', [])
                    approach_description = approach_obj.get('approach_description', '')
                    source_text = approach_obj.get('source_text', '')
                    source_location = approach_obj.get('source_location', '')
                    technical_details = approach_obj.get('technical_details', '')
                else:
                    # Handle Pydantic objects
                    solution_codes = getattr(approach_obj, 'solution_codes', [])
                    approach_description = getattr(approach_obj, 'approach_description', '')
                    source_text = getattr(approach_obj, 'source_text', '')
                    source_location = getattr(approach_obj, 'source_location', '')
                    technical_details = getattr(approach_obj, 'technical_details', '')
                
                approaches_summary += f"  {i}. Codes: {solution_codes}\n"
                approaches_summary += f"     Description: {approach_description}\n"
                approaches_summary += f"     Technical Details: {technical_details}\n"
                approaches_summary += f"     Source: {source_text[:200]}{'...' if len(source_text) > 200 else ''}\n"
                approaches_summary += f"     Location: {source_location}\n\n"
        else:
            approaches_summary = "  No solution development approaches identified\n"
        
        # Format assignee comments for verification
        assignee_comments = []
        
        for i, comment in enumerate(issue_data.get('comments', []), 1):
            roles = comment.get('role', [])
            if isinstance(roles, str):
                roles = [roles]
            
            if 'assignee' in roles:
                assignee_comments.append(f"comment_{i}: ASSIGNEE ({comment['author']}): {comment['body'][:400]}{'...' if len(comment['body']) > 400 else ''}")
        
        assignee_comments_text = "\n".join(assignee_comments) if assignee_comments else "No assignee comments found"
        
        return f"""
        **INITIAL ACCESSIBILITY SOLUTION DEVELOPMENT VALIDATION TASK - OPEN CODING**: Review the analysis below for accuracy and completeness.

        **ORIGINAL ANALYSIS TO VALIDATE:**
        - Issue Key: {issue_data.get('issue_key', 'N/A')}

        **Identified Solution Development Approaches with Open Codes:**
        {approaches_summary}

        **ASSIGNEE COMMENTS FOR VERIFICATION:**
        {assignee_comments_text}

        **VALIDATION INSTRUCTIONS:**

        **1. INITIAL SOLUTION FOCUS VERIFICATION:**
        - Verify that identified approaches represent INITIAL accessibility solution development only
        - Check that subsequent refinements and responses to feedback were properly excluded
        - Confirm focus on original accessibility problem-solving before external input
        - Ensure reactive solutions (fixes after reviewer feedback) were not included

        **2. OPEN CODING QUALITY ASSESSMENT:**
        - Are the open codes specific and descriptive of accessibility solution approaches?
        - Do codes capture authentic accessibility thinking processes from the evidence?
        - Are codes grounded in actual assignee comment content?
        - Do codes reflect accessibility-specific considerations (AT, WCAG, user impact, inclusive design)?
        - Are codes appropriately technical and action-oriented for accessibility work?

        **3. ACCESSIBILITY CODE APPROPRIATENESS:**
        - Do codes capture accessibility-specific solution development (not generic software development)?
        - Are assistive technology considerations properly reflected in codes?
        - Do codes represent accessibility standards application appropriately?
        - Are user barrier and inclusive design aspects captured in codes?

        **4. SCOPE BOUNDARY VALIDATION:**
        - SHOULD BE INCLUDED: First accessibility analysis, original solution design, initial accessibility reasoning
        - SHOULD BE EXCLUDED: Solutions after peer review feedback, integration problem fixes, test failure responses
        - Verify clear distinction between initial accessibility solutions and subsequent refinements

        **5. ASSIGNEE COMMENT VERIFICATION:**
        - Verify that identified approaches come from actual assignee comments only
        - Check that all substantial INITIAL accessibility solution development evidence was captured
        - Confirm no approaches were incorrectly attributed to non-assignee roles

        **6. EVIDENCE QUALITY FOR INITIAL ACCESSIBILITY SOLUTIONS:**
        - Verify that source_text accurately represents INITIAL accessibility solution approaches
        - Check that approach_description captures original accessibility solution development methods
        - Ensure technical_details reflect initial accessibility decisions and reasoning

        **7. COMPLETENESS OF INITIAL ACCESSIBILITY SOLUTION IDENTIFICATION:**
        - Are there any missed INITIAL accessibility solution development approaches in assignee comments?
        - Were all significant original accessibility solution articulations captured?
        - Are there substantial initial accessibility analyses not coded?

        **VALIDATION DECISION CRITERIA:**
        - **Agrees**: Original analysis accurately identified all INITIAL accessibility solution development approaches with appropriate open codes
        - **Partially_Agrees**: Minor issues in open coding quality or minor missed initial approaches
        - **Disagrees**: Major errors in initial vs. refinement distinction, inappropriate coding, or significant missed evidence

        **OPEN CODING VALIDATION EXAMPLES:**

        **GOOD ACCESSIBILITY CODES:**
        - ["ARIA_LABEL_MISUSE_IDENTIFICATION", "SEMANTIC_ROLE_REPLACEMENT_STRATEGY"]
        - ["SCREEN_READER_COMPATIBILITY_ANALYSIS", "NVDA_TESTING_APPROACH"]
        - ["WCAG_SUCCESS_CRITERIA_APPLICATION", "CONTRAST_RATIO_CALCULATION"]

        **POOR/GENERIC CODES:**
        - ["TECHNICAL_ANALYSIS", "IMPLEMENTATION_STRATEGY"] (too generic)
        - ["CODE_CHANGE", "PROBLEM_SOLVING"] (not accessibility-specific)
        - ["BEST_PRACTICES"] (too vague)

        **OUTPUT REQUIREMENTS:**
        - original_approaches: Copy the exact original approaches list from above
        - validated_approaches: Your final recommended approaches list with improved open codes if needed
        - validation_changes: List of specific changes made (use ["None"] if no changes)
        - validation_explanation: Brief explanation of your validation decision focusing on open coding quality
        - agreement_status: Must be exactly one of: "Agrees", "Partially_Agrees", or "Disagrees"

        **CRITICAL SCOPE REMINDER:**
        This research question focuses exclusively on INITIAL accessibility solution development using open coding methodology. Subsequent refinements, iterations, and responses to feedback are covered by the Role Sequence research question. Ensure this boundary is respected and that open codes authentically capture accessibility solution thinking.

        Provide your validation analysis in the specified JSON format with ALL required fields.
        """

    def validate_solution_development(self, original_result: Dict[str, Any], issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a solution development analysis with a second LLM"""
        try:
            validation_prompt = self._create_solution_validation_prompt(original_result, issue_data)
            
            parser = PydanticOutputParser(pydantic_object=SolutionValidationResult)
            system_message = SystemMessage(content=self._create_solution_validation_system_prompt())
            human_message = HumanMessage(content=f"{validation_prompt}\n\n{parser.get_format_instructions()}")
            
            with get_openai_callback() as cb:
                start_time = time.time()
                response = self.validation_llm.invoke([system_message, human_message])
                processing_time = time.time() - start_time
            
            validation_result = parser.parse(response.content)
            
            # Merge validation with original result
            return self._merge_solution_validation_result(original_result, validation_result, cb, processing_time)
        
        except Exception as e:
            logger.error(f"Solution development validation error for issue {issue_data.get('issue_key', 'unknown')}: {e}")
            original_result['solution_validation_error'] = str(e)
            return original_result

    def validate_non_a11y_solution_development(self, original_result: Dict[str, Any], issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a non-a11y solution development analysis with a second LLM (same validation logic)"""
        try:
            validation_prompt = self._create_solution_validation_prompt(original_result, issue_data)
            
            parser = PydanticOutputParser(pydantic_object=SolutionValidationResult)
            system_message = SystemMessage(content=self._create_solution_validation_system_prompt())
            human_message = HumanMessage(content=f"{validation_prompt}\n\n{parser.get_format_instructions()}")
            
            with get_openai_callback() as cb:
                start_time = time.time()
                response = self.validation_llm.invoke([system_message, human_message])
                processing_time = time.time() - start_time
            
            validation_result = parser.parse(response.content)
            
            # Merge validation with original result
            return self._merge_solution_validation_result(original_result, validation_result, cb, processing_time)
        
        except Exception as e:
            logger.error(f"Non-A11y solution development validation error for issue {issue_data.get('issue_key', 'unknown')}: {e}")
            original_result['solution_validation_error'] = str(e)
            return original_result

    def _merge_solution_validation_result(self, original_result: Dict[str, Any], 
                        validation_result: SolutionValidationResult,
                        callback_info, processing_time: float) -> Dict[str, Any]:
        """Merge solution validation results with original analysis"""
        merged_result = original_result.copy()
        
        # Add validation metadata
        merged_result['solution_validation'] = {
            'performed': True,
            'agreement_status': validation_result.agreement_status,
            'validation_changes': validation_result.validation_changes,
            'validation_explanation': validation_result.validation_explanation,
            'processing_time_seconds': round(processing_time, 2),
            'tokens_used': callback_info.total_tokens,
            'cost_estimate': callback_info.total_cost
        }
        
        # Update approaches if validation suggests changes
        if validation_result.agreement_status in ['Disagrees', 'Partially_Agrees']:
            merged_result['original_approaches'] = original_result.get('solution_approaches', [])
            # Convert Pydantic objects to dictionaries
            validated_approaches = []
            for approach in validation_result.validated_approaches:
                if hasattr(approach, 'model_dump'):
                    validated_approaches.append(approach.model_dump())
                elif hasattr(approach, 'dict'):
                    validated_approaches.append(approach.dict())
                else:
                    validated_approaches.append(approach)
            merged_result['solution_approaches'] = validated_approaches
        
        return merged_result

def register_research_questions_for_dataset(analyzer: MoodleIssueAnalyzer, config: AnalysisConfig):
    """Register appropriate research questions based on dataset type"""
    dataset_type = getattr(config, 'dataset_type', 'a11y')
    
    if dataset_type == 'a11y':
        # Accessibility issues
        if "role_sequence" in config.research_questions:
            analyzer.register_research_question(RoleSequenceQuestion())
            logger.info("Registered: Role Sequence Analysis (A11y)")
        
        if "wcag_categorization" in config.research_questions:
            analyzer.register_research_question(WCAGCategoryQuestion())
            logger.info("Registered: WCAG 2.2 Categorization")
        
        if "testing_verification" in config.research_questions:
            analyzer.register_research_question(TestingVerificationQuestion())
            logger.info("Registered: Testing & Verification Analysis (A11y)")
    
    else:
        # Non-accessibility issues
        if "role_sequence" in config.research_questions:
            analyzer.register_research_question(NonA11yRoleSequenceQuestion())
            logger.info("Registered: Role Sequence Analysis (Non-A11y)")
        
        if "wcag_categorization" in config.research_questions:
            logger.warning("WCAG categorization skipped for non-accessibility issues")
        
        if "testing_verification" in config.research_questions:
            analyzer.register_research_question(NonA11yTestingVerificationQuestion())
            logger.info("Registered: Testing & Verification Analysis (Non-A11y)")


def load_environment():
    """Load environment variables from .env file"""
    # Look for .env file in current directory and parent directories
    env_file = Path('.env')
    if not env_file.exists():
        # Try parent directory
        env_file = Path('../.env')
    
    if env_file.exists():
        load_dotenv(env_file)
        logger.info(f"Loaded environment variables from {env_file}")
    else:
        logger.warning("No .env file found. Make sure OPENAI_API_KEY is set in environment.")


def estimate_cost(num_issues: int, model_name: str, validation_enabled: bool = False, 
                  validate_only_non_default: bool = True) -> float:
    """Estimate the cost of analysis based on number of issues and model"""
    # Rough estimates based on typical token usage
    cost_per_issue = {
        "gpt-4o": 0.05,      # $0.05 per issue
        "gpt-4": 0.08,       # $0.08 per issue  
        "gpt-3.5-turbo": 0.01 # $0.01 per issue
    }
    
    base_cost = cost_per_issue.get(model_name, 0.05)
    total_cost = num_issues * base_cost
    
    if validation_enabled:
        # Estimate validation cost
        if validate_only_non_default:
            # Assume 30% of sequences are non-default and need validation
            validation_rate = 0.3
        else:
            # Validate all sequences
            validation_rate = 1.0
        
        validation_cost = num_issues * validation_rate * base_cost
        total_cost += validation_cost
    
    return total_cost


def confirm_analysis(config: AnalysisConfig, total_available_issues: int, issues_to_analyze: int) -> bool:
    """Show analysis summary and get user confirmation"""
    estimated_cost = estimate_cost(
        issues_to_analyze, 
        config.model_name, 
        config.enable_validation,
        config.validate_only_non_default
    )
    estimated_time = issues_to_analyze * (config.delay_seconds + 3)  # ~3 seconds per API call
    
    # Add validation time if enabled
    if config.enable_validation:
        validation_rate = 0.3 if config.validate_only_non_default else 1.0
        validation_time = issues_to_analyze * validation_rate * (config.validation_delay_seconds + 3)
        estimated_time += validation_time
    
    logger.info(f"{'='*50}")
    logger.info("ANALYSIS SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Total issues available: {total_available_issues}")
    logger.info(f"Issues to analyze: {issues_to_analyze}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Mode: {'Continue (append to existing)' if config.continue_analysis else 'Start Over (fresh analysis)'}")
    
    if config.enable_validation:
        logger.info(f"Validation: Enabled ({config.validation_model_name})")
        logger.info(f"Validation scope: {'Non-default sequences only' if config.validate_only_non_default else 'All sequences'}")
    else:
        logger.info("Validation: Disabled")
    
    logger.info(f"Estimated cost: ${estimated_cost:.2f}")
    logger.info(f"Estimated time: {estimated_time/60:.1f} minutes")
    logger.info(f"Rate limiting: {config.delay_seconds} seconds between requests")
    
    if config.max_issues and config.max_issues > total_available_issues:
        logger.warning(f"Note: Requested {config.max_issues} issues, but only {total_available_issues} available")
    
    logger.info(f"{'='*50}")
    
    while True:
        confirm = input("Proceed with analysis? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            return True
        elif confirm in ['n', 'no']:
            return False
        else:
            logger.warning("Please enter 'y' or 'n'")
            continue


def get_analysis_config() -> AnalysisConfig:
    """Get analysis configuration with interactive prompts or command line args"""
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        try:
            # Enhanced Command line usage with dataset-aware RQ selection
            
            if sys.argv[1].lower() == 'all':
                max_issues = None
            else:
                max_issues = int(sys.argv[1])
                if max_issues <= 0:
                    raise ValueError("Number of issues must be positive")
            
            model_name = sys.argv[2] if len(sys.argv) > 2 else "gpt-4o"
            
            # Parse continue/start_over parameter
            continue_analysis = False
            if len(sys.argv) > 3:
                mode_arg = sys.argv[3].lower()
                if mode_arg in ['continue', 'c']:
                    continue_analysis = True
                elif mode_arg in ['start_over', 'start', 's']:
                    continue_analysis = False
                else:
                    logger.warning(f"Unknown mode '{mode_arg}', using 'start_over'")
            
            # Parse dataset parameter FIRST to determine available RQs
            dataset_type = "a11y"  # default
            if '--dataset' in sys.argv:
                try:
                    dataset_idx = sys.argv.index('--dataset')
                    if dataset_idx + 1 < len(sys.argv):
                        dataset_arg = sys.argv[dataset_idx + 1].lower()
                        if dataset_arg in ['a11y', 'non_a11y']:
                            dataset_type = dataset_arg
                        else:
                            logger.warning(f"Unknown dataset '{dataset_arg}', using 'a11y'")
                except ValueError:
                    logger.warning("--dataset flag found but no dataset specified, using 'a11y'")
            
            # Set default research questions based on dataset
            if dataset_type == "a11y":
                default_rqs = ["role_sequence", "wcag_categorization", "testing_verification"]
            else:
                default_rqs = ["role_sequence", "testing_verification"]  # No WCAG for non-a11y
            
            research_questions = default_rqs.copy()
            
            # Parse research question parameter with dataset awareness
            if '--rq' in sys.argv:
                try:
                    rq_idx = sys.argv.index('--rq')
                    if rq_idx + 1 < len(sys.argv):
                        rq_arg = sys.argv[rq_idx + 1].lower()
                        
                        if rq_arg == 'role_sequence':
                            research_questions = ["role_sequence"]
                        elif rq_arg == 'wcag_categorization':
                            if dataset_type == "a11y":
                                research_questions = ["wcag_categorization"]
                            else:
                                logger.warning("WCAG categorization not available for non-a11y issues, using role_sequence")
                                research_questions = ["role_sequence"]
                        elif rq_arg == 'testing_verification':
                            research_questions = ["testing_verification"]
                        elif rq_arg == 'participant_influence':  # ADD THIS LINE
                            research_questions = ["participant_influence"]
                        elif rq_arg == 'solution_development':
                            research_questions = ["solution_development"]
                        elif rq_arg in ['both', 'all']:
                            research_questions = default_rqs  # Use dataset-appropriate defaults
                        else:
                            logger.warning(f"Unknown research question '{rq_arg}', using defaults for {dataset_type}")
                            research_questions = default_rqs
                except ValueError:
                    logger.warning("--rq flag found but no research question specified, using defaults")
            
            # Parse validation flags
            enable_validation = '--validate' in sys.argv
            validate_only_non_default = True
            validation_model_name = model_name
            
            if enable_validation:
                try:
                    validate_model_idx = sys.argv.index('--validation-model')
                    if validate_model_idx + 1 < len(sys.argv):
                        validation_model_name = sys.argv[validate_model_idx + 1]
                except ValueError:
                    pass
                
                if '--validate-all' in sys.argv:
                    validate_only_non_default = False
            
            # Validate model names
            valid_models = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
            if model_name not in valid_models:
                logger.warning(f"Unknown model '{model_name}', using 'gpt-4o'")
                model_name = "gpt-4o"
            
            if validation_model_name not in valid_models:
                logger.warning(f"Unknown validation model '{validation_model_name}', using '{model_name}'")
                validation_model_name = model_name
            
            mode_str = "Continue" if continue_analysis else "Start Over"
            validation_str = f"Enabled ({validation_model_name})" if enable_validation else "Disabled"
            rq_str = ", ".join(research_questions)
            logger.info(f"Using command line config: {max_issues or 'ALL'} issues with {model_name}")
            logger.info(f"Dataset: {dataset_type}")
            logger.info(f"Research Questions: {rq_str}")
            logger.info(f"Mode: {mode_str}, Validation: {validation_str}")
            
            config = AnalysisConfig(
                model_name=model_name,
                max_issues=max_issues,
                temperature=0,
                max_tokens=800,
                delay_seconds=0.5,
                continue_analysis=continue_analysis,
                research_questions=research_questions,
                enable_validation=enable_validation,
                validation_model_name=validation_model_name,
                validate_only_non_default=validate_only_non_default,
                validation_delay_seconds=0.5
            )
            
            # Store dataset type in config for later use
            config.dataset_type = dataset_type
            
            return config
            
        except (ValueError, IndexError) as e:
            logger.error(f"Invalid command line arguments: {e}")
            logger.info("Usage: python script.py <number_of_issues|all> [model_name] [continue|start_over] [--rq role_sequence|wcag_categorization|testing_verification|both|all] [--validate] [--validate-all] [--validation-model model_name] [--dataset a11y|non_a11y]")
            logger.info("Note: WCAG categorization is only available for a11y dataset")
            logger.info("Falling back to interactive mode...")
    
    # Enhanced interactive mode with dataset-aware RQ selection
    
    # First ask about dataset to determine available RQs
    logger.info("Dataset Selection:")
    logger.info("1. Accessibility Issues (a11y)")
    logger.info("2. Non-Accessibility Issues (non_a11y)")
    
    dataset_type = "a11y"
    while True:
        dataset_choice = input("Select dataset (1-2): ").strip()
        
        if dataset_choice == "1":
            dataset_type = "a11y"
            logger.info("Selected: Accessibility Issues")
            break
        elif dataset_choice == "2":
            dataset_type = "non_a11y"
            logger.info("Selected: Non-Accessibility Issues")
            break
        else:
            logger.warning("Please enter 1 or 2")
            continue
    
    # Ask about research questions based on dataset
    # In the interactive mode section:
    logger.info("Available Research Questions:")
    logger.info("1. Role Sequence Analysis")
    if dataset_type == "a11y":
        logger.info("2. WCAG 2.2 Categorization") 
    logger.info("3. Testing & Verification")
    logger.info("4. Participant Influence")
    logger.info("5. Solution Development")  # NEW
    logger.info("6. All Available Research Questions")  # UPDATE NUMBER
    
    while True:
        rq_choice = input("Select research questions to run (1-4): ").strip()
        
        if rq_choice == "1":
            research_questions = ["role_sequence"]
            logger.info("Selected: Role Sequence Analysis only")
            break
        elif rq_choice == "2" and dataset_type == "a11y":
            research_questions = ["wcag_categorization"]
            logger.info("Selected: WCAG 2.2 Categorization only")
            break
        elif rq_choice == "2" and dataset_type == "non_a11y":
            logger.warning("WCAG categorization not available for non-a11y issues. Please select 1, 3, or 4")
            continue
        elif rq_choice == "3":
            research_questions = ["testing_verification"]
            logger.info("Selected: Testing & Verification only")
            break
        # Update the selection logic:
        elif rq_choice == "4":
            research_questions = ["participant_influence"]
            logger.info("Selected: Participant Influence only")
            break
        elif rq_choice == "5":
            research_questions = ["solution_development"]
            logger.info("Selected: Solution Development only")
            break
        elif rq_choice == "6":  # Update this
            if dataset_type == "a11y":
                research_questions = ["role_sequence", "wcag_categorization", "testing_verification", "participant_influence", "solution_development"]
                logger.info("Selected: All Research Questions")
            else:
                research_questions = ["role_sequence", "testing_verification", "participant_influence", "solution_development"]
                logger.info("Selected: All Available Research Questions")
            break
        else:
            if dataset_type == "a11y":
                logger.warning("Please enter 1, 2, 3, or 4")
            else:
                logger.warning("Please enter 1, 3, or 4 (WCAG not available for non-a11y)")
            continue
    
    # Continue with the rest of the original interactive config...
    # Ask about continue vs start over
    logger.info("\nAnalysis Mode:")
    logger.info("1. Start Over (analyze all issues from scratch)")
    logger.info("2. Continue (skip already analyzed issues and append new results)")
    
    while True:
        mode_choice = input("Select mode (1-2): ").strip()
        
        if mode_choice == "1":
            continue_analysis = False
            logger.info("Selected: Start Over mode")
            break
        elif mode_choice == "2":
            continue_analysis = True
            logger.info("Selected: Continue mode")
            break
        else:
            logger.warning("Please enter 1 or 2")
            continue
    
    # Get number of issues from user
    while True:
        try:
            user_input = input("\nHow many issues would you like to analyze? (Enter number or 'all'): ").strip().lower()
            
            if user_input == 'all':
                max_issues = None
                logger.info("Will analyze ALL issues")
                break
            else:
                max_issues = int(user_input)
                if max_issues <= 0:
                    logger.warning("Please enter a positive number or 'all'")
                    continue
                logger.info(f"Will analyze {max_issues} issues")
                break
        except ValueError:
            logger.warning("Please enter a valid number or 'all'")
            continue
    
    # Get model choice
    logger.info("Available models:")
    logger.info("1. gpt-4o (recommended, more accurate)")
    logger.info("2. gpt-4 (good balance)")
    logger.info("3. gpt-3.5-turbo (faster, cheaper)")
    
    while True:
        model_choice = input("Select model (1-3) or press Enter for default (gpt-4o): ").strip()
        
        if model_choice == "" or model_choice == "1":
            model_name = "gpt-4o"
            break
        elif model_choice == "2":
            model_name = "gpt-4"
            break
        elif model_choice == "3":
            model_name = "gpt-3.5-turbo"
            break
        else:
            logger.warning("Please enter 1, 2, 3, or press Enter for default")
            continue
    
    logger.info(f"Selected model: {model_name}")
    
    # Ask about validation
    logger.info("\nValidation Options:")
    logger.info("1. No validation (faster, cheaper)")
    logger.info("2. Validate non-default sequences only (recommended)")
    logger.info("3. Validate all sequences (thorough, more expensive)")
    
    enable_validation = False
    validate_only_non_default = True
    validation_model_name = model_name
    
    while True:
        validation_choice = input("Select validation option (1-3) or press Enter for default (1): ").strip()
        
        if validation_choice == "" or validation_choice == "1":
            enable_validation = False
            logger.info("Selected: No validation")
            break
        elif validation_choice == "2":
            enable_validation = True
            validate_only_non_default = True
            logger.info("Selected: Validate non-default sequences only")
            break
        elif validation_choice == "3":
            enable_validation = True
            validate_only_non_default = False
            logger.info("Selected: Validate all sequences")
            break
        else:
            logger.warning("Please enter 1, 2, 3, or press Enter for default")
            continue
    
    # If validation enabled, ask about validation model
    if enable_validation:
        logger.info(f"\nValidation Model (current analysis model: {model_name}):")
        logger.info("1. Same as analysis model")
        logger.info("2. gpt-4o")
        logger.info("3. gpt-4")
        logger.info("4. gpt-3.5-turbo")
        
        while True:
            val_model_choice = input("Select validation model (1-4) or press Enter for same: ").strip()
            
            if val_model_choice == "" or val_model_choice == "1":
                validation_model_name = model_name
                break
            elif val_model_choice == "2":
                validation_model_name = "gpt-4o"
                break
            elif val_model_choice == "3":
                validation_model_name = "gpt-4"
                break
            elif val_model_choice == "4":
                validation_model_name = "gpt-3.5-turbo"
                break
            else:
                logger.warning("Please enter 1, 2, 3, 4, or press Enter for same")
                continue
        
        logger.info(f"Selected validation model: {validation_model_name}")
    
    config = AnalysisConfig(
        model_name=model_name,
        max_issues=max_issues,
        temperature=0,
        max_tokens=800,
        delay_seconds=1.0,
        continue_analysis=continue_analysis,
        research_questions=research_questions,
        enable_validation=enable_validation,
        validation_model_name=validation_model_name,
        validate_only_non_default=validate_only_non_default,
        validation_delay_seconds=0.5
    )
    
    # Store dataset type in config
    config.dataset_type = dataset_type
    
    return config


def main():
    """Main function with configurable research question selection"""
    
    """Enhanced main function with dataset selection"""
    
    # Load environment variables from .env file
    load_environment()
    
    # Get configuration (interactive or command line)
    config = get_analysis_config()
    
    # Determine input file based on dataset type
    if getattr(config, 'dataset_type', 'a11y') == 'non_a11y':
        input_file = "Data/detailed_similar_non_a11y_issues.json"
        output_suffix = "_non_a11y"
        logger.info("Analyzing NON-ACCESSIBILITY issues")
    else:
        input_file = "Data/processed_issues_rq1.json"
        output_suffix = ""
        logger.info("Analyzing ACCESSIBILITY issues")
    
    logger.info(f"Loading issues from {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            issues = json.load(f)
        logger.info(f"Loaded {len(issues)} issues")
    except FileNotFoundError:
        logger.error(f"Error: {input_file} not found. Please ensure the file exists.")
        return
    
    # Initialize analyzer and register research questions based on configuration and dataset
    analyzer = MoodleIssueAnalyzer(config)

    if getattr(config, 'dataset_type', 'a11y') == 'non_a11y':
        analyzer.output_dir = Path("Data/analysis_results_non_a11y")
    else:
        analyzer.output_dir = Path("Data/analysis_results")
    
    # In main() function, find this section and update:
    if "participant_influence" in config.research_questions:
        if getattr(config, 'dataset_type', 'a11y') == 'a11y':
            analyzer.register_research_question(ParticipantInfluenceQuestion())
            logger.info("Registered: Participant Influence Analysis (A11y)")
        else:
            analyzer.register_research_question(NonA11yParticipantInfluenceQuestion())
            logger.info("Registered: Participant Influence Analysis (Non-A11y)")
    
    # Register research questions based on configuration and dataset type
    if "role_sequence" in config.research_questions:
        if getattr(config, 'dataset_type', 'a11y') == 'a11y':
            analyzer.register_research_question(RoleSequenceQuestion())
            logger.info("Registered: Role Sequence Analysis (A11y)")
        else:
            analyzer.register_research_question(NonA11yRoleSequenceQuestion())
            logger.info("Registered: Role Sequence Analysis (Non-A11y)")
    
    # Only register WCAG for accessibility issues
    if "wcag_categorization" in config.research_questions and getattr(config, 'dataset_type', 'a11y') == 'a11y':
        analyzer.register_research_question(WCAGCategoryQuestion())
        logger.info("Registered: WCAG 2.2 Categorization")
    elif "wcag_categorization" in config.research_questions:
        logger.warning("WCAG categorization skipped for non-accessibility issues")
    
    if "testing_verification" in config.research_questions:
        if getattr(config, 'dataset_type', 'a11y') == 'a11y':
            analyzer.register_research_question(TestingVerificationQuestion())
            logger.info("Registered: Testing & Verification Analysis (A11y)")
        else:
            analyzer.register_research_question(NonA11yTestingVerificationQuestion())
            logger.info("Registered: Testing & Verification Analysis (Non-A11y)")
    
    if "solution_development" in config.research_questions:
        if getattr(config, 'dataset_type', 'a11y') == 'a11y':
            analyzer.register_research_question(SolutionDevelopmentQuestion())
            logger.info("Registered: Solution Development Analysis (A11y)")
        else:
            analyzer.register_research_question(NonA11ySolutionDevelopmentQuestion())
            logger.info("Registered: Solution Development Analysis (Non-A11y)")
    
    
    
    # Calculate how many issues will actually be analyzed
    if config.continue_analysis:
        total_to_analyze = 0
        for question_id in analyzer.research_questions:
            unanalyzed = analyzer._filter_unanalyzed_issues(issues, question_id)
            if config.max_issues:
                unanalyzed = unanalyzed[:config.max_issues]
            total_to_analyze = max(total_to_analyze, len(unanalyzed))
    else:
        total_to_analyze = min(config.max_issues or len(issues), len(issues))
    
    # Show analysis summary and get confirmation
    if not confirm_analysis(config, len(issues), total_to_analyze):
        logger.info("Analysis cancelled by user")
        return
    
    # Run analysis for selected research questions only
    all_results = analyzer.analyze_all_questions(issues)
    
    # Generate summaries
    summaries = analyzer.generate_summaries(all_results)
    
    # Save results with dataset-specific naming
    output_dir = f"Data/analysis_results{output_suffix}"
    analyzer.save_results(all_results, summaries, output_dir)
    
    # Print key findings
    dataset_label = "NON-ACCESSIBILITY" if getattr(config, 'dataset_type', 'a11y') == 'non_a11y' else "ACCESSIBILITY"
    logger.info(f"{'='*60}")
    logger.info(f"{dataset_label} ANALYSIS COMPLETE - KEY FINDINGS")
    logger.info(f"{'='*60}")
    
    for question_id, summary in summaries.items():
        logger.info(f"\n{question_id.upper()}:")
        
        if question_id == "role_sequence" and 'most_common_role_sequences' in summary:
            logger.info("Top role sequences:")
            for i, seq_data in enumerate(summary['most_common_role_sequences'][:5], 1):
                sequence_str = " → ".join(seq_data['sequence'])
                logger.info(f"  {i}. {sequence_str} ({seq_data['count']} issues, {seq_data['percentage']}%)")
        
        elif question_id == "wcag_categorization" and 'most_violated_success_criteria' in summary:
            logger.info("Most violated WCAG 2.2 Success Criteria:")
            for i, sc_data in enumerate(summary['most_violated_success_criteria'][:5], 1):
                logger.info(f"  {i}. {sc_data['success_criterion']} ({sc_data['count']} violations, {sc_data['percentage']}%)")
        
        elif question_id == "testing_verification" and 'most_used_tools_techniques' in summary:
            logger.info("Most used testing tools/techniques:")
            for i, tool_data in enumerate(summary['most_used_tools_techniques'][:5], 1):
                logger.info(f"  {i}. {tool_data['tool_technique']} ({tool_data['count']} uses, {tool_data['percentage']}%)")
            
            overview = summary.get('testing_overview', {})
            logger.info(f"\nOverview: {overview.get('issues_with_testing_methods', 0)} issues with testing methods")
        elif question_id in ["solution_development", "non_a11y_solution_development"] and 'approach_type_distribution' in summary:
            logger.info("Most common solution development approaches:")
            for i, approach_data in enumerate(summary['approach_type_distribution'][:5], 1):
                logger.info(f"  {i}. {approach_data['approach_type']} ({approach_data['count']} uses, {approach_data['percentage']}%)")
            
            logger.info("Most common reasoning patterns:")
            for i, pattern_data in enumerate(summary['reasoning_pattern_distribution'][:5], 1):
                logger.info(f"  {i}. {pattern_data['reasoning_pattern']} ({pattern_data['count']} uses, {pattern_data['percentage']}%)")
            
            overview = summary.get('solution_development_overview', {})
            logger.info(f"\nOverview: {overview.get('issues_with_solution_approaches', 0)} issues with solution development approaches")
        
        # Show validation statistics
        if 'validation_statistics' in summary:
            val_stats = summary['validation_statistics']
            if val_stats['total_validations_performed'] > 0:
                logger.info(f"\nValidation Statistics:")
                logger.info(f"  Total validations: {val_stats['total_validations_performed']}")
                logger.info(f"  Agreements: {val_stats['validation_agreements']}")
                logger.info(f"  Disagreements: {val_stats['validation_disagreements']}")
