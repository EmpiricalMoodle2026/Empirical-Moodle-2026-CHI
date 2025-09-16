import re
import json
import logging
import requests
import time
from typing import Dict, List, Any, Set, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Load environment variables
import os
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Retrieve credentials from .env file
USERNAME = os.getenv("USER_NAME")
PASSWORD = os.getenv("PASS_WORD")

# Thread-safe locks for API fetching
api_lock = Lock()
progress_lock = Lock()

@dataclass
class FilterConfig:
    """Configuration for filtering issues and timeline events for specific research questions"""
    name: str
    description: str
    # Issue-level filters
    created_date_min: datetime = None
    excluded_issue_types: Set[str] = None
    required_resolution_types: Set[str] = None
    # Timeline event field filters
    important_timeline_fields: Set[str] = None
    
    def __post_init__(self):
        # Set defaults if not provided
        if self.excluded_issue_types is None:
            self.excluded_issue_types = set()
        if self.required_resolution_types is None:
            self.required_resolution_types = set()
        if self.important_timeline_fields is None:
            # Default timeline fields for role sequence analysis
            self.important_timeline_fields = {
                'status', 'assignee', 'Peer reviewer', 'Integrator', 'Tester',
                'Testing Instructions', 'description', 'Attachment'
            }

# Predefined filter configurations for research questions
RESEARCH_QUESTION_FILTERS = {
    "RQ2-3": FilterConfig(
        name="RQ2_Analysis",
        description="Qualitative content analysis for accessibility issues",
        created_date_min=datetime(2019, 11, 19),
        excluded_issue_types={"Epic", "Task"},
        required_resolution_types={"Fixed"},
        important_timeline_fields={
            # Critical - role transitions and workflow
            'status', 'assignee', 'Peer reviewer', 'Integrator', 'Tester',
            # Moderately Important - progress indicators
            'Testing Instructions', 'description',
            # Attachment - evidence of work/progress
            'Attachment'
        }
    ),
}

class SimilarIssueFinder:
    """Find similar non-accessibility issues for accessibility issues"""
    
    def __init__(self):
        self.a11y_issue_keys = set()
        self.a11y_issues = {}
        self.non_a11y_issues = {}
        
    def load_data_files(self, processed_rq1_file: str, a11y_enhanced_file: str, 
                       non_a11y_file: str) -> bool:
        """Load all required data files"""
        try:
            # 1. Load processed RQ1 file to get accessibility issue keys
            logger.info(f"Loading processed RQ1 file: {processed_rq1_file}")
            with open(processed_rq1_file, 'r', encoding='utf-8') as f:
                processed_issues = json.load(f)
                self.a11y_issue_keys = {issue['issue_key'] for issue in processed_issues}
            logger.info(f"Loaded {len(self.a11y_issue_keys)} accessibility issue keys")
            
            # 2. Load enhanced accessibility issues file
            logger.info(f"Loading enhanced accessibility issues: {a11y_enhanced_file}")
            with open(a11y_enhanced_file, 'r', encoding='utf-8') as f:
                a11y_data = json.load(f)
                self.a11y_issues = {issue['issue_key']: issue for issue in a11y_data}
            logger.info(f"Loaded {len(self.a11y_issues)} enhanced accessibility issues")
            
            # 3. Load non-accessibility issues file
            logger.info(f"Loading non-accessibility issues: {non_a11y_file}")
            with open(non_a11y_file, 'r', encoding='utf-8') as f:
                non_a11y_data = json.load(f)
                # Filter out any issues that might be in the a11y dataset and only keep "Fixed" issues
                self.non_a11y_issues = {
                    issue['issue_key']: issue 
                    for issue in non_a11y_data 
                    if (issue['issue_key'] not in self.a11y_issues and 
                        issue.get('resolution_type') == 'Fixed')
                }
            logger.info(f"Loaded {len(self.non_a11y_issues)} non-accessibility issues (filtered)")
            
            return True
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading data files: {e}")
            return False
    
    def _create_non_a11y_index(self) -> Dict[Tuple, List[Dict[str, Any]]]:
        """Create an index of non-a11y issues by matching criteria (priority, issue_type, and affected_versions)"""
        exact_match_index = defaultdict(list)
        priority_type_index = defaultdict(list)  # For fallback intersection matching
        
        for issue_key, issue in self.non_a11y_issues.items():
            priority = issue.get('priority')
            issue_type = issue.get('issue_type')
            affected_versions = issue.get('affected_versions', [])
            
            # Index for exact affected versions matching
            affected_versions_key = tuple(sorted(affected_versions)) if affected_versions else ()
            exact_criteria_key = (priority, issue_type, affected_versions_key)
            exact_match_index[exact_criteria_key].append(issue)
            
            # Index for fallback intersection matching (just priority + issue_type)
            fallback_criteria_key = (priority, issue_type)
            priority_type_index[fallback_criteria_key].append(issue)
        
        logger.info(f"Created exact match index with {len(exact_match_index)} unique criteria combinations")
        logger.info(f"Created fallback index with {len(priority_type_index)} unique criteria combinations")
        
        # Log the distribution of exact match criteria combinations for debugging
        logger.info("Exact match index distribution (top 10):")
        for criteria, issues in sorted(exact_match_index.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            affected_versions_display = list(criteria[2]) if criteria[2] else "None"
            logger.info(f"  Priority: {criteria[0]}, Issue Type: {criteria[1]}, Affected Versions: {affected_versions_display} -> {len(issues)} issues")
        
        return {
            'exact_match': dict(exact_match_index),
            'fallback': dict(priority_type_index)
        }
    
    def _find_available_matching_issues(self, a11y_issue: Dict[str, Any], 
                                       non_a11y_index: Dict[str, Dict], 
                                       used_non_a11y_keys: set) -> Tuple[List[Dict[str, Any]], str]:
        """Find non-a11y issues that match the criteria and haven't been used yet
        
        Returns:
            Tuple of (available_matches, match_type) where match_type is 'exact' or 'intersection'
        """
        
        priority = a11y_issue.get('priority')
        issue_type = a11y_issue.get('issue_type')
        affected_versions = set(a11y_issue.get('affected_versions', []))
        
        # Try exact match first
        affected_versions_key = tuple(sorted(affected_versions)) if affected_versions else ()
        exact_criteria_key = (priority, issue_type, affected_versions_key)
        
        exact_matches = non_a11y_index['exact_match'].get(exact_criteria_key, [])
        exact_available = [
            issue for issue in exact_matches 
            if issue['issue_key'] not in used_non_a11y_keys
        ]
        
        if exact_available:
            logger.debug(f"Found {len(exact_available)} exact matches for {a11y_issue.get('issue_key')}")
            return exact_available, 'exact'
        
        # Fallback to intersection matching
        fallback_criteria_key = (priority, issue_type)
        potential_matches = non_a11y_index['fallback'].get(fallback_criteria_key, [])
        
        intersection_matches = []
        for issue in potential_matches:
            if issue['issue_key'] in used_non_a11y_keys:
                continue
                
            issue_affected_versions = set(issue.get('affected_versions', []))
            
            # Check for intersection (both must have affected versions and share at least one)
            if affected_versions and issue_affected_versions:
                intersection = affected_versions.intersection(issue_affected_versions)
                if intersection:
                    # Add intersection info to the issue for logging
                    issue_copy = issue.copy()
                    issue_copy['_intersection_versions'] = list(intersection)
                    intersection_matches.append(issue_copy)
            elif not affected_versions and not issue_affected_versions:
                # Both have no affected versions - consider this a match
                intersection_matches.append(issue)
        
        if intersection_matches:
            logger.debug(f"Found {len(intersection_matches)} intersection matches for {a11y_issue.get('issue_key')} after exact match failed")
            return intersection_matches, 'intersection'
        
        # No matches found
        affected_versions_display = list(affected_versions) if affected_versions else "None"
        logger.debug(f"No matches (exact or intersection) for {a11y_issue.get('issue_key')} with criteria:")
        logger.debug(f"  Priority: '{priority}'")
        logger.debug(f"  Issue Type: '{issue_type}'") 
        logger.debug(f"  Affected Versions: {affected_versions_display}")
        logger.debug(f"  Exact matches found: {len(exact_matches)}")
        logger.debug(f"  Potential fallback candidates: {len(potential_matches)}")
        
        return [], 'none'
    
    def find_similar_issues(self) -> Dict[str, Dict[str, Any]]:
        """
        Find one random similar non-accessibility issue for each accessibility issue
        Uses exact match first, then falls back to intersection matching
        Ensures no non-a11y issue is used more than once
        
        Returns:
            Dictionary mapping a11y issue_key to single similar non-a11y issue
        """
        import random
        
        similar_issues = {}
        used_non_a11y_keys = set()  # Track which non-a11y issues have been used
        
        # Create index of non-a11y issues by matching criteria for faster lookup
        non_a11y_index = self._create_non_a11y_index()
        
        logger.info("Finding one random similar issue per accessibility issue...")
        processed_count = 0
        exact_matched_count = 0
        intersection_matched_count = 0
        no_match_count = 0
        
        # Convert to list and shuffle for random processing order
        a11y_keys_list = list(self.a11y_issue_keys)
        random.shuffle(a11y_keys_list)
        
        # Process only the accessibility issues that are in RQ1
        for issue_key in a11y_keys_list:
            if issue_key not in self.a11y_issues:
                logger.warning(f"Issue {issue_key} from RQ1 not found in enhanced a11y data")
                continue
                
            a11y_issue = self.a11y_issues[issue_key]
            available_matches, match_type = self._find_available_matching_issues(
                a11y_issue, non_a11y_index, used_non_a11y_keys
            )
            
            if available_matches:
                # Randomly select one from available matches
                selected_issue = random.choice(available_matches)
                # Add match type to the selected issue for tracking
                selected_issue['_match_type'] = match_type
                if match_type == 'intersection' and '_intersection_versions' in selected_issue:
                    selected_issue['_intersection_versions'] = selected_issue['_intersection_versions']
                
                similar_issues[issue_key] = selected_issue
                used_non_a11y_keys.add(selected_issue['issue_key'])
                
                if match_type == 'exact':
                    exact_matched_count += 1
                    logger.debug(f"Exact match: {issue_key} with {selected_issue['issue_key']}")
                elif match_type == 'intersection':
                    intersection_matched_count += 1
                    intersection_versions = selected_issue.get('_intersection_versions', [])
                    logger.debug(f"Intersection match: {issue_key} with {selected_issue['issue_key']} (shared versions: {intersection_versions})")
            else:
                # No available matches (either no matches at all, or all matches already used)
                similar_issues[issue_key] = None
                no_match_count += 1
                logger.debug(f"No available match found for {issue_key}")
            
            processed_count += 1
            
            if processed_count % 50 == 0:
                logger.info(f"Processed {processed_count}/{len(self.a11y_issue_keys)} accessibility issues - "
                           f"Exact: {exact_matched_count}, Intersection: {intersection_matched_count}, No match: {no_match_count}")
        
        total_matched = exact_matched_count + intersection_matched_count
        logger.info(f"Matching completed:")
        logger.info(f"  Total a11y issues: {len(self.a11y_issue_keys)}")
        logger.info(f"  Successfully matched: {total_matched}")
        logger.info(f"    - Exact matches: {exact_matched_count}")
        logger.info(f"    - Intersection matches: {intersection_matched_count}")
        logger.info(f"  No match available: {no_match_count}")
        logger.info(f"  Unique non-a11y issues used: {len(used_non_a11y_keys)}")
        
        return similar_issues
    
    def save_similar_issues_mapping(self, similar_issues: Dict[str, Dict[str, Any]], 
                                   output_file: str = "Data/similar_issues_mapping.json"):
        """Save the mapping of accessibility issues to similar non-accessibility issues"""
        
        # Create summary statistics
        successful_matches = {k: v for k, v in similar_issues.items() if v is not None}
        failed_matches = {k: v for k, v in similar_issues.items() if v is None}
        
        stats = {
            'total_a11y_issues': len(similar_issues),
            'successful_matches': len(successful_matches),
            'failed_matches': len(failed_matches),
            'unique_non_a11y_used': len(set(issue['issue_key'] for issue in successful_matches.values()))
        }
        
        # Prepare output data
        output_data = {
            'metadata': {
                'description': 'One-to-one mapping of accessibility issues to similar non-accessibility issues',
                'matching_criteria': ['priority', 'issue_type', 'affected_versions'],
                'matching_strategy': 'Exact match first, then intersection fallback with no reuse',
                'statistics': stats,
                'generated_at': datetime.now().isoformat()
            },
            'similar_issues_mapping': similar_issues
        }
        
        # Save to file
        logger.info(f"Saving similar issues mapping to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Log statistics
        logger.info("Similar Issues Statistics:")
        logger.info(f"  Total a11y issues processed: {stats['total_a11y_issues']}")
        logger.info(f"  Successful matches: {stats['successful_matches']}")
        logger.info(f"  Failed matches (no available non-a11y): {stats['failed_matches']}")
        logger.info(f"  Unique non-a11y issues used: {stats['unique_non_a11y_used']}")
        
        return output_data
    
    def create_detailed_comparison_dataset(self, similar_issues: Dict[str, Dict[str, Any]], 
                                         output_file: str = "Data/a11y_vs_non_a11y_comparison.json"):
        """Create a detailed dataset for comparing a11y and non-a11y issues"""
        
        comparison_pairs = []
        
        for a11y_key, non_a11y_issue in similar_issues.items():
            # Skip if no matching non-a11y issue was found
            if non_a11y_issue is None:
                continue
                
            a11y_issue = self.a11y_issues[a11y_key]
            
            comparison_pair = {
                'pair_id': f"{a11y_key}_vs_{non_a11y_issue['issue_key']}",
                'a11y_issue': {
                    'issue_key': a11y_issue['issue_key'],
                    'title': a11y_issue.get('title', ''),
                    'priority': a11y_issue.get('priority'),
                    'issue_type': a11y_issue.get('issue_type'),
                    'affected_versions': a11y_issue.get('affected_versions', []),
                    'resolution_type': a11y_issue.get('resolution_type'),
                    'created_date': a11y_issue.get('created_date'),
                    'resolved_date': a11y_issue.get('resolved_date')
                },
                'non_a11y_issue': {
                    'issue_key': non_a11y_issue['issue_key'],
                    'title': non_a11y_issue.get('title', ''),
                    'priority': non_a11y_issue.get('priority'),
                    'issue_type': non_a11y_issue.get('issue_type'),
                    'affected_versions': non_a11y_issue.get('affected_versions', []),
                    'resolution_type': non_a11y_issue.get('resolution_type'),
                    'created_date': non_a11y_issue.get('created_date'),
                    'resolved_date': non_a11y_issue.get('resolved_date')
                },
                'matching_criteria': {
                    'priority_match': a11y_issue.get('priority') == non_a11y_issue.get('priority'),
                    'issue_type_match': a11y_issue.get('issue_type') == non_a11y_issue.get('issue_type'),
                    'affected_versions_match': (sorted(a11y_issue.get('affected_versions', [])) == 
                                               sorted(non_a11y_issue.get('affected_versions', [])))
                }
            }
            comparison_pairs.append(comparison_pair)
        
        output_data = {
            'metadata': {
                'description': 'Detailed one-to-one comparison dataset of accessibility vs non-accessibility issues',
                'total_pairs': len(comparison_pairs),
                'pairing_strategy': 'Random selection with no reuse of non-a11y issues',
                'generated_at': datetime.now().isoformat()
            },
            'comparison_pairs': comparison_pairs
        }
        
        logger.info(f"Saving detailed comparison dataset to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created {len(comparison_pairs)} comparison pairs")
        return output_data

    def analyze_matching_potential(self):
        """Analyze the matching potential between a11y and non-a11y issues for debugging"""
        logger.info("="*50)
        logger.info("MATCHING POTENTIAL ANALYSIS")
        logger.info("="*50)
        
        # Analyze a11y issues criteria distribution
        a11y_criteria_dist = defaultdict(int)
        for issue_key in self.a11y_issue_keys:
            if issue_key in self.a11y_issues:
                issue = self.a11y_issues[issue_key]
                affected_versions = issue.get('affected_versions', [])
                affected_versions_key = tuple(sorted(affected_versions)) if affected_versions else ()
                criteria = (issue.get('priority'), issue.get('issue_type'), affected_versions_key)
                a11y_criteria_dist[criteria] += 1
        
        # Analyze non-a11y issues criteria distribution
        non_a11y_criteria_dist = defaultdict(int)
        for issue in self.non_a11y_issues.values():
            affected_versions = issue.get('affected_versions', [])
            affected_versions_key = tuple(sorted(affected_versions)) if affected_versions else ()
            criteria = (issue.get('priority'), issue.get('issue_type'), affected_versions_key)
            non_a11y_criteria_dist[criteria] += 1
        
        # Find overlapping criteria
        overlapping_criteria = set(a11y_criteria_dist.keys()) & set(non_a11y_criteria_dist.keys())
        
        logger.info(f"A11y issues criteria combinations: {len(a11y_criteria_dist)}")
        logger.info(f"Non-a11y issues criteria combinations: {len(non_a11y_criteria_dist)}")
        logger.info(f"Overlapping criteria combinations: {len(overlapping_criteria)}")
        
        logger.info("\nTop 10 A11y criteria combinations:")
        for criteria, count in sorted(a11y_criteria_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
            available_non_a11y = non_a11y_criteria_dist.get(criteria, 0)
            affected_versions_display = list(criteria[2]) if criteria[2] else "None"
            logger.info(f"  Priority: {criteria[0]}, Issue Type: {criteria[1]}, Affected Versions: {affected_versions_display}")
            logger.info(f"    -> {count} a11y issues, {available_non_a11y} non-a11y available")
        
        logger.info("\nTop 10 Non-A11y criteria combinations:")
        for criteria, count in sorted(non_a11y_criteria_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
            a11y_demand = a11y_criteria_dist.get(criteria, 0)
            affected_versions_display = list(criteria[2]) if criteria[2] else "None"
            logger.info(f"  Priority: {criteria[0]}, Issue Type: {criteria[1]}, Affected Versions: {affected_versions_display}")
            logger.info(f"    -> {count} non-a11y issues, {a11y_demand} a11y demand")
        
        # Calculate theoretical matching potential
        total_possible_matches = 0
        for criteria in overlapping_criteria:
            a11y_count = a11y_criteria_dist[criteria]
            non_a11y_count = non_a11y_criteria_dist[criteria]
            possible_matches = min(a11y_count, non_a11y_count)
            total_possible_matches += possible_matches
        
        logger.info(f"\nTheoretical matching analysis:")
        logger.info(f"  Total a11y issues: {len(self.a11y_issue_keys)}")
        logger.info(f"  Total non-a11y issues: {len(self.non_a11y_issues)}")
        logger.info(f"  Maximum possible matches: {total_possible_matches}")
        logger.info(f"  Matching rate potential: {total_possible_matches / len(self.a11y_issue_keys) * 100:.1f}%")
        
        return {
            'a11y_criteria_dist': dict(a11y_criteria_dist),
            'non_a11y_criteria_dist': dict(non_a11y_criteria_dist),
            'overlapping_criteria': overlapping_criteria,
            'max_possible_matches': total_possible_matches
        }

def convert_to_email(email_address) -> Optional[str]:
    """Convert email address format, handling various input types"""
    # Handle None case
    if email_address is None:
        return None
    
    # Handle dictionary case (if the field contains a dict instead of string)
    if isinstance(email_address, dict):
        # Try to extract email from common dict structures
        if 'emailAddress' in email_address:
            return email_address['emailAddress']
        elif 'email' in email_address:
            return email_address['email']
        else:
            return None
    
    # Handle string case
    if isinstance(email_address, str):
        return email_address.strip() if email_address.strip() else None
    
    # Handle other types by converting to string
    try:
        return str(email_address) if email_address else None
    except:
        return None


def fetch_issue_data(issue_id: str = "mdl-85220") -> Dict:
    """
    Fetch issue data from Moodle tracker API.
    
    Args:
        issue_id: Issue ID (default: "mdl-85220")
        
    Returns:
        JSON response from API or empty dict if error
    """
    url = f"https://tracker.moodle.org/rest/api/2/issue/{issue_id}"
    params = {
        'expand': 'changelog,transitions',
        'fields': '*all'
    }
    
    try:
        logging.info(f"Fetching data for issue {issue_id}")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        logging.info(f"Successfully fetched data for issue {issue_id}")
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Error fetching issue {issue_id}: {e}")
        return {}


def get_changelog_info(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract changelog information from issue data"""
    changelog = json_data.get('changelog', {})
    histories = changelog.get('histories', [])
    
    changelog_summary = {
        'total_entries': len(histories),
        'total_from_api': changelog.get('total', 0),
        'status_changes': [],
        'assignee_changes': [],
        'time_logs': [],
        'contributors': [],
        'timeline': []
    }
    
    # Process each history entry
    for history in histories:
        timestamp = history.get('created')
        author_info = history.get('author', {})
        
        # Extract author name properly
        if isinstance(author_info, dict):
            author_name = author_info.get('displayName') or author_info.get('name') or 'Unknown'
        elif isinstance(author_info, str):
            author_name = author_info
        else:
            author_name = 'Unknown'
        
        # Add to contributors list (avoid duplicates)
        if author_name not in changelog_summary['contributors']:
            changelog_summary['contributors'].append(author_name)
        
        # Process each change item in this history entry
        for item in history.get('items', []):
            field = item.get('field')
            from_value = item.get('fromString', '')
            to_value = item.get('toString', '')
            
            # Track status changes
            if field == 'status':
                changelog_summary['status_changes'].append({
                    'timestamp': timestamp,
                    'author': author_name,
                    'from_status': from_value,
                    'to_status': to_value
                })
            
            # Track assignee changes
            elif field == 'assignee':
                changelog_summary['assignee_changes'].append({
                    'timestamp': timestamp,
                    'author': author_name,
                    'from_assignee': from_value,
                    'to_assignee': to_value
                })
            
            # Track time spent changes
            elif field == 'timespent':
                changelog_summary['time_logs'].append({
                    'timestamp': timestamp,
                    'author': author_name,
                    'time_spent_seconds': int(to_value) if to_value and to_value.isdigit() else 0
                })
            
            # Add to general timeline
            changelog_summary['timeline'].append({
                'timestamp': timestamp,
                'author': author_name,  # Use extracted name
                'field': field,
                'from_value': from_value,
                'to_value': to_value,
                'change_description': f"{field}: {from_value} → {to_value}" if from_value else f"{field}: {to_value}"
            })
    
    changelog_summary['status_changes'].sort(key=lambda x: x.get('timestamp') or '')
    changelog_summary['assignee_changes'].sort(key=lambda x: x.get('timestamp') or '')
    changelog_summary['time_logs'].sort(key=lambda x: x.get('timestamp') or '')
    changelog_summary['timeline'].sort(key=lambda x: x.get('timestamp') or '')

    return changelog_summary


def extract_issue_details_from_api(issue_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract detailed issue information from Moodle API response"""
    fields = issue_data.get("fields", {}) or {}
    
    # Extract comments
    comment_section = fields.get("comment", {})
    comments = comment_section.get("comments", [])
    
    # Extract changelog
    changelog_summary = get_changelog_info(issue_data)
    
    # Helper function to safely get display name
    def safe_get_display_name(field_data):
        if field_data is None:
            return None
        if isinstance(field_data, dict):
            return field_data.get("displayName")
        if isinstance(field_data, str):
            return field_data
        return str(field_data) if field_data else None
    
    # Helper function to safely get email address
    def safe_get_email(field_data):
        if field_data is None:
            return None
        if isinstance(field_data, dict):
            return convert_to_email(field_data.get("emailAddress"))
        return convert_to_email(field_data)
    
    # Extract basic fields with defensive programming
    extracted_data = {
        "issue_key": issue_data.get("key"),
        "priority": (fields.get("priority", {}) or {}).get("name") if fields.get("priority") else None,
        "title": fields.get("summary"),
        "test_instructions": fields.get("customfield_10214"),
        "description": fields.get("description"),
        "assignee": safe_get_display_name(fields.get("assignee")),
        "assignee_email": safe_get_email(fields.get("assignee")),
        "reporter": safe_get_display_name(fields.get("reporter")),
        "reporter_email": safe_get_email(fields.get("reporter")),
        "peer_reviewer": safe_get_display_name(fields.get("customfield_10179")),
        "peer_reviewer_email": safe_get_email(fields.get("customfield_10179")),
        "tester_name": safe_get_display_name(fields.get("customfield_10242")),
        "tester_email": safe_get_email(fields.get("customfield_10242")),
        "integrator": safe_get_display_name(fields.get("customfield_10224")),
        "integrator_email": safe_get_email(fields.get("customfield_10224")),
        "comments": comments,
        "changelog_summary": changelog_summary
    }
    
    return extracted_data


@dataclass
class DetailedFetchStats:
    """Statistics for the detailed fetch process"""
    total_issues: int = 0
    processed: int = 0
    successful_fetches: int = 0
    api_errors: int = 0
    start_time: float = 0

    def __init__(self):
        self.a11y_issue_keys = set()
        self.a11y_issues = {}
        self.non_a11y_issues = {}

    def load_data_files(self, processed_rq1_file: str, a11y_enhanced_file: str,
                        non_a11y_file: str) -> bool:
        """Load all required data files"""
        try:
            # 1. Load processed RQ1 file to get accessibility issue keys
            logger.info(f"Loading processed RQ1 file: {processed_rq1_file}")
            with open(processed_rq1_file, 'r', encoding='utf-8') as f:
                processed_issues = json.load(f)
                self.a11y_issue_keys = {issue['issue_key'] for issue in processed_issues}
            logger.info(f"Loaded {len(self.a11y_issue_keys)} accessibility issue keys")

            # 2. Load enhanced accessibility issues file
            logger.info(f"Loading enhanced accessibility issues: {a11y_enhanced_file}")
            with open(a11y_enhanced_file, 'r', encoding='utf-8') as f:
                a11y_data = json.load(f)
                self.a11y_issues = {issue['issue_key']: issue for issue in a11y_data}
            logger.info(f"Loaded {len(self.a11y_issues)} enhanced accessibility issues")

            # 3. Load non-accessibility issues file
            logger.info(f"Loading non-accessibility issues: {non_a11y_file}")
            with open(non_a11y_file, 'r', encoding='utf-8') as f:
                non_a11y_data = json.load(f)
                # Filter out any issues that might be in the a11y dataset and only keep "Fixed" issues
                self.non_a11y_issues = {
                    issue['issue_key']: issue
                    for issue in non_a11y_data
                    if (issue['issue_key'] not in self.a11y_issues and
                        issue.get('resolution_type') == 'Fixed')
                }
            logger.info(f"Loaded {len(self.non_a11y_issues)} non-accessibility issues (filtered)")

            return True

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading data files: {e}")
            return False

    def find_similar_issues(self) -> Dict[str, Dict[str, Any]]:
        """
        Find one random similar non-accessibility issue for each accessibility issue
        Ensures no non-a11y issue is used more than once

        Returns:
            Dictionary mapping a11y issue_key to single similar non-a11y issue
        """
        import random

        similar_issues = {}
        used_non_a11y_keys = set()  # Track which non-a11y issues have been used

        # Create index of non-a11y issues by matching criteria for faster lookup
        non_a11y_index = self._create_non_a11y_index()

        logger.info("Finding one random similar issue per accessibility issue...")
        processed_count = 0
        matched_count = 0
        no_match_count = 0

        # Convert to list and shuffle for random processing order
        a11y_keys_list = list(self.a11y_issue_keys)
        random.shuffle(a11y_keys_list)

        # Process only the accessibility issues that are in RQ1
        for issue_key in a11y_keys_list:
            if issue_key not in self.a11y_issues:
                logger.warning(f"Issue {issue_key} from RQ1 not found in enhanced a11y data")
                continue

            a11y_issue = self.a11y_issues[issue_key]
            available_matches = self._find_available_matching_issues(a11y_issue, non_a11y_index, used_non_a11y_keys)

            if available_matches:
                # Randomly select one from available matches
                selected_issue = random.choice(available_matches)
                similar_issues[issue_key] = selected_issue
                used_non_a11y_keys.add(selected_issue['issue_key'])
                matched_count += 1
                logger.debug(f"Matched {issue_key} with {selected_issue['issue_key']}")
            else:
                # No available matches (either no matches at all, or all matches already used)
                similar_issues[issue_key] = None
                no_match_count += 1
                logger.debug(f"No available match found for {issue_key}")

            processed_count += 1

            if processed_count % 50 == 0:
                logger.info(f"Processed {processed_count}/{len(self.a11y_issue_keys)} accessibility issues - "
                            f"Matched: {matched_count}, No match: {no_match_count}")

        logger.info(f"Matching completed:")
        logger.info(f"  Total a11y issues: {len(self.a11y_issue_keys)}")
        logger.info(f"  Successfully matched: {matched_count}")
        logger.info(f"  No match available: {no_match_count}")
        logger.info(f"  Unique non-a11y issues used: {len(used_non_a11y_keys)}")

        return similar_issues

    def _create_non_a11y_index(self) -> Dict[Tuple, List[Dict[str, Any]]]:
        """Create an index of non-a11y issues by matching criteria"""
        index = defaultdict(list)

        for issue_key, issue in self.non_a11y_issues.items():
            # Create a key tuple based on matching criteria
            criteria_key = (
                issue.get('type'),
                issue.get('priority'),
                issue.get('issue_type'),
                tuple(sorted(issue.get('affected_versions', []))) if issue.get('affected_versions') else ()
            )
            index[criteria_key].append(issue)

        logger.info(f"Created index with {len(index)} unique criteria combinations")
        return dict(index)

    def _find_available_matching_issues(self, a11y_issue: Dict[str, Any],
                                        non_a11y_index: Dict[Tuple, List[Dict[str, Any]]],
                                        used_non_a11y_keys: set) -> List[Dict[str, Any]]:
        """Find non-a11y issues that match the criteria and haven't been used yet"""

        # Create criteria key for the a11y issue
        criteria_key = (
            a11y_issue.get('type'),
            a11y_issue.get('priority'),
            a11y_issue.get('issue_type'),
            tuple(sorted(a11y_issue.get('affected_versions', []))) if a11y_issue.get('affected_versions') else ()
        )

        # Get all matching issues from index
        all_matches = non_a11y_index.get(criteria_key, [])

        # Filter out issues that have already been used
        available_matches = [
            issue for issue in all_matches
            if issue['issue_key'] not in used_non_a11y_keys
        ]

        return available_matches

    def save_similar_issues_mapping(self, similar_issues: Dict[str, Dict[str, Any]],
                                    output_file: str = "Data/similar_issues_mapping.json"):
        """Save the mapping of accessibility issues to similar non-accessibility issues"""

        # Create summary statistics
        successful_matches = {k: v for k, v in similar_issues.items() if v is not None}
        failed_matches = {k: v for k, v in similar_issues.items() if v is None}

        stats = {
            'total_a11y_issues': len(similar_issues),
            'successful_matches': len(successful_matches),
            'failed_matches': len(failed_matches),
            'unique_non_a11y_used': len(set(issue['issue_key'] for issue in successful_matches.values()))
        }

        # Prepare output data
        output_data = {
            'metadata': {
                'description': 'One-to-one mapping of accessibility issues to similar non-accessibility issues',
                'matching_criteria': ['type', 'priority', 'issue_type', 'affected_versions'],
                'matching_strategy': 'Random selection with no reuse',
                'statistics': stats,
                'generated_at': datetime.now().isoformat()
            },
            'similar_issues_mapping': similar_issues
        }

        # Save to file
        logger.info(f"Saving similar issues mapping to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        # Log statistics
        logger.info("Similar Issues Statistics:")
        logger.info(f"  Total a11y issues processed: {stats['total_a11y_issues']}")
        logger.info(f"  Successful matches: {stats['successful_matches']}")
        logger.info(f"  Failed matches (no available non-a11y): {stats['failed_matches']}")
        logger.info(f"  Unique non-a11y issues used: {stats['unique_non_a11y_used']}")

        return output_data

    def create_detailed_comparison_dataset(self, similar_issues: Dict[str, Dict[str, Any]],
                                           output_file: str = "Data/a11y_vs_non_a11y_comparison.json"):
        """Create a detailed dataset for comparing a11y and non-a11y issues"""

        comparison_pairs = []

        for a11y_key, non_a11y_issue in similar_issues.items():
            # Skip if no matching non-a11y issue was found
            if non_a11y_issue is None:
                continue

            a11y_issue = self.a11y_issues[a11y_key]

            comparison_pair = {
                'pair_id': f"{a11y_key}_vs_{non_a11y_issue['issue_key']}",
                'a11y_issue': {
                    'issue_key': a11y_issue['issue_key'],
                    'title': a11y_issue.get('title', ''),
                    'type': a11y_issue.get('type'),
                    'priority': a11y_issue.get('priority'),
                    'issue_type': a11y_issue.get('issue_type'),
                    'affected_versions': a11y_issue.get('affected_versions', []),
                    'resolution_type': a11y_issue.get('resolution_type'),
                    'created_date': a11y_issue.get('created_date'),
                    'resolved_date': a11y_issue.get('resolved_date')
                },
                'non_a11y_issue': {
                    'issue_key': non_a11y_issue['issue_key'],
                    'title': non_a11y_issue.get('title', ''),
                    'type': non_a11y_issue.get('type'),
                    'priority': non_a11y_issue.get('priority'),
                    'issue_type': non_a11y_issue.get('issue_type'),
                    'affected_versions': non_a11y_issue.get('affected_versions', []),
                    'resolution_type': non_a11y_issue.get('resolution_type'),
                    'created_date': non_a11y_issue.get('created_date'),
                    'resolved_date': non_a11y_issue.get('resolved_date')
                },
                'matching_criteria': {
                    'type_match': a11y_issue.get('type') == non_a11y_issue.get('type'),
                    'priority_match': a11y_issue.get('priority') == non_a11y_issue.get('priority'),
                    'issue_type_match': a11y_issue.get('issue_type') == non_a11y_issue.get('issue_type'),
                    'affected_versions_match': (sorted(a11y_issue.get('affected_versions', [])) ==
                                                sorted(non_a11y_issue.get('affected_versions', [])))
                }
            }
            comparison_pairs.append(comparison_pair)

        output_data = {
            'metadata': {
                'description': 'Detailed one-to-one comparison dataset of accessibility vs non-accessibility issues',
                'total_pairs': len(comparison_pairs),
                'pairing_strategy': 'Random selection with no reuse of non-a11y issues',
                'generated_at': datetime.now().isoformat()
            },
            'comparison_pairs': comparison_pairs
        }

        logger.info(f"Saving detailed comparison dataset to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Created {len(comparison_pairs)} comparison pairs")
        return output_data

    def __post_init__(self):
        self.start_time = time.time()
    
    def log_progress(self, force: bool = False):
        """Thread-safe progress logging"""
        with progress_lock:
            if self.processed % 5 == 0 or force:
                elapsed = time.time() - self.start_time
                rate = self.processed / elapsed if elapsed > 0 else 0
                remaining = (self.total_issues - self.processed) / rate if rate > 0 else 0
                
                logger.info(f"Progress: {self.processed}/{self.total_issues} "
                           f"({self.processed/self.total_issues*100:.1f}%) - "
                           f"Success: {self.successful_fetches}, Errors: {self.api_errors}, "
                           f"Rate: {rate:.1f}/s, ETA: {remaining/60:.1f}m")



def fetch_single_issue_details(issue_key: str, stats: DetailedFetchStats, extractor: 'MoodleIssueExtractor') -> Dict[str, Any]:
    """
    Fetch detailed information for a single issue and process it
    
    Args:
        issue_key: The issue key to fetch
        stats: Shared statistics object
        extractor: MoodleIssueExtractor instance for processing
    
    Returns:
        Processed issue data ready for LLM analysis
    """
    try:
        # Fetch detailed issue information from Moodle API
        logger.debug(f"Fetching detailed info for issue {issue_key}")
        api_data = fetch_issue_data(issue_key)
        
        if api_data:  # Successfully fetched data
            # Extract the raw issue details
            raw_issue_data = extract_issue_details_from_api(api_data)
            
            # Process using existing MoodleIssueExtractor
            essential_data = extractor.extract_essential_data(raw_issue_data)
            llm_ready_data = extractor.create_llm_ready_format(essential_data)
            
            logger.debug(f"Successfully processed detailed data for {issue_key}")
            
            with api_lock:
                stats.processed += 1
                stats.successful_fetches += 1
                
        else:  # Empty dict returned (API error handled in fetch_issue_data)
            logger.warning(f"Failed to fetch data for issue {issue_key}")
            llm_ready_data = {
                "issue_key": issue_key,
                "error": "Failed to fetch data from API",
                "title": "",
                "description": "",
                "priority": "",
                "num_comments": 0,
                "num_attachments": 0,
                "num_commits": 0,
                "test_instructions": "",
                "comments": [],
                "timeline_events": []
            }
            
            with api_lock:
                stats.processed += 1
                stats.api_errors += 1
        
        stats.log_progress()
        return llm_ready_data
        
    except Exception as e:
        logger.error(f"Error processing issue {issue_key}: {e}")
        with api_lock:
            stats.processed += 1
            stats.api_errors += 1
        stats.log_progress()
        return {
            "issue_key": issue_key,
            "error": f"Processing exception: {str(e)}",
            "title": "",
            "description": "",
            "priority": "",
            "num_comments": 0,
            "num_attachments": 0,
            "num_commits": 0,
            "test_instructions": "",
            "comments": [],
            "timeline_events": []
        }


def fetch_detailed_similar_issues(similar_issues_file: str = "Data/similar_issues_mapping.json",
                                 output_file: str = "Data/detailed_similar_non_a11y_issues_updated.json",
                                 max_workers: int = 8) -> bool:
    """
    Fetch detailed information for similar non-a11y issues using Moodle API
    
    Args:
        similar_issues_file: Path to the similar issues mapping file
        output_file: Path to save detailed issue data
        max_workers: Number of concurrent threads for API calls
    
    Returns:
        True if successful, False otherwise
    """
    
    # Validate credentials
    if not USERNAME or not PASSWORD:
        logger.error("USERNAME and PASSWORD must be set in .env file")
        return False
    
    try:
        # Load the similar issues mapping
        logger.info(f"Loading similar issues mapping from {similar_issues_file}")
        with open(similar_issues_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        similar_issues_mapping = mapping_data.get('similar_issues_mapping', {})
        
        # Extract unique non-a11y issue keys
        non_a11y_issue_keys = set()
        for a11y_key, non_a11y_issue in similar_issues_mapping.items():
            if non_a11y_issue is not None:
                non_a11y_issue_keys.add(non_a11y_issue['issue_key'])
        
        non_a11y_issue_keys = list(non_a11y_issue_keys)
        logger.info(f"Found {len(non_a11y_issue_keys)} unique non-a11y issues to fetch details for")
        
        # Initialize statistics and extractor
        stats = DetailedFetchStats(total_issues=len(non_a11y_issue_keys))
        extractor = MoodleIssueExtractor()
        
        # Fetch detailed information using multithreading
        detailed_issues = []
        
        logger.info(f"Starting detailed fetch process with {max_workers} worker threads")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_issue_key = {
                executor.submit(fetch_single_issue_details, issue_key, stats, extractor): issue_key 
                for issue_key in non_a11y_issue_keys
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_issue_key):
                try:
                    detailed_issue = future.result()
                    detailed_issues.append(detailed_issue)
                except Exception as e:
                    issue_key = future_to_issue_key[future]
                    logger.error(f"Error processing issue {issue_key}: {e}")
                    # Add fallback entry
                    detailed_issues.append({
                        "issue_key": issue_key,
                        "error": f"Processing error: {str(e)}",
                        "title": "",
                        "description": "",
                        "priority": "",
                        "num_comments": 0,
                        "num_attachments": 0,
                        "num_commits": 0,
                        "test_instructions": "",
                        "comments": [],
                        "timeline_events": []
                    })
        
        # Sort results by issue key for consistency
        detailed_issues.sort(key=lambda x: x['issue_key'])
        
        # Final statistics
        stats.log_progress(force=True)
        elapsed_total = time.time() - stats.start_time
        
        # Prepare output data
        output_data = {
            'metadata': {
                'description': 'Detailed information for similar non-accessibility issues',
                'source_mapping_file': similar_issues_file,
                'total_issues_fetched': len(detailed_issues),
                'successful_fetches': stats.successful_fetches,
                'api_errors': stats.api_errors,
                'processing_time_minutes': elapsed_total / 60,
                'generated_at': datetime.now().isoformat()
            },
            'detailed_issues': detailed_issues
        }
        
        # Save to file
        logger.info(f"Saving detailed issue data to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info("="*60)
        logger.info("DETAILED FETCH PROCESS COMPLETED")
        logger.info("="*60)
        logger.info(f"Total issues processed: {stats.total_issues}")
        logger.info(f"Successful fetches: {stats.successful_fetches}")
        logger.info(f"API errors: {stats.api_errors}")
        logger.info(f"Total time: {elapsed_total/60:.1f} minutes")
        logger.info(f"Average rate: {stats.total_issues/elapsed_total:.1f} issues/second")
        logger.info(f"Output saved to: {output_file}")
        
        return True
        
    except FileNotFoundError:
        logger.error(f"Similar issues mapping file {similar_issues_file} not found")
        return False
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {similar_issues_file}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during detailed fetch process: {e}")
        return False

class MoodleIssueExtractor:
    """Extract and clean essential Moodle issue data for role sequence analysis"""
    
    def __init__(self):
        # Jira markup patterns
        self.markup_patterns = {
            'color': re.compile(r'\{color:[^}]+\}(.*?)\{color\}'),
            'links': re.compile(r'\[([^|\]]+)\|([^\]]+)\]'),
            'images': re.compile(r'!([^!]+)!'),
            'checkmarks': re.compile(r'\(/\)'),
            'xmarks': re.compile(r'\(x\)'),
            'whitespace': re.compile(r'\n\s*\n'),
            'empty_dashes': re.compile(r'^\s*-\s*$', re.MULTILINE)
        }
    
    def clean_markup(self, text: str) -> str:
        """Clean Jira markup while preserving strikethrough meaning"""
        if not text:
            return ""
        
        # Handle strikethrough with context
        text = self._preserve_strikethrough_meaning(text)
        
        # Clean other markup
        text = self.markup_patterns['color'].sub(r'\1', text)
        text = self.markup_patterns['links'].sub(r'\1 (\2)', text)
        text = self.markup_patterns['images'].sub('[IMAGE]', text)
        text = self.markup_patterns['checkmarks'].sub('✓', text)
        text = self.markup_patterns['xmarks'].sub('✗', text)
        text = self.markup_patterns['empty_dashes'].sub('', text)
        text = self.markup_patterns['whitespace'].sub('\n\n', text)
        
        # Clean whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()
        
        return text
    
    def _preserve_strikethrough_meaning(self, text: str) -> str:
        """Convert strikethrough to semantic markers"""
        # Pattern 1: -Word- (status updates) -> [UPDATED: was Word]
        text = re.sub(r'-([^-\n]{1,20})-\s*([A-Z]\w*)', r'[UPDATED: was \1] \2', text)
        
        # Pattern 2: Long strikethrough (resolved/outdated)
        text = re.sub(r'~([^~]{30,})~', r'[RESOLVED: \1]', text)
        
        # Pattern 3: Short strikethrough (deprecated)
        text = re.sub(r'~([^~]{1,29})~', r'[DEPRECATED: \1]', text)
        
        return text
    
    def extract_essential_data(self, raw_issue: Dict[str, Any], filter_config: FilterConfig = None) -> Dict[str, Any]:
        """Extract only the essential fields for role sequence analysis"""
        
        # Get key participants for role determination
        participants = {
            'reporter': raw_issue.get('reporter'),
            'assignee': raw_issue.get('assignee'),
            'peer_reviewer': raw_issue.get('peer_reviewer'),
            'integrator': raw_issue.get('integrator'),
            'tester': raw_issue.get('tester_name')
        }
        
        essential_data = {
            # Core identifiers
            'issue_key': raw_issue.get('issue_key'),
            'title': self.clean_markup(raw_issue.get('title', '')),
            'description': self.clean_markup(raw_issue.get('description', '')),
            'priority': raw_issue.get('priority'),
            
            # Quantitative metrics
            'num_comments': raw_issue.get('num_comments', 0),
            'num_attachments': raw_issue.get('num_attachments', 0),
            'num_commits': raw_issue.get('num_commits', 0),
            
            # Test instructions (important for understanding process)
            'test_instructions': self.clean_markup(raw_issue.get('test_instructions', '')),
            
            # Comments (cleaned and structured)
            'comments': self._extract_comments(raw_issue.get('comments', []), participants),
            
            # Changelog (simplified with configurable field filtering)
            'changelog': self._extract_changelog(raw_issue.get('changelog_summary', {}), filter_config)
        }
        
        return essential_data
    
    def _extract_comments(self, raw_comments: List[Dict], participants: Dict[str, str]) -> List[Dict]:
        """Extract comments with author, body, and role only"""
        cleaned_comments = []
        
        for comment in raw_comments:
            # Extract author name properly - handle both string and dict cases
            author_data = comment.get('author')
            if isinstance(author_data, dict):
                author_name = author_data.get('displayName') or author_data.get('name') or 'Unknown'
            elif isinstance(author_data, str):
                author_name = author_data
            else:
                author_name = 'Unknown'
            
            cleaned_comment = {
                'author': author_name,
                'body': self.clean_markup(comment.get('body', '')),
                'role': self._determine_role(author_name, participants)
            }
            cleaned_comments.append(cleaned_comment)
        
        return cleaned_comments
    
    def _extract_changelog(self, raw_changelog: Dict, filter_config: FilterConfig = None) -> Dict:
        """Extract timeline events with configurable field filtering"""
        
        # Use provided filter config or default to RQ1 fields
        if filter_config and filter_config.important_timeline_fields:
            important_fields = filter_config.important_timeline_fields
        else:
            # Default fields for role sequence analysis
            important_fields = {
                'status', 'assignee', 'Peer reviewer', 'Integrator', 'Tester',
                'Testing Instructions', 'description', 'Attachment'
            }
        
        changelog = {
            'timeline_events': []
        }
        
        # Filter timeline events to only include important fields for this research question
        for event in raw_changelog.get('timeline', []):
            field = event.get('field')
            if field in important_fields:
                # Extract author name properly
                author_data = event.get('author')
                if isinstance(author_data, dict):
                    author_name = author_data.get('displayName') or author_data.get('name') or 'Unknown'
                elif isinstance(author_data, str):
                    author_name = author_data
                else:
                    author_name = 'Unknown'
                
                changelog['timeline_events'].append({
                    'timestamp': event.get('timestamp'),
                    'author': author_name,  # Use extracted name instead of raw author data
                    'field': field,
                    'from_value': event.get('from_value'),
                    'to_value': event.get('to_value'),
                    'change_description': event.get('change_description')
                })
        
            # Sort timeline events chronologically from oldest to newest
        changelog['timeline_events'].sort(key=lambda x: x.get('timestamp') or '')
        return changelog
    
    def create_llm_ready_format(self, essential_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format data for LLM analysis - let LLM analyze role transitions from actual content"""
        
        llm_format = {
            # Header information
            'issue_key': essential_data['issue_key'],
            'title': essential_data['title'],
            'description': essential_data['description'],
            'priority': essential_data['priority'],
            
            # Quantitative metrics
            'num_comments': essential_data['num_comments'],
            'num_attachments': essential_data['num_attachments'],
            'num_commits': essential_data['num_commits'],
            
            # Test instructions
            'test_instructions': essential_data['test_instructions'],
            
            # Comments (chronological with role info)
            'comments': essential_data['comments'],
            
            # Timeline events (chronological)
            'timeline_events': essential_data['changelog']['timeline_events']
        }
        
        return llm_format
    
    def _determine_role(self, person: str, participants: Dict[str, str]) -> List[str]:
        """Determine all roles of person"""
        if not person:
            return ['unknown']
        
        roles = []
    
        
        # Check for bot authors first
        if person == "CiBoT" or "bot" in person.lower() or "noreply" in person.lower():
            roles.append('bot')
        
        # Check all possible roles
        if person == participants.get('reporter'):
            roles.append('reporter')
        if person == participants.get('assignee'):
            roles.append('assignee')
        if person == participants.get('peer_reviewer'):
            roles.append('peer_reviewer')
        if person == participants.get('integrator'):
            roles.append('integrator')
        if person == participants.get('tester'):
            roles.append('tester')
        
        # If no specific roles found, they're a participant
        if not roles:
            roles.append('participant')
        
        return roles


def apply_filters(issue: Dict[str, Any], filter_config: FilterConfig) -> bool:
    """Apply configurable filters to determine if issue should be processed"""
    try:
        # Filter 1: Created date filter (if specified)
        if filter_config.created_date_min:
            created_date = datetime.strptime(issue["created_date"], "%Y-%m-%dT%H:%M:%S.%f%z")
            # Make filter_config.created_date_min timezone-aware to match created_date
            min_date_with_tz = filter_config.created_date_min.replace(tzinfo=created_date.tzinfo)
            if created_date < min_date_with_tz:
                return False
        
        # Filter 2: Exclude certain issue types (if specified)
        if filter_config.excluded_issue_types and issue.get("issue_type") in filter_config.excluded_issue_types:
            return False
        
        # Filter 3: Only include certain resolution types (if specified)
        if filter_config.required_resolution_types and issue.get("resolution_type") not in filter_config.required_resolution_types:
            return False
        
        return True
    
    except (ValueError, KeyError, TypeError) as e:
        logger.warning(f"Error applying filters to issue {issue.get('issue_key', 'unknown')}: {e}")
        return False


def process_moodle_issue_for_llm(raw_issue_data: Dict[str, Any], filter_config: FilterConfig = None) -> Dict[str, Any]:
    """Main function to process a Moodle issue for LLM analysis with configurable filtering"""
    extractor = MoodleIssueExtractor()
    
    # Extract essential data with filter config for timeline field filtering
    essential_data = extractor.extract_essential_data(raw_issue_data, filter_config)
    
    # Format for LLM
    llm_ready_data = extractor.create_llm_ready_format(essential_data)
    
    return llm_ready_data


def process_moodle_data_file(input_file_path: str, research_questions: List[str], 
                           output_dir: str = "Data") -> Dict[str, List[Dict[str, Any]]]:
    """
    Process the Moodle data file with configurable filters for multiple research questions
    
    Args:
        input_file_path: Path to the input JSON file
        research_questions: List of research question IDs (e.g., ["RQ1", "RQ2"])
        output_dir: Directory to save processed data
    
    Returns:
        Dictionary mapping research question ID to list of processed issues
    """
    results = {}
    
    logger.info(f"Reading data from {input_file_path}")
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        logger.info(f"Loaded {len(raw_data)} issues from file")
        logger.info(f"Processing for research questions: {research_questions}")
        
        # Process each research question
        for rq_id in research_questions:
            if rq_id not in RESEARCH_QUESTION_FILTERS:
                logger.error(f"Unknown research question: {rq_id}")
                continue
            
            filter_config = RESEARCH_QUESTION_FILTERS[rq_id]
            logger.info(f"\nProcessing {rq_id}: {filter_config.description}")
            
            processed_issues = []
            filtered_count = 0
            error_count = 0
            
            for i, issue in enumerate(raw_data):
                if i % 100 == 0 and i > 0:
                    logger.info(f"  Processing issue {i+1}/{len(raw_data)} for {rq_id}")
                
                # Apply filters for this research question
                if not apply_filters(issue, filter_config):
                    filtered_count += 1
                    continue
                
                try:
                    # Process the issue with the specific filter config for timeline field filtering
                    processed_issue = process_moodle_issue_for_llm(issue, filter_config)
                    processed_issues.append(processed_issue)
                    
                except Exception as e:
                    logger.error(f"Error processing issue {issue.get('issue_key', 'unknown')} for {rq_id}: {e}")
                    error_count += 1
                    continue
            
            logger.info(f"{rq_id} processing complete - Total: {len(raw_data)}, Filtered: {filtered_count}, "
                       f"Errors: {error_count}, Processed: {len(processed_issues)}")
            
            # Save processed data for this research question
            output_file = f"{output_dir}/processed_issues_{rq_id.lower()}.json"
            logger.info(f"Saving {rq_id} processed data to {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                # Save just the array of processed issues without metadata wrapper
                json.dump(processed_issues, f, indent=2, ensure_ascii=False)
            logger.info(f"{rq_id} data saved successfully")
            
            results[rq_id] = processed_issues
        
        return results
        
    except FileNotFoundError:
        logger.error(f"File {input_file_path} not found")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {input_file_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {}


def find_similar_non_a11y_issues(processed_rq1_file: str = "Data/processed_issues_rq1.json",
                                a11y_enhanced_file: str = "Data/combined_a11y_issues_enhanced.json",
                                non_a11y_file: str = "Data/non_a11y_issues_high_level_updated.json") -> Dict[str, Any]:
    """
    Main function to find one random similar non-accessibility issue for each accessibility issue
    
    Args:
        processed_rq1_file: Path to processed RQ1 issues
        a11y_enhanced_file: Path to enhanced accessibility issues
        non_a11y_file: Path to non-accessibility issues
    
    Returns:
        Dictionary containing the mapping and metadata
    """
    finder = SimilarIssueFinder()
    
    # Load all required data files
    if not finder.load_data_files(processed_rq1_file, a11y_enhanced_file, non_a11y_file):
        logger.error("Failed to load data files")
        return {}
    
    # Find similar issues (one per a11y issue, no reuse)
    similar_issues = finder.find_similar_issues()
    
    # Save the mapping
    mapping_result = finder.save_similar_issues_mapping(similar_issues)
    
    # Create detailed comparison dataset
    comparison_result = finder.create_detailed_comparison_dataset(similar_issues)
    
    return {
        'similar_issues_mapping': mapping_result,
        'detailed_comparison': comparison_result
    }


def add_research_question_filter(rq_id: str, filter_config: FilterConfig):
    """Add a new research question filter configuration"""
    RESEARCH_QUESTION_FILTERS[rq_id] = filter_config
    logger.info(f"Added filter configuration for {rq_id}: {filter_config.description}")


def list_available_research_questions():
    """List all available research question configurations"""
    logger.info("Available research question filters:")
    for rq_id, config in RESEARCH_QUESTION_FILTERS.items():
        logger.info(f"  {rq_id}: {config.description}")
        logger.info(f"    - Min date: {config.created_date_min}")
        logger.info(f"    - Excluded types: {config.excluded_issue_types}")
        logger.info(f"    - Required resolution: {config.required_resolution_types}")
        logger.info(f"    - Timeline fields: {config.important_timeline_fields}")


def main():
    """Main function to run the original data processing"""
    input_file = "Data/combined_a11y_issues_enhanced.json"
    
    # List available research questions
    list_available_research_questions()
    
    # Process data for specified research questions
    research_questions = ["RQ1"]  # Can add more: ["RQ1", "RQ2", "RQ3"]
    
    # Process accessibility issues
    results = process_moodle_data_file(input_file, research_questions)
    
    logger.info("Original data processing completed successfully")


def main_similar_issues():
    """Main function specifically for finding similar non-accessibility issues"""
    logger.info("="*60)
    logger.info("SIMILAR NON-ACCESSIBILITY ISSUES FINDER")
    logger.info("="*60)
    
    # Find similar non-accessibility issues
    similar_results = find_similar_non_a11y_issues()
    
    if similar_results:
        logger.info("\n" + "="*50)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("="*50)
        logger.info("Output files created:")
        logger.info("  - Data/similar_issues_mapping.json")
        logger.info("  - Data/a11y_vs_non_a11y_comparison.json")
        logger.info("\nUse these files for your accessibility vs non-accessibility analysis!")
    else:
        logger.error("Failed to complete similar issue analysis")
        logger.error("Please check the log messages above for details")


def main_fetch_detailed_similar_issues():
    """Main function to fetch detailed information for similar non-accessibility issues"""
    logger.info("="*70)
    logger.info("DETAILED SIMILAR NON-A11Y ISSUES FETCHER")
    logger.info("="*70)
    logger.info("This script will:")
    logger.info("1. Load the similar issues mapping file")
    logger.info("2. Extract unique non-a11y issue keys")
    logger.info("3. Fetch detailed information from Moodle API (comments, changelog, etc.)")
    logger.info("4. Process and clean the data using existing MoodleIssueExtractor")
    logger.info("5. Save LLM-ready detailed issue data")
    logger.info("="*70)
    
    # Configuration
    similar_issues_file = "Data/a11y_vs_non_a11y_comparison.json"  # Use the comparison file instead
    output_file = "Data/detailed_similar_non_a11y_issues.json"
    max_workers = 8  # Conservative to respect API limits
    
    logger.info(f"Configuration:")
    logger.info(f"  Input file: {similar_issues_file}")
    logger.info(f"  Output file: {output_file}")
    logger.info(f"  Max workers: {max_workers}")
    logger.info("")
    
    # Verify credentials first
    try:
        logger.info("Testing API connectivity...")
        test_data = fetch_issue_data("MDL-1")
        if test_data:
            logger.info("✓ API connectivity verified successfully")
        else:
            logger.info("✓ API is reachable (test issue may not exist, but API responds)")
    except Exception as e:
        logger.error(f"✗ Error testing API connectivity: {e}")
        return False
    
    # Load the comparison file instead of the mapping file
    try:
        logger.info(f"Loading comparison file from {similar_issues_file}")
        with open(similar_issues_file, 'r', encoding='utf-8') as f:
            comparison_data = json.load(f)
        
        # Extract unique non-a11y issue keys from comparison pairs
        comparison_pairs = comparison_data.get('comparison_pairs', [])
        non_a11y_issue_keys = set()
        
        for pair in comparison_pairs:
            non_a11y_issue = pair.get('non_a11y_issue', {})
            if non_a11y_issue and non_a11y_issue.get('issue_key'):
                non_a11y_issue_keys.add(non_a11y_issue['issue_key'])
        
        non_a11y_issue_keys = list(non_a11y_issue_keys)
        logger.info(f"Found {len(non_a11y_issue_keys)} unique non-a11y issues to fetch details for")
        
    except FileNotFoundError:
        logger.error(f"Comparison file {similar_issues_file} not found")
        return False
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {similar_issues_file}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error loading comparison file: {e}")
        return False
    
    # Initialize statistics and extractor
    stats = DetailedFetchStats(total_issues=len(non_a11y_issue_keys))
    extractor = MoodleIssueExtractor()
    
    # Fetch detailed information using multithreading
    detailed_issues = []
    
    logger.info(f"Starting detailed fetch process with {max_workers} worker threads")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_issue_key = {
            executor.submit(fetch_single_issue_details, issue_key, stats, extractor): issue_key 
            for issue_key in non_a11y_issue_keys
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_issue_key):
            try:
                detailed_issue = future.result()
                detailed_issues.append(detailed_issue)
            except Exception as e:
                issue_key = future_to_issue_key[future]
                logger.error(f"Error processing issue {issue_key}: {e}")
                # Add fallback entry
                detailed_issues.append({
                    "issue_key": issue_key,
                    "error": f"Processing error: {str(e)}",
                    "title": "",
                    "description": "",
                    "priority": "",
                    "num_comments": 0,
                    "num_attachments": 0,
                    "num_commits": 0,
                    "test_instructions": "",
                    "comments": [],
                    "timeline_events": []
                })
    
    # Sort results by issue key for consistency
    detailed_issues.sort(key=lambda x: x['issue_key'])
    
    # Final statistics
    stats.log_progress(force=True)
    elapsed_total = time.time() - stats.start_time
    
    # Prepare output data
    output_data = {
        'metadata': {
            'description': 'Detailed information for similar non-accessibility issues',
            'source_comparison_file': similar_issues_file,
            'total_issues_fetched': len(detailed_issues),
            'successful_fetches': stats.successful_fetches,
            'api_errors': stats.api_errors,
            'processing_time_minutes': elapsed_total / 60,
            'generated_at': datetime.now().isoformat()
        },
        'detailed_issues': detailed_issues
    }
    
    # Save to file
    logger.info(f"Saving detailed issue data to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info("="*60)
    logger.info("DETAILED FETCH PROCESS COMPLETED")
    logger.info("="*60)
    logger.info(f"Total issues processed: {stats.total_issues}")
    logger.info(f"Successful fetches: {stats.successful_fetches}")
    logger.info(f"API errors: {stats.api_errors}")
    logger.info(f"Total time: {elapsed_total/60:.1f} minutes")
    logger.info(f"Average rate: {stats.total_issues/elapsed_total:.1f} issues/second")
    logger.info(f"Output saved to: {output_file}")
    
    return True



def main_complete():
    """Main function to run both original processing and similar issue finding"""
    logger.info("="*60)
    logger.info("COMPLETE MOODLE ISSUE PROCESSING PIPELINE")
    logger.info("="*60)
    
    # Step 1: Original data processing
    logger.info("\nStep 1: Processing accessibility issues for research questions")
    main()
    
    # Step 2: Find similar non-accessibility issues
    logger.info("\nStep 2: Finding similar non-accessibility issues")
    main_similar_issues()
    
    # Step 3: Fetch detailed information for similar issues
    logger.info("\nStep 3: Fetching detailed information for similar non-a11y issues")
    main_fetch_detailed_similar_issues()
    
    logger.info("\n" + "="*60)
    logger.info("COMPLETE PIPELINE FINISHED")
    logger.info("="*60)