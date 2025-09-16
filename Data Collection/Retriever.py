import logging
import math
import os
import random
from collections import Counter
from datetime import datetime, timedelta
from itertools import combinations
from typing import Dict, List, Any
import certifi
from scipy.stats import norm
import sys
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import json
import re
from dotenv import load_dotenv
import numpy as np
from Scraper import scrape_epic_issue_links

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Retrieve credentials from .env file
USERNAME = os.getenv("USER_NAME")
PASSWORD = os.getenv("PASS_WORD")


def request_moodle_issues(username, password, query, start_index) -> requests.Response:
    url = f"https://tracker.moodle.org/rest/api/2/search?startAt={start_index}"
    auth = HTTPBasicAuth(username, password)

    headers = {
        "Accept": "application/json"
    }
    response = requests.get(
        url,
        headers=headers,
        params=query,
        auth=auth,
        verify=certifi.where()
    )
    return response


def convert_to_email(informal_email):
    # Replace ' at ' with '@' and ' dot ' with '.'
    if informal_email is not None:
        formal_email = informal_email.replace(' at ', '@').replace(' dot ', '.')
        return formal_email
    return None


def get_votes_and_watchers(issue):
    # URLs for votes and watchers
    url_votes = issue["fields"]["votes"]["self"]
    url_watchers = issue["fields"]["watches"]["self"]

    # Make requests to retrieve votes and watchers data
    try:
        response_votes = requests.get(url_votes)
        response_votes.raise_for_status()  # Check if the request was successful
        response_watchers = requests.get(url_watchers)
        response_watchers.raise_for_status()  # Check if the request was successful

        # Parse the JSON responses
        votes_data = response_votes.json()
        watchers_data = response_watchers.json()

        # Extract the number of votes and watchers
        num_votes = votes_data.get("votes", 0)
        num_watchers = watchers_data.get("watchCount", 0)

        return num_votes, num_watchers

    except requests.RequestException as e:
        logger.error(f"Error fetching data: {e}")
        return None, None


def get_attachment_info(detailed_report_url):
    try:
        details = requests.get(detailed_report_url)
        details.raise_for_status()
        details_data = details.json()

        # Ensure 'fields' and 'attachment' keys exist
        fields = details_data.get('fields', {})
        attachments = fields.get("attachment", [])

        # Form a dictionary where the filename is the key and the content URL is the value
        attachment_dict = {attachment.get('filename'): attachment.get('content') for attachment in attachments if
                           'filename' in attachment and 'content' in attachment}

        return attachment_dict
    except requests.RequestException as e:
        logger.error(f"Error fetching data: {e}")
        return None


def get_comments(detailed_report_url):
    try:
        details = requests.get(detailed_report_url)
        details.raise_for_status()
        details_data = details.json()

        # Ensure 'fields', 'comment', and 'comments' keys exist
        fields = details_data.get('fields', {})
        comment_section = fields.get("comment", {})
        comments = comment_section.get("comments", [])

        return comments
    except requests.RequestException as e:
        logger.error(f"Error fetching data: {e}")
        return None

def get_num_branches(customfield_14410):
    branch_pattern = r'branch=([^,\]]+)(.*?)(?=\],|}})'

    # Search for the 'branch=' value and related text
    branch_match = re.search(branch_pattern, customfield_14410)
    count_branch_values = None
    if branch_match:
        branch_value = branch_match.group(1)  # Extract the branch value

        # Find all 'count=' values within the branch-related section
        count_pattern = r'count=(\d+)'
        count_matches = re.findall(count_pattern, branch_value)
        count_branch_values = int(count_matches[0]) if count_matches else None
    return count_branch_values

def define_query_string(use_keyword: bool, keyword: str, fixed_resolution: bool, project: str, non_a11y: bool = False, issue_key: str = ""):
    query_str = ''
    if fixed_resolution:
        query_str += 'resolution = Fixed'
    if use_keyword:
        if len(query_str) > 0:
            query_str += ' AND '
        query_str += f'text ~ "\'{keyword}\'"'
    else:
        if not non_a11y:
            if len(query_str) > 0:
                query_str += ' AND '
            query_str += 'component = Accessibility'
        else:
            if len(query_str) > 0:
                query_str += ' AND '
            query_str += 'component != Accessibility AND component != "Accessibility toolkit"'
    if issue_key != "":
        if len(query_str) > 0:
            query_str += ' AND '
        query_str += f'id = {issue_key}'

    query_str += f' AND project = {project} ORDER BY createdDate DESC'
    query = {
        'jql': query_str
    }
    return query

"""
Get number of issues before, during and after the audit
"""
def get_a11y_issues_from_audits(created_date_list: list):
    # Dates of the accessibility audits
    audit_dates = [
        datetime(2020, 1, 1),
        datetime(2021, 9, 1),
        datetime(2022, 5, 1),
        datetime(2023, 8, 1)
    ]

    # Calculate the gap between today and the last audit date
    today = datetime.today()
    gap_days = (today - audit_dates[-1]).days

    # Calculate the start date by subtracting the average gap from the first audit date
    start_date = audit_dates[0] - timedelta(days=gap_days)

    # Define date ranges for each period
    date_ranges = {
        "before_first_audit": (start_date, audit_dates[0]),
        "first_to_second_audit": (audit_dates[0], audit_dates[1]),
        "second_to_third_audit": (audit_dates[1], audit_dates[2]),
        "third_to_latest_audit": (audit_dates[2], audit_dates[3]),
        "after_latest_audit": (audit_dates[3], today)
    }

    # Initialize raw issue counts per period
    raw_issue_counts = {
        "before_first_audit": 0,
        "first_to_second_audit": 0,
        "second_to_third_audit": 0,
        "third_to_latest_audit": 0,
        "after_latest_audit": 0
    }

    # Count issues based on date ranges
    for date_str in created_date_list:
        issue_date = datetime.fromisoformat(date_str.split('+')[0])  # Parse the created date

        # Assign issues to their corresponding period
        for period, (start, end) in date_ranges.items():
            if start <= issue_date < end:
                raw_issue_counts[period] += 1
                break

    # Calculate the duration (in days) of each period
    durations = {period: (end - start).days for period, (start, end) in date_ranges.items()}

    # Normalize the issue counts by the duration of each period
    normalized_issue_counts = {
        period: raw_count / durations[period] if durations[period] > 0 else 0
        for period, raw_count in raw_issue_counts.items()
    }

    return normalized_issue_counts

def get_a11y_issues_from_audits_epic_category(created_date_epic_list: list):
    # Dates of the accessibility audits
    audit_dates = [
        datetime(2020, 1, 1),
        datetime(2021, 9, 1),
        datetime(2022, 5, 1),
        datetime(2023, 8, 1)
    ]

    # Calculate today and the gap from the last audit
    today = datetime.today()
    gap_days = (today - audit_dates[-1]).days

    # Start date is adjusted backward by the gap duration
    start_date = audit_dates[0] - timedelta(days=gap_days)

    # Define date ranges for each period
    date_ranges = {
        "before_first_audit": (start_date, audit_dates[0]),
        "first_to_second_audit": (audit_dates[0], audit_dates[1]),
        "second_to_third_audit": (audit_dates[1], audit_dates[2]),
        "third_to_latest_audit": (audit_dates[2], audit_dates[3]),
        "after_latest_audit": (audit_dates[3], today)  # Closed period up to today
    }

    # Initialize counters for issue counts per period and audit relevance
    issue_counts = {
        period: {"direct audit": 0, "without audit": 0} for period in date_ranges.keys()
    }

    # Count issues by period and relevance
    for date_str, audit_type in created_date_epic_list:
        issue_date = datetime.fromisoformat(date_str.split('+')[0])  # Parse the created date

        # Determine the audit period for the issue
        for period, (start, end) in date_ranges.items():
            if start <= issue_date < end:
                if "audit" in audit_type:
                    issue_counts[period]["direct audit"] += 1
                else:
                    issue_counts[period]["without audit"] += 1
                break

    return issue_counts



def assign_epic_category_to_issues(epic_link_text:str):
    if epic_link_text == "Moodle Accessibility Audit":
        return "first audit"
    elif epic_link_text == "Moodle 3.11 Accessibility Audit":
        return "second audit"
    elif epic_link_text == "Moodle 4.0 Accessibility Audit":
        return "third audit"
    elif epic_link_text == "Moodle 4.2 Accessibility Audit":
        return "fourth audit"
    else:
        return "N/A"



def get_issue_info_into_dict(issue):
    fields = issue.get("fields", {}) or {}

    # Safely retrieve values with fallback handling
    resolution_info = fields.get("resolution") or {}
    comment_list = []
    comments = get_comments(issue.get("self"))
    for comment in comments:
        comment_author = comment["author"]["displayName"]
        comment_author_email = convert_to_email(comment["author"]["emailAddress"])
        comment_body = comment["body"]
        comment_created = comment["created"]
        comment_updated = comment["updated"]
        comment_updated_author = comment["updateAuthor"]["displayName"]
        comment_updated_author_email = convert_to_email(comment["updateAuthor"]["emailAddress"])
        comment_dict = {
            "author": comment_author,
            "author_email": comment_author_email,
            "body": comment_body,
            "created": comment_created,
            "updated": comment_updated,
            "updated_author": comment_updated_author,
            "updated_author_email": comment_updated_author_email
        }
        comment_list.append(comment_dict)
    issue_data = {
        "issue_key": issue.get("key"),
        "issue_type": fields.get("issuetype", {}).get("name"),
        "priority": fields.get("priority", {}).get("name"),
        "fix_versions": [fix.get('name') for fix in fields.get("fixVersions", [])],
        "affected_versions": [version.get('name') for version in fields.get("versions", [])],
        "affected_branches": fields.get("customfield_10070"),
        "fix_branches": fields.get("customfield_10071"),
        "pulled_repository": fields.get("customfield_10100"),
        "pulled_main_branch": fields.get("customfield_10111"),
        "pulled_main_diff": fields.get("customfield_10112"),
        "documentation_link": fields.get("customfield_10810"),
        "labels": fields.get("labels", []),
        "components": [component.get('name') for component in fields.get("components", [])],
        "created_date": fields.get("created"),
        "resolved_date": fields.get("resolutiondate"),
        "resolution_type": resolution_info.get("name", "Unresolved"),
        "title": fields.get("summary"),
        "time_logged": fields.get("timespent"),
        "test_instructions": fields.get("customfield_10117"),
        "description": fields.get("description"),
        "assignee": (fields.get("assignee") or {}).get("displayName"),
        "assignee_email": convert_to_email((fields.get("assignee") or {}).get("emailAddress")),
        "reporter": (fields.get("reporter") or {}).get("displayName"),
        "reporter_email": convert_to_email((fields.get("reporter") or {}).get("emailAddress")),
        "peer_reviewer": (fields.get("customfield_10118") or {}).get("displayName"),
        "peer_reviewer_email": convert_to_email((fields.get("customfield_10118") or {}).get("emailAddress")),
        "tester": (fields.get("customfield_10011") or {}).get("displayName"),
        "tester_email": convert_to_email((fields.get("customfield_10011") or {}).get("emailAddress")),
        "integrator": (fields.get("customfield_10110") or {}).get("displayName"),
        "integrator_email": convert_to_email((fields.get("customfield_10110") or {}).get("emailAddress")),
        "num_participants": len(fields.get("customfield_10020", [])),
        "num_votes": get_votes_and_watchers(issue)[0],
        "num_watchers": get_votes_and_watchers(issue)[1],
        "detailed_report_url": issue.get("self"),
        "attachment_dict": get_attachment_info(issue.get("self")),
        "num_attachments": len(get_attachment_info(issue.get("self")) or {}),
        "comments": comment_list,
        "num_comments": len(get_comments(issue.get("self")) or {}),
    }

    # Process issue links
    issue_links = fields.get("issuelinks", [])
    extracted_links = {}
    for issue_link in issue_links:
        if 'inwardIssue' in issue_link:
            inward_issue = issue_link.get('inwardIssue', {}) or {}
            inward_fields = inward_issue.get('fields', {}) or {}
            extracted_links[inward_issue.get('key')] = {
                'link': inward_issue.get('self'),
                'summary': inward_fields.get('summary'),
                'status': inward_fields.get('status', {}).get('name'),
                'priority': inward_fields.get('priority', {}).get('name'),
                'issue_type': issue_link.get('type', {}).get('inward'),
            }
    issue_data["issue_links"] = extracted_links

    # The number of branches involved in the issue
    branch_involved = fields.get("customfield_14410", "")  # Fallback to an empty string if missing
    branch_pattern = r'branch=([^,\]]+)(.*?)(?=\],|}})'

    # Search for the 'branch=' value and related text
    branch_match = re.search(branch_pattern, branch_involved)
    count_branch_values = None
    if branch_match:
        branch_value = branch_match.group(1)  # Extract the branch value

        # Find all 'count=' values within the branch-related section
        count_pattern = r'count=(\d+)'
        count_matches = re.findall(count_pattern, branch_value)
        count_branch_values = int(count_matches[0]) if count_matches else None

    # Regular expression to match the pattern
    commit_pattern = r"CommitOverallBean@\w+\[count=(\d+)"
    commit_matches = re.findall(commit_pattern, branch_involved)
    numb_commited = int(commit_matches[0]) if commit_matches else None
    issue_data["num_commits"] = numb_commited
    issue_data["num_branches"] = count_branch_values
    return issue_data



def save_issues_into_json_file(use_keyword: bool, keyword: str, fixed_resolution: bool, project: str, a11y: bool = True):
    start_index = 0
    query = define_query_string(use_keyword, keyword, fixed_resolution, project)
    all_issues = []  # List to store all issues
    logger.info("Starting to fetch issues from the Moodle Tracker...")
    while True:
        search_response = request_moodle_issues(USERNAME, PASSWORD, query, start_index)
        response_dict = json.loads(search_response.text)

        issues = response_dict.get("issues", [])

        # Break the loop if no issues are returned
        if not issues:
            break

        for issue in issues:
            issue_data = get_issue_info_into_dict(issue)
            # Add issue to the list
            all_issues.append(issue_data)

        logger.info("Fetching the set of issues starting at index: %d", start_index)
        start_index += 50  # Increment to fetch the next set of issues

    # Save all issues to a JSON file
    if a11y:
        file_name = f"{project}_a11y_issues.json"
    else:
        file_name = f"{project}_non_a11y_issues.json"
    with open(file_name, "w", encoding="utf-8") as json_file:
        json.dump(all_issues, json_file, indent=4, ensure_ascii=False)
    logger.info(f"Saved {len(all_issues)} issues to {file_name}")


def get_issues_audit(epic_key_list:list):
    audit_issue_dict = {}
    for epic_key in epic_key_list:
        logger.info(f"Fetching issues for epic: {epic_key}")
        issue_key_list = scrape_epic_issue_links(epic_key)
        issue_list = []
        for issue_key in issue_key_list:
            base_url = "https://tracker.moodle.org/rest/api/2/issue/"
            url = base_url + issue_key
            response = requests.get(url, auth=(USERNAME, PASSWORD))
            data = response.json()
            issue_data = get_issue_info_into_dict(data)
            issue_list.append(issue_data)
            if issue_key == "MDL-70032":
                print(data)
        if epic_key == "MDL-67688":
            audit_issue_dict["First Audit"] = issue_list
        elif epic_key == "MDL-72657":
            audit_issue_dict["Second Audit"] = issue_list
        elif epic_key == "MDL-74624":
            audit_issue_dict["Third Audit"] = issue_list
        elif epic_key == "MDL-78185":
            audit_issue_dict["Fourth Audit"] = issue_list
    with open("Data/audit_issues.json", "w", encoding="utf-8") as json_file:
        json.dump(audit_issue_dict, json_file, indent=4, ensure_ascii=False)
    logger.info(f"Saved {len(audit_issue_dict)} issues to audit_issues.json")

def get_non_a11y_issues_high_level(resume_saving_issues: bool = False):
    query = define_query_string(False, "", False, "MDL", True)
    all_issues = []  # List to store all issues
    if resume_saving_issues:
        with open("Data/next_index.json", "r", encoding="utf-8") as json_file:
            start_index = json.load(json_file)["next_index"]
        logger.info("Starting to fetch non-a11y issues from the Moodle Tracker at index: %d", start_index)
    else:
        start_index = 0
        logger.info("Starting to fetch non-a11y issues from the Moodle Tracker...")
    while True:
        search_response = request_moodle_issues(USERNAME, PASSWORD, query, start_index)
        response_dict = json.loads(search_response.text)
        issues = response_dict.get("issues", [])

        # Break the loop if no issues are returned
        if not issues:
            break

        for issue in issues:
            fields = issue.get("fields", {}) or {}
            resolution = fields.get("resolution")
            votes_watchers = get_votes_and_watchers(issue)
            issue_data = {
                "issue_key": issue.get("key"),
                "type": fields.get("issuetype", {}).get("name"),
                "created_date": fields.get("created"),
                "resolved_date": fields.get("resolutiondate"),
                "priority": fields.get("priority", {}).get("name"),
                "resolution_type": resolution.get("name", "Unresolved") if resolution else "Unresolved",
                "num_test_instructions": len(fields.get("customfield_10117")) if fields.get("customfield_10117") is not None else 0,
                "num_description": len(fields.get("description")) if fields.get("description") is not None else 0,
                "num_comments": len(get_comments(issue.get("self")) or []),
                "assignee": (fields.get("assignee") or {}).get("displayName"),
                "reporter": (fields.get("reporter") or {}).get("displayName"),
                "num_participants": len(fields.get("customfield_10020", [])),
                "num_votes": votes_watchers[0],
                "num_watchers": votes_watchers[1],
                "logged_time": fields.get("timespent"),
                "num_attachments": len(get_attachment_info(issue.get("self")) or {}),
                "num_branches": get_num_branches(fields.get("customfield_14410", "")),
                "num_commits": len(re.findall(r"CommitOverallBean@\w+\[count=(\d+)", fields.get("customfield_14410", ""))),
            }
            all_issues.append(issue_data)

            # Save every 50 issues
            if len(all_issues) >= 50:
                save_issues_to_file(all_issues, "Data/non_a11y_issues_high_level.json")
                all_issues = []  # Clear the list after saving

        start_index += 50  # Increment to fetch the next set of issues
        with open("Data/next_index.json", "w", encoding="utf-8") as json_file:
            json.dump({"next_index": start_index}, json_file, indent=4)
        logger.info("Fetching the set of non-a11y issues starting at index: %d", start_index)

    # Save any remaining issues
    if all_issues:
        save_issues_to_file(all_issues, "Data/non_a11y_issues_high_level.json")
    logger.info(f"Finished saving non-a11y issues.")


def save_issues_to_file(issues, file_name):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # Append issues to the file
    if os.path.exists(file_name):
        with open(file_name, "r+", encoding="utf-8") as json_file:
            existing_issues = json.load(json_file)
            existing_issues.extend(issues)
            json_file.seek(0)
            json.dump(existing_issues, json_file, indent=4, ensure_ascii=False)
    else:
        with open(file_name, "w", encoding="utf-8") as json_file:
            json.dump(issues, json_file, indent=4, ensure_ascii=False)


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

def save_to_json(data: Dict, filename: str = "moodle_issue_data.json"):
    """
    Save extracted data to JSON file.
    
    Args:
        data: Extracted data dictionary
        filename: Output filename
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Data saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving to {filename}: {e}")

def save_batch_to_json(data: List[Dict], filename: str, is_first_batch: bool = False):
    """
    Save data batch to JSON file (append or create new).
    
    Args:
        data: List of data to save
        filename: Output filename
        is_first_batch: True if this is the first batch (creates new file)
    """
    try:
        if is_first_batch:
            # Create new file with first batch
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logging.info(f"Created new file {filename} with {len(data)} items")
        else:
            # Read existing file, append new data, and save
            with open(filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            existing_data.extend(data)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Appended {len(data)} items to {filename} (total: {len(existing_data)})")
            
    except Exception as e:
        logging.error(f"Error saving batch to {filename}: {e}")


if __name__ == '__main__':
    save_issues_into_json_file(False, "", False, "MDL")
    save_issues_into_json_file(False, "", False, "MDL", a11y=False)
    get_issues_audit(["MDL-67688", "MDL-72657", "MDL-74624", "MDL-78185"])
    get_non_a11y_issues_high_level(True)

