import os
import pickle
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve credentials from .env file
USERNAME = os.getenv("USER_NAME")
PASSWORD = os.getenv("PASS_WORD")



# Function to save cookies
def save_cookies(driver, cookies_path):
    with open(cookies_path, "wb") as file:
        pickle.dump(driver.get_cookies(), file)


# Function to load cookies
def load_cookies(driver, cookies_path):
    if os.path.exists(cookies_path):
        with open(cookies_path, "rb") as file:
            cookies = pickle.load(file)
            driver.delete_all_cookies()  # Clear existing cookies to avoid conflicts
            for cookie in cookies:
                driver.add_cookie(cookie)
        return True
    return False


def get_issue_commit_history(issue_key):
    # Path to store cookies
    COOKIES_PATH = "Cookies/jira_cookies.pkl"

    # Set up Selenium WebDriver with a persistent profile
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode
    # Works for macOS
    # options.add_argument("--user-data-dir=./chrome_data")  # Sets up persistent profile directory
    # Works for Windows
    options.add_argument("--user-data-dir=C:/Users/10332/AppData/Local/Google/Chrome/User Data/Custom")

    driver = webdriver.Chrome(options=options)
    # Start browser session and navigate to the main Jira tracker page
    driver.get("https://tracker.moodle.org/")

    # Check if cookies are saved and load them if present
    if load_cookies(driver, COOKIES_PATH):
        driver.refresh()  # Refresh to apply cookies
        time.sleep(3)
    else:
        # If no cookies are saved, perform login
        login_button = driver.find_element(By.LINK_TEXT, "Log In")
        login_button.click()

        # Enter login credentials
        username = driver.find_element(By.ID, "login-form-username")
        password = driver.find_element(By.ID, "login-form-password")

        username.send_keys(USERNAME)
        password.send_keys(PASSWORD)
        password.send_keys(Keys.RETURN)

        # Give time for login to process and redirect
        time.sleep(3)

        # Save cookies after successful login
        save_cookies(driver, COOKIES_PATH)

    # Navigate to the specific issue URL
    issue_url = f"https://tracker.moodle.org/browse/{issue_key}?page=com.atlassian.jira.plugin.system.issuetabpanels%3Aworklog-tabpanel&devStatusDetailDialog=repository"
    driver.get(issue_url)

    # Wait for the page to load completely
    time.sleep(20)

    # Extract information (example of extracting all text from the page)
    page_content = driver.page_source
    soup = BeautifulSoup(page_content, "html.parser")

    # Find the table by class and iterate over rows with "data-commit-index" attribute
    table_rows = soup.find_all("tr", attrs={"data-commit-index": True})

    commit_list = []

    for row in table_rows:
        # Extract the commit link
        commit_link = row.find("a", class_="changesetid")
        commit_link_text = commit_link.text if commit_link else "N/A"
        commit_link_url = commit_link['href'] if commit_link else "N/A"

        # Extract the commit message
        message_span = row.find("span", class_="ellipsis")
        message_text = message_span.get("title", "N/A") if message_span else "N/A"

        # Extract the commit timestamp
        timestamp = row.find("time", class_="livestamp")
        timestamp_text = timestamp.text if timestamp else "N/A"

        commit_list.append({
            "commit_link_text": commit_link_text,
            "commit_link_url": commit_link_url,
            "message_text": message_text,
            "timestamp": timestamp_text
        })
    # Close the browser session
    driver.quit()
    return commit_list


def scrape_epic_issue_links(epic_key):
    # Path to store cookies
    COOKIES_PATH = "Cookies/jira_cookies.pkl"

    # Set up Selenium WebDriver with a persistent profile
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode
    # Works for macOS
    options.add_argument("--user-data-dir=./chrome_data")  # Sets up persistent profile directory
    # Works for Windows
    # options.add_argument("--user-data-dir=C:/Users/10332/AppData/Local/Google/Chrome/User Data/Custom")

    driver = webdriver.Chrome(options=options)
    # Start browser session and navigate to the main Jira tracker page
    driver.get("https://tracker.moodle.org/")

    # Check if cookies are saved and load them if present
    if load_cookies(driver, COOKIES_PATH):
        driver.refresh()  # Refresh to apply cookies
        time.sleep(3)
    else:
        # If no cookies are saved, perform login
        login_button = driver.find_element(By.LINK_TEXT, "Log In")
        login_button.click()

        # Enter login credentials
        username = driver.find_element(By.ID, "login-form-username")
        password = driver.find_element(By.ID, "login-form-password")

        username.send_keys(USERNAME)
        password.send_keys(PASSWORD)
        password.send_keys(Keys.RETURN)

        # Give time for login to process and redirect
        time.sleep(3)

        # Save cookies after successful login
        save_cookies(driver, COOKIES_PATH)

    # Navigate to the specific epic URL
    epic_url = f"https://tracker.moodle.org/browse/{epic_key}"
    driver.get(epic_url)

    # Extract information (example of extracting all text from the page)
    page_content = driver.page_source
    soup = BeautifulSoup(page_content, "html.parser")

    # Find the table by class and iterate over rows with "issuerow" class
    table_rows = soup.find_all("tr", attrs={"data-issuekey": True})

    issue_list = []

    for row in table_rows:
        # Extract the commit link
        issue_link = row.find("a")
        issue_link_key = issue_link.text if issue_link else "N/A"
        issue_list.append(issue_link_key)

    # Close the browser session
    driver.quit()
    return issue_list