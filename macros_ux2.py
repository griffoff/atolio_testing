from playwright.sync_api import sync_playwright
from io import BytesIO
import pandas as pd
from bert_score import score
from datetime import datetime
import time
import sys
import os

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def save_auth_session(playwright, url):
    print("Launching browser for manual login...")
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    print("Please log in manually and ensure the chatbot interface is ready.")
    page.goto(url)  # URL taken from sys.argv
    input("Press Enter once you have logged in and the chatbot interface is ready for use...")
    return browser, context, page


def get_atolio_answers(playwright, context, url, questions):
    atolio_answers = []
    print("Fetching Atolio Answers...")
    input_selector = "textarea[data-testid='ask-textarea']"
    response_selector = "div.whitespace-pre-wrap.chat-markdown-content p"

    for question in questions:
        page = context.new_page()  # Start with a fresh page for each question
        page.goto(url)  # Using URL passed from sys.argv
        try:
            page.wait_for_selector(input_selector, timeout=30000)
            print(f"Sending Query: {question}")
            page.fill(input_selector, question)
            page.keyboard.press("Enter")
            page.wait_for_timeout(5000)
            time.sleep(30)  # Wait for chatbot response
            response = page.locator(response_selector).all_inner_texts()
            atolio_answer = "\n".join(response) if response else "No response"
            print(f"Chatbot Answer: {atolio_answer}")
            atolio_answers.append(atolio_answer)
        except Exception as e:
            print(f"Failed to retrieve the answer for '{question}': {e}")
            atolio_answers.append("Error: Could not retrieve answer")
        page.close()  # Close the page to clean up for the next question
    return atolio_answers

def get_user_name(): 
    username = "Unknown"
    try:
        # os.getlogin() gets the user logged into the controlling terminal
        username = os.getlogin()
        print(f"The current user is: {username}")
    except OSError:
        # Fallback for environments without a controlling terminal (e.g., some services)
        # On Windows, 'USERNAME' is the standard environment variable.
        username = os.environ.get('USERNAME')
        print(f"The current user (from environment variable) is: {username}")
    
    return username


def main():
    try:
        # Parse command-line arguments
        excel_file = sys.argv[1]  # Path to the Excel file
        sheet_name = sys.argv[2]  # Sheet name
        url = sys.argv[3]  # URL (e.g., https://test-atolio.cengage.info/)
        
        print(excel_file)
        print(sheet_name)
        print(f"URL: {url}")

        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Load Excel file
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        print(df.head())

        question_column_name = "QUESTION"
        answer_column_name = "EXPECTED ANSWER"
        # question_column_name = "USER PROMPT/QUESTION (input)\n(pre-populate your questions or copy/paste as you test here)"
        # answer_column_name = "EXPECTED ANSWER (uses the 'expected knowledge base url' to answer the question - provided by testing team)"
        

        # Ensure required columns are present
        required_columns = [
            question_column_name,
            answer_column_name
        ]
        if not set(required_columns).issubset(df.columns):
            raise ValueError(f"Excel file must contain the required columns: {required_columns}")

        # Extract questions and references
        questions = df[question_column_name].astype(str).tolist()
        references = df[answer_column_name].astype(str).tolist()

        # Run Playwright and fetch results
        with sync_playwright() as playwright:
            browser, context, page = save_auth_session(playwright, url)  # Pass the URL
            atolio_answers = get_atolio_answers(playwright, context, url, questions)  # Pass the URL
            browser.close()

        # Append fetched answers to the dataframe
        df["Atolio ANSWER"] = atolio_answers
        print("Fetched Atolio answers added to DataFrame.")

        # Calculate BERTScore
        print("Calculating BERTScore...")
        P, R, F1 = score(atolio_answers, references, lang='en', verbose=True)
        df["Auto Test User"] = get_user_name()
        df["Auto Test Date"] = datetime.date()        
        df["Auto Test Confidence"] = F1.numpy() * 100  # Convert F1 score to percentage
        df["Auto Test Outcome"] = df["percentage"].apply(lambda x: "pass" if x >= 90 else "fail")
        
        # Save results to Excel
        output_filename = os.path.join(script_dir, f"{sheet_name}_ux2_{timestamp}.xlsx")
        df.to_excel(output_filename, index=False)
        print(f"Final results saved to: {output_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()