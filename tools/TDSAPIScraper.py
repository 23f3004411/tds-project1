import requests
import openpyxl
from datetime import datetime

def create_session_with_browser_cookies(discourse_url, cookies):
    session = requests.Session()
    
    for name, value in cookies.items():
        session.cookies.set(name, value, domain=discourse_url.split('//')[1])
    
    return session


def get_discourse_posts_and_responses(session, discourse_url, category_id, start_date, end_date):
    """
    Fetches Discourse posts and their responses from a specific category within a date range.
    """
    all_posts_data = []
    page = 0
    
    # Convert string dates to datetime objects for comparison
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    while True:
        topics_url = f"{discourse_url}/c/{category_id}.json"
        params = {
            'page': page,
            'order': 'created',
            'ascending': 'true'
        }
        
        response = session.get(topics_url, params=params)
        if response.status_code == 200:
            response.raise_for_status()  
            topics_data = response.json() 
        else:
            print(f"Failed to get topics: {response.status_code}")

        topics = topics_data.get('topic_list', {}).get('topics', [])

        if not topics:
            break 

        for topic in topics:
            topic_created_at_str = topic.get('created_at')
            print(topic_created_at_str)
            if topic_created_at_str:
                topic_created_at = datetime.strptime(topic_created_at_str.split('T')[0], '%Y-%m-%d')
                
                if start_dt <= topic_created_at <= end_dt:
                    topic_id = topic.get('id')
                    topic_slug = topic.get('slug')
                    
                    if topic_id and topic_slug:
                        post_url = f"{discourse_url}/t/{topic_slug}/{topic_id}.json"
                        
                        try:
                            post_response = session.get(post_url)
                            post_response.raise_for_status()
                            post_details = post_response.json()

                            posts_in_topic = post_details.get('post_stream', {}).get('posts', [])
                            for post in posts_in_topic:
                                all_posts_data.append({
                                    'Topic ID': topic_id,
                                    'Post ID': post.get('id'),
                                    'Post URL': f"{discourse_url}/t/{topic_slug}/{topic_id}/{post.get('post_number')}",
                                    'Username': post.get('username'),
                                    'Created At': post.get('created_at'),
                                    'Content': post.get('cooked') # 'cooked' contains the HTML rendered content
                                })
                        except requests.exceptions.RequestException as e:
                            print(f"Error fetching posts for topic {topic_id}: {e}")
                elif topic_created_at > end_dt:
                    break 


        page += 1

    return all_posts_data

def save_to_excel(data, filename="scraped_data/tds_discourse_posts.xlsx"):
    """
    Saves the extracted data into an Excel spreadsheet.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "TDS Discourse Posts"

    headers = ["Topic ID", "Post ID", "Post URL", "Username", "Created At", "Content"]
    ws.append(headers)

    for row_data in data:
        row = [
            row_data.get('Topic ID'),
            row_data.get('Post ID'),
            row_data.get('Post URL'),
            row_data.get('Username'),
            row_data.get('Created At'),
            row_data.get('Content')
        ]
        ws.append(row)

    wb.save(filename)
    print(f"Data successfully saved to {filename}")


# IIT Madras BS Program Discourse URL
DISCOURSE_URL = "https://discourse.onlinedegree.iitm.ac.in"

browser_cookies = {
    '_t': '_t',
    '_forum_session': '_forum_session'
}

session = create_session_with_browser_cookies(DISCOURSE_URL, browser_cookies)

response = session.get(f"{DISCOURSE_URL}/session/current.json")
if response.status_code == 200:
    user_data = response.json()
    print(f"Authenticated as: {user_data['current_user']['username']}")
    print(f"Successfully connected to IIT Madras BS Program forum")

    start_date = "2025-01-01"
    end_date = "2025-04-14"
    category_id = 34 

    posts_and_responses = get_discourse_posts_and_responses(session, DISCOURSE_URL, category_id, start_date, end_date)

    if posts_and_responses:
        save_to_excel(posts_and_responses)
    else:
        print("No posts found for the specified criteria.")

else:
    print("Authentication failed using browser cookies")
