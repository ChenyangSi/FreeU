import requests
import os

REPO_OWNER = "ChenyangSi"
REPO_NAME = "FreeU"

# GitHub API endpoint for pull requests and users
API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls"
USER_API_URL = "https://api.github.com/users/{username}"

# Headers to handle GitHub rate limits (if necessary, add your personal access token)
HEADERS = {
    'Accept': 'application/vnd.github.v3+json',
    # Uncomment the following line if you're using a personal access token
    # 'Authorization': 'token YOUR_GITHUB_TOKEN',
}

def get_user_profile(username):
    """Fetch GitHub user profile information including name and avatar URL."""
    user_response = requests.get(USER_API_URL.format(username=username), headers=HEADERS)
    user_data = user_response.json()
    
    # Fallback to username if the full name is missing (None)
    full_name = user_data.get('name') if user_data.get('name') else username
    avatar_url = user_data.get('avatar_url')
    
    return full_name, avatar_url

def get_contributors_and_prs():
    """Fetch all closed pull requests from the repository."""
    contributors = {}
    page = 1
    while True:
        response = requests.get(API_URL, headers=HEADERS, params={'state': 'closed', 'page': page, 'per_page': 100})
        pulls = response.json()
        
        if not pulls:
            break

        for pr in pulls:
            user = pr['user']['login']
            pr_title = pr['title']
            pr_url = pr['html_url']
            
            if user not in contributors:
                # Fetch user's full name and avatar
                full_name, avatar_url = get_user_profile(user)
                contributors[user] = {'name': full_name, 'avatar_url': avatar_url, 'prs': []}
                
            contributors[user]['prs'].append({'title': pr_title, 'url': pr_url})
        
        page += 1  # Increment page for pagination

    return contributors

def generate_contributors_markdown(contributors):
    """Append the contributors and their PRs to CONTRIBUTING.md, avoid re-adding existing contributors."""
    # Load current CONTRIBUTING.md to avoid duplicate data
    if os.path.exists('CONTRIBUTING.md'):
        with open('CONTRIBUTING.md', 'r') as f:
            existing_content = f.read()
    else:
        existing_content = ""
    
    with open('CONTRIBUTING.md', 'a') as f:
        if "# Contributors" not in existing_content:
            f.write("\n\n# Contributors\n\n")
        
        for user, data in contributors.items():
            # Skip contributors who are already in the file
            if f"[@{user}]" in existing_content:
                continue
            
            # Embed profile link into the name (or username if name is missing)
            f.write(f"<img src='{data['avatar_url']}' width='80' height='80' align='left'>\n")
            f.write(f"**[{data['name']}](https://github.com/{user})**\n\n")  # Embed name with profile link
            f.write(f"- **Pull Requests**: {len(data['prs'])}\n")
            
            # Add collapsible dropdown for the PR titles
            f.write(f"<details>\n  <summary>View Pull Requests</summary>\n  <ul>\n")
            for pr in data['prs']:
                f.write(f"    <li><a href='{pr['url']}'>{pr['title']}</a></li>\n")
            f.write("  </ul>\n</details>\n\n")
            f.write("<br clear='all'/>\n\n")  # To clear floating image

if __name__ == "__main__":
    contributors = get_contributors_and_prs()
    generate_contributors_markdown(contributors)
