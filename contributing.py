import requests

REPO_OWNER = "ChenyangSi"
REPO_NAME = "FreeU"

# GitHub API endpoint for pull requests
API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls"

# Headers to handle GitHub rate limits (if necessary, add your personal access token)
HEADERS = {
    'Accept': 'application/vnd.github.v3+json',
    # Uncomment the following line if you're using a personal access token
    # 'Authorization': 'token YOUR_GITHUB_TOKEN',
}

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
                contributors[user] = []
            contributors[user].append({'title': pr_title, 'url': pr_url})
        
        page += 1  # Increment page for pagination

    return contributors

def update_contributors_markdown(contributors):
    """Update the contributors and their PRs in CONTRIBUTING.md."""
    # Read the existing content of the file
    with open('CONTRIBUTING.md', 'r') as f:
        content = f.readlines()

    # Prepare a new content list
    new_content = []
    contributors_section_found = False

    for line in content:
        # Check for the contributors section
        if line.startswith("# Contributors"):
            contributors_section_found = True
            new_content.append(line)  # Add the header
            new_content.append("\n")  # Add a newline
            
            # Add updated contributors
            for user, prs in contributors.items():
                pr_links = ', '.join([f"[{pr['title']}]({pr['url']})" for pr in prs])
                new_content.append(f"[![{user}](https://img.shields.io/badge/{user}-blue?style=flat-square)](https://github.com/{user}) {pr_links}\n\n")
            continue  # Skip the old contributors section

        # Add lines before contributors section as is
        if not contributors_section_found:
            new_content.append(line)  
       
    # If the contributors section was not found, append it
    if not contributors_section_found:
        new_content.append("\n\n# Contributors\n\n")
        for user, prs in contributors.items():
            pr_links = ', '.join([f"[{pr['title']}]({pr['url']})" for pr in prs])
            new_content.append(f"[![{user}](https://img.shields.io/badge/{user}-blue?style=flat-square)](https://github.com/{user}) {pr_links}\n\n")
    
    # Write the updated content back to the file
    with open('CONTRIBUTING.md', 'w') as f:
        f.writelines(new_content)

if __name__ == "__main__":
    contributors = get_contributors_and_prs()
    update_contributors_markdown(contributors)
