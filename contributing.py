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

def generate_contributors_markdown(contributors):
    """Append the contributors and their PRs to CONTRIBUTING.md."""
    with open('CONTRIBUTING.md', 'a') as f:
        f.write("\n\n# Contributors\n\n")
        f.write("| Contributor | Pull Requests |\n")
        f.write("|-------------|----------------|\n")

        for user, prs in contributors.items():
            pr_links = ', '.join([f"[{pr['title']}]({pr['url']})" for pr in prs])
            f.write(f"| [@{user}](https://github.com/{user}) | {pr_links} |\n")

if __name__ == "__main__":
    contributors = get_contributors_and_prs()
    generate_contributors_markdown(contributors)
