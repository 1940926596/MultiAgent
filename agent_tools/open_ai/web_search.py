from duckduckgo_search import DDGS

def search_wikipedia_summary(company_name: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(f"{company_name} site:en.wikipedia.org", max_results=1)
        for r in results:
            return r.get("body", "No Wikipedia summary found.")
    return "No Wikipedia data found."

def search_industry_news(company_name: str, year="2024") -> str:
    with DDGS() as ddgs:
        results = ddgs.text(f"{company_name} industry trend {year} site:investopedia.com OR site:forbes.com", max_results=2)
        combined = "\n".join([r.get("body", "") for r in results])
        return combined if combined else "No relevant industry news found."