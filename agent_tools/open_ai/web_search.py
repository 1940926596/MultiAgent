from duckduckgo_search import DDGS

def search_wikipedia_summary(company_name: str) -> str:
    # with DDGS() as ddgs:
    #     results = ddgs.text(f"{company_name} site:en.wikipedia.org", max_results=10)
    #     for r in results:
    #         return r.get("body", "No Wikipedia summary found.")
    
    return """
        Apple Inc. is an American multinational technology company founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne. Originally named Apple Computer, Inc., the company was established to develop and sell the Apple I personal computer designed by Wozniak. (Wikipedia, Apple Inc.)
        Apple is known for its innovative consumer electronics, including the iPhone smartphone, iPad tablet, Mac personal computer, iPod music player, Apple Watch smartwatch, Apple Vision Pro spatial computing device, Apple TV digital media player, AirPods wireless earbuds, and HomePod smart speaker. (Wikipedia, Apple Inc. Products)
        On the software side, Apple develops multiple operating systems such as macOS, iOS, iPadOS, watchOS, tvOS, and visionOS, along with software including the iTunes media player, Safari web browser, and the Shazam music identification app. The company also offers various online services like the iTunes Store, App Store, Apple Music, Apple TV+, iCloud, iMessage, and Apple Pay. (Wikipedia, Apple Inc. Software and Services)
        Apple’s headquarters is located in Cupertino, California, at the Apple Park campus, also known as the “Spaceship” campus, designed by architect Norman Foster and situated at 1 Apple Park Way. (Wikipedia, Apple Park)
        As of 2024, Apple’s market capitalization exceeds $3.74 trillion, making it one of the world’s most valuable companies. (Wikipedia, Apple Inc.)
        """
    

def search_industry_news(company_name: str, year="2024") -> str:
    # with DDGS() as ddgs:
    #     results = ddgs.text(f"{company_name} industry trend {year} site:investopedia.com OR site:forbes.com", max_results=2)
    #     combined = "\n".join([r.get("body", "") for r in results])
    #     return combined if combined else "No relevant industry news found."
    if year==2022:
        return """
        Apple Inc. Annual News Summary (2022–2024)
        2022: Financial Growth and Product Expansion
        Financial Performance: Apple reported a record revenue of $394.3 billion for the fiscal year 2022, marking an 8% increase year-over-year. The fourth quarter alone saw a revenue of $90.1 billion, up 8% from the previous year, with earnings per share rising by 4% to $1.29. 
        Product Launches: The company introduced the iPhone 14 series, the Apple Watch Series 8, and the AirPods Pro (2nd generation), expanding its product lineup and reinforcing its ecosystem.
        Environmental Initiatives: Apple continued its commitment to sustainability, operating on 100% renewable energy and launching its first carbon-neutral Apple Watch models.
        """
    
    if year==2023:
        return """
        2023: Innovation and Strategic Acquisitions
        Financial Results: In Q2 2023, Apple reported a revenue of $94.8 billion, with services revenue reaching an all-time high. The installed base of active devices surpassed 2.2 billion, setting a new record. 
        Product Developments: Apple unveiled the Vision Pro, a mixed-reality headset, at WWDC 2023. The device was released in early 2024, with initial shipments selling out rapidly. 
        Acquisitions: The company acquired Mira, an AR headset startup, to enhance its mixed-reality capabilities. 
        """
    
    if year==2024:
        return """
        2024: AI Integration and Legal Challenges
        Financial Performance: In Q1 2024, Apple achieved a revenue of $119.6 billion, with earnings per share increasing by 16% to $2.18. The company also announced a $95 million settlement in a lawsuit alleging Siri's unauthorized recording of private conversations. 
        Product Launches: The iPhone 16 series, Apple Watch Series 10, and AirPods 4 were introduced, featuring advanced health monitoring capabilities. 
        AI Developments: Apple launched Apple Intelligence, an on-device AI system, and introduced the A18 and A18 Pro chips, enhancing AI performance across devices. 
        Legal Issues: The U.S. Department of Justice filed an antitrust lawsuit against Apple, accusing the company of monopolistic practices in the smartphone market. 
        """