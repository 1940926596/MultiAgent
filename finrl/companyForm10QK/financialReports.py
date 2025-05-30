from sec_edgar_downloader import Downloader

# Initialize a downloader instance. Download filings to the current
# working directory. Must declare company name and email address
# to form a user-agent string that complies with the SEC Edgar's
# programmatic downloading fair access policy.
# More info: https://www.sec.gov/os/webmaster-faq#code-support
# Company name and email are used to form a user-agent of the form:
# User-Agent: <Company Name> <Email Address>
dl = Downloader("MyCompanyName", "my.email@domain.com")

# # 下载最新的两个10-K文件
# dl.get("10-K", "AAPL", limit=5)

# 下载最新的两个10-Q文件
# dl.get("10-Q", "AAPL", limit=20)

# # Get all 8-K filings for Apple (ticker: AAPL)
dl.get("8-K", "AAPL",limit=20)

# # Get all 8-K filings for Apple, including filing amends (8-K/A)
# dl.get("8-K", "AAPL", include_amends=True)

# # Get all 8-K filings for Apple after January 1, 2017 and before March 25, 2017
# # Note: after and before strings must be in the form "YYYY-MM-DD"
# dl.get("8-K", "AAPL", after="2017-01-01", before="2017-03-25")

# # Get the five most recent 8-K filings for Apple
# dl.get("8-K", "AAPL", limit=5)

# # Get all 10-K filings for Microsoft
# dl.get("10-K", "MSFT")

# # Get the latest 10-K filing for Microsoft
# dl.get("10-K", "MSFT", limit=1)

# # Get all 10-Q filings for Visa
# dl.get("10-Q", "V")

# # Get all 13F-NT filings for the Vanguard Group
# dl.get("13F-NT", "0000102909")

# # Get all 13F-HR filings for the Vanguard Group
# dl.get("13F-HR", "0000102909")

# # Get all SC 13G filings for Apple
# dl.get("SC 13G", "AAPL")

# # Get all SD filings for Apple
# dl.get("SD", "AAPL")