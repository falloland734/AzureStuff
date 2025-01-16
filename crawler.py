import asyncio
from crawl4ai import AsyncWebCrawler
from openai import OpenAI

class CrawledResult:
    """
    A small wrapper class that holds the crawled data (dict)
    and provides methods to post-process it in a fluent style.
    """
    def __init__(self, data_dict: dict):
        self.data_dict = data_dict

    def check_with_ai(self, api_key: str) -> "CrawledResult":
        """
        Calls the AI-based check on the held data_dict to remove “useless” entries.
        Returns self so you can chain calls if desired.
        """
        self.data_dict = Crawler.check_with_ai(self.data_dict, api_key)
        return self  # Return the same instance, so you can chain


class Crawler:
    """
    A class that provides two main crawling methods:
      1) crawl_links(websiteURL) -> returns a CrawledResult object
      2) check_with_ai(clean_data_dict, api_key) -> filters dict items using OpenAI
    """

    @staticmethod
    def get_links(websiteURL: str) -> set[str]:
        """
        Synchronously retrieve all internal links from the given websiteURL.
        
        :param websiteURL: The initial website URL to crawl.
        :return: A set of internal links (strings).
        
        This method blocks until the crawling is complete.
        """
        return asyncio.run(Crawler._async_get_links(websiteURL))

    @staticmethod
    async def _async_get_links(websiteURL: str) -> set[str]:
        """
        The internal async version of link retrieval.
        """
        async with AsyncWebCrawler(verbose=True) as crawler:
            result = await crawler.arun(
                url=websiteURL,
                exclude_external_links=True,
                remove_overlay_elements=True
            )
        # Build a set of unique internal links (removing trailing slashes).
        return {item['href'].rstrip("/") for item in result.links["internal"]}

    @staticmethod
    def crawl_links(websiteURL: str) -> CrawledResult:
        """
        Synchronously crawl all internal links for the given URL
        and retrieve their Markdown content, returning a CrawledResult.
        
        :param websiteURL: The initial website URL to crawl.
        :return: A CrawledResult object containing { short_link: markdown_content }.
        """
        data_dict = asyncio.run(Crawler._async_crawl_links(websiteURL))
        return CrawledResult(data_dict)

    @staticmethod
    async def _async_crawl_links(websiteURL: str) -> dict:
        """
        The internal async version of the crawling workflow:
          1) Gather all links from websiteURL (async).
          2) For each link, fetch the markdown content.
          3) Return a dict of short_link -> markdown content.
        """
        # First, gather the link set
        link_set = await Crawler._async_get_links(websiteURL)

        clean_data_dict = {}
        async with AsyncWebCrawler() as crawler:
            for link in link_set:
                result = await crawler.arun(
                    url=link,
                    exclude_external_links=True,
                    remove_overlay_elements=True,
                    process_iframes=False,
                    excluded_tags=['form', 'header', 'footer', 'nav'],
                    magic=True,
                    exclude_external_images=True
                )
                # Create a readable key name from the link
                short_link = (
                    link
                    .replace("https://www.", "")
                    .replace(".com", "")
                    .replace("https:", "")
                    .replace("/", "_")
                )

                clean_data_dict[short_link] = result.markdown
        return clean_data_dict

    @staticmethod
    def check_with_ai(clean_data_dict: dict, api_key: str) -> dict:
        """
        Passes each dict entry's content to OpenAI and removes entries
        classified as 'useless' based on the system instructions.
        """
        keys_to_remove = []
        
        client = OpenAI(api_key=api_key)

        for key, value in clean_data_dict.items():
            system_message = (
                """
                System Prompt
                You are an assistant that classifies text as either "useless" or "useful" for a knowledge base.

                A file is "useless" if the majority or essence of its content falls into one or more of the following categories:
                1. It’s a placeholder or error page, such as:
                  - “# This page isn’t available,” with phrases like “Please check that the URL entered is correct,” 
                    “try loading the page again,” or “please contact the website owner.”
                  - A JSON error, for example:
                      {
                        "status":"error",
                        "message":"Cannot parse path tel: 18005462070",
                        "correlationId":"..."
                      }
                  - A page saying only “Loading,” “Close dialog,” or similarly trivial placeholders.
                  - Essentially empty, containing only whitespace, \\n, or a few filler words.
                2. It’s solely composed of error or placeholder text with no substantial content.

                A file is “useful” if there is any significant content beyond trivial or placeholder text, 
                even if it includes some placeholder or error fragments. Do not classify it as “useless” 
                if there is meaningful text that might be relevant to a knowledge base or a RAG system.

                Your output must be a single word: "useless" or "useful".
                """
            )
            user_message = (
                f"Decide whether the following text is 'useless' or 'useful' based on the above instructions: \n\n{value}"
            )
            
            response = client.chat.completions.create(
                model="gpt-4o",  # or whichever model you want
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=20,
                temperature=0.0,
            )

            answer = response.choices[0].message.content.lower()

            if "useless" in answer:
                keys_to_remove.append(key)

        for useless in keys_to_remove:
            clean_data_dict.pop(useless, None)

        return clean_data_dict
