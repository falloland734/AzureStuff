import asyncio
from crawl4ai import AsyncWebCrawler
from openai import OpenAI

class CrawledResult:
    """
    A small wrapper class that holds the crawled data (dict) 
    and provides methods to post-process it in a fluent (chained) style.

    :param data_dict: A dictionary of `{ short_link: markdown_content }`.
    :type data_dict: dict

    Example::
    
        # Suppose you have some data dict
        data = {"about_us": "Some content...", "contact_us": "Contact details..."}
        result = CrawledResult(data)
        clean_result = result.check_with_ai("YOUR_OPENAI_API_KEY")

        # `clean_result.data_dict` is now filtered / updated in-place.
    """

    def __init__(self, data_dict: dict):
        """
        Initialize the CrawledResult with a data dictionary.

        :param data_dict: A dict containing the crawled data, 
                          typically { short_link: markdown_content }.
        :type data_dict: dict
        """
        self.data_dict = data_dict

    def check_with_ai(self, model : str, api_key: str) -> "CrawledResult":
        """
        Calls the AI-based check on the `data_dict` to remove “useless” entries.
        This method mutates the internal `data_dict` and returns `self`, 
        enabling a fluent (chainable) usage pattern.

        :param model: Your OpenAI model that you want to use to analyze and generate a response. (gpt-3.5, etc)
        :type model: str
        :param api_key: Your OpenAI API key to authenticate requests.
        :type api_key: str
        :return: Returns the same CrawledResult instance, so you can chain calls.
        :rtype: CrawledResult

        Example usage (in a chained style)::

            result = Crawler.crawl_links("https://example.com").check_with_ai("YOUR_OPENAI_API_KEY")
            # Now `result.data_dict` is filtered to remove "useless" entries.
        """
        # Delegate the actual logic to Crawler.check_with_ai(...)
        self.data_dict = Crawler.check_with_ai(self.data_dict, model, api_key)
        return self


class Crawler:
    """
    A class that provides crawling methods for a given URL, returning
    crawled Markdown data and an AI-driven filtering mechanism.

    Public static methods:
      1) crawl_links(websiteURL) -> CrawledResult
         - Crawls the given URL for internal links and scrapes their Markdown content.
      2) check_with_ai(clean_data_dict, api_key) -> dict
         - Filters out "useless" content from a dictionary by querying OpenAI.

    Internal (async) methods:
      1) _async_crawl_links(websiteURL) -> dict
      2) _async_get_links(websiteURL) -> set[str]
         - Called internally via `asyncio.run()`.
    """

    @staticmethod
    def get_links(websiteURL: str) -> set[str]:
        """
        Synchronously retrieve all internal links from the given website URL.

        This method runs `_async_get_links(...)` under the hood using `asyncio.run()`,
        so the caller does not need to worry about async/await.

        :param websiteURL: The initial website URL to crawl, e.g. "https://example.com".
        :type websiteURL: str
        :return: A set of internal links, each being a string URL.
        :rtype: set[str]
        """
        return asyncio.run(Crawler._async_get_links(websiteURL))

    @staticmethod
    async def _async_get_links(websiteURL: str) -> set[str]:
        """
        Internal async method to get all internal links from the given website URL.

        :param websiteURL: The initial website URL to crawl, e.g. "https://example.com".
        :type websiteURL: str
        :return: A set of internal links, each being a string URL.
        :rtype: set[str]

        You typically won't call this method directly; use `get_links(...)` instead.
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
        and retrieve their Markdown content, returning a `CrawledResult`.

        This method runs `_async_crawl_links(...)` under the hood using `asyncio.run()`.

        :param websiteURL: The initial website URL to crawl, e.g. "https://example.com".
        :type websiteURL: str
        :return: A CrawledResult object containing a dict of { short_link: markdown_content }.
        :rtype: CrawledResult

        Example usage::

            # Returns a CrawledResult object
            result = Crawler.crawl_links("https://example.com")
            
            # You can optionally filter out "useless" content next
            result.check_with_ai(api_key="YOUR_OPENAI_API_KEY")
        """
        data_dict = asyncio.run(Crawler._async_crawl_links(websiteURL))
        return CrawledResult(data_dict)

    @staticmethod
    async def _async_crawl_links(websiteURL: str) -> dict:
        """
        Internal async crawling workflow:
          1) Gather all internal links from `websiteURL`.
          2) For each link, fetch the markdown content.
          3) Return a dict of short_link -> markdown content.

        :param websiteURL: The initial website URL to crawl, e.g. "https://example.com".
        :type websiteURL: str
        :return: A dictionary where each key is a "short_link" version of the URL,
                 and each value is the crawled Markdown content.
        :rtype: dict
        """
        # Step 1) Gather internal links
        link_set = await Crawler._async_get_links(websiteURL)

        # Step 2) For each link, fetch and store the markdown content
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
                # Generate a short, file-friendly link key
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
    def check_with_ai(clean_data_dict: dict, model : str, api_key: str) -> dict:
        """
        Passes each dict entry's content to OpenAI for classification
        and removes entries classified as 'useless' based on the system instructions.
        
        :param clean_data_dict: A dictionary of { short_link: markdown_content }.
        :type clean_data_dict: dict
        :param model: Your OpenAI model that you want to use to analyze and generate a response. (gpt-3.5, etc)
        :type model: str
        :param api_key: Your OpenAI API key to authenticate requests.
        :type api_key: str
        :return: The filtered dictionary with "useless" entries removed.
        :rtype: dict

        The classification logic:
          - If the system determines the majority of the text is
            error pages, placeholders, or empty, it is labeled "useless".
          - Otherwise, it's labeled "useful".
        """
        # We'll collect keys that are "useless" and pop them from the dict later
        keys_to_remove = []
        
        # Instantiate OpenAI client
        client = OpenAI(api_key=api_key)

        # Iterate over each piece of content
        for key, value in clean_data_dict.items():
            # Define the system prompt that explains how to classify text
            system_message = """
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

            # The user prompt for classification
            user_message = (
                "Decide whether the following text is 'useless' or 'useful' based on the above instructions:\n\n"
                + value
            )
            
            # Make the request to OpenAI
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=20,
                temperature=0.0,
            )

            # Process the AI response (make it lowercase for easy checks)
            answer = response.choices[0].message.content.lower()

            if "useless" in answer:
                # Mark for removal
                keys_to_remove.append(key)

        # Remove all keys determined to be useless
        for useless_key in keys_to_remove:
            clean_data_dict.pop(useless_key, None)

        return clean_data_dict
