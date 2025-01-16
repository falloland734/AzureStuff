from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient
from urllib.parse import urlparse
from crawler import Crawler

class BlobUploader:
    """
    A class that handles crawling a website for data and uploading it to an Azure Blob container.
    The container name is automatically derived from the domain name of the target website
    (e.g., 'https://www.example.com/' -> 'example').
    """

    def __init__(self, website_link: str, account_url: str, credential: str):
        """
        Initialize the BlobUploader with the website link to crawl and
        the Azure Blob Storage account details.

        :param website_link: The full URL of the website to crawl 
                             (e.g., "https://www.example.com/").
        :param account_url:  The URL endpoint for the Azure Storage account 
                             (e.g., "https://<account_name>.blob.core.windows.net/").
        :param credential:   The access key or SAS token used to authenticate with
                             the Azure Storage account.
        """
        self.website_link = website_link
        self.account_url = account_url
        self.credential = credential
        
        # Create the BlobServiceClient once at initialization, so we can reuse it.
        self.blob_service_client = BlobServiceClient(
            account_url=self.account_url,
            credential=self.credential
        )
        
        # Will store the crawler results as a dictionary: { "page_name": "page_content", ... }
        self.files = {}

    def gather_data(self) -> None:
        """
        Use the Crawler class to retrieve text content from each link found
        on the given website_link. The results are stored in 'self.files'.
        
        The dictionary keys typically correspond to link identifiers,
        while the values are the extracted text or HTML content.
        """
        self.files = Crawler().crawl_links(self.website_link)

    def get_container_name(self) -> str:
        """
        Parse the website_link to strip out the main body of the domain.
        For example, given 'https://www.example.com/', we return 'example'.

        :return: The extracted domain portion suitable for an Azure container name.
        """
        parsed_url = urlparse(self.website_link)
        netloc = parsed_url.netloc  # e.g., "www.example.com"
        
        # If the domain starts with 'www.', remove it.
        if netloc.startswith("www."):
            netloc = netloc[4:]  # "example.com"
        
        # Split on '.' and take the first part for the container name (e.g., "example").
        parts = netloc.split(".")
        container_name = parts[0]
        
        return container_name

    def create_or_update_blob(self, container_name: str) -> None:
        """
        Creates or updates the specified container, then uploads each file in 'self.files'
        as a text blob. If the container already exists, it deletes all existing blobs
        prior to uploading the new data.

        :param container_name: The name of the container to use in Azure Blob Storage.
        :raises ResourceExistsError: If the container creation fails due to existing container.
        """
        try:
            # Attempt to create a new container. If it doesn't exist, it will be created;
            # otherwise, a ResourceExistsError is raised. We clear the container and upload new blobs
            self.blob_service_client.create_container(name=container_name)
            print(f"Container '{container_name}' created.")
        except ResourceExistsError:
            print(f"Container '{container_name}' already exists. Deleting existing blobs...")
            container_client = self.blob_service_client.get_container_client(container_name)

            # Remove all existing blobs in that container
            for blob in container_client.list_blobs():
                container_client.delete_blob(blob.name)

        # Upload the new data from self.files
        for key, value in self.files.items():
            # Each file is uploaded as a separate blob named '<key>.txt'
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name, 
                blob=f"{key}.txt"
            )
            data = value.encode("utf-8")  # Convert text to bytes
            blob_client.upload_blob(data, blob_type="BlockBlob")

        print("Upload complete!")

    def run(self) -> None:
        """
        Coordinates the entire process of:
        1) Gathering data by crawling the website_link.
        2) Determining a suitable container name from the domain name.
        3) Creating or clearing the container, then uploading the crawled data as text blobs.
        """
        self.gather_data()             # Collect link data from the website
        container_name = self.get_container_name()  # Derive container name from domain
        self.create_or_update_blob(container_name)  # Upload the data to Azure Blob Storage
