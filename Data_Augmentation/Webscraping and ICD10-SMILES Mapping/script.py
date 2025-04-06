import requests
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor
import argparse

class ClinicalTrialsDownloader:
    """
    A class to download data from ClinicalTrials.gov API for a list of NCT numbers.
    """
    
    def __init__(self, output_dir="downloaded_trials", max_workers=5, use_modern_api=True):
        """
        Initialize the downloader with configuration options.
        
        Args:
            output_dir (str): Directory where files will be saved
            max_workers (int): Maximum number of concurrent downloads
            use_modern_api (bool): Whether to use modern API (True) or legacy API (False)
        """
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.use_modern_api = use_modern_api
        
        # Base URLs for API endpoints
        self.modern_api_base = "https://clinicaltrials.gov/api/v2/studies"
        self.legacy_api_base = "https://clinicaltrials.gov/api/legacy/public-xml"
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def download_single_study(self, nct_id):
        """
        Download data for a single NCT ID.
        
        Args:
            nct_id (str): The NCT ID to download
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.use_modern_api:
            # Modern API - JSON format
            url = f"{self.modern_api_base}/{nct_id}"
            output_file = os.path.join(self.output_dir, f"{nct_id}.json")
            headers = {"Accept": "application/json"}
        else:
            # Legacy API - XML format
            url = f"{self.legacy_api_base}/{nct_id}"
            output_file = os.path.join(self.output_dir, f"{nct_id}.xml")
            headers = {"Accept": "application/xml"}
        
        try:
            print(f"Downloading {nct_id}...")
            response = requests.get(url, headers=headers)
            
            # Check if request was successful
            if response.status_code == 200:
                # Save the response content to file
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                print(f"Successfully downloaded {nct_id}")
                return True
            else:
                print(f"Failed to download {nct_id}: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error downloading {nct_id}: {str(e)}")
            return False
    
    def download_multiple_studies(self, nct_ids):
        """
        Download data for multiple NCT IDs using parallel requests.
        
        Args:
            nct_ids (list): List of NCT IDs to download
            
        Returns:
            dict: Results with counts of successful and failed downloads
        """
        successful = 0
        failed = 0
        
        # Use ThreadPoolExecutor to download in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self.download_single_study, nct_ids))
            
            # Count successes and failures
            for result in results:
                if result:
                    successful += 1
                else:
                    failed += 1
        
        return {
            "total": len(nct_ids),
            "successful": successful,
            "failed": failed
        }
    
    def download_bulk_query(self, nct_ids, fields=None):
        """
        Download data for multiple NCT IDs using a bulk query approach.
        This is more efficient for large numbers of NCT IDs.
        
        Args:
            nct_ids (list): List of NCT IDs to download
            fields (list): Optional list of specific fields to retrieve (modern API only)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # This approach is only available with the modern API
        if not self.use_modern_api:
            print("Bulk query is only available with the modern API. Please set use_modern_api=True")
            return False
        
        try:
            # Prepare NCT IDs as a comma-separated list for the query parameter
            nct_id_list = ",".join(nct_ids)
            
            # Construct the URL with query parameters
            url = f"{self.modern_api_base}?query.term=AREA[NCTId]{nct_id_list}"
            
            # Add fields parameter if specified
            if fields:
                url += f"&fields={','.join(fields)}"
                
            # Add format parameter for JSON
            url += "&format=json"
            
            print(f"Performing bulk query for {len(nct_ids)} NCT IDs...")
            response = requests.get(url)
            
            if response.status_code == 200:
                # Save the response as a single JSON file
                output_file = os.path.join(self.output_dir, "bulk_results.json")
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                print(f"Successfully downloaded bulk data to {output_file}")
                return True
            else:
                print(f"Failed to perform bulk query: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error performing bulk query: {str(e)}")
            return False


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Download clinical trial data from ClinicalTrials.gov")
    parser.add_argument("--input", required=True, help="Input file with NCT IDs (one per line)")
    parser.add_argument("--output", default="downloaded_trials", help="Output directory for downloaded files")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent downloads")
    parser.add_argument("--legacy", action="store_true", help="Use legacy API (XML format)")
    parser.add_argument("--bulk", action="store_true", help="Use bulk query method instead of individual downloads")
    
    args = parser.parse_args()
    
    # Read NCT IDs from input file
    try:
        with open(args.input, 'r') as f:
            nct_ids = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(nct_ids)} NCT IDs from {args.input}")
        
        # Initialize downloader
        downloader = ClinicalTrialsDownloader(
            output_dir=args.output,
            max_workers=args.workers,
            use_modern_api=not args.legacy
        )
        
        # Download the data
        if args.bulk and not args.legacy:
            # Use bulk query method
            result = downloader.download_bulk_query(nct_ids)
            if result:
                print("Bulk download completed successfully")
            else:
                print("Bulk download failed")
        else:
            # Download each NCT ID individually
            results = downloader.download_multiple_studies(nct_ids)
            print(f"\nDownload summary:")
            print(f"  Total: {results['total']}")
            print(f"  Successful: {results['successful']}")
            print(f"  Failed: {results['failed']}")
            
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()