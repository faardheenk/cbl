#!/usr/bin/env python3

import logging
from match_excel import run_matching_process

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for the application.
    TODO: Implement SharePoint integration
    """
    logger.info("Starting application...")
    
    # TODO: Add SharePoint integration code here
    # This will include:
    # 1. Connecting to SharePoint
    # 2. Downloading files
    # 3. Running the matching process
    # 4. Uploading results back to SharePoint
    
    # For now, we'll just run the matching process directly
    run_matching_process()

if __name__ == "__main__":
    main() 