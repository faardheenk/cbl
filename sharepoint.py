#!/usr/bin/env python3

import os
import json
from office365.sharepoint.client_context import ClientContext
from dotenv import load_dotenv
import logging
import re
from match_excel import run_matching_process
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# log_file = "E:\\FRCI\\AuditLog.txt"
# message = f"Script ran at: {datetime.datetime.now()}\n"
 
# with open(log_file, "a") as f:
#     f.write(message)

def sanitize_json_string(s):
    # Remove trailing commas before closing } or ]
    s = re.sub(r',\s*([}\]])', r'\1', s)
    return s
class SharePointService:
    def __init__(self):
        """Initialize SharePoint service with credentials from environment variables."""
        # Load environment variables
        load_dotenv(override=True)
        
        # Get SharePoint configuration from environment variables
        # self.site_url = os.getenv('SITE_URL')
        # self.client_username = os.getenv('SP_USERNAME')
        # self.client_password = os.getenv('PASSWORD')
        self.site_url = os.getenv('SITE_URL')
        self.cert_thumbprint = os.getenv('CERT_THUMBPRINT')
        self.client_id = os.getenv('CLIENT_ID')
        self.tenant=os.getenv('TENANT')
        self.cert_path=os.getenv('CERT_PATH')

        # self.site_url = "https://citybrokersltdmu.sharepoint.com/sites/statementrecon"
        # self.cert_thumbprint = "B3F2EB224794D54AF99FD443D1E4ABFEF8E10C7B"
        # self.client_id = "74de1033-3314-49cc-8f5a-829e0ec76b27"
        # self.tenant= "citybrokersltdmu.onmicrosoft.com"
        # self.cert_path= "E:\\FRCI\\certificate\\cert.pem"


        self.cert_credentials = { 
            'tenant': self.tenant,
            'client_id': self.client_id,
            'thumbprint': self.cert_thumbprint,
            'cert_path': self.cert_path
        }
        self.ctx = None
        
        # # Validate required environment variables
        # if not all([self.site_url, self.client_username, self.client_password]):
        #     raise ValueError("Missing required SharePoint configuration in .env file")
        
        # # Initialize SharePoint credentials
        # self.credentials = UserCredential(self.client_username, self.client_password)
        # self.ctx = None


    def get_client_context(self):
        """Get SharePoint client context."""
        print(self.site_url)
        try:
            if not self.ctx:
                self.ctx = ClientContext(self.site_url).with_client_certificate(**self.cert_credentials)
            return self.ctx
        except Exception as ex:
            logger.error(f"Failed to get SharePoint client context: {str(ex)}")
            raise

    def get_column_mappings(self,  insurer_name = None) :
        try: 
            library_name = "Mappings"
            ctx = self.get_client_context()
            library = ctx.web.lists.get_by_title(library_name)
            
            query_cbl = f"Title eq 'CBL'"
            query_insurer = f"Title eq '{insurer_name}'"

            items_insurer = library.items.filter(query_insurer).select(['ColumnMappings']).get().execute_query()
            items_cbl = library.items.filter(query_cbl).select(['ColumnMappings']).get().execute_query()
            
            # Parse the raw column mappings from SharePoint
            insurer_mappings_raw = items_insurer[0].properties['ColumnMappings']
            cbl_mappings_raw = items_cbl[0].properties['ColumnMappings']
            
            # Parse JSON strings into Python dictionaries
            insurer_mappings = json.loads(sanitize_json_string(insurer_mappings_raw)) if insurer_mappings_raw else {}
            cbl_mappings = json.loads(sanitize_json_string(cbl_mappings_raw)) if cbl_mappings_raw else {}

            print(insurer_mappings)
            print(cbl_mappings)
            
            # Return the mappings in the expected format
            return {
                'insurer_mappings': insurer_mappings,
                'cbl_mappings': cbl_mappings
            }
        except Exception as ex:
            logger.error(f"Failed to get column mappings: {str(ex)}")
            raise

    def get_matrix(self, insurer_name=None):
        try:
            ctx = self.get_client_context()
            library = ctx.web.lists.get_by_title("Matrix")
            ctx.load(library)
            ctx.execute_query()

            # Get the root folder of the library
            root_folder = library.root_folder
            ctx.load(root_folder)
            ctx.execute_query()

            # Get all subfolders in the root folder
            folders = root_folder.folders
            ctx.load(folders)
            ctx.execute_query()

            # Find the folder matching the insurer_name
            insurer_folder_url = None
            for folder in folders:
                if folder.name.lower() == insurer_name.lower():
                    insurer_folder_url = folder.serverRelativeUrl
                    break

            if not insurer_folder_url:
                raise ValueError(f"No folder found for insurer: {insurer_name}")

            matrix_folder = ctx.web.get_folder_by_server_relative_url(insurer_folder_url)
            ctx.load(matrix_folder)
            ctx.execute_query()

            # Get all list items in the folder
            list_items = library.items.filter(f"FileDirRef eq '{insurer_folder_url}'").get().execute_query()

            # Collect all list item details
            all_items = []
            for item in list_items:
                try:
                    # Collect item details
                    item_details = {
                        'matrixKey': item.properties.get('Title'),
                    }
                    all_items.append(item_details)
                except Exception as e:
                    logger.error(f"Error processing list item {item.properties.get('ID')}: {str(e)}")
                    continue

            return all_items

        except Exception as ex:
            logger.error(f"Error getting matrix: {str(ex)}")
            raise

    def get_pending_folders(self, library_name="Reconciliation Library"):
        """
        Get all folders from the specified library that have 'pending' status.
        Processes all insurance folders and their subfolders.
        
        Args:
            library_name (str): Name of the SharePoint library to search in
            
        Returns:
            list: List of folder objects with pending status
        """
        try:
            ctx = self.get_client_context()
            
            # Get the library
            library = ctx.web.lists.get_by_title(library_name)
            ctx.load(library)
            ctx.execute_query()
            
            # Get the root folder
            root_folder = library.root_folder
            ctx.load(root_folder)
            ctx.execute_query()
            
            pending_folders = []
            
            def process_folder(folder, parent_name=None):
                """Process folders and their subfolders."""
                print('PROCESSING FOLDER >> ', folder.name)
                try:
                    # Load folder properties
                    ctx.load(folder)
                    ctx.execute_query()
                    
                    # Get status from list item
                    try:
                        list_item = folder.list_item_all_fields
                        ctx.load(list_item)
                        ctx.execute_query()
                        status = list_item.properties.get('Status')
                        
                        if status and status.lower() == 'pending':
                            pending_folders.append({
                                'name': folder.name,
                                'url': folder.serverRelativeUrl,
                                'status': status,
                                'created': list_item.properties.get('Created', ''),
                                'modified': list_item.properties.get('Modified', ''),
                                'parent_folder': parent_name,
                                'library': library_name
                            })
                            print('PENDING FOLDER >> ', pending_folders)
                    except Exception as e:
                        logger.error(f"Error getting list item for folder {folder.name}: {str(e)}")
                    
                    # Get and process subfolders
                    subfolders = folder.folders
                    ctx.load(subfolders)
                    ctx.execute_query()
                    
                    for subfolder in subfolders:
                        process_folder(subfolder, folder.name)
                        
                except Exception as ex:
                    logger.error(f"Error processing folder {folder.name}: {str(ex)}")
            
            # Get all folders in the root
            root_folders = root_folder.folders
            ctx.load(root_folders)
            ctx.execute_query()
            
            # Process all folders -- PARENT FOLDER
            for folder in root_folders:
                process_folder(folder, None)
            
            logger.info(f"Found {len(pending_folders)} folders with pending status")
            return pending_folders
                
        except Exception as ex:
            logger.error(f"Error getting pending folders: {str(ex)}")
            raise

    def get_excel_files_from_folder(self, folder_url):
        try:
            ctx = self.get_client_context()
            folder = ctx.web.get_folder_by_server_relative_url(folder_url)
            ctx.load(folder)
            ctx.execute_query()
            
            # Update folder status to "In Progress"
            try:
                list_item = folder.list_item_all_fields
                ctx.load(list_item)
                ctx.execute_query()
                list_item.set_property('Status', 'In Progress')
                list_item.update()
                ctx.execute_query()
                logger.info(f"Updated folder status to 'In Progress': {folder.name}")
            except Exception as e:
                logger.error(f"Error updating folder status: {str(e)}")
            
            # Get files in the folder
            files = folder.files
            ctx.load(files)
            ctx.execute_query()
            
            # Filter for Excel files
            excel_files = []
            for file in files:
                if file.name.endswith(('.xlsx', '.xls')):
                    # Update file status to "In Progress"
                    try:
                        file_list_item = file.listItemAllFields
                        ctx.load(file_list_item)
                        ctx.execute_query()
                        file_list_item.set_property('Status', 'In Progress')
                        file_list_item.update()
                        ctx.execute_query()
                        logger.info(f"Updated file status to 'In Progress': {file.name}")
                    except Exception as e:
                        logger.error(f"Error updating file status: {str(e)}")
                    
                    # Download the file content
                    file_content = file.read()
                    excel_files.append({
                        'name': file.name,
                        'content': file_content,
                        'url': file.serverRelativeUrl
                    })
            
            return excel_files
            
        except Exception as ex:
            logger.error(f"Error getting Excel files from folder {folder_url}: {str(ex)}")
            return []
        
    def upload_to_sharepoint(self, file_path, library_name, folder_path):
        try:
            ctx = self.get_client_context()
            library = ctx.web.lists.get_by_title(library_name)
            ctx.load(library)
            ctx.execute_query()
            
            # Get the root folder
            root_folder = library.root_folder
            ctx.load(root_folder)
            ctx.execute_query()
            
            logger.info(f"Root folder URL: {root_folder.serverRelativeUrl}")
            
            # Get the target folder
            target_folder = root_folder
            if folder_path:
                # Extract just the folder path after the library name
                parts = folder_path.split(library_name)
                if len(parts) > 1:
                    folder_path = parts[1]
                
                # Remove leading/trailing slashes
                folder_path = folder_path.strip('/')
                
                logger.info(f"Target folder path: {folder_path}")
                
                # Get the folder by server relative URL
                target_folder = ctx.web.get_folder_by_server_relative_url(f"{root_folder.serverRelativeUrl}/{folder_path}")
                ctx.load(target_folder)
                ctx.execute_query()
            
            # Upload the file
            with open(file_path, 'rb') as content_file:
                file_content = content_file.read()

            file_name = os.path.basename(file_path)
            target_file = target_folder.upload_file(file_name, file_content).execute_query()

            # Get the file object to update its status
            file_obj = ctx.web.get_file_by_server_relative_url(target_file.serverRelativeUrl)
            file_list_item = file_obj.listItemAllFields
            ctx.load(file_list_item)
            ctx.execute_query()
            file_list_item.set_property('Status', 'Manual Review')
            file_list_item.update()
            ctx.execute_query()
            
            logger.info(f"✓ File uploaded successfully to SharePoint: {target_file.serverRelativeUrl}")

            return target_file.serverRelativeUrl
                
        except Exception as ex:
            logger.error(f"Error uploading to SharePoint: {str(ex)}")
            raise

def main():
    try:
        # Initialize SharePoint service
        sharepoint_service = SharePointService()
        
        # Test connection
        ctx = sharepoint_service.get_client_context()
        logger.info("Successfully connected to SharePoint")

        web = ctx.web
        ctx.load(web)
        ctx.execute_query()
        logger.info(f"Connected to SharePoint site: {web.properties['Title']}")
        
        # Get pending folders
        pending_folders = sharepoint_service.get_pending_folders()
        logger.info(f"Found {len(pending_folders)} folders with pending status")
        
        # Process each pending folder
        for folder in pending_folders:
            logger.info(f"\nProcessing folder: {folder['name']}")
            excel_files = sharepoint_service.get_excel_files_from_folder(folder['url'])
    
            if len(excel_files) == 2:
                column_mappings = sharepoint_service.get_column_mappings(folder['parent_folder'])
                # matrix_keys = sharepoint_service.get_matrix(folder['parent_folder'])
                logger.info(f"Found exactly 2 Excel files in {folder['name']}:")
                for file in excel_files:
                    logger.info(f"  - {file['name']}")
                
                # Save files temporarily and run matching process
                try:
                    # Create temporary directory if it doesn't exist
                    temp_dir = "temp_excel_files"
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # Save files and determine which is CBL
                    file_paths = []
                    cbl_index = None
                    for i, file in enumerate(excel_files):
                        file_path = os.path.join(temp_dir, file['name'])
                        with open(file_path, 'wb') as f:
                            f.write(file['content'])
                        file_paths.append(file_path)
                        if 'cbl' in file['name'].lower():
                            cbl_index = i
                    
                    # Ensure CBL file is first
                    if cbl_index == 1:
                        file_paths[0], file_paths[1] = file_paths[1], file_paths[0]
                    
                    # Run matching process
                    result = run_matching_process(
                        column_mappings=column_mappings,
                        matrix_keys=[],
                        cbl_file=file_paths[0],
                        insurer_file=file_paths[1], 
                        output_file="output.xlsx"
                    )
                    # Upload the result to SharePoint
                    if result and result.get('output_file'):
                        sharepoint_url = sharepoint_service.upload_to_sharepoint(
                            result['output_file'],
                            folder['library'],
                            folder['url']
                        )
                        logger.info(f"✓ Results uploaded to SharePoint: {sharepoint_url}")
                        
                        # Update folder status to "Completed"
                        try:
                            folder_obj = ctx.web.get_folder_by_server_relative_url(folder['url'])
                            folder_list_item = folder_obj.list_item_all_fields
                            ctx.load(folder_list_item)
                            ctx.execute_query()
                            folder_list_item.set_property('Status', 'Manual Review')
                            folder_list_item.update()
                            ctx.execute_query()
                            logger.info(f"✓ Updated folder status to 'Manual Review': {folder['name']}")
                        except Exception as e:
                            logger.error(f"Error updating folder status: {str(e)}")
                        
                        # Update status of both input Excel files to "Completed"
                        for file in excel_files:
                            try:
                                file_obj = ctx.web.get_file_by_server_relative_url(file['url'])
                                file_list_item = file_obj.listItemAllFields
                                ctx.load(file_list_item)
                                ctx.execute_query()
                                file_list_item.set_property('Status', 'Manual Review')
                                file_list_item.update()
                                ctx.execute_query()
                                logger.info(f"✓ Updated file status to 'Manual Review': {file['name']}")
                            except Exception as e:
                                logger.error(f"Error updating file status: {str(e)}")
                    
                    # Clean up temporary files
                    try:
                        # First remove all files in the directory
                        for file_path in file_paths:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                        
                        # Then remove the output file if it exists
                        output_path = os.path.join(temp_dir, "output.xlsx")
                        if os.path.exists(output_path):
                            os.remove(output_path)
                        
                        # Finally remove the directory
                        if os.path.exists(temp_dir):
                            os.rmdir(temp_dir)
                            
                    except Exception as cleanup_error:
                        logger.error(f"Error during cleanup: {str(cleanup_error)}")
                        # Continue execution even if cleanup fails
                
                except Exception as e:
                    logger.error(f"Error processing Excel files: {str(e)}")
                    # Try to clean up even if there was an error
                    try:
                        for file_path in file_paths:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                        if os.path.exists(temp_dir):
                            os.rmdir(temp_dir)
                    except:
                        pass
            else:
                logger.info(f"Found {len(excel_files)} Excel files (expected 2) in {folder['name']}")

    except Exception as ex:
        logger.error(f"Error in main: {str(ex)}")
        raise

if __name__ == "__main__":
    main()

