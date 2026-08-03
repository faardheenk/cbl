#!/usr/bin/env python3

import os
import json
from office365.sharepoint.client_context import ClientContext
try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    load_dotenv = None  # not available (e.g. in PyInstaller bundle); use env vars or defaults
import logging
import re
import pandas as pd
from matching.orchestrator import run_matching_process
from matching.data_processing import create_dynamic_column_mappings, read_excel_with_smart_headers
import datetime
from sharepoint_audit_logger import SharePointAuditLogger, create_audit_statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def sanitize_json_string(s):
    # Remove trailing commas before closing } or ]
    s = re.sub(r',\s*([}\]])', r'\1', s)
    return s

class SharePointService:
    def __init__(self):
        """Initialize SharePoint service with credentials from environment variables."""
        # Load environment variables
        if load_dotenv:
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

        # self.site_url = "https://frcidevtest.sharepoint.com/sites/CityBroker2"
        # self.cert_thumbprint = "C8F25C2F82B6F5712662D175C42FDD3E355B953B"
        # self.client_id = "8f9f6a06-b2b8-4f3b-9d10-9a0390e94ea7"
        # self.tenant= "frcidevtest.onmicrosoft.com"
        # self.cert_path= "C:\\Users\\boshavg.SERVICES\\Desktop\\Projects\\CBL\\certificate\\cert.pem"

        # self.site_url = "https://citybrokersltdmu.sharepoint.com/sites/statementrecon"
        # self.cert_thumbprint = "B3F2EB224794D54AF99FD443D1E4ABFEF8E10C7B"
        # self.client_id = "74de1033-3314-49cc-8f5a-829e0ec76b27"
        # self.tenant= "citybrokersltdmu.onmicrosoft.com"
        # self.cert_path= "E:\\FRCI\\certificate\\cert.pem"

        self.cert_credentials = {
            "tenant": self.tenant,
            "client_id": self.client_id,
            "thumbprint": self.cert_thumbprint,
            "cert_path": self.cert_path,
        }
        self.ctx = None
        
        # Initialize audit logger
        self.audit_logger = SharePointAuditLogger(self, audit_list_name="Audit Log")
        
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

    def ensure_audit_log_list_exists(self):
        """
        Ensure the Audit Log list exists in SharePoint, create it if it doesn't.
        Also ensures all required columns exist.
        
        Returns:
            bool: True if list exists or was created successfully
        """
        try:
            ctx = self.get_client_context()
            
            audit_list = ctx.web.lists.get_by_title("Audit Log")
            ctx.load(audit_list)
            ctx.execute_query()
            logger.info("✅ Audit Log list already exists")
            
            return True
        except Exception as ex:
            logger.error(f"❌ Failed to ensure Audit Log list exists: {str(ex)}")
            return False


    def get_column_mappings(self, insurer_name=None):
        """Get column mappings from SharePoint."""
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

    def create_hybrid_column_mappings_from_content(self, cbl_file_content, insurer_file_content, insurer_name):
        """
        Create column mappings using both SharePoint config and dynamic detection from file content.
        
        Args:
            cbl_file_content: CBL Excel file content as bytes
            insurer_file_content: Insurer Excel file content as bytes
            insurer_name: Name of insurer for SharePoint lookup
            
        Returns:
            dict: Combined column mappings
        """
        try:
            # Step 1: Get SharePoint-configured mappings (if available)
            logger.info(f"🔍 Getting SharePoint column mappings for {insurer_name}...")
            sharepoint_mappings = None
            try:
                sharepoint_mappings = self.get_column_mappings(insurer_name)
                logger.info(f"✅ Retrieved SharePoint mappings:")
                logger.info(f"   CBL: {sharepoint_mappings.get('cbl_mappings', {})}")
                logger.info(f"   Insurer: {sharepoint_mappings.get('insurer_mappings', {})}")
            except Exception as e:
                logger.warning(f"⚠️ Could not get SharePoint mappings for {insurer_name}: {str(e)}")
                logger.info("Will use pure dynamic detection instead")
            
            # Step 2: Read files to analyze column structure with smart header detection
            logger.info(f"📊 Analyzing file structure with smart header detection...")
            from matching.data_processing import read_excel_with_smart_headers
            cbl_df = read_excel_with_smart_headers(cbl_file_content)
            insurer_df = read_excel_with_smart_headers(insurer_file_content)
            
            cbl_columns = list(cbl_df.columns)
            insurer_columns = list(insurer_df.columns)
            
            logger.info(f"CBL columns found ({len(cbl_columns)}): {cbl_columns}")
            logger.info(f"Insurer columns found ({len(insurer_columns)}): {insurer_columns}")
            
            # Step 3: Create dynamic mappings with SharePoint overrides
            logger.info(f"🎯 Creating hybrid column mappings...")
            from matching.data_processing import create_dynamic_column_mappings
            dynamic_mappings = create_dynamic_column_mappings(
                cbl_columns=cbl_columns,
                insurer_columns=insurer_columns,
                custom_mappings=sharepoint_mappings  # Use SharePoint mappings as overrides
            )
            
            logger.info(f"✅ Final hybrid mappings created:")
            logger.info(f"   CBL mappings: {dynamic_mappings['cbl_mappings']}")
            logger.info(f"   Insurer mappings: {dynamic_mappings['insurer_mappings']}")
            
            # Log the source of mappings
            if sharepoint_mappings:
                logger.info(f"💡 Using hybrid approach: SharePoint config + dynamic detection")
            else:
                logger.info(f"💡 Using pure dynamic detection (no SharePoint config)")
            
            return dynamic_mappings
            
        except Exception as e:
            logger.error(f"❌ Error creating hybrid mappings: {str(e)}")
            
            # Fallback to pure dynamic detection if everything fails
            try:
                logger.info("🔄 Falling back to pure dynamic detection with smart headers...")
                from matching.data_processing import read_excel_with_smart_headers, create_dynamic_column_mappings
                cbl_df = read_excel_with_smart_headers(cbl_file_content)
                insurer_df = read_excel_with_smart_headers(insurer_file_content)
                
                fallback_mappings = create_dynamic_column_mappings(
                    cbl_columns=list(cbl_df.columns),
                    insurer_columns=list(insurer_df.columns)
                )
                
                logger.info(f"✅ Fallback mappings created: {fallback_mappings}")
                return fallback_mappings
                
            except Exception as fallback_error:
                logger.error(f"❌ Even fallback failed: {str(fallback_error)}")
                raise


    def get_dynamic_buckets(self):
        """
        Fetch dynamic bucket definitions from SharePoint 'Buckets' list.
        Buckets are global and apply to all insurance companies.

        Returns:
            list of {"BucketName": str, "BucketKey": str}
        """
        try:
            ctx = self.get_client_context()
            bucket_list = ctx.web.lists.get_by_title("Buckets")
            items = bucket_list.items.select(
                ['BucketName', 'BucketKey', 'Rematch']
            ).get().execute_query()

            buckets = []
            for item in items:
                bucket_name = item.properties.get('BucketName', '')
                bucket_key = item.properties.get('BucketKey', '')
                rematch_raw = item.properties.get('Rematch', False)
                rematch = rematch_raw is True or str(rematch_raw).strip().lower() in ('yes', 'true', '1')
                if bucket_name and bucket_key:
                    buckets.append({
                        'BucketName': bucket_name,
                        'BucketKey': bucket_key,
                        'Rematch': rematch,
                    })

            logger.info(f"[BUCKETS] Fetched {len(buckets)} dynamic buckets: {[b['BucketKey'] for b in buckets]}")
            return buckets
        except Exception as e:
            logger.warning(f"[BUCKETS] Could not fetch dynamic buckets: {e}")
            return []

    def get_history_file(self, insurer_name):
        """
        Download history.xlsx for an insurer from the Matrix library.
        Path: Matrix/{INSURER_NAME}/history.xlsx

        Args:
            insurer_name: Insurer folder name (e.g. "ALLIANZ")

        Returns:
            bytes or None: File content as bytes, or None if not found.
        """
        try:
            ctx = self.get_client_context()
            library = ctx.web.lists.get_by_title("Matrix")
            root_folder = library.root_folder
            ctx.load(root_folder)
            ctx.execute_query()

            # First, list files in the insurer folder to debug
            insurer_folder_url = f"{root_folder.serverRelativeUrl}/{insurer_name}"
            logger.info(f"[HISTORY] Looking in folder: {insurer_folder_url}")
            try:
                insurer_folder = ctx.web.get_folder_by_server_relative_url(insurer_folder_url)
                files = insurer_folder.files
                ctx.load(files)
                ctx.execute_query()
                file_names = [f.name for f in files]
                logger.info(f"[HISTORY] Files in Matrix/{insurer_name}/: {file_names}")
            except Exception as list_err:
                logger.error(f"[HISTORY] Could not list folder Matrix/{insurer_name}/: {type(list_err).__name__}: {list_err}")

            # Download history.xlsx from the folder files we already loaded
            for f in files:
                if f.name.lower() == "history.xlsx":
                    logger.info(f"[HISTORY] Found history.xlsx — downloading...")
                    file_content = f.read()
                    ctx.execute_query()

                    if hasattr(file_content, 'value'):
                        file_content = file_content.value

                    logger.info(f"[HISTORY] history.xlsx downloaded — type={type(file_content).__name__}")
                    return file_content

            logger.info(f"[HISTORY] history.xlsx not found in file listing for {insurer_name}")
            return None
        except Exception as e:
            import traceback
            logger.error(f"[HISTORY] Failed to download history.xlsx for {insurer_name}: {type(e).__name__}: {e}")
            logger.error(f"[HISTORY] Traceback: {traceback.format_exc()}")
            return None

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
        
    def update_folder_status(self, folder_url, status):
        """
        Update the status of a SharePoint folder.
        
        Args:
            folder_url (str): Server relative URL of the folder
            status (str): New status value (e.g., 'Pending', 'In Progress', 'Manual Review', 'Failed')
        """
        try:
            ctx = self.get_client_context()
            folder_obj = ctx.web.get_folder_by_server_relative_url(folder_url)
            folder_list_item = folder_obj.list_item_all_fields
            ctx.load(folder_list_item)
            ctx.execute_query()
            folder_list_item.set_property('Status', status)
            folder_list_item.update()
            ctx.execute_query()
            logger.info(f"✅ Updated folder status to '{status}': {folder_url}")
        except Exception as e:
            logger.error(f"Error updating folder status to '{status}': {str(e)}")
            raise

    def update_file_status(self, file_url, status):
        """
        Update the status of a SharePoint file.
        
        Args:
            file_url (str): Server relative URL of the file
            status (str): New status value (e.g., 'Pending', 'In Progress', 'Manual Review', 'Failed')
        """
        try:
            ctx = self.get_client_context()
            file_obj = ctx.web.get_file_by_server_relative_url(file_url)
            file_list_item = file_obj.listItemAllFields
            ctx.load(file_list_item)
            ctx.execute_query()
            file_list_item.set_property('Status', status)
            file_list_item.update()
            ctx.execute_query()
            logger.info(f"✅ Updated file status to '{status}': {file_url}")
        except Exception as e:
            logger.error(f"Error updating file status to '{status}': {str(e)}")
            raise

    def upload_content_to_sharepoint(self, file_content, filename, library_name, folder_path):
        """
        Upload file content directly to SharePoint without saving locally.
        
        Args:
            file_content (bytes): File content as bytes
            filename (str): Name of the file to upload
            library_name (str): Name of the SharePoint library
            folder_path (str): Path to the folder in SharePoint
            
        Returns:
            str: Server relative URL of the uploaded file
        """
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
            
            # Upload the file content directly
            target_file = target_folder.upload_file(filename, file_content).execute_query()

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
            logger.error(f"Error uploading content to SharePoint: {str(ex)}")
            raise

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
    sharepoint_service = None
    folders_found = 0
    folders_processed = 0
    folders_failed = 0
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

        # Ensure Audit Log list exists
        logger.info("🔍 Ensuring Audit Log list exists...")
        if not sharepoint_service.ensure_audit_log_list_exists():
            logger.error("❌ Failed to create or access Audit Log list")
            return

        # Start script-level audit
        sharepoint_service.audit_logger.start_script_audit()

        # Get pending folders
        pending_folders = sharepoint_service.get_pending_folders()
        folders_found = len(pending_folders)
        logger.info(f"Found {folders_found} folders with pending status")

        # Process each pending folder
        for folder in pending_folders:
            logger.info(f"\n🚀 Processing folder: {folder['name']} (Parent: {folder['parent_folder']})")
            excel_files = sharepoint_service.get_excel_files_from_folder(folder['url'])
    
            if len(excel_files) == 2:
                logger.info(f"Found exactly 2 Excel files in {folder['name']}:")
                for file in excel_files:
                    logger.info(f"  - {file['name']}")
                
                # Start audit logging for this folder
                file_names = [file['name'] for file in excel_files]
                audit_id = sharepoint_service.audit_logger.start_audit_entry(
                    folder_id=folder['name'],
                    insurer_name=folder['parent_folder'],
                    file_names=file_names
                )

                # Process files directly from memory without saving to disk
                try:
                    # Determine which file is CBL and which is insurer
                    cbl_file_content = None
                    insurer_file_content = None
                    cbl_file_name = None
                    insurer_file_name = None
                    
                    for file in excel_files:
                        if file['name'].lower() == 'cbl.xlsx':
                            cbl_file_content = file['content']
                            cbl_file_name = file['name']
                        elif file['name'].lower() == 'insurer.xlsx':
                            insurer_file_content = file['content']
                            insurer_file_name = file['name']

                    if not cbl_file_content or not insurer_file_content:
                        raise ValueError("Expected files named 'cbl.xlsx' and 'insurer.xlsx'")
                    
                    # 🆕 CREATE DYNAMIC COLUMN MAPPINGS
                    logger.info(f"🎯 Creating dynamic column mappings for {folder['parent_folder']}...")
                    column_mappings = sharepoint_service.create_hybrid_column_mappings_from_content(
                        cbl_file_content=cbl_file_content,
                        insurer_file_content=insurer_file_content,
                        insurer_name=folder['parent_folder']
                    )
                    
                    # Fetch dynamic buckets (global, applies to all insurers)
                    insurer_name = folder['parent_folder']
                    dynamic_buckets = sharepoint_service.get_dynamic_buckets()

                    # Download match history for this insurer (if it exists)
                    logger.info(f"[HISTORY] Attempting to download history.xlsx for insurer: '{insurer_name}'")
                    logger.info(f"[HISTORY] Expected SharePoint path: Matrix/{insurer_name}/history.xlsx")
                    history_content = sharepoint_service.get_history_file(insurer_name)
                    if history_content is not None:
                        logger.info(f"[HISTORY] SUCCESS — history.xlsx fetched, type={type(history_content).__name__}")
                    else:
                        logger.info(f"[HISTORY] FAILED — history.xlsx not found for '{insurer_name}'")

                    # Run matching process
                    logger.info(f"Running matching process...")
                    try:
                        result = run_matching_process(
                            column_mappings=column_mappings,
                            cbl_file=cbl_file_content,
                            insurer_file=insurer_file_content,
                            output_file="output.xlsx",
                            history_file=history_content,
                            dynamic_buckets=dynamic_buckets,
                        )
                        
                        # Upload the result to SharePoint
                        if result and result.get('output_content'):
                            logger.info(f"📤 Uploading results to SharePoint...")
                            sharepoint_url = sharepoint_service.upload_content_to_sharepoint(
                                result['output_content'],
                                result['output_file'],
                                folder['library'],
                                folder['url']
                            )
                            logger.info(f"✅ Results uploaded to SharePoint: {sharepoint_url}")
                            
                            # Enhanced result summary
                            logger.info(f"\n🎉 Processing complete for {folder['name']}!")
                            logger.info(f"📊 Results Summary:")
                            logger.info(f"   ✅ CBL Exact Matches: {result['cbl_stats']['exact_matches']}")
                            logger.info(f"   🔄 CBL Partial Matches: {result['cbl_stats']['partial_matches']}")
                            logger.info(f"   ❌ CBL No Matches: {result['cbl_stats']['no_matches']}")
                            logger.info(f"   📈 Insurer Match Rate: {result['insurer_stats']['exact_match_rate']:.1f}%")
                            logger.info(f"   💰 Exact Match Amount: ${result['cbl_stats']['exact_match_amount']:,.2f}")
                            
                            # Complete audit logging with success
                            audit_stats = create_audit_statistics(result)
                            sharepoint_service.audit_logger.complete_audit_entry(
                                status='Completed',
                                match_statistics=audit_stats
                            )
                            
                            # Update folder status to "Manual Review"
                            try:
                                sharepoint_service.update_folder_status(folder['url'], 'Manual Review')
                            except Exception as e:
                                logger.error(f"Error updating folder status: {str(e)}")
                            
                            # Update status of both input Excel files to "Manual Review"
                            for file in excel_files:
                                try:
                                    sharepoint_service.update_file_status(file['url'], 'Manual Review')
                                except Exception as e:
                                    logger.error(f"Error updating file status: {str(e)}")

                            folders_processed += 1
                        else:
                            logger.error(f"❌ Matching process completed but no output file generated")
                            # Complete audit logging with failure
                            sharepoint_service.audit_logger.complete_audit_entry(
                                status='Failed',
                                error_details="Matching process failed to generate output file"
                            )
                            raise Exception("Matching process failed to generate output file")
                            
                    except Exception as matching_error:
                        logger.error(f"❌ Matching process failed for {folder['name']}: {str(matching_error)}")
                        logger.error(f"Error details: {type(matching_error).__name__}")
                        
                        # Complete audit logging with failure
                        sharepoint_service.audit_logger.complete_audit_entry(
                            status='Failed',
                            error_details=f"Matching process failed: {str(matching_error)}"
                        )
                        
                        # Update folder status to "Failed"
                        try:
                            sharepoint_service.update_folder_status(folder['url'], 'Failed')
                        except Exception as e:
                            logger.error(f"Error updating folder status to Failed: {str(e)}")
                        
                        # Update status of both input Excel files to "Failed"
                        for file in excel_files:
                            try:
                                sharepoint_service.update_file_status(file['url'], 'Failed')
                            except Exception as e:
                                logger.error(f"Error updating file status to Failed: {str(e)}")
                        
                        # Re-raise the error to be caught by the outer exception handler
                        raise
                    
                    # No cleanup needed - files were processed from memory
                
                except Exception as e:
                    logger.error(f"❌ Error processing Excel files for {folder['name']}: {str(e)}")
                    logger.error(f"Error type: {type(e).__name__}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    
                    # Complete audit logging with failure if audit was started
                    try:
                        if 'audit_id' in locals():
                            sharepoint_service.audit_logger.complete_audit_entry(
                                status='Failed',
                                error_details=f"Excel file processing failed: {str(e)}"
                            )
                    except Exception as audit_error:
                        logger.error(f"Error completing audit log: {str(audit_error)}")
                    
                    # Update folder and file statuses to "Failed" on any processing error
                    try:
                        # Update folder status to "Failed"
                        sharepoint_service.update_folder_status(folder['url'], 'Failed')
                    except Exception as status_error:
                        logger.error(f"Error updating folder status to Failed: {str(status_error)}")
                    
                    # Update status of both input Excel files to "Failed"
                    for file in excel_files:
                        try:
                            sharepoint_service.update_file_status(file['url'], 'Failed')
                        except Exception as file_status_error:
                            logger.error(f"Error updating file status to Failed: {str(file_status_error)}")
                    
                    # No cleanup needed - files were processed from memory

                    folders_failed += 1

                    # Continue processing other folders even if this one failed
                    continue
            else:
                logger.info(f"⚠️ Found {len(excel_files)} Excel files (expected 2) in {folder['name']}")

    except Exception as ex:
        logger.error(f"❌ Error in main: {str(ex)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Complete script-level audit
        if sharepoint_service and sharepoint_service.audit_logger.script_audit_entry:
            sharepoint_service.audit_logger.complete_script_audit(
                folders_found=folders_found,
                folders_processed=folders_processed,
                folders_failed=folders_failed
            )

        # Log script completion
        logger.info(f"\n🏁 Script execution completed at {datetime.datetime.now()}")
        logger.info("📝 All audit logs have been saved to SharePoint Audit Log list")

if __name__ == "__main__":
    main()
