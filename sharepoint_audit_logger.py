#!/usr/bin/env python3

import datetime
import json
import logging
from typing import Dict, Any, Optional
from office365.sharepoint.client_context import ClientContext

logger = logging.getLogger(__name__)


class SharePointAuditLogger:
    """
    SharePoint-based audit logging system for the reconciliation process.
    
    This class handles creating audit log entries in a SharePoint list with the following columns:
    - Audit Id (SharePoint default)
    - Started_Date
    - Processing_Time
    - Parent_Folder
    - Insurer_Name
    - File_Names
    - Match_Statistics
    - Status
    - Error_Details
    """
    
    def __init__(self, sharepoint_service, audit_list_name="Audit Log"):
        """
        Initialize the SharePoint audit logger.
        
        Args:
            sharepoint_service: SharePointService instance for SharePoint operations
            audit_list_name: Name of the SharePoint list to store audit logs
        """
        self.sharepoint_service = sharepoint_service
        self.audit_list_name = audit_list_name
        self.audit_entry = None
        self.start_time = None
        self.script_audit_entry = None
        self.script_start_time = None
        
    def start_script_audit(self):
        """Create a script-level audit entry that logs every run, even when there is no work."""
        try:
            self.script_start_time = datetime.datetime.now()
            unique_id = f"SCRIPT_RUN_{self.script_start_time.strftime('%Y%m%d_%H%M%S')}"

            audit_data = {
                'Title': unique_id,
                'Started_Date': self.script_start_time.isoformat(),
                'Insurer_Name': '-',
                'Folder_Id': '-',
                'File_Names': '[]',
                'Status': 'Running',
            }

            self.script_audit_entry = self._create_audit_entry(audit_data)
            logger.info(f"Started script audit entry {unique_id}")

        except Exception as e:
            logger.error(f"Failed to start script audit entry: {str(e)}")

    def complete_script_audit(self, folders_found: int = 0, folders_processed: int = 0,
                              folders_failed: int = 0, error_details: str = None):
        """Complete the script-level audit entry with a run summary."""
        if not self.script_audit_entry:
            return

        try:
            processing_time = None
            if self.script_start_time:
                processing_time = (datetime.datetime.now() - self.script_start_time).total_seconds()

            if error_details:
                status = 'Failed'
            elif folders_found == 0:
                status = 'No Work'
            elif folders_failed > 0 and folders_processed == 0:
                status = 'Failed'
            elif folders_failed > 0:
                status = 'Partial'
            else:
                status = 'Completed'

            summary = {
                'script_run': True,
                'folders_found': folders_found,
                'folders_processed': folders_processed,
                'folders_failed': folders_failed,
            }

            update_data = {
                'Status': status,
                'Match_Statistics': json.dumps(summary),
            }

            if processing_time is not None:
                update_data['Processing_Time'] = processing_time

            if error_details:
                update_data['Error_Details'] = error_details

            self._update_audit_entry(self.script_audit_entry, update_data)
            logger.info(f"Completed script audit: {status} (found={folders_found}, processed={folders_processed}, failed={folders_failed})")

        except Exception as e:
            logger.error(f"Failed to complete script audit entry: {str(e)}")
        finally:
            self.script_audit_entry = None
            self.script_start_time = None

    def start_audit_entry(self, folder_id: str, insurer_name: str, file_names: list) -> str:
        """
        Start a new audit entry and return the audit ID.
        
        Args:
            folder_id: ID of the folder being processed
            insurer_name: Name of the insurer
            file_names: List of file names being processed
            
        Returns:
            str: Audit ID of the created entry
        """
        try:
            self.start_time = datetime.datetime.now()
            
            # Generate a unique audit ID for the Title field
            unique_audit_id = f"AUDIT_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{folder_id}_{insurer_name}".replace(' ', '_').replace('/', '_')
            
            # Create audit entry data
            audit_data = {
                'Title': unique_audit_id,  # Use Title field as Audit_Id
                'Started_Date': self.start_time.isoformat(),
                'Folder_Id': folder_id,
                'Insurer_Name': insurer_name,
                'File_Names': json.dumps(file_names) if file_names else "[]",
                'Status': 'In Progress',
                'Error_Details': None
            }
            
            # Create the audit entry in SharePoint
            sharepoint_id = self._create_audit_entry(audit_data)
            self.audit_entry = sharepoint_id
            
            logger.info(f"📝 Started audit entry {unique_audit_id} (SharePoint ID: {sharepoint_id}) for {insurer_name} in {folder_id}")
            return sharepoint_id
            
        except Exception as e:
            logger.error(f"❌ Failed to start audit entry: {str(e)}")
            raise
    
    def update_audit_entry(self, status: str = None, match_statistics: Dict[str, Any] = None, 
                          error_details: str = None, processing_time: float = None):
        """
        Update the current audit entry with new information.
        
        Args:
            status: New status (e.g., 'Completed', 'Failed', 'Manual Review')
            match_statistics: Dictionary containing match statistics
            error_details: Error details if any errors occurred
            processing_time: Processing time in seconds (if not provided, will be calculated)
        """
        if not self.audit_entry:
            logger.warning("⚠️ No active audit entry to update")
            return
            
        try:
            update_data = {}
            
            # Calculate processing time if not provided
            if processing_time is None and self.start_time:
                processing_time = (datetime.datetime.now() - self.start_time).total_seconds()
            
            if processing_time is not None:
                update_data['Processing_Time'] = processing_time
            
            if status is not None:
                update_data['Status'] = status
                
            if match_statistics is not None:
                update_data['Match_Statistics'] = json.dumps(match_statistics)
                
            if error_details is not None:
                update_data['Error_Details'] = error_details
            
            # Update the audit entry in SharePoint
            self._update_audit_entry(self.audit_entry, update_data)
            
            logger.info(f"📝 Updated audit entry {self.audit_entry} with status: {status}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update audit entry: {str(e)}")
            raise
    
    def complete_audit_entry(self, status: str = 'Completed', match_statistics: Dict[str, Any] = None, 
                           error_details: str = None):
        """
        Complete the current audit entry.
        
        Args:
            status: Final status (default: 'Completed')
            match_statistics: Final match statistics
            error_details: Any error details
        """
        if not self.audit_entry:
            logger.warning("⚠️ No active audit entry to complete")
            return
            
        try:
            # Calculate final processing time
            processing_time = None
            if self.start_time:
                processing_time = (datetime.datetime.now() - self.start_time).total_seconds()
            
            # Update with final information
            self.update_audit_entry(
                status=status,
                match_statistics=match_statistics,
                error_details=error_details,
                processing_time=processing_time
            )
            
            logger.info(f"✅ Completed audit entry {self.audit_entry} with status: {status}")
            
        except Exception as e:
            logger.error(f"❌ Failed to complete audit entry: {str(e)}")
            raise
        finally:
            # Clear the current audit entry
            self.audit_entry = None
            self.start_time = None
    
    def _create_audit_entry(self, audit_data: Dict[str, Any]) -> str:
        """
        Create a new audit entry in the SharePoint list.
        
        Args:
            audit_data: Dictionary containing audit entry data
            
        Returns:
            str: ID of the created audit entry
        """
        try:
            ctx = self.sharepoint_service.get_client_context()
            audit_list = ctx.web.lists.get_by_title(self.audit_list_name)
            
            # Create the list item
            item_properties = audit_data.copy()
            item = audit_list.add_item(item_properties)
            ctx.execute_query()
            
            # Get the SharePoint ID of the created item
            sharepoint_id = str(item.properties['Id'])
            logger.debug(f"Created audit entry with SharePoint ID: {sharepoint_id}")
            
            return sharepoint_id
            
        except Exception as e:
            logger.error(f"Failed to create audit entry in SharePoint: {str(e)}")
            raise
    
    def _update_audit_entry(self, audit_id: str, update_data: Dict[str, Any]):
        """
        Update an existing audit entry in the SharePoint list.
        
        Args:
            audit_id: ID of the audit entry to update
            update_data: Dictionary containing data to update
        """
        try:
            ctx = self.sharepoint_service.get_client_context()
            audit_list = ctx.web.lists.get_by_title(self.audit_list_name)
            
            # Get the item by ID
            item = audit_list.get_item_by_id(int(audit_id))
            
            # Update the item properties
            for key, value in update_data.items():
                item.set_property(key, value)
            
            item.update()
            ctx.execute_query()
            
            logger.debug(f"Updated audit entry {audit_id} with data: {update_data}")
            
        except Exception as e:
            logger.error(f"Failed to update audit entry {audit_id} in SharePoint: {str(e)}")
            raise
    
    def get_audit_entry(self, audit_id: str) -> Dict[str, Any]:
        """
        Retrieve an audit entry from SharePoint by SharePoint ID.
        
        Args:
            audit_id: SharePoint ID of the audit entry to retrieve
            
        Returns:
            Dict containing the audit entry data
        """
        try:
            ctx = self.sharepoint_service.get_client_context()
            audit_list = ctx.web.lists.get_by_title(self.audit_list_name)
            
            # Get the item by SharePoint ID
            item = audit_list.get_item_by_id(int(audit_id))
            ctx.load(item)
            ctx.execute_query()
            
            # Convert to dictionary
            audit_data = dict(item.properties)
            
            # Parse JSON fields
            if 'File_Names' in audit_data and audit_data['File_Names']:
                try:
                    audit_data['File_Names'] = json.loads(audit_data['File_Names'])
                except json.JSONDecodeError:
                    pass
            
            if 'Match_Statistics' in audit_data and audit_data['Match_Statistics']:
                try:
                    audit_data['Match_Statistics'] = json.loads(audit_data['Match_Statistics'])
                except json.JSONDecodeError:
                    pass
            
            return audit_data
            
        except Exception as e:
            logger.error(f"Failed to retrieve audit entry {audit_id} from SharePoint: {str(e)}")
            raise

    def get_audit_entry_by_audit_id(self, audit_id: str) -> Dict[str, Any]:
        """
        Retrieve an audit entry from SharePoint by Title (used as Audit_Id).
        
        Args:
            audit_id: Title/Audit_Id of the audit entry to retrieve
            
        Returns:
            Dict containing the audit entry data
        """
        try:
            ctx = self.sharepoint_service.get_client_context()
            audit_list = ctx.web.lists.get_by_title(self.audit_list_name)
            
            # Query for items with the specific Title (which serves as Audit_Id)
            items = audit_list.items.filter(f"Title eq '{audit_id}'").get().execute_query()
            
            if not items:
                raise ValueError(f"No audit entry found with Title/Audit_Id: {audit_id}")
            
            # Get the first (and should be only) item
            item = items[0]
            audit_data = dict(item.properties)
            
            # Parse JSON fields
            if 'File_Names' in audit_data and audit_data['File_Names']:
                try:
                    audit_data['File_Names'] = json.loads(audit_data['File_Names'])
                except json.JSONDecodeError:
                    pass
            
            if 'Match_Statistics' in audit_data and audit_data['Match_Statistics']:
                try:
                    audit_data['Match_Statistics'] = json.loads(audit_data['Match_Statistics'])
                except json.JSONDecodeError:
                    pass
            
            return audit_data
            
        except Exception as e:
            logger.error(f"Failed to retrieve audit entry with Title/Audit_Id {audit_id} from SharePoint: {str(e)}")
            raise
    
    def list_recent_audit_entries(self, limit: int = 10) -> list:
        """
        List recent audit entries from SharePoint.
        
        Args:
            limit: Maximum number of entries to retrieve
            
        Returns:
            List of audit entry dictionaries
        """
        try:
            ctx = self.sharepoint_service.get_client_context()
            audit_list = ctx.web.lists.get_by_title(self.audit_list_name)
            
            # Get recent items, ordered by creation date descending
            # Note: SharePoint order_by doesn't support ascending parameter, so we'll get all and sort
            items = audit_list.items.top(limit * 2).get().execute_query()  # Get more items to ensure we have enough after sorting
            
            audit_entries = []
            for item in items:
                audit_data = dict(item.properties)
                
                # Parse JSON fields
                if 'File_Names' in audit_data and audit_data['File_Names']:
                    try:
                        audit_data['File_Names'] = json.loads(audit_data['File_Names'])
                    except json.JSONDecodeError:
                        pass
                
                if 'Match_Statistics' in audit_data and audit_data['Match_Statistics']:
                    try:
                        audit_data['Match_Statistics'] = json.loads(audit_data['Match_Statistics'])
                    except json.JSONDecodeError:
                        pass
                
                audit_entries.append(audit_data)
            
            # Sort by Created date descending (most recent first)
            # Handle cases where Created might be None or in different formats
            def sort_key(entry):
                created = entry.get('Created')
                if created is None:
                    return datetime.datetime.min
                # Handle different date formats
                if isinstance(created, str):
                    try:
                        return datetime.datetime.fromisoformat(created.replace('Z', '+00:00'))
                    except:
                        return datetime.datetime.min
                elif hasattr(created, 'isoformat'):
                    return created
                else:
                    return datetime.datetime.min
            
            audit_entries.sort(key=sort_key, reverse=True)
            
            # Return only the requested limit
            return audit_entries[:limit]
            
        except Exception as e:
            logger.error(f"Failed to list recent audit entries from SharePoint: {str(e)}")
            raise


def create_audit_statistics(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create standardized audit statistics from matching process results.
    
    Args:
        result: Result dictionary from the matching process
        
    Returns:
        Dict containing standardized audit statistics
    """
    try:
        cbl_stats = result.get('cbl_stats', {})
        insurer_stats = result.get('insurer_stats', {})
        
        audit_stats = {
            'cbl_summary': {
                'total_records': cbl_stats.get('exact_matches', 0) + cbl_stats.get('partial_matches', 0) + cbl_stats.get('no_matches', 0),
                'exact_matches': cbl_stats.get('exact_matches', 0),
                'partial_matches': cbl_stats.get('partial_matches', 0),
                'no_matches': cbl_stats.get('no_matches', 0),
                'exact_match_amount': round(float(cbl_stats.get('exact_match_amount', 0)), 2),
                'partial_match_amount': round(float(cbl_stats.get('partial_match_amount', 0)), 2),
                'no_match_amount': round(float(cbl_stats.get('no_match_amount', 0)), 2)
            },
            'insurer_summary': {
                'total_records': insurer_stats.get('total_rows', 0),
                'exact_matches': insurer_stats.get('exact_match_rows', 0),
                'partial_matches': insurer_stats.get('partial_match_rows', 0),
                'no_matches': insurer_stats.get('unmatched_rows', 0),
                'exact_match_amount': round(float(insurer_stats.get('exact_match_amount', 0)), 2),
                'partial_match_amount': round(float(insurer_stats.get('partial_match_amount', 0)), 2),
                'no_match_amount': round(float(insurer_stats.get('unmatched_amount', 0)), 2)
            },
            'processing_info': {
                'timestamp': datetime.datetime.now().isoformat()
            }
        }
        
        return audit_stats
        
    except Exception as e:
        logger.error(f"Failed to create audit statistics: {str(e)}")
        return {
            'error': f"Failed to create audit statistics: {str(e)}",
            'timestamp': datetime.datetime.now().isoformat()
        }
