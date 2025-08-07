"""
Catalog parser for penetration testing guidelines
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime


class CatalogParser:
    """
    Parses and manages penetration testing guidelines and methodologies.
    """
    
    def __init__(self, catalog_file: str = "data/guidelines/pentest_catalog.json"):
        """
        Initialize the catalog parser.
        
        Args:
            catalog_file: Path to the guidelines catalog file
        """
        self.catalog_file = catalog_file
        self.guidelines = {}
        self.methodologies = {}
        self.compliance_frameworks = {}
        self.test_cases = {}
        
        self._load_catalog()
    
    def _load_catalog(self):
        """
        Load the guidelines catalog from file.
        """
        if os.path.exists(self.catalog_file):
            try:
                with open(self.catalog_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.guidelines = data.get('guidelines', {})
                    self.methodologies = data.get('methodologies', {})
                    self.compliance_frameworks = data.get('compliance_frameworks', {})
                    self.test_cases = data.get('test_cases', {})
            except Exception as e:
                print(f"Error loading catalog: {e}")
                self._initialize_default_catalog()
        else:
            self._initialize_default_catalog()
    
    def _initialize_default_catalog(self):
        """
        Initialize with default penetration testing guidelines.
        """
        default_data = {
            'guidelines': {
                'owasp_testing_guide': {
                    'name': 'OWASP Testing Guide',
                    'version': '4.2',
                    'description': 'Comprehensive web application security testing methodology',
                    'categories': {
                        'injection': {
                            'sql_injection': {
                                'description': 'Testing for SQL Injection vulnerabilities',
                                'test_cases': [
                                    'test_reflected_sql_injection',
                                    'test_blind_sql_injection',
                                    'test_stored_sql_injection'
                                ],
                                'severity': 'high',
                                'cwe_id': 'CWE-89'
                            },
                            'nosql_injection': {
                                'description': 'Testing for NoSQL Injection vulnerabilities',
                                'test_cases': ['test_nosql_injection'],
                                'severity': 'high',
                                'cwe_id': 'CWE-943'
                            }
                        },
                        'authentication': {
                            'bypass_authentication': {
                                'description': 'Testing for authentication bypass',
                                'test_cases': ['test_auth_bypass'],
                                'severity': 'critical',
                                'cwe_id': 'CWE-287'
                            }
                        }
                    }
                },
                'ptes': {
                    'name': 'Penetration Testing Execution Standard',
                    'version': '1.1',
                    'description': 'Standard for penetration testing execution',
                    'phases': [
                        'reconnaissance',
                        'scanning',
                        'enumeration',
                        'vulnerability_assessment',
                        'exploitation',
                        'post_exploitation',
                        'reporting'
                    ]
                }
            },
            'methodologies': {
                'sql_injection_testing': {
                    'name': 'SQL Injection Testing Methodology',
                    'steps': [
                        {
                            'step': 1,
                            'name': 'identification',
                            'description': 'Identify injection points',
                            'techniques': ['parameter_fuzzing', 'error_analysis']
                        },
                        {
                            'step': 2,
                            'name': 'confirmation',
                            'description': 'Confirm vulnerability existence',
                            'techniques': ['boolean_blind', 'time_based', 'error_based']
                        },
                        {
                            'step': 3,
                            'name': 'exploitation',
                            'description': 'Exploit the vulnerability',
                            'techniques': ['union_based', 'data_extraction', 'privilege_escalation']
                        },
                        {
                            'step': 4,
                            'name': 'assessment',
                            'description': 'Assess impact and risk',
                            'techniques': ['data_enumeration', 'system_fingerprinting']
                        }
                    ]
                }
            },
            'test_cases': {
                'test_reflected_sql_injection': {
                    'name': 'Test for Reflected SQL Injection',
                    'description': 'Test input fields for reflected SQL injection vulnerabilities',
                    'priority': 'high',
                    'test_vectors': [
                        "' OR '1'='1",
                        "' UNION SELECT 1,2,3--",
                        "'; DROP TABLE users--"
                    ],
                    'expected_behaviors': [
                        'error_messages',
                        'data_disclosure',
                        'boolean_responses'
                    ],
                    'remediation': 'Use parameterized queries and input validation'
                },
                'test_blind_sql_injection': {
                    'name': 'Test for Blind SQL Injection',
                    'description': 'Test for blind SQL injection using boolean and time-based techniques',
                    'priority': 'high',
                    'test_vectors': [
                        "' AND 1=1--",
                        "' AND 1=2--",
                        "' AND SLEEP(5)--"
                    ],
                    'expected_behaviors': [
                        'response_time_differences',
                        'content_length_differences',
                        'status_code_variations'
                    ],
                    'remediation': 'Implement proper error handling and use prepared statements'
                }
            }
        }
        
        self.guidelines = default_data['guidelines']
        self.methodologies = default_data['methodologies']
        self.test_cases = default_data['test_cases']
        
        self._save_catalog()
    
    def get_guideline(self, guideline_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific guideline by ID.
        
        Args:
            guideline_id: Guideline identifier
            
        Returns:
            Guideline data or None if not found
        """
        return self.guidelines.get(guideline_id)
    
    def get_methodology(self, methodology_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific methodology by ID.
        
        Args:
            methodology_id: Methodology identifier
            
        Returns:
            Methodology data or None if not found
        """
        return self.methodologies.get(methodology_id)
    
    def get_test_case(self, test_case_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific test case by ID.
        
        Args:
            test_case_id: Test case identifier
            
        Returns:
            Test case data or None if not found
        """
        return self.test_cases.get(test_case_id)
    
    def get_sql_injection_guidelines(self) -> Dict[str, Any]:
        """
        Get SQL injection specific guidelines.
        
        Returns:
            SQL injection guidelines and test cases
        """
        sql_guidelines = {}
        
        # Extract SQL injection related guidelines
        for guideline_id, guideline_data in self.guidelines.items():
            categories = guideline_data.get('categories', {})
            injection_category = categories.get('injection', {})
            sql_injection = injection_category.get('sql_injection')
            
            if sql_injection:
                sql_guidelines[guideline_id] = {
                    'guideline_name': guideline_data.get('name'),
                    'sql_injection_tests': sql_injection
                }
        
        # Add relevant test cases
        sql_test_cases = {}
        for test_id, test_data in self.test_cases.items():
            if 'sql' in test_id.lower() or 'injection' in test_data.get('name', '').lower():
                sql_test_cases[test_id] = test_data
        
        return {
            'guidelines': sql_guidelines,
            'test_cases': sql_test_cases,
            'methodologies': {
                k: v for k, v in self.methodologies.items() 
                if 'sql' in k.lower()
            }
        }
    
    def generate_test_plan(self, target_type: str = 'web_application', 
                          focus_areas: List[str] = None) -> Dict[str, Any]:
        """
        Generate a test plan based on guidelines and focus areas.
        
        Args:
            target_type: Type of target (web_application, api, etc.)
            focus_areas: List of areas to focus on
            
        Returns:
            Generated test plan
        """
        if focus_areas is None:
            focus_areas = ['sql_injection', 'authentication']
        
        test_plan = {
            'target_type': target_type,
            'focus_areas': focus_areas,
            'test_phases': [],
            'test_cases': [],
            'estimated_duration': '2-4 hours',
            'required_tools': [],
            'success_criteria': []
        }
        
        # Add phases based on PTES methodology
        ptes = self.methodologies.get('sql_injection_testing', {})
        if ptes:
            test_plan['test_phases'] = ptes.get('steps', [])
        
        # Add test cases based on focus areas
        for focus_area in focus_areas:
            for test_id, test_data in self.test_cases.items():
                if focus_area in test_id or focus_area in test_data.get('description', ''):
                    test_plan['test_cases'].append({
                        'id': test_id,
                        'name': test_data.get('name'),
                        'priority': test_data.get('priority', 'medium'),
                        'test_vectors': test_data.get('test_vectors', [])
                    })
        
        # Add tools and success criteria
        if 'sql_injection' in focus_areas:
            test_plan['required_tools'].extend([
                'sqlmap', 'burp_suite', 'custom_payloads'
            ])
            test_plan['success_criteria'].extend([
                'Identify all injection points',
                'Confirm exploitability',
                'Assess data exposure risk'
            ])
        
        return test_plan
    
    def validate_test_execution(self, test_case_id: str, 
                               results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate test execution against guidelines.
        
        Args:
            test_case_id: Test case identifier
            results: Test execution results
            
        Returns:
            Validation results
        """
        test_case = self.get_test_case(test_case_id)
        if not test_case:
            return {'valid': False, 'error': 'Test case not found'}
        
        validation = {
            'valid': True,
            'completeness_score': 0.0,
            'coverage_analysis': {},
            'recommendations': [],
            'missing_elements': []
        }
        
        # Check if required test vectors were used
        expected_vectors = test_case.get('test_vectors', [])
        used_vectors = results.get('payloads_used', [])
        
        if expected_vectors:
            coverage = len(set(used_vectors) & set(expected_vectors)) / len(expected_vectors)
            validation['coverage_analysis']['test_vectors'] = coverage
        
        # Check if expected behaviors were observed
        expected_behaviors = test_case.get('expected_behaviors', [])
        observed_behaviors = results.get('observed_behaviors', [])
        
        if expected_behaviors:
            behavior_coverage = len(set(observed_behaviors) & set(expected_behaviors)) / len(expected_behaviors)
            validation['coverage_analysis']['behaviors'] = behavior_coverage
        
        # Calculate overall completeness
        coverage_scores = list(validation['coverage_analysis'].values())
        if coverage_scores:
            validation['completeness_score'] = sum(coverage_scores) / len(coverage_scores)
        
        # Generate recommendations
        if validation['completeness_score'] < 0.8:
            validation['recommendations'].append(
                'Test coverage is below 80%. Consider testing additional vectors.'
            )
        
        for vector in expected_vectors:
            if vector not in used_vectors:
                validation['missing_elements'].append(f'Test vector: {vector}')
        
        for behavior in expected_behaviors:
            if behavior not in observed_behaviors:
                validation['missing_elements'].append(f'Expected behavior: {behavior}')
        
        return validation
    
    def get_compliance_mapping(self, framework: str = 'owasp_top10') -> Dict[str, Any]:
        """
        Get compliance framework mapping.
        
        Args:
            framework: Compliance framework name
            
        Returns:
            Compliance mapping information
        """
        mappings = {
            'owasp_top10': {
                'A03_injection': {
                    'test_cases': ['test_reflected_sql_injection', 'test_blind_sql_injection'],
                    'severity': 'high',
                    'description': 'Injection flaws occur when untrusted data is sent to an interpreter'
                }
            },
            'cwe_top25': {
                'cwe_89': {
                    'name': 'SQL Injection',
                    'test_cases': ['test_reflected_sql_injection', 'test_blind_sql_injection'],
                    'mitigation': 'Use parameterized queries'
                }
            }
        }
        
        return mappings.get(framework, {})
    
    def export_guidelines(self, format: str = 'json', 
                         guideline_ids: List[str] = None) -> str:
        """
        Export guidelines in specified format.
        
        Args:
            format: Export format (json, markdown, html)
            guideline_ids: Specific guidelines to export
            
        Returns:
            Exported guidelines as string
        """
        if guideline_ids:
            export_data = {gid: self.guidelines[gid] for gid in guideline_ids if gid in self.guidelines}
        else:
            export_data = self.guidelines
        
        if format == 'json':
            return json.dumps(export_data, indent=2)
        
        elif format == 'markdown':
            md_content = []
            md_content.append('# Penetration Testing Guidelines\n')
            
            for guideline_id, guideline_data in export_data.items():
                md_content.append(f"## {guideline_data.get('name', guideline_id)}\n")
                md_content.append(f"**Version:** {guideline_data.get('version', 'N/A')}\n")
                md_content.append(f"**Description:** {guideline_data.get('description', 'N/A')}\n")
                
                categories = guideline_data.get('categories', {})
                if categories:
                    md_content.append('\n### Categories\n')
                    for category_name, category_data in categories.items():
                        md_content.append(f"#### {category_name.replace('_', ' ').title()}\n")
                        if isinstance(category_data, dict):
                            for item_name, item_data in category_data.items():
                                md_content.append(f"- **{item_name.replace('_', ' ').title()}**: {item_data.get('description', 'N/A')}\n")
                
                md_content.append('\n---\n')
            
            return ''.join(md_content)
        
        else:
            return str(export_data)
    
    def search_guidelines(self, query: str, category: str = None) -> List[Dict[str, Any]]:
        """
        Search guidelines based on query.
        
        Args:
            query: Search query
            category: Filter by category
            
        Returns:
            List of matching guidelines
        """
        results = []
        query_lower = query.lower()
        
        for guideline_id, guideline_data in self.guidelines.items():
            # Search in guideline name and description
            if (query_lower in guideline_data.get('name', '').lower() or
                query_lower in guideline_data.get('description', '').lower()):
                
                result = {
                    'id': guideline_id,
                    'name': guideline_data.get('name'),
                    'description': guideline_data.get('description'),
                    'match_type': 'guideline'
                }
                results.append(result)
            
            # Search in categories
            categories = guideline_data.get('categories', {})
            for category_name, category_data in categories.items():
                if category and category != category_name:
                    continue
                
                if isinstance(category_data, dict):
                    for item_name, item_data in category_data.items():
                        if (query_lower in item_name.lower() or
                            query_lower in item_data.get('description', '').lower()):
                            
                            result = {
                                'id': f"{guideline_id}.{category_name}.{item_name}",
                                'name': item_data.get('description', item_name),
                                'description': f"From {guideline_data.get('name')} - {category_name}",
                                'match_type': 'category_item',
                                'parent_guideline': guideline_id
                            }
                            results.append(result)
        
        return results
    
    def _save_catalog(self):
        """
        Save the catalog to file.
        """
        data = {
            'guidelines': self.guidelines,
            'methodologies': self.methodologies,
            'compliance_frameworks': self.compliance_frameworks,
            'test_cases': self.test_cases,
            'metadata': {
                'last_updated': datetime.now().isoformat(),
                'version': '1.0'
            }
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.catalog_file), exist_ok=True)
        
        try:
            with open(self.catalog_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving catalog: {e}")
    
    def add_custom_guideline(self, guideline_id: str, guideline_data: Dict[str, Any]):
        """
        Add a custom guideline to the catalog.
        
        Args:
            guideline_id: Unique identifier for the guideline
            guideline_data: Guideline data
        """
        self.guidelines[guideline_id] = guideline_data
        self._save_catalog()
    
    def add_custom_test_case(self, test_case_id: str, test_case_data: Dict[str, Any]):
        """
        Add a custom test case to the catalog.
        
        Args:
            test_case_id: Unique identifier for the test case
            test_case_data: Test case data
        """
        self.test_cases[test_case_id] = test_case_data
        self._save_catalog()
