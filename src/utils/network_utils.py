"""
Network utilities for HTTP requests and response handling
"""

import requests
import time
import urllib.parse
from typing import Dict, List, Any, Optional, Tuple
import ssl
import socket
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class NetworkUtils:
    """
    Utility class for network operations and HTTP request handling.
    """
    
    def __init__(self, timeout: int = 10, max_retries: int = 3):
        """
        Initialize network utilities.
        
        Args:
            timeout: Default timeout for requests
            max_retries: Maximum number of retries for failed requests
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = self._create_session()
        self.request_history = []
    
    def _create_session(self) -> requests.Session:
        """
        Create a configured requests session.
        
        Returns:
            Configured requests session
        """
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        return session
    
    def send_request(self, url: str, method: str = 'GET', 
                    params: Optional[Dict[str, str]] = None,
                    data: Optional[Dict[str, str]] = None,
                    headers: Optional[Dict[str, str]] = None,
                    cookies: Optional[Dict[str, str]] = None,
                    timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Send HTTP request and return detailed response information.
        
        Args:
            url: Target URL
            method: HTTP method
            params: URL parameters
            data: POST data
            headers: Additional headers
            cookies: Cookies to send
            timeout: Request timeout
            
        Returns:
            Response information dictionary
        """
        start_time = time.time()
        timeout = timeout or self.timeout
        
        try:
            # Prepare request
            request_headers = {}
            if headers:
                request_headers.update(headers)
            
            if cookies:
                self.session.cookies.update(cookies)
            
            # Send request
            if method.upper() == 'GET':
                response = self.session.get(
                    url, 
                    params=params, 
                    headers=request_headers, 
                    timeout=timeout,
                    allow_redirects=False
                )
            elif method.upper() == 'POST':
                response = self.session.post(
                    url, 
                    data=data, 
                    params=params, 
                    headers=request_headers, 
                    timeout=timeout,
                    allow_redirects=False
                )
            elif method.upper() == 'PUT':
                response = self.session.put(
                    url, 
                    data=data, 
                    params=params, 
                    headers=request_headers, 
                    timeout=timeout,
                    allow_redirects=False
                )
            elif method.upper() == 'DELETE':
                response = self.session.delete(
                    url, 
                    params=params, 
                    headers=request_headers, 
                    timeout=timeout,
                    allow_redirects=False
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response_time = time.time() - start_time
            
            # Build response information
            response_info = {
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'content': response.text,
                'content_length': len(response.content),
                'response_time': response_time,
                'url': response.url,
                'final_url': response.url,
                'cookies': dict(response.cookies),
                'encoding': response.encoding,
                'reason': response.reason,
                'request_headers': dict(response.request.headers),
                'request_method': method.upper(),
                'redirects': len(response.history),
                'ssl_info': self._get_ssl_info(url) if url.startswith('https') else None
            }
            
            # Record request
            self._record_request(url, method, params, data, response_info)
            
            return response_info
            
        except requests.exceptions.Timeout:
            response_time = time.time() - start_time
            error_response = {
                'status_code': 0,
                'headers': {},
                'content': '',
                'content_length': 0,
                'response_time': response_time,
                'url': url,
                'final_url': url,
                'cookies': {},
                'encoding': None,
                'reason': 'Timeout',
                'request_headers': request_headers,
                'request_method': method.upper(),
                'redirects': 0,
                'error': 'timeout',
                'ssl_info': None
            }
            
            self._record_request(url, method, params, data, error_response)
            return error_response
            
        except requests.exceptions.ConnectionError as e:
            response_time = time.time() - start_time
            error_response = {
                'status_code': 0,
                'headers': {},
                'content': '',
                'content_length': 0,
                'response_time': response_time,
                'url': url,
                'final_url': url,
                'cookies': {},
                'encoding': None,
                'reason': 'Connection Error',
                'request_headers': request_headers,
                'request_method': method.upper(),
                'redirects': 0,
                'error': f'connection_error: {str(e)}',
                'ssl_info': None
            }
            
            self._record_request(url, method, params, data, error_response)
            return error_response
            
        except Exception as e:
            response_time = time.time() - start_time
            error_response = {
                'status_code': 0,
                'headers': {},
                'content': '',
                'content_length': 0,
                'response_time': response_time,
                'url': url,
                'final_url': url,
                'cookies': {},
                'encoding': None,
                'reason': 'Unknown Error',
                'request_headers': request_headers,
                'request_method': method.upper(),
                'redirects': 0,
                'error': f'unknown_error: {str(e)}',
                'ssl_info': None
            }
            
            self._record_request(url, method, params, data, error_response)
            return error_response
    
    def _get_ssl_info(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Get SSL certificate information for HTTPS URLs.
        
        Args:
            url: HTTPS URL
            
        Returns:
            SSL information or None
        """
        try:
            parsed_url = urllib.parse.urlparse(url)
            hostname = parsed_url.hostname
            port = parsed_url.port or 443
            
            context = ssl.create_default_context()
            with socket.create_connection((hostname, port), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    
                    return {
                        'subject': dict(x[0] for x in cert['subject']),
                        'issuer': dict(x[0] for x in cert['issuer']),
                        'version': cert['version'],
                        'serial_number': cert['serialNumber'],
                        'not_before': cert['notBefore'],
                        'not_after': cert['notAfter'],
                        'cipher': ssock.cipher(),
                        'protocol_version': ssock.version()
                    }
        except Exception:
            return None
    
    def _record_request(self, url: str, method: str, params: Optional[Dict[str, str]], 
                       data: Optional[Dict[str, str]], response: Dict[str, Any]):
        """
        Record request details for analysis.
        
        Args:
            url: Request URL
            method: HTTP method
            params: URL parameters
            data: POST data
            response: Response information
        """
        record = {
            'timestamp': time.time(),
            'url': url,
            'method': method,
            'params': params,
            'data': data,
            'status_code': response.get('status_code'),
            'response_time': response.get('response_time'),
            'content_length': response.get('content_length'),
            'error': response.get('error')
        }
        
        self.request_history.append(record)
        
        # Keep only last 1000 requests
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
    
    def analyze_response_patterns(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze patterns in multiple responses.
        
        Args:
            responses: List of response dictionaries
            
        Returns:
            Analysis results
        """
        if not responses:
            return {}
        
        analysis = {
            'status_code_distribution': {},
            'response_time_stats': {
                'min': float('inf'),
                'max': 0,
                'avg': 0,
                'median': 0
            },
            'content_length_stats': {
                'min': float('inf'),
                'max': 0,
                'avg': 0,
                'median': 0
            },
            'error_rate': 0,
            'common_headers': {},
            'response_patterns': []
        }
        
        response_times = []
        content_lengths = []
        error_count = 0
        header_counts = {}
        
        # Collect statistics
        for response in responses:
            # Status codes
            status = response.get('status_code', 0)
            analysis['status_code_distribution'][status] = \
                analysis['status_code_distribution'].get(status, 0) + 1
            
            # Response times
            resp_time = response.get('response_time', 0)
            response_times.append(resp_time)
            
            # Content lengths
            content_len = response.get('content_length', 0)
            content_lengths.append(content_len)
            
            # Errors
            if response.get('error') or status == 0:
                error_count += 1
            
            # Headers
            headers = response.get('headers', {})
            for header_name in headers.keys():
                header_counts[header_name] = header_counts.get(header_name, 0) + 1
        
        # Calculate statistics
        if response_times:
            analysis['response_time_stats']['min'] = min(response_times)
            analysis['response_time_stats']['max'] = max(response_times)
            analysis['response_time_stats']['avg'] = sum(response_times) / len(response_times)
            
            sorted_times = sorted(response_times)
            mid = len(sorted_times) // 2
            analysis['response_time_stats']['median'] = sorted_times[mid]
        
        if content_lengths:
            analysis['content_length_stats']['min'] = min(content_lengths)
            analysis['content_length_stats']['max'] = max(content_lengths)
            analysis['content_length_stats']['avg'] = sum(content_lengths) / len(content_lengths)
            
            sorted_lengths = sorted(content_lengths)
            mid = len(sorted_lengths) // 2
            analysis['content_length_stats']['median'] = sorted_lengths[mid]
        
        # Error rate
        analysis['error_rate'] = error_count / len(responses) if responses else 0
        
        # Common headers (present in >50% of responses)
        threshold = len(responses) * 0.5
        analysis['common_headers'] = {
            header: count for header, count in header_counts.items() 
            if count > threshold
        }
        
        return analysis
    
    def detect_rate_limiting(self, url: str, requests_per_second: int = 10, 
                           duration: int = 60) -> Dict[str, Any]:
        """
        Detect rate limiting by sending multiple requests.
        
        Args:
            url: Target URL
            requests_per_second: Number of requests per second
            duration: Test duration in seconds
            
        Returns:
            Rate limiting detection results
        """
        results = {
            'rate_limiting_detected': False,
            'requests_sent': 0,
            'successful_requests': 0,
            'blocked_requests': 0,
            'average_response_time': 0,
            'blocking_status_codes': [],
            'blocking_threshold': None
        }
        
        start_time = time.time()
        interval = 1.0 / requests_per_second
        responses = []
        
        while time.time() - start_time < duration:
            response = self.send_request(url)
            responses.append(response)
            results['requests_sent'] += 1
            
            status_code = response.get('status_code', 0)
            
            if status_code in [429, 503, 403]:  # Common rate limiting codes
                results['blocked_requests'] += 1
                if status_code not in results['blocking_status_codes']:
                    results['blocking_status_codes'].append(status_code)
            elif 200 <= status_code < 300:
                results['successful_requests'] += 1
            
            time.sleep(interval)
        
        # Analysis
        if results['blocked_requests'] > 0:
            results['rate_limiting_detected'] = True
            results['blocking_threshold'] = results['requests_sent'] - results['blocked_requests']
        
        if responses:
            total_time = sum(r.get('response_time', 0) for r in responses)
            results['average_response_time'] = total_time / len(responses)
        
        return results
    
    def test_http_methods(self, url: str) -> Dict[str, Dict[str, Any]]:
        """
        Test different HTTP methods on a URL.
        
        Args:
            url: Target URL
            
        Returns:
            Results for each HTTP method
        """
        methods = ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'PATCH']
        results = {}
        
        for method in methods:
            try:
                response = self.send_request(url, method=method)
                results[method] = {
                    'supported': response.get('status_code') not in [405, 501],
                    'status_code': response.get('status_code'),
                    'response_time': response.get('response_time'),
                    'content_length': response.get('content_length'),
                    'headers': response.get('headers', {}),
                    'error': response.get('error')
                }
            except Exception as e:
                results[method] = {
                    'supported': False,
                    'error': str(e)
                }
        
        return results
    
    def get_request_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about recorded requests.
        
        Returns:
            Request statistics
        """
        if not self.request_history:
            return {}
        
        stats = {
            'total_requests': len(self.request_history),
            'unique_urls': len(set(r['url'] for r in self.request_history)),
            'method_distribution': {},
            'status_code_distribution': {},
            'average_response_time': 0,
            'error_rate': 0,
            'request_rate': 0  # requests per minute
        }
        
        # Collect data
        response_times = []
        error_count = 0
        
        for request in self.request_history:
            # Methods
            method = request['method']
            stats['method_distribution'][method] = \
                stats['method_distribution'].get(method, 0) + 1
            
            # Status codes
            status = request['status_code']
            stats['status_code_distribution'][status] = \
                stats['status_code_distribution'].get(status, 0) + 1
            
            # Response times
            if request['response_time']:
                response_times.append(request['response_time'])
            
            # Errors
            if request['error'] or status == 0:
                error_count += 1
        
        # Calculate averages
        if response_times:
            stats['average_response_time'] = sum(response_times) / len(response_times)
        
        stats['error_rate'] = error_count / len(self.request_history)
        
        # Calculate request rate
        if len(self.request_history) > 1:
            time_span = (self.request_history[-1]['timestamp'] - 
                        self.request_history[0]['timestamp'])
            if time_span > 0:
                stats['request_rate'] = len(self.request_history) / (time_span / 60)
        
        return stats
    
    def clear_history(self):
        """
        Clear the request history.
        """
        self.request_history.clear()
    
    def set_proxy(self, proxy_url: str):
        """
        Set proxy for requests.
        
        Args:
            proxy_url: Proxy URL (e.g., 'http://proxy:8080')
        """
        self.session.proxies.update({
            'http': proxy_url,
            'https': proxy_url
        })
    
    def set_user_agent(self, user_agent: str):
        """
        Set custom User-Agent header.
        
        Args:
            user_agent: User agent string
        """
        self.session.headers.update({'User-Agent': user_agent})
    
    def add_custom_header(self, name: str, value: str):
        """
        Add a custom header to all requests.
        
        Args:
            name: Header name
            value: Header value
        """
        self.session.headers.update({name: value})
    
    def remove_custom_header(self, name: str):
        """
        Remove a custom header.
        
        Args:
            name: Header name to remove
        """
        if name in self.session.headers:
            del self.session.headers[name]
