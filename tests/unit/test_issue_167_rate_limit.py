import unittest
from unittest.mock import MagicMock, patch
import requests
from src.core.ccxt_client import CcxtClient

class TestIssue167RateLimit(unittest.TestCase):
    def test_make_authenticated_bingx_request_retries_on_429(self):
        """
        Test that _make_authenticated_bingx_request retries on 429 Too Many Requests.
        """
        # Mock BingXAuthenticator
        mock_auth = MagicMock()
        mock_auth.prepare_authenticated_request.return_value = {
            'params': {},
            'headers': {}
        }
        
        # Initialize CcxtClient
        # We need to mock ccxt.bingx to avoid actual initialization
        with patch('ccxt.bingx') as MockBingX:
            client = CcxtClient('bingx', {'apiKey': 'test', 'secret': 'test'})
            client.bingx_auth = mock_auth
            
            # Mock requests.get
            with patch('requests.get') as mock_get:
                # Setup mock response for 429 then 200
                response_429 = MagicMock()
                response_429.status_code = 429
                response_429.raise_for_status.side_effect = requests.exceptions.HTTPError("429 Client Error")
                
                response_200 = MagicMock()
                response_200.status_code = 200
                response_200.json.return_value = {'code': 0, 'data': 'success'}
                
                mock_get.side_effect = [response_429, response_200]
                
                # Call the method
                # We expect it to retry and succeed
                # We might need to patch time.sleep to speed up test
                with patch('time.sleep') as mock_sleep:
                    result = client._make_authenticated_bingx_request('/test')
                    
                    # Verify result
                    self.assertEqual(result, {'code': 0, 'data': 'success'})
                    
                    # Verify it was called twice
                    self.assertEqual(mock_get.call_count, 2)
                    
                    # Verify sleep was called
                    mock_sleep.assert_called()

if __name__ == '__main__':
    unittest.main()
