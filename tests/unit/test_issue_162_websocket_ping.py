import unittest
from unittest.mock import MagicMock, patch
import threading
from src.core.bingx_websocket import BingXWebSocket

class TestIssue162WebSocketPing(unittest.TestCase):
    def test_run_forever_called_with_ping_params(self):
        """
        Test that run_forever is called with ping_interval and ping_timeout
        to prevent silent connection drops.
        """
        # Mock websocket.WebSocketApp
        with patch('src.core.bingx_websocket.websocket.WebSocketApp') as MockWebSocketApp:
            # Mock the instance
            mock_ws_app = MagicMock()
            MockWebSocketApp.return_value = mock_ws_app
            
            # Initialize BingXWebSocket
            client = BingXWebSocket(testnet=True)
            
            # We need to mock threading.Thread to avoid actually starting a thread
            with patch('threading.Thread') as MockThread:
                # Call _start_async (we can call it synchronously for testing logic)
                # But _start_async is async, so we need to run it or simulate it.
                # Actually, we can just inspect the code or call the internal logic if possible.
                # But _start_async creates the WebSocketApp and Thread.
                
                # Let's simulate what _start_async does manually to verify the fix pattern,
                # or better, run it in a loop.
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # We need to mock the lock too or just run it
                loop.run_until_complete(client._start_async())
                
                # Verify WebSocketApp was created
                MockWebSocketApp.assert_called()
                
                # Verify Thread was created with target=mock_ws_app.run_forever
                # But we want to check the KWARGS passed to run_forever.
                # Wait, run_forever is called INSIDE the thread target.
                # The thread target is `self.ws.run_forever`.
                # The arguments to run_forever are NOT passed to Thread constructor unless we use args/kwargs there.
                # In the current code: 
                # self._ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
                # So run_forever is called with NO arguments.
                
                # To fix this, the code should be:
                # self._ws_thread = threading.Thread(target=self.ws.run_forever, kwargs={'ping_interval': 30, 'ping_timeout': 10}, daemon=True)
                
                # So we check MockThread call args.
                args, kwargs = MockThread.call_args
                print(f"Thread kwargs: {kwargs}")
                
                target = kwargs.get('target')
                target_kwargs = kwargs.get('kwargs', {})
                
                # Check if ping_interval is in target_kwargs
                self.assertIn('ping_interval', target_kwargs, "ping_interval missing in run_forever kwargs")
                self.assertIn('ping_timeout', target_kwargs, "ping_timeout missing in run_forever kwargs")
                self.assertEqual(target_kwargs['ping_interval'], 30)
                self.assertEqual(target_kwargs['ping_timeout'], 10)

if __name__ == '__main__':
    unittest.main()
