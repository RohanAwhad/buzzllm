import json
import socket
import traceback
import threading
import time
from jupyter_client import KernelManager


class KernelServer:
    def __init__(self):
        self.km = KernelManager()
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        
        # Wait for kernel to be ready
        while True:
            try:
                self.kc.kernel_info()
                break
            except Exception:
                time.sleep(0.1)

    def execute(self, code):
        """Execute code and return stdout, stderr, and result"""
        msg_id = self.kc.execute(code)
        outs, errs, result = [], [], None
        
        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=30)
                
                if msg['header']['msg_type'] == 'execute_result':
                    result = msg['content']['data'].get('text/plain', '')
                elif msg['header']['msg_type'] == 'stream':
                    if msg['content']['name'] == 'stdout':
                        outs.append(msg['content']['text'])
                    else:
                        errs.append(msg['content']['text'])
                elif msg['header']['msg_type'] == 'error':
                    errs.append('\n'.join(msg['content']['traceback']))
                elif (msg['header']['msg_type'] == 'status' and 
                      msg['content']['execution_state'] == 'idle'):
                    break
                    
            except Exception as e:
                errs.append(f"Kernel error: {str(e)}")
                break
        
        return {
            'stdout': ''.join(outs),
            'stderr': ''.join(errs),
            'result': result
        }

    def handle_client(self, conn):
        """Handle a single client connection"""
        try:
            data = conn.recv(65536)
            if not data:
                return
                
            req = json.loads(data.decode())
            code = req.get('code', '')
            
            if not code:
                resp = {'stdout': '', 'stderr': 'No code provided', 'result': None}
            else:
                resp = self.execute(code)
            
            conn.sendall(json.dumps(resp).encode())
            
        except Exception as e:
            error_resp = {
                'stdout': '',
                'stderr': f'Server error: {str(e)}',
                'result': None
            }
            try:
                conn.sendall(json.dumps(error_resp).encode())
            except:
                pass
        finally:
            conn.close()

    def run_server(self):
        """Run the kernel server"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', 8787))
        server_socket.listen(5)
        
        print("Kernel server started on port 8787")
        
        while True:
            try:
                conn, addr = server_socket.accept()
                # Handle each connection in a separate thread
                client_thread = threading.Thread(
                    target=self.handle_client, 
                    args=(conn,)
                )
                client_thread.daemon = True
                client_thread.start()
                
            except Exception as e:
                print(f"Server error: {e}")
                break


if __name__ == "__main__":
    server = KernelServer()
    server.run_server()
