"""Exercise the shipped HTTP and Stratum code on native sockets, without a GPU."""
import json
import socket
import subprocess
import sys
import urllib.request

with socket.socket() as listener, socket.socket() as reserve:
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    listener.settimeout(10)
    reserve.bind(('127.0.0.1', 0))
    http_port = reserve.getsockname()[1]
    reserve.close()
    child = subprocess.Popen([sys.argv[1], str(listener.getsockname()[1]), str(http_port)])
    try:
        conn, _ = listener.accept()
        with conn, conn.makefile('rwb', buffering=0) as stream:
            conn.settimeout(10)
            sub = json.loads(stream.readline())
            auth = json.loads(stream.readline())
            assert sub['method'] == 'mining.subscribe' and sub['params'][0] == 'supr-meow-tsc/0.7.0'
            assert auth['method'] == 'mining.authorize' and auth['params'] == ['test.worker', 'x']
            with urllib.request.urlopen(f'http://127.0.0.1:{http_port}/summary', timeout=3) as response:
                assert json.load(response) == {'test': True}
            for row in [{'id': 1, 'result': True, 'error': None}, {'id': 2, 'result': True, 'error': None},
                        {'method': 'mining.notify', 'params': ['job', '00'*76, 'ff'*32, 1, 27000, 9999999999, True]}]:
                stream.write(json.dumps(row).encode() + b'\n')
            share = json.loads(stream.readline())
            assert share['method'] == 'mining.submit' and share['params'][:4] == ['test.worker', 'job', 42, 'TEST']
            stream.write(json.dumps({'id': share['id'], 'result': True, 'error': None}).encode() + b'\n')
            assert child.wait(timeout=10) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
print('Native socket contract passed')
