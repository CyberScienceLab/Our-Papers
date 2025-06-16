import subprocess
import time
import requests


# pstree -p <PID>

class RedHitServer:
    process = None
    port = 7070
    model_path = ''
    timeout = 240

    def __init__(self, model_path: str, port: int = 7070, timeout=240):
        RedHitServer.model_path = model_path
        RedHitServer.port = port
        RedHitServer.timeout = timeout

    @classmethod
    def start_vllm_server(cls):
        if cls.process is not None:
            print('VLLM is running')
            return

        command = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            cls.model_path,
            "--port",
            str(cls.port),
            '--max_num_seqs',
            '1',
            '--load-format',
            'bitsandbytes',
            '--quantization',
            'bitsandbytes',
            '--kv_cache_dtype',
            'fp8',
            '--gpu_memory_utilization',
            '0.45',
            '--uvicorn-log-level',
            'critical'
        ]

        cls.process = subprocess.Popen(command)

        start_time = time.time()
        while time.time() - start_time < cls.timeout:
            try:
                response = requests.get(
                    f"http://0.0.0.0:{cls.port}/health")
                if response.status_code == 200:
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)

    @classmethod
    def stop_vllm_server(cls):
        if cls.process is not None:
            try:
                cls.process.terminate()
                cls.process.wait(timeout=10)

            except subprocess.TimeoutExpired:
                cls.process.kill()

            cls.process = None

    @classmethod
    def restart(cls):
        cls.stop_vllm_server()
        cls.start_vllm_server()
