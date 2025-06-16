"""garak entry point wrapper"""

import os
from utility import disable_logs, seed_torch
import sys
from config import RedHitConf
from redhit_server.redhit_server import RedHitServer
from garak import cli

os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

seed_torch()

disable_logs()

model_name = RedHitConf.get('default_model_path')
server = RedHitServer(model_name, port=7070)
RedHitServer.start_vllm_server()


def main():
    cli.main(sys.argv[1:])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
