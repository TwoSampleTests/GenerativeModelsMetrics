Visual Studio Code (1.87.2, ssh-remote, desktop)
Jupyter Extension Version: 2024.2.0.
Python Extension Version: 2024.2.1.
Pylance Extension Version: 2024.3.1.
Platform: linux (x64).
Workspace folder /mnt/project_mnt/teo_fs/sgrossi/GenerativeModelsMetrics, Home = /home/sgrossi
11:51:57.583 [warn] Exception while attempting zmq : /lib64/libstdc++.so.6: version `GLIBCXX_3.4.20' not found (required by /mnt/project_mnt/home_fs/<username>/.vscode-server/extensions/ms-toolsai.jupyter-2024.2.0-linux-x64/dist/node_modules/zeromq/prebuilds/linux-x64/node.napi.glibc.node)
11:51:57.584 [warn] Exception while attempting zmq (fallback) : /lib64/libstdc++.so.6: version `GLIBCXX_3.4.20' not found (required by /mnt/project_mnt/home_fs/<username>/.vscode-server/extensions/ms-toolsai.jupyter-2024.2.0-linux-x64/dist/node_modules/zeromqold/prebuilds/linux-x64/node.napi.glibc.node)
11:51:57.977 [info] Start refreshing Kernel Picker (1710931917977)
11:51:58.005 [info] Using Pylance
11:51:58.621 [info] Start refreshing Interpreter Kernel Picker
11:52:00.741 [warn] Publisher  is allowed to access the Kernel API with a message.
11:52:05.216 [info] Starting Kernel startUsingPythonInterpreter, .jvsc74a57bd01376128a827615767986518bfb33281eafa68a5e48cb8581eb40a42ba3659816./local_data/scratch/rtorre/anaconda3/envs/tf2_6/python./local_data/scratch/rtorre/anaconda3/envs/tf2_6/python.-m#ipykernel_launcher  (Python Path: /local_data/scratch/rtorre/anaconda3/envs/tf2_6/bin/python, Conda, tf2_6, 3.8.6) for '/mnt/project_mnt/teo_fs/<username>/GenerativeModelsMetrics/notebooks/Test_metrics_Samuele.ipynb' (disableUI=true)
11:52:06.299 [info] Process Execution: /local_data/scratch/rtorre/anaconda3/envs/tf2_12/bin/python -c "import jupyter;print('6af208d0-cb9c-427f-b937-ff563e17efdf')"
11:52:06.313 [info] Process Execution: /local_data/scratch/rtorre/anaconda3/envs/tf2_12/bin/python -c "import notebook;print('6af208d0-cb9c-427f-b937-ff563e17efdf')"
11:52:06.331 [info] Process Execution: /local_data/scratch/rtorre/anaconda3/envs/tf2_12/bin/python -m pip list
11:52:06.332 [info] End refreshing Kernel Picker (1710931917977)
11:52:06.423 [info] Process Execution: /local_data/scratch/rtorre/anaconda3/envs/tf2_12/bin/python -m jupyter kernelspec --version
11:52:06.730 [info] Launching server
11:52:06.735 [info] Starting Notebook
11:52:06.737 [info] Generating custom default config at /tmp/3f08edb1-13e5-4624-9c45-a6c716c070ff/jupyter_notebook_config.py
11:52:06.761 [info] Process Execution: /local_data/scratch/rtorre/anaconda3/envs/tf2_12/bin/python -m jupyter notebook --no-browser --notebook-dir="/mnt/project_mnt/teo_fs/<username>/GenerativeModelsMetrics/notebooks" --config=/tmp/3f08edb1-13e5-4624-9c45-a6c716c070ff/jupyter_notebook_config.py --NotebookApp.iopub_data_rate_limit=10000000000.0
11:52:06.905 [error] Failed to start the Jupyter Server Error: Jupyter Server crashed. Unable to connect. 
Error code from Jupyter: 1
usage: jupyter.py [-h] [--version] [--config-dir] [--data-dir] [--runtime-dir]
                  [--paths] [--json] [--debug]
                  [subcommand]

Jupyter: Interactive Computing

positional arguments:
  subcommand     the subcommand to launch

options:
  -h, --help     show this help message and exit
  --version      show the versions of core jupyter packages and exit
  --config-dir   show Jupyter config dir
  --data-dir     show Jupyter data dir
  --runtime-dir  show Jupyter runtime dir
  --paths        show all Jupyter paths. Add --json for machine-readable
                 format.
  --json         output paths as machine-readable json
  --debug        output debug information about paths

Available subcommands: kernel kernelspec migrate run troubleshoot trust

Jupyter command `jupyter-notebook` not found.
    > at $_.rejectStartPromise (/mnt/project_mnt/home_fs/<username>/.vscode-server/extensions/ms-toolsai.jupyter-2024.2.0-linux-x64/dist/extension.node.js:308:2207)
    > interpreter = [object Object]
11:52:06.909 [warn] Error occurred while trying to start the kernel, options.disableUI=true Error: Jupyter Server crashed. Unable to connect. 
Error code from Jupyter: 1
usage: jupyter.py [-h] [--version] [--config-dir] [--data-dir] [--runtime-dir]
                  [--paths] [--json] [--debug]
                  [subcommand]

Jupyter: Interactive Computing

positional arguments:
  subcommand     the subcommand to launch

options:
  -h, --help     show this help message and exit
  --version      show the versions of core jupyter packages and exit
  --config-dir   show Jupyter config dir
  --data-dir     show Jupyter data dir
  --runtime-dir  show Jupyter runtime dir
  --paths        show all Jupyter paths. Add --json for machine-readable
                 format.
  --json         output paths as machine-readable json
  --debug        output debug information about paths

Available subcommands: kernel kernelspec migrate run troubleshoot trust

Jupyter command `jupyter-notebook` not found.
    > at $_.rejectStartPromise (/mnt/project_mnt/home_fs/<username>/.vscode-server/extensions/ms-toolsai.jupyter-2024.2.0-linux-x64/dist/extension.node.js:308:2207)
    > interpreter = [object Object]
11:52:07.320 [info] Process Execution: /local_data/scratch/rtorre/anaconda3/envs/tf2_6/bin/python -m pip list
11:52:41.698 [info] Restart requested /mnt/project_mnt/teo_fs/<username>/GenerativeModelsMetrics/notebooks/Test_metrics_Samuele.ipynb
11:52:41.698 [info] No kernel session to interrupt
11:52:41.710 [info] Starting Kernel startUsingPythonInterpreter, .jvsc74a57bd01376128a827615767986518bfb33281eafa68a5e48cb8581eb40a42ba3659816./local_data/scratch/rtorre/anaconda3/envs/tf2_6/python./local_data/scratch/rtorre/anaconda3/envs/tf2_6/python.-m#ipykernel_launcher  (Python Path: /local_data/scratch/rtorre/anaconda3/envs/tf2_6/bin/python, Conda, tf2_6, 3.8.6) for '/mnt/project_mnt/teo_fs/<username>/GenerativeModelsMetrics/notebooks/Test_metrics_Samuele.ipynb' (disableUI=false)
11:52:41.713 [info] Launching server
11:52:41.753 [info] Starting Notebook
11:52:41.755 [info] Generating custom default config at /tmp/78c7e751-373d-47ec-b2d8-96e79b5bcbda/jupyter_notebook_config.py
11:52:41.791 [info] Process Execution: /local_data/scratch/rtorre/anaconda3/envs/tf2_12/bin/python -m jupyter notebook --no-browser --notebook-dir="/mnt/project_mnt/teo_fs/<username>/GenerativeModelsMetrics/notebooks" --config=/tmp/78c7e751-373d-47ec-b2d8-96e79b5bcbda/jupyter_notebook_config.py --NotebookApp.iopub_data_rate_limit=10000000000.0
11:52:41.976 [error] Failed to start the Jupyter Server Error: Jupyter Server crashed. Unable to connect. 
Error code from Jupyter: 1
usage: jupyter.py [-h] [--version] [--config-dir] [--data-dir] [--runtime-dir]
                  [--paths] [--json] [--debug]
                  [subcommand]

Jupyter: Interactive Computing

positional arguments:
  subcommand     the subcommand to launch

options:
  -h, --help     show this help message and exit
  --version      show the versions of core jupyter packages and exit
  --config-dir   show Jupyter config dir
  --data-dir     show Jupyter data dir
  --runtime-dir  show Jupyter runtime dir
  --paths        show all Jupyter paths. Add --json for machine-readable
                 format.
  --json         output paths as machine-readable json
  --debug        output debug information about paths

Available subcommands: kernel kernelspec migrate run troubleshoot trust

Jupyter command `jupyter-notebook` not found.
    > at $_.rejectStartPromise (/mnt/project_mnt/home_fs/<username>/.vscode-server/extensions/ms-toolsai.jupyter-2024.2.0-linux-x64/dist/extension.node.js:308:2207)
    > interpreter = [object Object]
11:52:41.978 [error] Restart failed /mnt/project_mnt/teo_fs/<username>/GenerativeModelsMetrics/notebooks/Test_metrics_Samuele.ipynb Error: Jupyter Server crashed. Unable to connect. 
Error code from Jupyter: 1
usage: jupyter.py [-h] [--version] [--config-dir] [--data-dir] [--runtime-dir]
                  [--paths] [--json] [--debug]
                  [subcommand]

Jupyter: Interactive Computing

positional arguments:
  subcommand     the subcommand to launch

options:
  -h, --help     show this help message and exit
  --version      show the versions of core jupyter packages and exit
  --config-dir   show Jupyter config dir
  --data-dir     show Jupyter data dir
  --runtime-dir  show Jupyter runtime dir
  --paths        show all Jupyter paths. Add --json for machine-readable
                 format.
  --json         output paths as machine-readable json
  --debug        output debug information about paths

Available subcommands: kernel kernelspec migrate run troubleshoot trust

Jupyter command `jupyter-notebook` not found.
    > at $_.rejectStartPromise (/mnt/project_mnt/home_fs/<username>/.vscode-server/extensions/ms-toolsai.jupyter-2024.2.0-linux-x64/dist/extension.node.js:308:2207)
    > interpreter = [object Object]
11:52:41.995 [error] Failed to restart kernel /mnt/project_mnt/teo_fs/<username>/GenerativeModelsMetrics/notebooks/Test_metrics_Samuele.ipynb Error: Jupyter Server crashed. Unable to connect. 
Error code from Jupyter: 1
usage: jupyter.py [-h] [--version] [--config-dir] [--data-dir] [--runtime-dir]
                  [--paths] [--json] [--debug]
                  [subcommand]

Jupyter: Interactive Computing

positional arguments:
  subcommand     the subcommand to launch

options:
  -h, --help     show this help message and exit
  --version      show the versions of core jupyter packages and exit
  --config-dir   show Jupyter config dir
  --data-dir     show Jupyter data dir
  --runtime-dir  show Jupyter runtime dir
  --paths        show all Jupyter paths. Add --json for machine-readable
                 format.
  --json         output paths as machine-readable json
  --debug        output debug information about paths

Available subcommands: kernel kernelspec migrate run troubleshoot trust

Jupyter command `jupyter-notebook` not found.
    > at $_.rejectStartPromise (/mnt/project_mnt/home_fs/<username>/.vscode-server/extensions/ms-toolsai.jupyter-2024.2.0-linux-x64/dist/extension.node.js:308:2207)
    > interpreter = [object Object]
11:52:41.997 [warn] Error occurred while trying to restart the kernel, options.disableUI=false Error: Jupyter Server crashed. Unable to connect. 
Error code from Jupyter: 1
usage: jupyter.py [-h] [--version] [--config-dir] [--data-dir] [--runtime-dir]
                  [--paths] [--json] [--debug]
                  [subcommand]

Jupyter: Interactive Computing

positional arguments:
  subcommand     the subcommand to launch

options:
  -h, --help     show this help message and exit
  --version      show the versions of core jupyter packages and exit
  --config-dir   show Jupyter config dir
  --data-dir     show Jupyter data dir
  --runtime-dir  show Jupyter runtime dir
  --paths        show all Jupyter paths. Add --json for machine-readable
                 format.
  --json         output paths as machine-readable json
  --debug        output debug information about paths

Available subcommands: kernel kernelspec migrate run troubleshoot trust

Jupyter command `jupyter-notebook` not found.
    > at $_.rejectStartPromise (/mnt/project_mnt/home_fs/<username>/.vscode-server/extensions/ms-toolsai.jupyter-2024.2.0-linux-x64/dist/extension.node.js:308:2207)
    > interpreter = [object Object]
11:52:41.998 [warn] Kernel Error, context = restart Error: Jupyter Server crashed. Unable to connect. 
Error code from Jupyter: 1
usage: jupyter.py [-h] [--version] [--config-dir] [--data-dir] [--runtime-dir]
                  [--paths] [--json] [--debug]
                  [subcommand]

Jupyter: Interactive Computing

positional arguments:
  subcommand     the subcommand to launch

options:
  -h, --help     show this help message and exit
  --version      show the versions of core jupyter packages and exit
  --config-dir   show Jupyter config dir
  --data-dir     show Jupyter data dir
  --runtime-dir  show Jupyter runtime dir
  --paths        show all Jupyter paths. Add --json for machine-readable
                 format.
  --json         output paths as machine-readable json
  --debug        output debug information about paths

Available subcommands: kernel kernelspec migrate run troubleshoot trust

Jupyter command `jupyter-notebook` not found.
    > at $_.rejectStartPromise (/mnt/project_mnt/home_fs/<username>/.vscode-server/extensions/ms-toolsai.jupyter-2024.2.0-linux-x64/dist/extension.node.js:308:2207)
    > interpreter = [object Object]
11:52:42.023 [info] Process Execution: /local_data/scratch/rtorre/anaconda3/envs/tf2_6/bin/python -c "import ipykernel;print('6af208d0-cb9c-427f-b937-ff563e17efdf')"
11:52:42.394 [info] Dispose Kernel '/mnt/project_mnt/teo_fs/<username>/GenerativeModelsMetrics/notebooks/Test_metrics_Samuele.ipynb' associated with '/mnt/project_mnt/teo_fs/<username>/GenerativeModelsMetrics/notebooks/Test_metrics_Samuele.ipynb'
11:52:52.917 [info] Starting Kernel startUsingPythonInterpreter, .jvsc74a57bd02fabc9d06fbc88b0e6f8c81f91c976828a53ad7c4d928b5740ae40a4ff507bda./local_data/scratch/rtorre/anaconda3/envs/tf2_12/python./local_data/scratch/rtorre/anaconda3/envs/tf2_12/python.-m#ipykernel_launcher  (Python Path: /local_data/scratch/rtorre/anaconda3/envs/tf2_12/bin/python, Conda, 3.10.11) for '/mnt/project_mnt/teo_fs/<username>/GenerativeModelsMetrics/notebooks/Test_metrics_Samuele.ipynb' (disableUI=true)
11:52:52.919 [info] Launching server
11:52:52.940 [info] Starting Notebook
11:52:52.950 [info] Generating custom default config at /tmp/9d2bb190-5474-4053-97e6-70083967fcc4/jupyter_notebook_config.py
11:52:52.975 [info] Process Execution: /local_data/scratch/rtorre/anaconda3/envs/tf2_12/bin/python -m jupyter notebook --no-browser --notebook-dir="/mnt/project_mnt/teo_fs/<username>/GenerativeModelsMetrics/notebooks" --config=/tmp/9d2bb190-5474-4053-97e6-70083967fcc4/jupyter_notebook_config.py --NotebookApp.iopub_data_rate_limit=10000000000.0
11:52:53.117 [error] Failed to start the Jupyter Server Error: Jupyter Server crashed. Unable to connect. 
Error code from Jupyter: 1
usage: jupyter.py [-h] [--version] [--config-dir] [--data-dir] [--runtime-dir]
                  [--paths] [--json] [--debug]
                  [subcommand]

Jupyter: Interactive Computing

positional arguments:
  subcommand     the subcommand to launch

options:
  -h, --help     show this help message and exit
  --version      show the versions of core jupyter packages and exit
  --config-dir   show Jupyter config dir
  --data-dir     show Jupyter data dir
  --runtime-dir  show Jupyter runtime dir
  --paths        show all Jupyter paths. Add --json for machine-readable
                 format.
  --json         output paths as machine-readable json
  --debug        output debug information about paths

Available subcommands: kernel kernelspec migrate run troubleshoot trust

Jupyter command `jupyter-notebook` not found.
    > at $_.rejectStartPromise (/mnt/project_mnt/home_fs/<username>/.vscode-server/extensions/ms-toolsai.jupyter-2024.2.0-linux-x64/dist/extension.node.js:308:2207)
    > interpreter = [object Object]
11:52:53.124 [warn] Error occurred while trying to start the kernel, options.disableUI=true Error: Jupyter Server crashed. Unable to connect. 
Error code from Jupyter: 1
usage: jupyter.py [-h] [--version] [--config-dir] [--data-dir] [--runtime-dir]
                  [--paths] [--json] [--debug]
                  [subcommand]

Jupyter: Interactive Computing

positional arguments:
  subcommand     the subcommand to launch

options:
  -h, --help     show this help message and exit
  --version      show the versions of core jupyter packages and exit
  --config-dir   show Jupyter config dir
  --data-dir     show Jupyter data dir
  --runtime-dir  show Jupyter runtime dir
  --paths        show all Jupyter paths. Add --json for machine-readable
                 format.
  --json         output paths as machine-readable json
  --debug        output debug information about paths

Available subcommands: kernel kernelspec migrate run troubleshoot trust

Jupyter command `jupyter-notebook` not found.
    > at $_.rejectStartPromise (/mnt/project_mnt/home_fs/<username>/.vscode-server/extensions/ms-toolsai.jupyter-2024.2.0-linux-x64/dist/extension.node.js:308:2207)
    > interpreter = [object Object]
11:52:54.906 [info] Handle Execution of Cells 2 for /mnt/project_mnt/teo_fs/<username>/GenerativeModelsMetrics/notebooks/Test_metrics_Samuele.ipynb
11:52:54.911 [info] Starting Kernel startUsingPythonInterpreter, .jvsc74a57bd02fabc9d06fbc88b0e6f8c81f91c976828a53ad7c4d928b5740ae40a4ff507bda./local_data/scratch/rtorre/anaconda3/envs/tf2_12/python./local_data/scratch/rtorre/anaconda3/envs/tf2_12/python.-m#ipykernel_launcher  (Python Path: /local_data/scratch/rtorre/anaconda3/envs/tf2_12/bin/python, Conda, 3.10.11) for '/mnt/project_mnt/teo_fs/<username>/GenerativeModelsMetrics/notebooks/Test_metrics_Samuele.ipynb' (disableUI=false)
11:52:54.912 [info] Launching server
11:52:54.917 [info] Starting Notebook
11:52:54.919 [info] Generating custom default config at /tmp/388a2f0f-ef59-4085-baf9-26ee566b626e/jupyter_notebook_config.py
11:52:54.944 [info] Process Execution: /local_data/scratch/rtorre/anaconda3/envs/tf2_12/bin/python -m jupyter notebook --no-browser --notebook-dir="/mnt/project_mnt/teo_fs/<username>/GenerativeModelsMetrics/notebooks" --config=/tmp/388a2f0f-ef59-4085-baf9-26ee566b626e/jupyter_notebook_config.py --NotebookApp.iopub_data_rate_limit=10000000000.0
11:52:55.057 [error] Failed to start the Jupyter Server Error: Jupyter Server crashed. Unable to connect. 
Error code from Jupyter: 1
usage: jupyter.py [-h] [--version] [--config-dir] [--data-dir] [--runtime-dir]
                  [--paths] [--json] [--debug]
                  [subcommand]

Jupyter: Interactive Computing

positional arguments:
  subcommand     the subcommand to launch

options:
  -h, --help     show this help message and exit
  --version      show the versions of core jupyter packages and exit
  --config-dir   show Jupyter config dir
  --data-dir     show Jupyter data dir
  --runtime-dir  show Jupyter runtime dir
  --paths        show all Jupyter paths. Add --json for machine-readable
                 format.
  --json         output paths as machine-readable json
  --debug        output debug information about paths

Available subcommands: kernel kernelspec migrate run troubleshoot trust

Jupyter command `jupyter-notebook` not found.
    > at $_.rejectStartPromise (/mnt/project_mnt/home_fs/<username>/.vscode-server/extensions/ms-toolsai.jupyter-2024.2.0-linux-x64/dist/extension.node.js:308:2207)
    > interpreter = [object Object]
11:52:55.059 [warn] Error occurred while trying to start the kernel, options.disableUI=false Error: Jupyter Server crashed. Unable to connect. 
Error code from Jupyter: 1
usage: jupyter.py [-h] [--version] [--config-dir] [--data-dir] [--runtime-dir]
                  [--paths] [--json] [--debug]
                  [subcommand]

Jupyter: Interactive Computing

positional arguments:
  subcommand     the subcommand to launch

options:
  -h, --help     show this help message and exit
  --version      show the versions of core jupyter packages and exit
  --config-dir   show Jupyter config dir
  --data-dir     show Jupyter data dir
  --runtime-dir  show Jupyter runtime dir
  --paths        show all Jupyter paths. Add --json for machine-readable
                 format.
  --json         output paths as machine-readable json
  --debug        output debug information about paths

Available subcommands: kernel kernelspec migrate run troubleshoot trust

Jupyter command `jupyter-notebook` not found.
    > at $_.rejectStartPromise (/mnt/project_mnt/home_fs/<username>/.vscode-server/extensions/ms-toolsai.jupyter-2024.2.0-linux-x64/dist/extension.node.js:308:2207)
    > interpreter = [object Object]
11:52:55.059 [warn] Kernel Error, context = start Error: Jupyter Server crashed. Unable to connect. 
Error code from Jupyter: 1
usage: jupyter.py [-h] [--version] [--config-dir] [--data-dir] [--runtime-dir]
                  [--paths] [--json] [--debug]
                  [subcommand]

Jupyter: Interactive Computing

positional arguments:
  subcommand     the subcommand to launch

options:
  -h, --help     show this help message and exit
  --version      show the versions of core jupyter packages and exit
  --config-dir   show Jupyter config dir
  --data-dir     show Jupyter data dir
  --runtime-dir  show Jupyter runtime dir
  --paths        show all Jupyter paths. Add --json for machine-readable
                 format.
  --json         output paths as machine-readable json
  --debug        output debug information about paths

Available subcommands: kernel kernelspec migrate run troubleshoot trust

Jupyter command `jupyter-notebook` not found.
    > at $_.rejectStartPromise (/mnt/project_mnt/home_fs/<username>/.vscode-server/extensions/ms-toolsai.jupyter-2024.2.0-linux-x64/dist/extension.node.js:308:2207)
    > interpreter = [object Object]
11:52:55.082 [info] Process Execution: /local_data/scratch/rtorre/anaconda3/envs/tf2_12/bin/python -c "import ipykernel;print('6af208d0-cb9c-427f-b937-ff563e17efdf')"
11:52:55.256 [info] Dispose Kernel '/mnt/project_mnt/teo_fs/<username>/GenerativeModelsMetrics/notebooks/Test_metrics_Samuele.ipynb' associated with '/mnt/project_mnt/teo_fs/<username>/GenerativeModelsMetrics/notebooks/Test_metrics_Samuele.ipynb'
11:52:55.258 [error] Error in execution Error: Jupyter Server crashed. Unable to connect. 
Error code from Jupyter: 1
usage: jupyter.py [-h] [--version] [--config-dir] [--data-dir] [--runtime-dir]
                  [--paths] [--json] [--debug]
                  [subcommand]

Jupyter: Interactive Computing

positional arguments:
  subcommand     the subcommand to launch

options:
  -h, --help     show this help message and exit
  --version      show the versions of core jupyter packages and exit
  --config-dir   show Jupyter config dir
  --data-dir     show Jupyter data dir
  --runtime-dir  show Jupyter runtime dir
  --paths        show all Jupyter paths. Add --json for machine-readable
                 format.
  --json         output paths as machine-readable json
  --debug        output debug information about paths

Available subcommands: kernel kernelspec migrate run troubleshoot trust

Jupyter command `jupyter-notebook` not found.
    > at $_.rejectStartPromise (/mnt/project_mnt/home_fs/<username>/.vscode-server/extensions/ms-toolsai.jupyter-2024.2.0-linux-x64/dist/extension.node.js:308:2207)
    > interpreter = [object Object]
11:52:55.260 [error] Error in execution (get message for cell) Error: Jupyter Server crashed. Unable to connect. 
Error code from Jupyter: 1
usage: jupyter.py [-h] [--version] [--config-dir] [--data-dir] [--runtime-dir]
                  [--paths] [--json] [--debug]
                  [subcommand]

Jupyter: Interactive Computing

positional arguments:
  subcommand     the subcommand to launch

options:
  -h, --help     show this help message and exit
  --version      show the versions of core jupyter packages and exit
  --config-dir   show Jupyter config dir
  --data-dir     show Jupyter data dir
  --runtime-dir  show Jupyter runtime dir
  --paths        show all Jupyter paths. Add --json for machine-readable
                 format.
  --json         output paths as machine-readable json
  --debug        output debug information about paths

Available subcommands: kernel kernelspec migrate run troubleshoot trust

Jupyter command `jupyter-notebook` not found.
    > at $_.rejectStartPromise (/mnt/project_mnt/home_fs/<username>/.vscode-server/extensions/ms-toolsai.jupyter-2024.2.0-linux-x64/dist/extension.node.js:308:2207)
    > interpreter = [object Object]
11:52:55.417 [info] End cell 2 execution after 0s, completed @ undefined, started @ undefined
