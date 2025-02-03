uv venv

source .venv/bin/activate

uv pip install -e ".[devel,google]" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-main/constraints-source-providers-3.9.txt"

uv sync --extra devel --extra devel-tests --extra google


  airflow webserver --port 8080
