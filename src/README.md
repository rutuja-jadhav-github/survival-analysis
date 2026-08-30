rn Survival App

## Documentation
Full write-up (model build, evaluation, deployment): <link>

## Running the app

1. Clone this repo and make sure [Docker Desktop](https://www.docker.com/products/docker-desktop) is installed and running.
2. From the `src` folder, run:

docker compose build --no-cache
docker compose up


3. Once you see `Application startup complete.` and `You can now view your Streamlit app`, open:

http://localhost:8501


**Note:** `--no-cache` forces a full rebuild from scratch — slower, but guarantees you're not running stale code or dependencies. Once the app is working and you haven't changed anything, `docker compose up` alone is enough and starts much faster.

To stop the app: `Ctrl+C`, or `docker compose down` from another terminal.