import subprocess


def start_api():

    subprocess.run([
        "uvicorn",
        "src.api.api:app",
        "--reload"
    ])


if __name__ == "__main__":

    start_api()