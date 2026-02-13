from api import run_api
from config.settings import settings

if __name__ == "__main__":
    run_api(settings.host, settings.port)
