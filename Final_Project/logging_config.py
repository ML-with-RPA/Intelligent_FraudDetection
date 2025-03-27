import logging
import os

# Cria a pasta de logs, se não existir
os.makedirs("results/logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("results/logs/experiment.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("Configuração de logging iniciada.")


