from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RTE_DIR = DATA_DIR / "rte"
SNCF_DIR = DATA_DIR / "sncf"

RTE_PROD_PARQUET = RTE_DIR / "RealisationDonneesProduction_2023_processed.parquet"
RTE_CONS_PARQUET = RTE_DIR / "conso_mix_RTE_2023_processed.parquet"

SNCF_EMISSIONS_1 = SNCF_DIR / "bilans-des-emissions-de-gaz-a-effet-de-serre-sncf.parquet"
SNCF_EMISSIONS_2 = SNCF_DIR / "emission-co2-perimetre-complet.parquet"
