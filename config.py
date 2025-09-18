# config.py
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Azure SQL Database
    AZURE_SQL_SERVER = os.getenv("AZURE_SQL_SERVER")
    AZURE_SQL_DATABASE = os.getenv("AZURE_SQL_DATABASE")
    AZURE_SQL_USERNAME = os.getenv("AZURE_SQL_USERNAME")
    AZURE_SQL_PASSWORD = os.getenv("AZURE_SQL_PASSWORD")

    # Connection string
    AZURE_SQL_CONNECTION_STRING = (
        f"mssql+pyodbc://{AZURE_SQL_USERNAME}:{AZURE_SQL_PASSWORD}@"
        f"{AZURE_SQL_SERVER}/{AZURE_SQL_DATABASE}?"
        f"driver=ODBC+Driver+17+for+SQL+Server"
    )

    # Azure ML Workspace
    AZURE_ML_SUBSCRIPTION_ID = os.getenv("AZURE_ML_SUBSCRIPTION_ID")
    AZURE_ML_RESOURCE_GROUP = os.getenv("AZURE_ML_RESOURCE_GROUP")
    AZURE_ML_WORKSPACE_NAME = os.getenv("AZURE_ML_WORKSPACE_NAME")

    # Model Configuration
    MODEL_NAME = "insurance_approval_model"
    MODEL_VERSION = "1.0.0"

    # Data Configuration
    TRAIN_TEST_SPLIT = 0.2
    RANDOM_STATE = 42

    # API Configuration
    API_HOST = "0.0.0.0"
    API_PORT = 5000

    # Feature Engineering
    FEATURE_COLUMNS = [
        "age",
        "gender_encoded",
        "bmi",
        "bmi_category_encoded",
        "children",
        "smoker_encoded",
        "region_encoded",
        "risk_score",
        "income_to_charges_ratio",
    ]

    TARGET_COLUMN = "approval_status"


# .env template file
# Create a .env file with these values:

# AZURE_SQL_SERVER=your-server.database.windows.net
# AZURE_SQL_DATABASE=your-database
# AZURE_SQL_USERNAME=your-username
# AZURE_SQL_PASSWORD=your-password
# AZURE_ML_SUBSCRIPTION_ID=your-subscription-id
# AZURE_ML_RESOURCE_GROUP=your-resource-group
# AZURE_ML_WORKSPACE_NAME=your-ml-workspace
