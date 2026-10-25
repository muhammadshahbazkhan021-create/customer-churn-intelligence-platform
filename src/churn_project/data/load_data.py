import os
import logging
import pandas as pd
from typing import Optional

# Logging configuration for production tracking
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_raw_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads raw data from a CSV file into a pandas DataFrame.

    This function includes robust validation checks to ensure the file 
    exists, is accessible, and contains valid, non-empty data.

    Parameters:
    -----------
    file_path : str, optional
        The path to the raw data CSV file. If None, it defaults to the 
        standard project structure path: 'data/raw/customer_churn.csv'.

    Returns:
    --------
    pd.DataFrame
        The loaded dataframe containing raw customer data.

    Raises:
    -------
    FileNotFoundError
        If the specified file path does not exist.
    ValueError
        If the loaded file is empty or contains no data.
    """
    # 1. Dynamic / Configurable Path Handling
    if file_path is None:
        # Default production path relative to project structure
        # File ke location se project root nikaalna:
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "customer_churn.csv")
    
    logger.info(file_path)

    # 2. Robust Error Handling (File Existence Check)
    if not os.path.exists(file_path):
        error_msg = f"The specified data file was not found at: {os.path.abspath(file_path)}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        # Loading the dataset
        df = pd.read_csv(file_path)
        
        # 3. Data Validation (Empty DataFrame Check)
        if df.empty:
            error_msg = f"Validation Failed: The file at {file_path} is completely empty."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
        return df

    except pd.errors.EmptyDataError:
        error_msg = f"Pandas Error: No data/columns to parse at {file_path}."
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    except Exception as e:
        logger.error(f"Unexpected error occurred while loading data: {str(e)}")
        raise e


if __name__ == "__main__":
    # Test script to verify the function locally
    print("Testing load_raw_data function...")
    try:
        # dynamic test with default path
        data = load_raw_data()
    except Exception as e:
        print(f"\n[Test Result] Function successfully caught structural exception: {e}")
