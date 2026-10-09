import sys
from src.data_loader import load_all_data
from src.validation import validate_all_tables
from src.data_cleaner import clean_data, save_cleaned_data
from src.logger import log_pipeline_step, log_error

def run_pipeline():
    """Main orchestrator that runs the entire ETL data cleaning process."""
    try:
        log_pipeline_step("--- MOTOR INSURANCE DATA CLEANING PIPELINE RUN STARTED ---")
        print("\n==================================================")
        print("Starting data engineering pipeline engine...")
        print("==================================================")
        
        # 1. Loading Module Ingestion
        raw_data = load_all_data(data_dir="data/raw")
        print("✔ Step 1: Raw data files read successfully.")
        
        # 2. Strict Relational and Structural Checks
        validate_all_tables(raw_data)
        print("✔ Step 2: Data structural schema validation passed.")
        
        # 3. Clean Engine Transformation & Metrics Processing
        processed_data = clean_data(raw_data)
        print("✔ Step 3: Data cleaning & derived column generation completed.")
        
        # 4. Save Finalized Artifacts
        save_cleaned_data(processed_data, output_dir="data/cleaned")
        print("✔ Step 4: Cleaned data exported to 'data/cleaned/'.")
        
        log_pipeline_step("--- MOTOR INSURANCE DATA CLEANING PIPELINE COMPLETED SUCCESSFULLY ---")
        print("==================================================")
        print("SUCCESS: Pipeline executed flawlessly!")
        print("==================================================\n")
        
    except Exception as e:
        error_msg = f"Pipeline execution failed: {str(e)}"
        print(f"\n[CRITICAL ERROR]: {error_msg}")
        log_error(error_msg)
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()
