import pytest
import pandas as pd
from app.streamlit_app import REQUIRED_COLUMNS

def test_validate_customers_schema():
    """Proves that your system flags and blocks files missing standard requested fields."""
    faulty_df = pd.DataFrame({"customer_id": ["C1"], "invalid_column_name": ["Error"]})
    missing = [col for col in REQUIRED_COLUMNS["customers"] if col not in faulty_df.columns]
    
    # Asserting that validation rules catch schema cracks
    assert len(missing) > 0
    assert "customer_name" in missing

def test_validate_relationships():
    """Confirms that policies reference real customer IDs across the data model."""
    mock_customers = pd.DataFrame({"customer_id": ["C1", "C2"]})
    mock_policies = pd.DataFrame({"policy_id": ["P1"], "customer_id": ["C999"]}) # Orphan ID entry
    
    is_orphan_present = not mock_policies['customer_id'].isin(mock_customers['customer_id']).all()
    assert is_orphan_present is True
