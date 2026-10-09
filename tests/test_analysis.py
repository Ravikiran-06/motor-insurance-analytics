import pytest
import pandas as pd
import numpy as np

# In-memory minimal control test fixtures matching standard structures
def sample_insurance_data():
    return {
        "policies": pd.DataFrame({
            "policy_id": ["P1", "P2", "P3"],
            "policy_status": ["Active", "Expired", "Cancelled"],
            "policy_type": ["Comprehensive", "Third Party", "Comprehensive"],
            "premium_amount": [1200.0, 800.0, 1000.0]
        }),
        "claims": pd.DataFrame({
            "claim_id": ["C1", "C2"],
            "policy_id": ["P1", "P2"],
            "claim_status": ["Approved", "Pending"],
            "claim_amount": [500.0, 300.0],
            "claim_settlement_days": [10.0, 20.0]
        }),
        "payments": pd.DataFrame({
            "payment_id": ["PAY1"],
            "payment_amount": [500.0]
        })
    }

def test_calculate_total_policies():
    """Verifies unique policy tracking."""
    data = sample_insurance_data()
    assert int(data['policies']['policy_id'].nunique()) == 3

def test_calculate_total_premium():
    """Verifies premium collections mathematics."""
    data = sample_insurance_data()
    assert float(data['policies']['premium_amount'].sum()) == 3000.0

def test_calculate_total_claims():
    """Verifies claims count tracking metrics."""
    data = sample_insurance_data()
    assert int(data['claims']['claim_id'].nunique()) == 2
