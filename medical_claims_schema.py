#!/usr/bin/env python3

"""
Medical Claims Data Structure for Claim Rejection Prediction Model

This schema includes key features that influence claim approval/rejection decisions:
- Patient demographics and history
- Provider characteristics
- Clinical details and diagnosis codes
- Billing and administrative factors
- Risk indicators for fraud detection
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker
import uuid

class MedicalClaimsGenerator:
    def __init__(self):
        self.fake = Faker()

        # Common ICD-10 diagnosis codes
        self.icd10_codes = [
            'Z00.00', 'Z12.11', 'M79.3', 'K21.9', 'E11.9', 'I10', 'J44.1',
            'M25.511', 'F41.1', 'K59.00', 'Z51.11', 'M54.5', 'E78.5',
            'N39.0', 'H52.4', 'J06.9', 'K76.0', 'M19.90', 'F32.9', 'G43.909'
        ]

        # Common CPT procedure codes
        self.cpt_codes = [
            '99213', '99214', '99215', '99203', '99204', '99205', '99212',
            '99232', '99233', '99231', '36415', '80053', '85025', '80048',
            '93000', '71020', '73721', '76700', '99281', '99282', '99283'
        ]

        # Rejection reasons
        self.rejection_reasons = [
            'Medical necessity not established',
            'Pre-authorization required',
            'Out of network provider',
            'Duplicate claim',
            'Incorrect coding',
            'Missing documentation',
            'Policy exclusion',
            'Benefit exhausted',
            'Experimental treatment',
            'Fraudulent activity suspected'
        ]

    def generate_claims_data(self, num_records: int = 1000) -> pd.DataFrame:
        """Generate synthetic medical claims data"""

        data = []

        for _ in range(num_records):
            # Basic claim information
            claim_id = f"CLM_{uuid.uuid4().hex[:10].upper()}"
            submission_date = self.fake.date_time_between(start_date='-2y', end_date='now')

            # Patient demographics
            patient_id = f"PAT_{uuid.uuid4().hex[:8].upper()}"
            patient_age = random.randint(18, 85)
            patient_gender = random.choice(['M', 'F'])
            patient_state = self.fake.state_abbr()

            # Policy information
            policy_type = random.choice(['HMO', 'PPO', 'EPO', 'POS', 'HDHP'])
            deductible = random.choice([500, 1000, 1500, 2000, 3000, 5000])
            copay = random.choice([10, 15, 20, 25, 30, 40, 50])

            # Provider information
            provider_id = f"PRV_{uuid.uuid4().hex[:8].upper()}"
            provider_specialty = random.choice([
                'Family Medicine', 'Internal Medicine', 'Cardiology', 'Orthopedics',
                'Emergency Medicine', 'Radiology', 'Pathology', 'Dermatology',
                'Psychiatry', 'Oncology', 'Neurology', 'Gastroenterology'
            ])
            provider_network_status = random.choice(['In-Network', 'Out-of-Network'])
            provider_years_experience = random.randint(1, 40)

            # Clinical information
            primary_diagnosis = random.choice(self.icd10_codes)
            procedure_code = random.choice(self.cpt_codes)
            claim_type = random.choice(['Inpatient', 'Outpatient', 'Emergency', 'Pharmacy', 'Lab'])

            # Financial details
            billed_amount = round(random.uniform(50, 25000), 2)

            # Risk factors and patterns
            is_weekend_submission = submission_date.weekday() >= 5
            days_between_service_and_submission = random.randint(0, 90)
            service_date = submission_date - timedelta(days=days_between_service_and_submission)

            # Provider behavior patterns
            provider_claims_this_month = random.randint(1, 200)
            avg_claim_amount_provider = round(random.uniform(100, 5000), 2)

            # Patient history
            previous_claims_count = random.randint(0, 50)
            previous_rejections_count = random.randint(0, min(5, previous_claims_count))

            # Documentation quality indicators
            documentation_score = random.uniform(0.3, 1.0)  # 0-1 scale
            supporting_docs_count = random.randint(0, 8)

            # Geographic and demographic risk factors
            high_fraud_zip = random.choice([True, False])  # Based on historical fraud data

            # Timing patterns
            submission_hour = submission_date.hour
            is_after_hours = submission_hour < 8 or submission_hour > 18

            # Calculate risk score (for synthetic data generation)
            risk_score = 0

            # Add risk based on various factors
            if provider_network_status == 'Out-of-Network':
                risk_score += 0.3
            if days_between_service_and_submission > 30:
                risk_score += 0.2
            if billed_amount > 10000:
                risk_score += 0.2
            if previous_rejections_count > 2:
                risk_score += 0.2
            if documentation_score < 0.6:
                risk_score += 0.3
            if high_fraud_zip:
                risk_score += 0.1
            if is_after_hours:
                risk_score += 0.1
            if provider_claims_this_month > 150:
                risk_score += 0.1

            # Determine claim status based on risk score
            rejection_probability = min(risk_score, 0.9)
            is_rejected = random.random() < rejection_probability

            rejection_reason = None
            if is_rejected:
                rejection_reason = random.choice(self.rejection_reasons)

            # Create claim record
            claim_record = {
                # Claim identifiers
                'claim_id': claim_id,
                'patient_id': patient_id,
                'provider_id': provider_id,

                # Dates
                'service_date': service_date,
                'submission_date': submission_date,
                'days_to_submission': days_between_service_and_submission,

                # Patient demographics
                'patient_age': patient_age,
                'patient_gender': patient_gender,
                'patient_state': patient_state,

                # Policy details
                'policy_type': policy_type,
                'deductible_amount': deductible,
                'copay_amount': copay,

                # Provider information
                'provider_specialty': provider_specialty,
                'provider_network_status': provider_network_status,
                'provider_years_experience': provider_years_experience,
                'provider_monthly_claims': provider_claims_this_month,
                'provider_avg_claim_amount': avg_claim_amount_provider,

                # Clinical details
                'primary_diagnosis_code': primary_diagnosis,
                'procedure_code': procedure_code,
                'claim_type': claim_type,

                # Financial
                'billed_amount': billed_amount,
                'amount_per_age': round(billed_amount / patient_age, 2),

                # Patient history
                'patient_previous_claims': previous_claims_count,
                'patient_previous_rejections': previous_rejections_count,
                'patient_rejection_rate': round(previous_rejections_count / max(1, previous_claims_count), 3),

                # Documentation and quality
                'documentation_quality_score': round(documentation_score, 3),
                'supporting_documents_count': supporting_docs_count,

                # Risk indicators
                'is_weekend_submission': is_weekend_submission,
                'is_after_hours_submission': is_after_hours,
                'submission_hour': submission_hour,
                'high_fraud_area': high_fraud_zip,

                # Derived features
                'billed_amount_zscore': 0,  # Will calculate after generating all data
                'provider_specialty_risk': 0,  # Will calculate after generating all data

                # Target variable
                'is_rejected': is_rejected,
                'rejection_reason': rejection_reason,
                'risk_score': round(risk_score, 3)
            }

            data.append(claim_record)

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Calculate z-scores and specialty risk after all data is generated
        df['billed_amount_zscore'] = (df['billed_amount'] - df['billed_amount'].mean()) / df['billed_amount'].std()

        # Calculate specialty risk based on rejection rates
        specialty_rejection_rates = df.groupby('provider_specialty')['is_rejected'].mean()
        df['provider_specialty_risk'] = df['provider_specialty'].map(specialty_rejection_rates)

        return df

def create_database_schema():
    """SQL schema for creating the claims table"""

    schema = """
    CREATE TABLE medical_claims (
        claim_id VARCHAR(50) PRIMARY KEY,
        patient_id VARCHAR(50) NOT NULL,
        provider_id VARCHAR(50) NOT NULL,

        -- Dates
        service_date DATETIME NOT NULL,
        submission_date DATETIME NOT NULL,
        days_to_submission INT,

        -- Patient demographics
        patient_age INT,
        patient_gender CHAR(1),
        patient_state VARCHAR(2),

        -- Policy details
        policy_type VARCHAR(10),
        deductible_amount DECIMAL(10,2),
        copay_amount DECIMAL(10,2),

        -- Provider information
        provider_specialty VARCHAR(50),
        provider_network_status VARCHAR(20),
        provider_years_experience INT,
        provider_monthly_claims INT,
        provider_avg_claim_amount DECIMAL(10,2),

        -- Clinical details
        primary_diagnosis_code VARCHAR(10),
        procedure_code VARCHAR(10),
        claim_type VARCHAR(20),

        -- Financial
        billed_amount DECIMAL(10,2),
        amount_per_age DECIMAL(10,2),

        -- Patient history
        patient_previous_claims INT,
        patient_previous_rejections INT,
        patient_rejection_rate DECIMAL(5,3),

        -- Documentation and quality
        documentation_quality_score DECIMAL(4,3),
        supporting_documents_count INT,

        -- Risk indicators
        is_weekend_submission BIT,
        is_after_hours_submission BIT,
        submission_hour INT,
        high_fraud_area BIT,

        -- Derived features
        billed_amount_zscore DECIMAL(10,6),
        provider_specialty_risk DECIMAL(5,3),

        -- Target variable
        is_rejected BIT NOT NULL,
        rejection_reason VARCHAR(100),
        risk_score DECIMAL(4,3),

        -- Audit fields
        created_date DATETIME DEFAULT GETDATE(),

        INDEX idx_patient_id (patient_id),
        INDEX idx_provider_id (provider_id),
        INDEX idx_service_date (service_date),
        INDEX idx_is_rejected (is_rejected)
    );
    """

    return schema

if __name__ == "__main__":
    # Generate sample data
    generator = MedicalClaimsGenerator()
    claims_df = generator.generate_claims_data(5000)

    print("Medical Claims Dataset Generated!")
    print(f"Total records: {len(claims_df)}")
    print(f"Rejection rate: {claims_df['is_rejected'].mean():.2%}")
    print(f"\nColumns: {list(claims_df.columns)}")
    print(f"\nSample data:")
    print(claims_df.head())

    # Save to CSV
    claims_df.to_csv('medical_claims_data.csv', index=False)
    print(f"\nData saved to medical_claims_data.csv")

    # Print schema
    print(f"\nDatabase Schema:")
    print(create_database_schema())