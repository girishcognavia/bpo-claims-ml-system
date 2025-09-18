#!/usr/bin/env python3

"""
RCM BPO Intermediary Claims Processing System

The BPO acts as an authorized intermediary that:
1. Receives raw clinical data from healthcare providers
2. Processes and prepares claims using coding expertise
3. Validates claims for accuracy and compliance
4. Submits claims to payors on behalf of providers
5. Manages follow-up and appeals processes

Key Business Model: BPO owns the claim preparation quality and pay rate outcomes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import random
from faker import Faker
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BPOIntermediaryClaimsGenerator:
    def __init__(self):
        self.fake = Faker()

        # BPO-specific data elements
        self.bpo_coders = [f"CODER_{i:03d}" for i in range(1, 51)]
        self.client_providers = [f"CLIENT_{i:04d}" for i in range(1, 201)]

        # Common issues in raw provider data
        self.raw_data_issues = [
            'incomplete_documentation',
            'missing_diagnosis_details',
            'unclear_procedure_notes',
            'missing_patient_demographics',
            'incomplete_insurance_info',
            'missing_referring_physician',
            'unclear_service_dates',
            'missing_medical_necessity'
        ]

        # BPO processing outcomes
        self.bpo_actions = [
            'enhanced_documentation',
            'added_supporting_codes',
            'clarified_medical_necessity',
            'obtained_missing_authorizations',
            'corrected_patient_demographics',
            'validated_insurance_eligibility',
            'optimized_coding_selection',
            'added_modifier_codes'
        ]

        # Quality metrics BPO tracks
        self.quality_flags = [
            'coding_accuracy_verified',
            'medical_necessity_documented',
            'authorization_obtained',
            'eligibility_verified',
            'documentation_complete',
            'compliance_checked',
            'billing_rules_applied'
        ]

    def generate_bpo_claims_data(self, num_records: int = 1000) -> pd.DataFrame:
        """Generate claims data from BPO intermediary perspective"""

        data = []

        for _ in range(num_records):
            # Basic claim identifiers
            claim_id = f"BPO_{uuid.uuid4().hex[:10].upper()}"
            internal_ref = f"INT_{uuid.uuid4().hex[:8].upper()}"

            # BPO client relationship
            client_provider_id = random.choice(self.client_providers)
            provider_specialty = random.choice([
                'Family Medicine', 'Internal Medicine', 'Cardiology', 'Orthopedics',
                'Emergency Medicine', 'Radiology', 'Pathology', 'Dermatology'
            ])

            # BPO processing team
            assigned_coder = random.choice(self.bpo_coders)
            coder_experience_years = random.randint(1, 15)
            coder_specialty_expertise = random.choice(['General', 'Surgical', 'Specialty'])

            # Dates - BPO processing timeline
            raw_data_received = self.fake.date_time_between(start_date='-60d', end_date='-30d')
            service_date = raw_data_received - timedelta(days=random.randint(1, 30))
            claim_prepared_date = raw_data_received + timedelta(days=random.randint(1, 5))
            submission_date = claim_prepared_date + timedelta(days=random.randint(0, 2))

            # Raw data quality from provider (major BPO challenge)
            raw_data_completeness = random.uniform(0.3, 0.95)
            missing_elements = random.randint(0, 8)
            requires_provider_clarification = random.choice([True, False])

            # BPO processing quality
            bpo_processing_time_hours = random.randint(2, 48)
            coding_accuracy_score = random.uniform(0.85, 0.99)
            documentation_enhancement_level = random.uniform(0.0, 0.8)

            # BPO value-add services
            prior_auth_obtained = random.choice([True, False]) if random.random() < 0.3 else True
            eligibility_verified = random.choice([True, False]) if random.random() < 0.1 else True
            medical_necessity_enhanced = random.choice([True, False])

            # Financial data
            raw_charges_from_provider = round(random.uniform(100, 15000), 2)
            bpo_coding_adjustments = round(random.uniform(-500, 1000), 2)  # Can be positive or negative
            final_billed_amount = max(50, raw_charges_from_provider + bpo_coding_adjustments)

            # BPO fee structure
            bpo_fee_percentage = random.uniform(0.05, 0.12)  # 5-12% of collected amount
            bpo_flat_fee = random.choice([0, 25, 50])  # Some claims have flat fees

            # Quality assurance metrics
            qa_review_required = random.choice([True, False]) if final_billed_amount > 5000 else False
            qa_reviewer = random.choice(self.bpo_coders) if qa_review_required else None
            compliance_score = random.uniform(0.9, 1.0)

            # Client satisfaction metrics
            client_feedback_score = random.uniform(3.5, 5.0)  # 1-5 scale
            turnaround_time_met = bpo_processing_time_hours <= 24

            # Payor information
            primary_payor = random.choice(['Anthem', 'Humana', 'Aetna', 'BCBS', 'Medicare', 'Medicaid', 'UnitedHealth'])
            payor_tier = random.choice(['Tier_1', 'Tier_2', 'Tier_3'])  # BPO's payor relationship strength

            # Patient demographics (scrubbed for BPO)
            patient_age_group = random.choice(['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
            patient_state = self.fake.state_abbr()

            # Clinical data (as processed by BPO)
            primary_diagnosis = random.choice(['Z00.00', 'Z12.11', 'M79.3', 'K21.9', 'E11.9', 'I10'])
            procedure_codes = random.choice(['99213', '99214', '99215', '99203', '99204'])
            modifier_codes_added = random.choice([None, '25', '59', 'GT', 'TC'])

            # BPO-specific risk assessment
            claim_complexity = random.choice(['Simple', 'Moderate', 'Complex', 'High_Risk'])
            audit_risk_score = random.uniform(0.1, 0.9)

            # Calculate BPO success probability based on their processing quality
            success_probability = 0.7  # Base success rate

            # BPO processing quality adjustments
            if coding_accuracy_score > 0.95:
                success_probability += 0.15
            if raw_data_completeness > 0.9:
                success_probability += 0.1
            if prior_auth_obtained:
                success_probability += 0.1
            if eligibility_verified:
                success_probability += 0.05
            if qa_review_required and qa_reviewer:
                success_probability += 0.05
            if documentation_enhancement_level > 0.5:
                success_probability += 0.05

            # Risk factors that reduce success
            if missing_elements > 3:
                success_probability -= 0.2
            if requires_provider_clarification:
                success_probability -= 0.15
            if payor_tier == 'Tier_3':
                success_probability -= 0.1

            success_probability = max(0.1, min(0.98, success_probability))

            # Determine actual outcome
            is_successful = random.random() < success_probability

            # BPO performance metrics
            if is_successful:
                actual_reimbursement = final_billed_amount * random.uniform(0.7, 0.95)
                bpo_earned_fee = actual_reimbursement * bpo_fee_percentage + bpo_flat_fee
                days_to_payment = random.randint(14, 45)
            else:
                actual_reimbursement = 0
                bpo_earned_fee = bpo_flat_fee  # Only flat fee, no percentage
                days_to_payment = None

            # Appeal/rework tracking
            requires_appeal = not is_successful and random.random() < 0.6
            appeal_success_probability = success_probability * 0.7 if requires_appeal else 0

            claim_record = {
                # Claim identifiers
                'claim_id': claim_id,
                'bpo_internal_ref': internal_ref,
                'client_provider_id': client_provider_id,

                # Provider information
                'provider_specialty': provider_specialty,
                'client_relationship_duration_months': random.randint(3, 60),
                'provider_data_quality_score': round(raw_data_completeness, 3),

                # BPO team assignment
                'assigned_coder': assigned_coder,
                'coder_experience_years': coder_experience_years,
                'coder_specialty': coder_specialty_expertise,
                'qa_reviewer': qa_reviewer,

                # Timeline
                'service_date': service_date,
                'raw_data_received_date': raw_data_received,
                'claim_prepared_date': claim_prepared_date,
                'submission_date': submission_date,
                'bpo_processing_hours': bpo_processing_time_hours,
                'turnaround_sla_met': turnaround_time_met,

                # Data quality metrics
                'raw_data_completeness_score': round(raw_data_completeness, 3),
                'missing_elements_count': missing_elements,
                'required_provider_clarification': requires_provider_clarification,
                'documentation_enhancement_score': round(documentation_enhancement_level, 3),

                # BPO value-added services
                'prior_authorization_obtained': prior_auth_obtained,
                'eligibility_verification_completed': eligibility_verified,
                'medical_necessity_enhanced': medical_necessity_enhanced,
                'coding_accuracy_score': round(coding_accuracy_score, 3),
                'compliance_score': round(compliance_score, 3),

                # Financial data
                'raw_provider_charges': raw_charges_from_provider,
                'bpo_coding_adjustments': bpo_coding_adjustments,
                'final_billed_amount': final_billed_amount,
                'bpo_fee_percentage': round(bpo_fee_percentage, 4),
                'bpo_flat_fee': bpo_flat_fee,

                # Payor information
                'primary_payor': primary_payor,
                'payor_relationship_tier': payor_tier,
                'payor_specific_rules_applied': random.choice([True, False]),

                # Patient demographics (anonymized)
                'patient_age_group': patient_age_group,
                'patient_state': patient_state,
                'insurance_type': random.choice(['Commercial', 'Medicare', 'Medicaid', 'Other']),

                # Clinical data (BPO processed)
                'primary_diagnosis_code': primary_diagnosis,
                'procedure_code': procedure_codes,
                'modifier_codes': modifier_codes_added,
                'claim_complexity': claim_complexity,

                # Risk and quality assessment
                'audit_risk_score': round(audit_risk_score, 3),
                'qa_review_required': qa_review_required,
                'client_satisfaction_score': round(client_feedback_score, 2),

                # Outcomes and performance
                'bpo_success_probability': round(success_probability, 3),
                'claim_successful': is_successful,
                'actual_reimbursement': round(actual_reimbursement, 2),
                'bpo_fee_earned': round(bpo_earned_fee, 2),
                'days_to_payment': days_to_payment,
                'net_collection_rate': round(actual_reimbursement / final_billed_amount, 3) if final_billed_amount > 0 else 0,

                # Follow-up actions
                'requires_appeal': requires_appeal,
                'appeal_success_probability': round(appeal_success_probability, 3),
                'provider_feedback_required': not is_successful and raw_data_completeness < 0.7,

                # BPO business metrics
                'bpo_roi': round((bpo_earned_fee / (bpo_processing_time_hours * 25)) * 100, 2) if bpo_processing_time_hours > 0 else 0,  # Assuming $25/hour cost
                'created_timestamp': datetime.now()
            }

            data.append(claim_record)

        return pd.DataFrame(data)

def create_bpo_database_schema():
    """SQL schema for BPO intermediary claims processing"""

    return """
    CREATE TABLE bpo_claims_processing (
        -- Primary identifiers
        claim_id VARCHAR(50) PRIMARY KEY,
        bpo_internal_ref VARCHAR(50) UNIQUE NOT NULL,
        client_provider_id VARCHAR(50) NOT NULL,

        -- Provider relationship
        provider_specialty VARCHAR(50),
        client_relationship_duration_months INT,
        provider_data_quality_score DECIMAL(4,3),

        -- BPO team assignment
        assigned_coder VARCHAR(50),
        coder_experience_years INT,
        coder_specialty VARCHAR(20),
        qa_reviewer VARCHAR(50),

        -- Processing timeline
        service_date DATETIME NOT NULL,
        raw_data_received_date DATETIME NOT NULL,
        claim_prepared_date DATETIME NOT NULL,
        submission_date DATETIME NOT NULL,
        bpo_processing_hours INT,
        turnaround_sla_met BIT,

        -- Data quality metrics
        raw_data_completeness_score DECIMAL(4,3),
        missing_elements_count INT,
        required_provider_clarification BIT,
        documentation_enhancement_score DECIMAL(4,3),

        -- BPO value-added services
        prior_authorization_obtained BIT,
        eligibility_verification_completed BIT,
        medical_necessity_enhanced BIT,
        coding_accuracy_score DECIMAL(4,3),
        compliance_score DECIMAL(4,3),

        -- Financial tracking
        raw_provider_charges DECIMAL(10,2),
        bpo_coding_adjustments DECIMAL(10,2),
        final_billed_amount DECIMAL(10,2),
        bpo_fee_percentage DECIMAL(6,4),
        bpo_flat_fee DECIMAL(8,2),

        -- Payor information
        primary_payor VARCHAR(50),
        payor_relationship_tier VARCHAR(10),
        payor_specific_rules_applied BIT,

        -- Patient demographics (anonymized)
        patient_age_group VARCHAR(10),
        patient_state VARCHAR(2),
        insurance_type VARCHAR(20),

        -- Clinical data
        primary_diagnosis_code VARCHAR(10),
        procedure_code VARCHAR(10),
        modifier_codes VARCHAR(20),
        claim_complexity VARCHAR(20),

        -- Risk and QA
        audit_risk_score DECIMAL(4,3),
        qa_review_required BIT,
        client_satisfaction_score DECIMAL(3,2),

        -- Outcomes
        bpo_success_probability DECIMAL(4,3),
        claim_successful BIT NOT NULL,
        actual_reimbursement DECIMAL(10,2),
        bpo_fee_earned DECIMAL(10,2),
        days_to_payment INT,
        net_collection_rate DECIMAL(4,3),

        -- Follow-up
        requires_appeal BIT,
        appeal_success_probability DECIMAL(4,3),
        provider_feedback_required BIT,

        -- Business metrics
        bpo_roi DECIMAL(8,2),
        created_timestamp DATETIME DEFAULT GETDATE(),

        -- Indexes
        INDEX idx_client_provider (client_provider_id),
        INDEX idx_assigned_coder (assigned_coder),
        INDEX idx_submission_date (submission_date),
        INDEX idx_claim_successful (claim_successful),
        INDEX idx_payor (primary_payor),
        INDEX idx_bpo_roi (bpo_roi)
    );
    """

def analyze_bpo_performance(df: pd.DataFrame) -> Dict:
    """Analyze BPO performance metrics"""

    total_claims = len(df)
    successful_claims = df['claim_successful'].sum()
    success_rate = successful_claims / total_claims if total_claims > 0 else 0

    # Financial metrics
    total_billed = df['final_billed_amount'].sum()
    total_collected = df['actual_reimbursement'].sum()
    total_bpo_fees = df['bpo_fee_earned'].sum()
    net_collection_rate = total_collected / total_billed if total_billed > 0 else 0

    # Operational metrics
    avg_processing_time = df['bpo_processing_hours'].mean()
    sla_compliance = df['turnaround_sla_met'].mean()
    qa_required_rate = df['qa_review_required'].mean()

    # Quality metrics
    avg_coding_accuracy = df['coding_accuracy_score'].mean()
    avg_data_completeness = df['raw_data_completeness_score'].mean()
    avg_client_satisfaction = df['client_satisfaction_score'].mean()

    # Top performing coders
    coder_performance = df.groupby('assigned_coder').agg({
        'claim_successful': 'mean',
        'bpo_fee_earned': 'sum',
        'coding_accuracy_score': 'mean'
    }).sort_values('claim_successful', ascending=False)

    return {
        'overall_metrics': {
            'total_claims_processed': total_claims,
            'success_rate': round(success_rate, 3),
            'net_collection_rate': round(net_collection_rate, 3),
            'total_revenue_collected': round(total_collected, 2),
            'total_bpo_fees_earned': round(total_bpo_fees, 2),
            'avg_processing_hours': round(avg_processing_time, 2),
            'sla_compliance_rate': round(sla_compliance, 3)
        },
        'quality_metrics': {
            'avg_coding_accuracy': round(avg_coding_accuracy, 3),
            'avg_raw_data_completeness': round(avg_data_completeness, 3),
            'avg_client_satisfaction': round(avg_client_satisfaction, 2),
            'qa_review_rate': round(qa_required_rate, 3)
        },
        'top_coders': coder_performance.head().to_dict('index')
    }

if __name__ == "__main__":
    print("🏥 BPO INTERMEDIARY CLAIMS PROCESSING SYSTEM")
    print("=" * 80)

    # Generate sample data
    generator = BPOIntermediaryClaimsGenerator()
    claims_df = generator.generate_bpo_claims_data(5000)

    print(f"Generated {len(claims_df)} BPO-processed claims")
    print(f"Success Rate: {claims_df['claim_successful'].mean():.1%}")

    # Analyze performance
    performance = analyze_bpo_performance(claims_df)

    print(f"\n📊 BPO PERFORMANCE METRICS:")
    print(f"Success Rate: {performance['overall_metrics']['success_rate']:.1%}")
    print(f"Collection Rate: {performance['overall_metrics']['net_collection_rate']:.1%}")
    print(f"Total BPO Revenue: ${performance['overall_metrics']['total_bpo_fees_earned']:,.2f}")
    print(f"Average Processing Time: {performance['overall_metrics']['avg_processing_hours']:.1f} hours")
    print(f"SLA Compliance: {performance['overall_metrics']['sla_compliance_rate']:.1%}")

    print(f"\n⭐ QUALITY METRICS:")
    print(f"Coding Accuracy: {performance['quality_metrics']['avg_coding_accuracy']:.1%}")
    print(f"Client Satisfaction: {performance['quality_metrics']['avg_client_satisfaction']:.2f}/5.0")

    # Save data
    claims_df.to_csv('bpo_intermediary_claims.csv', index=False)
    print(f"\n✅ Data saved to bpo_intermediary_claims.csv")

    print(f"\n🎯 KEY BPO SUCCESS FACTORS:")
    print(f"1. Raw data quality from providers")
    print(f"2. Coder experience and accuracy")
    print(f"3. Turnaround time compliance")
    print(f"4. Payor relationship management")
    print(f"5. Quality assurance processes")