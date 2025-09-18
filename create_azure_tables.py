#!/usr/bin/env python3

"""
Azure SQL Table Creation Script

This script ensures all required tables are created in Azure SQL Database
for the BPO Claims ML System. It handles:
1. Raw applications table (from original data ingestion)
2. Medical claims table (detailed medical claims structure)
3. BPO claims processing table (BPO intermediary perspective)
4. BPO training dataset table (ML training data)

The script is idempotent - safe to run multiple times.
"""

from sqlalchemy import create_engine, text
import logging
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureTableCreator:
    def __init__(self):
        """Initialize Azure Table Creator"""
        self.connection_string = os.getenv("AZURE_SQL_CONNECTION_STRING")
        if not self.connection_string:
            raise ValueError("AZURE_SQL_CONNECTION_STRING environment variable not set")

        self.engine = create_engine(self.connection_string, echo=False)

    def create_raw_applications_table(self):
        """Create raw_applications table for basic insurance data"""
        logger.info("Creating raw_applications table...")

        sql = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='raw_applications' AND xtype='U')
        CREATE TABLE raw_applications (
            id INT IDENTITY(1,1) PRIMARY KEY,
            customer_id NVARCHAR(50),
            age INT,
            gender NVARCHAR(10),
            bmi FLOAT,
            children INT,
            smoker NVARCHAR(10),
            region NVARCHAR(20),
            charges FLOAT,
            medical_history NVARCHAR(200),
            employment_status NVARCHAR(50),
            annual_income INT,
            application_date DATETIME,
            created_date DATETIME DEFAULT GETDATE()
        );
        """

        try:
            with self.engine.begin() as conn:
                conn.execute(text(sql))
            logger.info("✅ raw_applications table created successfully")
        except Exception as e:
            logger.error(f"Failed to create raw_applications table: {e}")

    def create_medical_claims_table(self):
        """Create medical_claims table for detailed medical claims"""
        logger.info("Creating medical_claims table...")

        sql = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='medical_claims' AND xtype='U')
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
            created_date DATETIME DEFAULT GETDATE()
        );

        -- Create indexes for medical_claims table
        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_medical_claims_patient_id')
            CREATE INDEX idx_medical_claims_patient_id ON medical_claims (patient_id);

        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_medical_claims_provider_id')
            CREATE INDEX idx_medical_claims_provider_id ON medical_claims (provider_id);

        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_medical_claims_service_date')
            CREATE INDEX idx_medical_claims_service_date ON medical_claims (service_date);

        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_medical_claims_is_rejected')
            CREATE INDEX idx_medical_claims_is_rejected ON medical_claims (is_rejected);
        """

        try:
            with self.engine.begin() as conn:
                conn.execute(text(sql))
            logger.info("✅ medical_claims table and indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to create medical_claims table: {e}")

    def create_bpo_claims_processing_table(self):
        """Create bpo_claims_processing table for BPO intermediary data"""
        logger.info("Creating bpo_claims_processing table...")

        sql = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='bpo_claims_processing' AND xtype='U')
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
            created_timestamp DATETIME DEFAULT GETDATE()
        );

        -- Create indexes for bpo_claims_processing table
        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_bpo_client_provider')
            CREATE INDEX idx_bpo_client_provider ON bpo_claims_processing (client_provider_id);

        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_bpo_assigned_coder')
            CREATE INDEX idx_bpo_assigned_coder ON bpo_claims_processing (assigned_coder);

        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_bpo_submission_date')
            CREATE INDEX idx_bpo_submission_date ON bpo_claims_processing (submission_date);

        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_bpo_claim_successful')
            CREATE INDEX idx_bpo_claim_successful ON bpo_claims_processing (claim_successful);

        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_bpo_primary_payor')
            CREATE INDEX idx_bpo_primary_payor ON bpo_claims_processing (primary_payor);
        """

        try:
            with self.engine.begin() as conn:
                conn.execute(text(sql))
            logger.info("✅ bpo_claims_processing table and indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to create bpo_claims_processing table: {e}")

    def create_bpo_claims_training_table(self):
        """Create bpo_claims_training table for ML training data"""
        logger.info("Creating bpo_claims_training table...")

        sql = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='bpo_claims_training' AND xtype='U')
        CREATE TABLE bpo_claims_training (
            -- Identifiers
            claim_id VARCHAR(50) PRIMARY KEY,
            provider_id VARCHAR(50),
            bpo_coder VARCHAR(50),

            -- Temporal features
            service_date DATETIME,
            submission_date DATETIME,
            days_service_to_submission INT,
            submission_day_of_week INT,
            submission_hour INT,

            -- Patient demographics (anonymized)
            patient_age INT,
            patient_gender CHAR(1),
            patient_state VARCHAR(2),

            -- Financial features
            billed_amount DECIMAL(10,2),
            expected_reimbursement DECIMAL(10,2),
            amount_per_age DECIMAL(8,2),

            -- Clinical features
            primary_diagnosis_code VARCHAR(10),
            procedure_code VARCHAR(10),
            claim_type VARCHAR(20),
            provider_specialty VARCHAR(30),

            -- BPO processing quality
            coder_experience_years INT,
            coding_accuracy_score DECIMAL(5,3),
            documentation_enhancement_score DECIMAL(5,3),
            processing_time_hours INT,

            -- BPO services
            prior_authorization_obtained BIT,
            eligibility_verification_completed BIT,
            medical_necessity_documented BIT,
            qa_review_completed BIT,

            -- Provider data quality
            provider_data_completeness DECIMAL(5,3),
            missing_documentation_count INT,
            provider_response_time_hours INT,

            -- Payor and network
            primary_payor VARCHAR(30),
            network_status VARCHAR(20),
            payor_tier VARCHAR(10),

            -- Risk indicators
            high_dollar_claim BIT,
            complex_diagnosis BIT,
            weekend_submission BIT,
            after_hours_submission BIT,

            -- Historical features
            provider_historical_approval_rate DECIMAL(5,3),
            patient_previous_rejections INT,
            bpo_coder_approval_rate DECIMAL(5,3),

            -- Derived features
            bpo_quality_score DECIMAL(5,3),
            provider_risk_score DECIMAL(5,3),
            claim_complexity_score DECIMAL(5,3),

            -- GROUND TRUTH LABELS
            is_approved BIT NOT NULL,
            approval_probability_actual DECIMAL(6,4),
            rejection_reason VARCHAR(50),
            rejection_category VARCHAR(20),
            days_to_decision INT,
            appeal_filed BIT,
            final_payment_amount DECIMAL(10,2),

            -- BPO performance tracking
            bpo_prediction_confidence DECIMAL(5,3),
            manual_review_flagged BIT,
            qa_override_occurred BIT,

            created_timestamp DATETIME DEFAULT GETDATE()
        );

        -- Create indexes for ML training queries
        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_training_approval')
            CREATE INDEX idx_training_approval ON bpo_claims_training (is_approved);

        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_training_rejection_category')
            CREATE INDEX idx_training_rejection_category ON bpo_claims_training (rejection_category);

        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_training_payor')
            CREATE INDEX idx_training_payor ON bpo_claims_training (primary_payor);

        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_training_bpo_coder')
            CREATE INDEX idx_training_bpo_coder ON bpo_claims_training (bpo_coder);
        """

        try:
            with self.engine.begin() as conn:
                conn.execute(text(sql))
            logger.info("✅ bpo_claims_training table and indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to create bpo_claims_training table: {e}")

    def create_all_tables(self):
        """Create all required tables for the BPO Claims ML System"""
        logger.info("🚀 Starting Azure SQL table creation for BPO Claims ML System")
        logger.info("=" * 80)

        try:
            # Test connection first
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT @@VERSION"))
                version = result.fetchone()[0]
                logger.info(f"✅ Connected to Azure SQL: {version[:50]}...")

            # Create all tables
            self.create_raw_applications_table()
            self.create_medical_claims_table()
            self.create_bpo_claims_processing_table()
            self.create_bpo_claims_training_table()

            logger.info("=" * 80)
            logger.info("🎉 All tables created successfully!")
            logger.info("📊 Ready for BPO Claims ML System deployment")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to create tables: {e}")
            return False

    def verify_tables(self):
        """Verify all tables exist and show their structure"""
        logger.info("🔍 Verifying table creation...")

        tables_to_check = [
            'raw_applications',
            'medical_claims',
            'bpo_claims_processing',
            'bpo_claims_training'
        ]

        try:
            with self.engine.connect() as conn:
                for table_name in tables_to_check:
                    # Check if table exists
                    result = conn.execute(text(f"""
                        SELECT COUNT(*)
                        FROM INFORMATION_SCHEMA.TABLES
                        WHERE TABLE_NAME = '{table_name}'
                    """))
                    exists = result.fetchone()[0] > 0

                    if exists:
                        # Get column count
                        result = conn.execute(text(f"""
                            SELECT COUNT(*)
                            FROM INFORMATION_SCHEMA.COLUMNS
                            WHERE TABLE_NAME = '{table_name}'
                        """))
                        column_count = result.fetchone()[0]
                        logger.info(f"✅ {table_name}: {column_count} columns")
                    else:
                        logger.error(f"❌ {table_name}: NOT FOUND")

            logger.info("🎯 Table verification complete!")

        except Exception as e:
            logger.error(f"❌ Failed to verify tables: {e}")

def main():
    """Main execution function"""
    print("🏥 AZURE SQL TABLE CREATOR FOR BPO CLAIMS ML SYSTEM")
    print("=" * 80)

    try:
        # Create table creator
        creator = AzureTableCreator()

        # Create all tables
        success = creator.create_all_tables()

        if success:
            # Verify tables were created
            creator.verify_tables()

            print(f"\n🎉 SUCCESS! All tables are ready for:")
            print(f"   • Medical claims data ingestion")
            print(f"   • BPO claims processing workflows")
            print(f"   • ML model training and predictions")
            print(f"   • Production deployment")

        else:
            print(f"\n❌ FAILED! Please check your:")
            print(f"   • Azure SQL connection string")
            print(f"   • Database permissions")
            print(f"   • Network connectivity")

    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Make sure AZURE_SQL_CONNECTION_STRING is set in your .env file")

if __name__ == "__main__":
    main()