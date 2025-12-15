"""
Generate Synthetic Customer Churn Dataset

This script generates a realistic customer dataset for churn prediction,
designed for interview preparation and ML pipeline demonstration.

Features generated:
- Demographics (age, gender, location)
- Subscription info (plan type, tenure, payment method)
- Usage patterns (logins, sessions, content consumption)
- Engagement metrics (support interactions, ratings)
- Transaction history (payment failures, plan changes)

Dataset characteristics:
- 10,000 customers
- ~15% churn rate (imbalanced)
- Realistic feature distributions
- Temporal patterns (recent behavior matters)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import random
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


class ChurnDataGenerator:
    """Generate synthetic customer churn data"""

    def __init__(self, n_customers=10000, churn_rate=0.15):
        """
        Initialize data generator

        Args:
            n_customers: Number of customers to generate
            churn_rate: Target churn rate (0.15 = 15% churn)
        """
        self.n_customers = n_customers
        self.churn_rate = churn_rate
        self.today = datetime.now()

    def generate_demographics(self):
        """Generate demographic features"""
        print("Generating demographics...")

        # Age: Normal distribution, 18-75 years
        age = np.random.normal(35, 12, self.n_customers).astype(int)
        age = np.clip(age, 18, 75)

        # Gender: ~50/50 split
        gender = np.random.choice(['M', 'F', 'Other'], self.n_customers, p=[0.48, 0.48, 0.04])

        # Location: Major cities
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                  'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
        location = np.random.choice(cities, self.n_customers)

        return pd.DataFrame({
            'age': age,
            'gender': gender,
            'location': location
        })

    def generate_subscription_info(self):
        """Generate subscription-related features"""
        print("Generating subscription info...")

        # Signup date: Last 2 years
        signup_days_ago = np.random.exponential(180, self.n_customers).astype(int)
        signup_days_ago = np.clip(signup_days_ago, 30, 730)  # 30 days to 2 years
        signup_date = [self.today - timedelta(days=int(d)) for d in signup_days_ago]

        # Tenure in days
        tenure_days = signup_days_ago

        # Plan type: Basic, Standard, Premium
        # Older customers more likely to have Premium
        plan_type = []
        for i in range(self.n_customers):
            if tenure_days[i] > 365:
                probs = [0.2, 0.3, 0.5]  # Premium more likely for older customers
            else:
                probs = [0.5, 0.3, 0.2]  # Basic more likely for new customers
            plan_type.append(np.random.choice(['Basic', 'Standard', 'Premium'], p=probs))

        # Monthly price based on plan
        price_map = {'Basic': 9.99, 'Standard': 14.99, 'Premium': 19.99}
        monthly_price = [price_map[p] for p in plan_type]

        # Payment method
        payment_method = np.random.choice(
            ['Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer'],
            self.n_customers,
            p=[0.50, 0.25, 0.15, 0.10]
        )

        return pd.DataFrame({
            'signup_date': signup_date,
            'tenure_days': tenure_days,
            'plan_type': plan_type,
            'monthly_price': monthly_price,
            'payment_method': payment_method
        })

    def generate_usage_patterns(self, tenure_days, will_churn):
        """
        Generate usage patterns (key predictor of churn!)

        Args:
            tenure_days: Customer tenure in days
            will_churn: Boolean array indicating churn
        """
        print("Generating usage patterns...")

        n = len(tenure_days)

        # Base login frequency depends on tenure (engagement decreases over time)
        base_logins = 15 * np.exp(-tenure_days / 365)  # Exponential decay
        base_logins = np.clip(base_logins, 2, 30)

        # Churners have significantly lower recent activity
        churn_multiplier = np.where(will_churn, 0.3, 1.0)  # 70% drop for churners

        # Last 7 days logins
        logins_last_7d = np.random.poisson(base_logins * churn_multiplier / 4)
        logins_last_7d = np.clip(logins_last_7d, 0, 15)

        # Last 30 days logins
        logins_last_30d = np.random.poisson(base_logins * churn_multiplier)
        logins_last_30d = np.clip(logins_last_30d, logins_last_7d, 50)

        # Last 90 days logins
        logins_last_90d = np.random.poisson(base_logins * churn_multiplier * 3)
        logins_last_90d = np.clip(logins_last_90d, logins_last_30d, 150)

        # Session duration (minutes)
        base_session = 25
        session_multiplier = np.where(will_churn, 0.5, 1.0)
        avg_session_duration = np.random.gamma(2, base_session * session_multiplier / 2)
        avg_session_duration = np.clip(avg_session_duration, 5, 120)

        # Content consumed (hours)
        content_multiplier = np.where(will_churn, 0.4, 1.0)
        total_watch_time_30d = np.random.gamma(3, 10 * content_multiplier)
        total_watch_time_30d = np.clip(total_watch_time_30d, 0, 100)

        # Unique content items accessed
        unique_content_30d = np.random.poisson(20 * content_multiplier)
        unique_content_30d = np.clip(unique_content_30d, 1, 100)

        # Days since last login (churners have higher values)
        days_since_last_login = np.where(
            will_churn,
            np.random.exponential(10, n),  # Churners: mean 10 days
            np.random.exponential(2, n)    # Active: mean 2 days
        )
        days_since_last_login = np.clip(days_since_last_login, 0, 60).astype(int)

        return pd.DataFrame({
            'logins_last_7d': logins_last_7d,
            'logins_last_30d': logins_last_30d,
            'logins_last_90d': logins_last_90d,
            'avg_session_duration': avg_session_duration.round(1),
            'total_watch_time_30d': total_watch_time_30d.round(1),
            'unique_content_30d': unique_content_30d,
            'days_since_last_login': days_since_last_login
        })

    def generate_engagement_features(self, will_churn):
        """Generate engagement-related features"""
        print("Generating engagement features...")

        n = len(will_churn)

        # Support interactions (churners contact support more)
        support_rate = np.where(will_churn, 2.0, 0.5)
        support_tickets_30d = np.random.poisson(support_rate)

        # Customer satisfaction rating (1-5 scale)
        satisfaction_mean = np.where(will_churn, 2.5, 4.0)
        satisfaction_score = np.random.normal(satisfaction_mean, 0.8, n)
        satisfaction_score = np.clip(satisfaction_score, 1, 5).round(1)

        # App rating (1-5 scale)
        app_rating_mean = np.where(will_churn, 2.8, 4.2)
        app_rating = np.random.normal(app_rating_mean, 0.7, n)
        app_rating = np.clip(app_rating, 1, 5).round(1)

        # Social connections (friends on platform)
        connection_mean = np.where(will_churn, 3, 12)
        social_connections = np.random.poisson(connection_mean)
        social_connections = np.clip(social_connections, 0, 50)

        # Feature adoption (% of features used)
        adoption_mean = np.where(will_churn, 0.3, 0.7)
        feature_adoption_rate = np.random.beta(2, 2, n) * adoption_mean + 0.1
        feature_adoption_rate = np.clip(feature_adoption_rate, 0, 1).round(2)

        return pd.DataFrame({
            'support_tickets_30d': support_tickets_30d,
            'satisfaction_score': satisfaction_score,
            'app_rating': app_rating,
            'social_connections': social_connections,
            'feature_adoption_rate': feature_adoption_rate
        })

    def generate_transaction_features(self, will_churn):
        """Generate transaction-related features"""
        print("Generating transaction features...")

        n = len(will_churn)

        # Payment failures (strong churn indicator!)
        failure_rate = np.where(will_churn, 0.8, 0.1)
        payment_failures_6m = np.random.poisson(failure_rate)
        payment_failures_6m = np.clip(payment_failures_6m, 0, 5)

        # Plan changes (downgrades indicate churn)
        plan_changes_12m = np.where(
            will_churn,
            np.random.choice([0, 1, 2], n, p=[0.3, 0.5, 0.2]),  # Churners change more
            np.random.choice([0, 1, 2], n, p=[0.8, 0.15, 0.05])  # Active rarely change
        )

        # Billing disputes
        dispute_rate = np.where(will_churn, 0.3, 0.05)
        billing_disputes = np.random.binomial(1, dispute_rate)

        # Auto-renew enabled (churners disable this)
        auto_renew_prob = np.where(will_churn, 0.3, 0.85)
        auto_renew_enabled = np.random.binomial(1, auto_renew_prob)

        # Days until next billing
        days_until_renewal = np.random.randint(1, 31, n)

        return pd.DataFrame({
            'payment_failures_6m': payment_failures_6m,
            'plan_changes_12m': plan_changes_12m,
            'billing_disputes': billing_disputes,
            'auto_renew_enabled': auto_renew_enabled,
            'days_until_renewal': days_until_renewal
        })

    def generate_derived_features(self, df):
        """Generate derived/engineered features"""
        print("Generating derived features...")

        # Login trend (recent vs historical)
        df['login_frequency_trend'] = (
            df['logins_last_7d'] / (df['logins_last_30d'] + 1)
        ).round(3)

        # Engagement score (composite metric)
        df['engagement_score'] = (
            0.3 * (df['logins_last_30d'] / 30) +
            0.3 * (df['total_watch_time_30d'] / 50) +
            0.2 * (df['unique_content_30d'] / 50) +
            0.2 * df['feature_adoption_rate']
        ).round(3)
        df['engagement_score'] = df['engagement_score'].clip(0, 1)

        # Risk flags
        df['high_support_flag'] = (df['support_tickets_30d'] >= 3).astype(int)
        df['payment_issue_flag'] = (df['payment_failures_6m'] > 0).astype(int)
        df['low_engagement_flag'] = (df['logins_last_30d'] < 5).astype(int)

        return df

    def generate_churn_labels(self, df):
        """
        Generate churn labels with realistic logic

        Churn probability depends on:
        - Low usage (most important)
        - Payment failures
        - Low satisfaction
        - Low tenure (new customers churn more)
        """
        print("Generating churn labels...")

        # Calculate churn probability based on features
        churn_prob = np.zeros(len(df))

        # Usage impact (strongest predictor)
        churn_prob += 0.4 * (1 - df['logins_last_30d'] / 30)

        # Payment issues
        churn_prob += 0.2 * (df['payment_failures_6m'] > 0)

        # Satisfaction
        churn_prob += 0.15 * (1 - df['satisfaction_score'] / 5)

        # Engagement
        churn_prob += 0.15 * (1 - df['engagement_score'])

        # Tenure (new customers churn more)
        churn_prob += 0.1 * np.exp(-df['tenure_days'] / 180)

        # Normalize to [0, 1]
        churn_prob = churn_prob / churn_prob.max()

        # Generate churn labels
        # Add some randomness so it's not perfectly deterministic
        noise = np.random.uniform(-0.1, 0.1, len(df))
        churn_prob_noisy = np.clip(churn_prob + noise, 0, 1)

        # Adjust to target churn rate
        threshold = np.percentile(churn_prob_noisy, (1 - self.churn_rate) * 100)
        churned = (churn_prob_noisy >= threshold).astype(int)

        # Add churn probability (useful for threshold tuning)
        df['churn_probability_true'] = churn_prob_noisy.round(3)
        df['churned'] = churned

        print(f"Actual churn rate: {churned.mean():.1%}")

        return df

    def generate_dataset(self):
        """Generate complete dataset"""
        print(f"\nGenerating {self.n_customers:,} customer records...\n")

        # Generate customer IDs
        customer_ids = [f"CUST_{str(i).zfill(6)}" for i in range(1, self.n_customers + 1)]

        # Pre-generate churn labels (needed for correlated features)
        will_churn = np.random.rand(self.n_customers) < self.churn_rate

        # Generate feature groups
        demographics = self.generate_demographics()
        subscription = self.generate_subscription_info()
        usage = self.generate_usage_patterns(subscription['tenure_days'].values, will_churn)
        engagement = self.generate_engagement_features(will_churn)
        transactions = self.generate_transaction_features(will_churn)

        # Combine all features
        df = pd.concat([
            pd.DataFrame({'customer_id': customer_ids}),
            demographics,
            subscription,
            usage,
            engagement,
            transactions
        ], axis=1)

        # Generate derived features
        df = self.generate_derived_features(df)

        # Generate final churn labels (based on actual features)
        df = self.generate_churn_labels(df)

        # Add timestamp
        df['data_date'] = self.today.strftime('%Y-%m-%d')

        print(f"\n✓ Dataset generated successfully!")
        print(f"  - Total customers: {len(df):,}")
        print(f"  - Features: {len(df.columns) - 2}")  # Exclude customer_id and churned
        print(f"  - Churn rate: {df['churned'].mean():.1%}")
        print(f"  - Date range: {df['signup_date'].min().date()} to {df['signup_date'].max().date()}")

        return df


def save_dataset(df, output_dir='data'):
    """Save dataset in multiple formats"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving dataset to {output_dir}/...")

    # Convert dates to string for JSON
    df_export = df.copy()
    df_export['signup_date'] = df_export['signup_date'].dt.strftime('%Y-%m-%d')

    # Save as CSV
    csv_path = output_path / 'customers.csv'
    df_export.to_csv(csv_path, index=False)
    print(f"  ✓ Saved CSV: {csv_path} ({csv_path.stat().st_size / 1024:.1f} KB)")

    # Save as JSON
    json_path = output_path / 'customers.json'
    df_export.to_json(json_path, orient='records', indent=2)
    print(f"  ✓ Saved JSON: {json_path} ({json_path.stat().st_size / 1024:.1f} KB)")

    # Save summary statistics
    summary_path = output_path / 'dataset_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CUSTOMER CHURN DATASET SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Records: {len(df):,}\n")
        f.write(f"Total Features: {len(df.columns)}\n")
        f.write(f"Churn Rate: {df['churned'].mean():.2%}\n\n")

        f.write("=" * 70 + "\n")
        f.write("FEATURE STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        f.write(df.describe().to_string())

        f.write("\n\n" + "=" * 70 + "\n")
        f.write("CATEGORICAL DISTRIBUTIONS\n")
        f.write("=" * 70 + "\n\n")

        for col in ['gender', 'plan_type', 'payment_method']:
            f.write(f"\n{col}:\n")
            f.write(df[col].value_counts().to_string())
            f.write("\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("CHURN ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        # Churn by plan type
        f.write("Churn Rate by Plan Type:\n")
        f.write(df.groupby('plan_type')['churned'].agg(['mean', 'count']).to_string())
        f.write("\n\n")

        # Top features correlated with churn
        f.write("Top 10 Features Correlated with Churn:\n")
        correlations = df.select_dtypes(include=[np.number]).corrwith(df['churned']).abs().sort_values(ascending=False)
        f.write(correlations.head(10).to_string())
        f.write("\n")

    print(f"  ✓ Saved summary: {summary_path}")

    return csv_path, json_path


def main():
    """Main execution"""
    print("\n" + "=" * 70)
    print("CUSTOMER CHURN DATASET GENERATOR")
    print("=" * 70)

    # Generate dataset
    generator = ChurnDataGenerator(n_customers=10000, churn_rate=0.15)
    df = generator.generate_dataset()

    # Save dataset
    csv_path, json_path = save_dataset(df)

    print("\n" + "=" * 70)
    print("DATASET PREVIEW")
    print("=" * 70)
    print(df.head(10).to_string())

    print("\n" + "=" * 70)
    print("SAMPLE STATISTICS")
    print("=" * 70)
    print(f"\nChurned customers: {df['churned'].sum():,} ({df['churned'].mean():.1%})")
    print(f"Active customers: {(1 - df['churned']).sum():,} ({(1 - df['churned']).mean():.1%})")

    print("\n✓ Complete! Dataset ready for ML pipeline.\n")

    # Usage instructions
    print("=" * 70)
    print("USAGE")
    print("=" * 70)
    print(f"""
To use this dataset:

1. Load in Python:
   import pandas as pd
   df = pd.read_csv('{csv_path}')

2. Load in JSON:
   import json
   with open('{json_path}') as f:
       data = json.load(f)

3. Key features to use:
   - Usage: logins_last_*d, avg_session_duration
   - Engagement: satisfaction_score, feature_adoption_rate
   - Transactions: payment_failures_6m, auto_renew_enabled
   - Derived: engagement_score, login_frequency_trend

4. Target variable: 'churned' (0 = retained, 1 = churned)

5. Class imbalance: ~15% churn (typical in subscription businesses)
   - Use class_weight='balanced' or SMOTE
   - Optimize threshold for business objectives

6. Temporal split: Use 'signup_date' for time-based train/test split
""")


if __name__ == '__main__':
    main()
