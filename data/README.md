# Data Directory

This directory is for storing transaction datasets.

## Supported Formats

The system expects CSV files with transaction data. Common column names that are automatically recognized:

### Required or Generated Columns:
- `sender` / `nameOrig`: Transaction sender/originator ID
- `receiver` / `nameDest`: Transaction receiver/destination ID
- `amount`: Transaction amount

### Optional Columns:
- `isFraud` / `is_fraud` / `fraud` / `Class` / `label`: Binary fraud label (0=legitimate, 1=fraud)
- `timestamp` / `step`: Transaction timestamp or time step
- `sender_balance_before` / `oldbalanceOrg`: Sender's balance before transaction
- `sender_balance_after` / `newbalanceOrig`: Sender's balance after transaction
- `receiver_balance_before` / `oldbalanceDest`: Receiver's balance before transaction
- `receiver_balance_after` / `newbalanceDest`: Receiver's balance after transaction

## Example Datasets

You can use synthetic financial transaction datasets from Kaggle or similar sources:

1. **PaySim Synthetic Financial Dataset**
   - Synthetic mobile money transaction dataset
   - Available on Kaggle
   
2. **Credit Card Fraud Detection Dataset**
   - Real credit card transaction dataset
   - Available on Kaggle

3. **IEEE-CIS Fraud Detection Dataset**
   - Transaction data with fraud labels
   - Available on Kaggle

## Usage

Place your CSV file in this directory and run:

```bash
python main.py --data-path data/your_dataset.csv
```

Or use the synthetic data generator for testing:

```bash
python main.py --use-synthetic --n-transactions 10000
```

## Data Privacy

**Important**: Do not commit actual transaction data to version control. This directory is in `.gitignore` to protect sensitive information.
