#!/usr/bin/env python
import os
import sys
import argparse

# Append project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from churn_project.models.train import train_churn_model


def main():
    parser = argparse.ArgumentParser(
        description="CLI Tool to orchestrate and retrain the Customer Churn ML Pipeline."
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/customer_churn.csv",
        help="Path to the raw input customer churn dataset (CSV)."
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default="artifacts/models/best_model.joblib",
        help="Destination path to save the trained model artifact."
    )
    parser.add_argument(
        "--preprocessor-output",
        type=str,
        default="artifacts/preprocessors/preprocessor.joblib",
        help="Destination path to save the fitted preprocessor pipeline artifact."
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross-validation splits to perform."
    )

    args = parser.parse_args()

    print("=" * 60)
    print("STARTING CUSTOMER CHURN MODEL TRAINING RUNNER")
    print("=" * 60)
    print(f"[*] Input Data Path:          {args.data_path}")
    print(f"[*] Model Output Destination: {args.model_output}")
    print(f"[*] Preprocessor Destination: {args.preprocessor_output}")
    print(f"[*] Cross-Validation Splits:  {args.cv_splits}")
    print("-" * 60)

    try:
        results = train_churn_model(
            data_path=args.data_path,
            model_save_path=args.model_output,
            preprocessor_save_path=args.preprocessor_output,
            cv_splits=args.cv_splits
        )
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        for metric, val in results.items():
            print(f" - {metric.upper()}: {val:.4f}")
            
    except Exception as e:
        print(f"\n[!] Execution Failure: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()