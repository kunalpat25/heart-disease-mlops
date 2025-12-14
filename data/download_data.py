"""
Heart Disease UCI Dataset Download Script
Supports multiple methods: local files, UCI ML Repository API, or direct download
"""

import os
import pandas as pd

# Column names for the dataset
COLUMN_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]


def load_from_local_folder(folder_path: str = None) -> pd.DataFrame:
    """
    Load dataset from locally downloaded UCI folder.
    
    Args:
        folder_path: Path to the heart+disease folder
    
    Returns:
        DataFrame with the heart disease data
    """
    if folder_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(script_dir, "heart+disease")
    
    # Try processed.cleveland.data first (most commonly used)
    cleveland_path = os.path.join(folder_path, "processed.cleveland.data")
    
    if os.path.exists(cleveland_path):
        print(f"Loading from local file: {cleveland_path}")
        df = pd.read_csv(cleveland_path, names=COLUMN_NAMES, na_values='?')
        
        # Convert target to binary (0 = no disease, 1 = disease present)
        df['target'] = (df['target'] > 0).astype(int)
        
        print(f"Loaded {len(df)} samples from Cleveland dataset")
        return df
    
    raise FileNotFoundError(f"Could not find processed.cleveland.data in {folder_path}")


def load_from_ucimlrepo() -> pd.DataFrame:
    """
    Load dataset using the official UCI ML Repository API.
    
    Returns:
        DataFrame with the heart disease data
    """
    try:
        from ucimlrepo import fetch_ucirepo
        
        print("Fetching Heart Disease dataset from UCI ML Repository...")
        heart_disease = fetch_ucirepo(id=45)
        
        # Get features and target
        X = heart_disease.data.features
        y = heart_disease.data.targets
        
        # Combine into single DataFrame
        df = X.copy()
        df['target'] = y
        
        # Rename columns to match expected format
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Convert target to binary if needed
        if 'target' in df.columns or 'num' in df.columns:
            target_col = 'target' if 'target' in df.columns else 'num'
            df['target'] = (df[target_col] > 0).astype(int)
        
        print(f"Loaded {len(df)} samples from UCI ML Repository")
        print(f"Metadata: {heart_disease.metadata.num_instances} instances")
        
        return df
        
    except ImportError:
        print("ucimlrepo not installed. Install with: pip install ucimlrepo")
        raise
    except Exception as e:
        print(f"Error fetching from UCI ML Repository: {e}")
        raise


def download_dataset(save_path: str = None) -> pd.DataFrame:
    """
    Download/load the Heart Disease UCI dataset.
    Tries multiple sources in order:
    1. Local heart+disease folder
    2. UCI ML Repository API
    3. Direct URL download
    
    Args:
        save_path: Path to save the CSV file. If None, saves to data/raw/heart.csv
    
    Returns:
        DataFrame with the heart disease data
    """
    if save_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, "raw", "heart.csv")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    df = None
    
    # Method 1: Try local folder first
    try:
        df = load_from_local_folder()
    except FileNotFoundError as e:
        print(f"Local folder not found: {e}")
    
    # Method 2: Try UCI ML Repository API
    if df is None:
        try:
            df = load_from_ucimlrepo()
        except Exception as e:
            print(f"UCI ML Repository failed: {e}")
    
    # Method 3: Direct URL download as fallback
    if df is None:
        try:
            import requests
            from io import StringIO
            
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            print(f"Downloading from {url}...")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text), names=COLUMN_NAMES, na_values='?')
            df['target'] = (df['target'] > 0).astype(int)
            
            print(f"Downloaded {len(df)} samples")
            
        except Exception as e:
            print(f"Direct download failed: {e}")
    
    if df is None:
        raise RuntimeError("Could not load dataset from any source!")
    
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        print(f"Found {df.isnull().sum().sum()} missing values, filling with median...")
        df = df.fillna(df.median())
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"\nDataset saved to {save_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df


def load_dataset(path: str = None) -> pd.DataFrame:
    """
    Load the heart disease dataset from local file.
    
    Args:
        path: Path to the CSV file. If None, looks in data/raw/heart.csv
    
    Returns:
        DataFrame with the heart disease data
    """
    if path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, "raw", "heart.csv")
    
    if not os.path.exists(path):
        print("Dataset not found locally. Downloading...")
        return download_dataset(path)
    
    return pd.read_csv(path)


if __name__ == "__main__":
    df = download_dataset()
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nTarget distribution:")
    print(df['target'].value_counts())
