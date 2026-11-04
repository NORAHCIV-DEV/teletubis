from pathlib import Path
from kmeans_service import train

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    result = train(root / "Dataset_10000_Bolivia_Sintetico.csv", root / "artifacts")
    print(f"K-Means entrenado: k={result['selected_k']}, silhouette={result['silhouette']:.4f}, "
          f"registros={result['records']}")
