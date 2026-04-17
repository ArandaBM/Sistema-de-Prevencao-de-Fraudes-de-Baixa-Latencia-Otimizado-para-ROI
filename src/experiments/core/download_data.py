import kagglehub

# Download latest version
path = kagglehub.dataset_download("kartik2112/fraud-detection", output_dir='data/raw', force_download=True)

print("Path to dataset files:", path)