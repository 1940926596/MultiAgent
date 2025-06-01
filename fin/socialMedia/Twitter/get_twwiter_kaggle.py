import kagglehub

# Download latest version
path = kagglehub.dataset_download("kirolosatef/netflex-stock-dataset-with-twitter-sentiment")

print("Path to dataset files:", path)