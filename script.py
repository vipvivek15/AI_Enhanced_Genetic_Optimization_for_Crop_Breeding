import os

# Define the project structure
project_structure = {
    "data": [
        "raw_data/",  # Folder for datasets
        "preprocess.py",
        "data_loader.py"
    ],
    "models": [
        "train_model.py",
        "model.py",
        "hyperparameter_tuning.py",
        "validation.py"
    ],
    "app": [
        "app.py",
        "templates/",  # Frontend HTML templates
        "static/"  # CSS, JS, and images
    ],
    "deployment": [
        "Dockerfile",
        "kubernetes.yaml",
        "server_config.py"
    ]
}

# Function to create directories and files
def create_project_structure(base_path="."):
    for folder, items in project_structure.items():
        folder_path = os.path.join(base_path, folder) if folder != "root" else base_path

        # Create directory if it doesn't exist
        if folder != "root":
            os.makedirs(folder_path, exist_ok=True)

        for item in items:
            item_path = os.path.join(folder_path, item)

            if item.endswith("/"):  # Create subdirectories
                os.makedirs(item_path, exist_ok=True)
            else:  # Create files with initial content
                with open(item_path, "w") as f:
                    f.write(f"# {item}\n\nGenerated automatically by setup script.\n")

# Run the script
if __name__ == "__main__":
    create_project_structure()
    print("✅ Project structure created successfully!")
