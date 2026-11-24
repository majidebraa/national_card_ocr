# Windows/PowerShell version of download_dataset.sh
pip install kaggle
New-Item -ItemType Directory -Force data\raw | Out-Null
kaggle datasets download -d kibordevents/iranian-national-id-card-dataset -p data/raw --unzip
Write-Host "Done. Inspect the license and the metadata CSV before use. Never commit data/."
