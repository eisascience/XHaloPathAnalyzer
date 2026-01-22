# XHaloPathAnalyzer 🔬

**Web-Based GUI for Halo Digital Pathology Image Analysis**

A comprehensive, OS-agnostic application for custom image analysis on whole-slide images stored in the Halo digital pathology platform. Integrates MedSAM AI segmentation with Halo's API for seamless export-analyze-import workflows.

## Features

- 🔐 **Secure Authentication**: Connect to Halo via GraphQL API
- 🔬 **Slide Management**: Browse, search, and select slides
- 🤖 **AI Analysis**: MedSAM segmentation on regions of interest
- 📊 **Visualization**: Side-by-side comparison and overlay views
- 📤 **GeoJSON Export**: Convert results to Halo-compatible annotations
- 🖥️ **Cross-Platform**: Works on Windows, macOS, and Linux
- 🚀 **GPU Accelerated**: Automatic CUDA detection and optimization

## Quick Start

### 1. Install Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt

# Install OpenSlide (platform-specific)
# macOS: brew install openslide
# Ubuntu: sudo apt-get install openslide-tools
# Windows: Download from https://openslide.org/download/
```

### 2. Download MedSAM Model
```bash
# Create models directory and download checkpoint (1.7GB)
mkdir -p models
wget -O models/medsam_vit_b.pth \
  https://zenodo.org/record/8408663/files/medsam_vit_b.pth
```

### 3. Configure Environment
```bash
# Copy example environment file
cp .env.example .env

# Edit with your Halo credentials
nano .env
```

### 4. Run Application
```bash
# Start Streamlit app
streamlit run app.py

# Open browser to http://localhost:8501
```

## Project Structure

```
XHaloPathAnalyzer/
├── app.py                 # Main Streamlit application
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── GUIDE.md              # Comprehensive 5000+ word guide
├── utils/
│   ├── halo_api.py       # Halo GraphQL API integration
│   ├── image_proc.py     # Image processing utilities
│   ├── ml_models.py      # MedSAM model wrapper
│   └── geojson_utils.py  # GeoJSON conversion
├── models/               # Model weights directory
└── temp/                 # Temporary files directory
```

## Documentation

See **[GUIDE.md](GUIDE.md)** for comprehensive documentation including:
- Detailed architecture explanation
- Complete setup instructions
- Code implementation details
- Advanced features and extensions
- Testing and debugging guide
- Deployment instructions
- Full example workflows

## Requirements

- **Python**: 3.10 or higher
- **RAM**: 16GB minimum (32GB recommended)
- **GPU**: Optional (NVIDIA CUDA for acceleration)
- **Storage**: 10GB for models and cache
- **Halo**: API access with valid token

## Key Technologies

- **Streamlit**: Web framework for interactive UI
- **PyTorch**: Deep learning framework
- **MedSAM**: Medical image segmentation model
- **GraphQL**: API communication with Halo
- **OpenSlide**: Whole-slide image processing
- **scikit-image**: Image processing and analysis

## Usage Example

```python
from config import Config
from utils.halo_api import HaloAPI
from utils.ml_models import MedSAMPredictor
from utils.image_proc import *
from utils.geojson_utils import *
import asyncio

# Setup
Config.validate()
api = HaloAPI(Config.HALO_API_ENDPOINT, Config.HALO_API_TOKEN)

# Get slides
slides = asyncio.run(api.get_slides())
slide = slides[0]

# Download region
data = api.download_region(slide['id'], 0, 0, 1024, 1024)
image = load_image_from_bytes(data)

# Analyze with MedSAM
predictor = MedSAMPredictor(Config.MEDSAM_CHECKPOINT)
preprocessed, metadata = preprocess_for_medsam(image)
mask = predictor.predict(preprocessed)
final_mask = postprocess_mask(mask, metadata)

# Export to GeoJSON
polygons = mask_to_polygons(final_mask)
geojson = polygons_to_geojson(polygons)
save_geojson(geojson, "annotations.geojson")
```

## License

This project is provided as-is for research and educational purposes.

## Support

For questions, issues, or contributions, please open an issue on GitHub.

---

**Built with ❤️ for the digital pathology community**
X-Halo-Patho-Analyzer
