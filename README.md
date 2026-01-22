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

# 🔬 XHalo Path Analyzer

**Halo AI Workflow: A web-based GUI for digital pathology analysis**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

XHalo Path Analyzer is a powerful, OS-agnostic workflow tool that bridges Halo's digital pathology platform with external AI/ML capabilities. It enables researchers to:

- 🔌 **Export WSIs/ROIs** from Halo via GraphQL API
- 🤖 **Run external ML models** (e.g., MedSAM segmentation) in Python
- 📊 **Import results back** to Halo for visualization and analysis
- 🎨 **Process large images** using intelligent tiling strategies
- 📄 **Generate GeoJSON** exports for interoperability
- 🌐 **Work independently** of vendor-specific tools

Built for exploratory AI in digital pathology, this tool provides a flexible, interactive environment for developing and deploying machine learning workflows.

## Key Features

### 🔬 Digital Pathology Integration
- **Halo GraphQL API Integration**: Direct connection to Halo for slide management
- **WSI/ROI Export**: Export whole slide images and regions of interest
- **Annotation Import**: Push AI-generated annotations back to Halo

### 🤖 AI/ML Capabilities
- **MedSAM Integration**: Medical Segment Anything Model for tissue segmentation
- **Tiled Processing**: Handle large pathology images efficiently
- **Custom Model Support**: Extensible architecture for other ML models

### 🎨 Visualization & Analysis
- **Interactive Web UI**: Built with Streamlit for ease of use
- **Real-time Visualization**: See segmentation results immediately
- **Overlay Views**: Compare original images with segmentation masks
- **Statistics**: Automatic calculation of coverage metrics

### 📄 Data Export
- **GeoJSON Export**: Industry-standard format for annotations
- **Mask Export**: Save binary segmentation masks
- **Halo Import**: Direct upload of results to Halo platform

## Technology Stack

- **Frontend**: Streamlit
- **API Integration**: gql (GraphQL client)
- **Image Processing**: large_image, Pillow, OpenCV
- **Machine Learning**: PyTorch, MedSAM
- **Geospatial**: geojson, shapely
- **Visualization**: matplotlib, plotly

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster inference

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/eisascience/XHaloPathAnalyzer.git
cd XHaloPathAnalyzer
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install the package**
```bash
pip install -e .
```

## Usage

### Web Interface (Recommended)

Launch the interactive web application:

```bash
streamlit run app.py
```

Or use the CLI:

```bash
xhalo-analyzer web --port 8501 --host localhost
```

Then open your browser to `http://localhost:8501`

### Web Interface Workflow

1. **Configure Halo API Connection**
   - Enter your Halo API URL and authentication key in the sidebar
   - Or use the Mock API for testing without a real Halo instance

2. **Initialize MedSAM**
   - Select device (CPU/CUDA)
   - Optionally provide path to MedSAM checkpoint
   - Click "Initialize MedSAM"

3. **Select/Upload Image**
   - Load slides from Halo using the GraphQL API
   - Or upload a local image file for analysis

4. **Run Segmentation**
   - Adjust processing parameters (tile size, overlap, min area)
   - Click "Run Segmentation"
   - View results with mask and overlay visualizations

5. **Export Results**
   - Export as GeoJSON for interoperability
   - Download segmentation mask as PNG
   - Import annotations directly back to Halo

### Command-Line Interface

Process images directly from the command line:

```bash
# Process an image and save results
xhalo-analyzer process input.tif \
    --output mask.png \
    --geojson annotations.geojson \
    --tile-size 1024
```

### Python API

Use XHalo Path Analyzer programmatically:

```python
from xhalo.api import MockHaloAPIClient
from xhalo.ml import MedSAMPredictor, segment_tissue
from xhalo.utils import load_image, mask_to_geojson
import asyncio

# Initialize API client
client = MockHaloAPIClient()

# Load slides
slides = asyncio.run(client.list_slides())
print(f"Found {len(slides)} slides")

# Load an image
image = load_image("path/to/image.tif")

# Run segmentation
predictor = MedSAMPredictor(device="cpu")
mask = predictor.predict_tiles(image, tile_size=1024)

# Export to GeoJSON
geojson_data = mask_to_geojson(mask, min_area=100)

# Import back to Halo
annotations = convert_to_halo_annotations(mask)
success = asyncio.run(
    client.import_annotations(slide_id, annotations)
)
```

## Configuration

### Halo API Setup

To connect to a real Halo instance:

1. Obtain your Halo GraphQL API endpoint URL
2. Generate an API key/token from your Halo instance
3. Enter these credentials in the web UI sidebar

### MedSAM Model

To use the full MedSAM model:

1. Download the MedSAM checkpoint from the [official repository](https://github.com/bowang-lab/MedSAM)
2. Provide the path to the checkpoint in the web UI
3. Select appropriate device (CUDA for GPU acceleration)

Note: The application includes a mock segmentation mode for testing without the full model.

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
├── app.py                          # Main Streamlit application
├── setup.py                        # Package configuration
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── xhalo/                          # Main package
│   ├── __init__.py
│   ├── cli.py                      # Command-line interface
│   ├── api/                        # Halo API integration
│   │   ├── __init__.py
│   │   └── halo_client.py         # GraphQL client
│   ├── ml/                         # Machine learning models
│   │   ├── __init__.py
│   │   └── medsam.py              # MedSAM integration
│   └── utils/                      # Utility functions
│       ├── __init__.py
│       ├── image_utils.py         # Image processing
│       └── geojson_utils.py       # GeoJSON conversion
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# Run with coverage
pytest --cov=xhalo tests/
```

### Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Deployment

### Local Deployment

The application can be run locally as described in the Usage section.

### Cloud Deployment

#### Streamlit Cloud

1. Push your repository to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy directly from your repository

#### Docker (Coming Soon)

```bash
# Build Docker image
docker build -t xhalo-analyzer .

# Run container
docker run -p 8501:8501 xhalo-analyzer
```

#### Other Cloud Platforms

The application can be deployed to:
- AWS EC2 / ECS
- Google Cloud Run
- Azure Web Apps
- Heroku

See the [deployment documentation](docs/deployment.md) for detailed instructions.

## Use Cases

### Research Applications
- Automated tissue segmentation in pathology studies
- High-throughput analysis of large slide collections
- Validation of manual annotations
- Exploratory analysis with AI models

### Clinical Workflows
- Pre-screening of samples for pathologist review
- Quantitative assessment of tissue characteristics
- Standardized measurement protocols
- Integration with existing LIMS/pathology systems

### AI/ML Development
- Rapid prototyping of segmentation models
- Model validation and comparison
- Ground truth generation for training data
- Deployment of custom models in production

## Limitations

- Mock MedSAM implementation for demonstration (full model integration requires checkpoint)
- GraphQL API schema may need adaptation for specific Halo versions
- Large image processing requires adequate RAM
- Real-time processing depends on hardware capabilities

## Roadmap

- [ ] Support for additional ML models
- [ ] Enhanced visualization options
- [ ] Batch processing capabilities
- [ ] Multi-user collaboration features
- [ ] Docker containerization
- [ ] Comprehensive test suite
- [ ] CI/CD pipeline
- [ ] Extended documentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MedSAM](https://github.com/bowang-lab/MedSAM) - Medical Segment Anything Model
- [Halo](https://indicalab.com/halo/) - Digital pathology platform by Indica Labs
- [Streamlit](https://streamlit.io/) - Web application framework

## Support

For issues, questions, or contributions:
- 📧 Open an issue on [GitHub](https://github.com/eisascience/XHaloPathAnalyzer/issues)
- 📖 Check the [documentation](docs/)
- 💬 Join discussions in the repository

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{xhalo_path_analyzer,
  title = {XHalo Path Analyzer: Web-based AI Workflow for Digital Pathology},
  author = {Eisa Science},
  year = {2026},
  url = {https://github.com/eisascience/XHaloPathAnalyzer}
}
```

---

**Built for exploratory AI in digital pathology** 🔬
