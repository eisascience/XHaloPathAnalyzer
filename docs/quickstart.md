# Quick Start Guide

Get up and running with XHalo Path Analyzer in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster inference

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/eisascience/XHaloPathAnalyzer.git
cd XHaloPathAnalyzer
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

## Running the Application

### Option 1: Web Interface (Recommended)

Launch the interactive web application:

```bash
streamlit run app.py
```

Then open your browser to: http://localhost:8501

### Option 2: Command Line Interface

Process images directly from the command line:

```bash
xhalo-analyzer process input.tif --output mask.png --geojson annotations.geojson
```

### Option 3: Python API

Use in your own Python scripts:

```python
from xhalo.ml import segment_tissue
from xhalo.utils import load_image, mask_to_geojson

# Load and process image
image = load_image("path/to/image.tif")
mask = segment_tissue(image)
geojson = mask_to_geojson(mask)
```

## First Steps with the Web Interface

1. **Connect to Halo API**
   - In the sidebar, check "Use Mock API" for testing
   - Or enter your Halo API credentials
   - Click "Connect to Halo"

2. **Initialize MedSAM**
   - Select device (CPU or CUDA)
   - Click "Initialize MedSAM"

3. **Load an Image**
   - Go to "Slide Selection" tab
   - Upload a local image file
   - Or load slides from Halo

4. **Run Segmentation**
   - Go to "Segmentation" tab
   - Adjust parameters if needed
   - Click "Run Segmentation"

5. **View Results**
   - See segmentation mask and overlay
   - Check statistics

6. **Export Results**
   - Go to "Export/Import" tab
   - Export as GeoJSON or PNG
   - Import back to Halo

## Example Workflow

Try the included examples:

```bash
# Basic API usage example
python examples/basic_usage.py

# Image processing example
python examples/process_image.py
```

## Testing

Run the test suite:

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

## Docker Deployment

If you prefer Docker:

```bash
# Build and run
docker-compose up -d

# Access at http://localhost:8501
```

## Troubleshooting

### Import Error

If you get import errors, make sure you installed the package:
```bash
pip install -e .
```

### CUDA Not Available

If you want GPU acceleration but CUDA is not available:
1. Check GPU drivers: `nvidia-smi`
2. Install PyTorch with CUDA support from [pytorch.org](https://pytorch.org)

### Port Already in Use

If port 8501 is busy, use a different port:
```bash
streamlit run app.py --server.port=8502
```

## Next Steps

- Read the full [README](../README.md) for detailed documentation
- Check out [deployment options](deployment.md)
- Explore the [API documentation](api.md)
- Join the discussion on [GitHub Issues](https://github.com/eisascience/XHaloPathAnalyzer/issues)

## Getting Help

- 📖 [Documentation](../README.md)
- 💬 [GitHub Issues](https://github.com/eisascience/XHaloPathAnalyzer/issues)
- 🔍 [Examples](../examples/)

Happy analyzing! 🔬
