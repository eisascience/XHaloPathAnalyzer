"""
Halo AI Workflow - Main Streamlit Application
Web-based GUI for digital pathology analysis with Halo API integration
"""

import streamlit as st
import asyncio
import numpy as np
from PIL import Image
import io
import logging
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import local modules
from xhalo.api import HaloAPIClient, MockHaloAPIClient
from xhalo.ml import MedSAMPredictor, segment_tissue
from xhalo.utils import (
    load_image,
    resize_image,
    overlay_mask,
    mask_to_geojson,
    convert_to_halo_annotations,
    save_geojson
)

# Page configuration
st.set_page_config(
    page_title="Halo AI Workflow",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize session state variables"""
    if 'halo_client' not in st.session_state:
        st.session_state.halo_client = None
    if 'medsam_predictor' not in st.session_state:
        st.session_state.medsam_predictor = None
    if 'selected_slide' not in st.session_state:
        st.session_state.selected_slide = None
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'segmentation_mask' not in st.session_state:
        st.session_state.segmentation_mask = None
    if 'slides_list' not in st.session_state:
        st.session_state.slides_list = []


def sidebar_config():
    """Configure sidebar with API and model settings"""
    with st.sidebar:
        st.title("🔬 Halo AI Workflow")
        st.markdown("---")
        
        # Halo API Configuration
        st.header("Halo API Configuration")
        
        use_mock = st.checkbox("Use Mock API (for testing)", value=True)
        
        if use_mock:
            api_url = "http://mock-halo-api"
            api_key = None
            st.info("Using mock Halo API with sample data")
        else:
            api_url = st.text_input(
                "Halo API URL",
                value="https://your-halo-instance/graphql",
                help="GraphQL endpoint for your Halo instance"
            )
            api_key = st.text_input(
                "API Key",
                type="password",
                help="Authentication token for Halo API"
            )
        
        if st.button("Connect to Halo"):
            with st.spinner("Connecting to Halo API..."):
                try:
                    if use_mock:
                        st.session_state.halo_client = MockHaloAPIClient(api_url, api_key)
                    else:
                        st.session_state.halo_client = HaloAPIClient(api_url, api_key)
                    st.success("✅ Connected to Halo API")
                except Exception as e:
                    st.error(f"❌ Connection failed: {e}")
        
        st.markdown("---")
        
        # MedSAM Configuration
        st.header("MedSAM Configuration")
        
        device = st.selectbox(
            "Device",
            ["cpu", "cuda"],
            help="Select device for ML inference"
        )
        
        model_path = st.text_input(
            "Model Path (optional)",
            help="Path to MedSAM checkpoint file"
        )
        
        if st.button("Initialize MedSAM"):
            with st.spinner("Initializing MedSAM predictor..."):
                try:
                    st.session_state.medsam_predictor = MedSAMPredictor(
                        model_path=model_path if model_path else None,
                        device=device
                    )
                    st.success("✅ MedSAM initialized")
                except Exception as e:
                    st.error(f"❌ Initialization failed: {e}")
        
        st.markdown("---")
        
        # Processing Parameters
        st.header("Processing Parameters")
        
        tile_size = st.slider(
            "Tile Size",
            min_value=256,
            max_value=2048,
            value=1024,
            step=256,
            help="Size of tiles for processing large images"
        )
        
        overlap = st.slider(
            "Tile Overlap",
            min_value=0,
            max_value=512,
            value=128,
            step=64,
            help="Overlap between tiles to avoid boundary artifacts"
        )
        
        min_area = st.slider(
            "Minimum Object Area",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="Minimum area for detected objects"
        )
        
        return {
            'tile_size': tile_size,
            'overlap': overlap,
            'min_area': min_area
        }


def slide_selection_tab():
    """Tab for selecting and loading slides from Halo"""
    st.header("📁 Slide Selection")
    
    if st.session_state.halo_client is None:
        st.warning("⚠️ Please connect to Halo API first (see sidebar)")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        project_id = st.text_input(
            "Project ID (optional)",
            help="Filter slides by project"
        )
    
    with col2:
        if st.button("🔄 Load Slides"):
            with st.spinner("Loading slides from Halo..."):
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    slides = loop.run_until_complete(
                        st.session_state.halo_client.list_slides(
                            project_id if project_id else None
                        )
                    )
                    st.session_state.slides_list = slides
                    st.success(f"✅ Loaded {len(slides)} slides")
                except Exception as e:
                    st.error(f"❌ Error loading slides: {e}")
    
    if st.session_state.slides_list:
        st.subheader("Available Slides")
        
        # Display slides in a table
        slide_names = [slide['name'] for slide in st.session_state.slides_list]
        selected_idx = st.selectbox(
            "Select a slide",
            range(len(slide_names)),
            format_func=lambda i: f"{slide_names[i]} (ID: {st.session_state.slides_list[i]['id']})"
        )
        
        if selected_idx is not None:
            st.session_state.selected_slide = st.session_state.slides_list[selected_idx]
            
            # Display slide info
            slide = st.session_state.selected_slide
            
            st.info(f"""
            **Slide Information:**
            - Name: {slide['name']}
            - ID: {slide['id']}
            - Dimensions: {slide['width']} x {slide['height']}
            - Magnification: {slide.get('magnification', 'N/A')}x
            """)
    
    # Image upload option
    st.markdown("---")
    st.subheader("Or Upload Local Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        help="Upload a local image for analysis"
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.session_state.current_image = np.array(image.convert('RGB'))
            
            # Display image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.success(f"✅ Image loaded: {image.size[0]} x {image.size[1]}")
        except Exception as e:
            st.error(f"❌ Error loading image: {e}")


def segmentation_tab(params: Dict[str, Any]):
    """Tab for running segmentation and visualizing results"""
    st.header("🎯 Segmentation & Analysis")
    
    if st.session_state.current_image is None:
        st.warning("⚠️ Please load an image first (see Slide Selection tab)")
        return
    
    # Display current image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(st.session_state.current_image, use_column_width=True)
    
    with col2:
        st.subheader("Segmentation Controls")
        
        # Segmentation button
        if st.button("🚀 Run Segmentation", type="primary"):
            if st.session_state.medsam_predictor is None:
                st.warning("Initializing default MedSAM predictor...")
                st.session_state.medsam_predictor = MedSAMPredictor()
            
            with st.spinner("Running segmentation..."):
                try:
                    # Run segmentation
                    mask = st.session_state.medsam_predictor.predict_tiles(
                        st.session_state.current_image,
                        tile_size=params['tile_size'],
                        overlap=params['overlap']
                    )
                    
                    st.session_state.segmentation_mask = mask
                    st.success("✅ Segmentation complete!")
                
                except Exception as e:
                    st.error(f"❌ Segmentation failed: {e}")
                    logger.exception(e)
    
    # Display segmentation results
    if st.session_state.segmentation_mask is not None:
        st.markdown("---")
        st.subheader("Segmentation Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Mask**")
            # Display mask
            mask_img = (st.session_state.segmentation_mask * 255).astype(np.uint8)
            st.image(mask_img, use_column_width=True, clamp=True)
        
        with col2:
            st.write("**Overlay**")
            # Create overlay
            overlay = overlay_mask(
                st.session_state.current_image,
                st.session_state.segmentation_mask,
                alpha=0.4,
                color=(255, 0, 0)
            )
            st.image(overlay, use_column_width=True)
        
        # Statistics
        st.markdown("---")
        st.subheader("Statistics")
        
        total_pixels = st.session_state.segmentation_mask.size
        positive_pixels = np.sum(st.session_state.segmentation_mask > 0)
        coverage = (positive_pixels / total_pixels) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Pixels", f"{total_pixels:,}")
        with col2:
            st.metric("Segmented Pixels", f"{positive_pixels:,}")
        with col3:
            st.metric("Coverage", f"{coverage:.2f}%")


def export_tab(params: Dict[str, Any]):
    """Tab for exporting results and importing back to Halo"""
    st.header("💾 Export & Import")
    
    if st.session_state.segmentation_mask is None:
        st.warning("⚠️ Please run segmentation first")
        return
    
    st.subheader("Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export as GeoJSON
        st.write("**Export as GeoJSON**")
        
        annotation_type = st.text_input("Annotation Type", value="tissue")
        layer_name = st.text_input("Layer Name", value="AI Annotations")
        
        if st.button("📄 Generate GeoJSON"):
            with st.spinner("Converting to GeoJSON..."):
                try:
                    geojson_data = mask_to_geojson(
                        st.session_state.segmentation_mask,
                        min_area=params['min_area'],
                        simplify_tolerance=1.0,
                        properties={
                            "type": annotation_type,
                            "layer": layer_name,
                            "source": "MedSAM"
                        }
                    )
                    
                    st.session_state.geojson_data = geojson_data
                    
                    # Show preview
                    st.json(geojson_data, expanded=False)
                    
                    # Download button
                    import json
                    json_str = json.dumps(geojson_data, indent=2)
                    st.download_button(
                        label="⬇️ Download GeoJSON",
                        data=json_str,
                        file_name="segmentation.geojson",
                        mime="application/json"
                    )
                    
                    st.success("✅ GeoJSON generated!")
                
                except Exception as e:
                    st.error(f"❌ Error generating GeoJSON: {e}")
    
    with col2:
        # Export mask as image
        st.write("**Export Mask as Image**")
        
        if st.button("🖼️ Download Mask"):
            # Convert mask to PNG
            mask_img = Image.fromarray(
                (st.session_state.segmentation_mask * 255).astype(np.uint8)
            )
            
            # Save to bytes
            buf = io.BytesIO()
            mask_img.save(buf, format='PNG')
            buf.seek(0)
            
            st.download_button(
                label="⬇️ Download PNG",
                data=buf,
                file_name="segmentation_mask.png",
                mime="image/png"
            )
    
    # Import back to Halo
    st.markdown("---")
    st.subheader("Import to Halo")
    
    if st.session_state.halo_client is None:
        st.warning("⚠️ Please connect to Halo API first")
    elif st.session_state.selected_slide is None:
        st.warning("⚠️ Please select a slide first")
    else:
        if st.button("⬆️ Import Annotations to Halo", type="primary"):
            with st.spinner("Importing annotations to Halo..."):
                try:
                    # Convert to Halo format
                    annotations = convert_to_halo_annotations(
                        st.session_state.segmentation_mask,
                        annotation_type=annotation_type,
                        layer_name=layer_name,
                        min_area=params['min_area']
                    )
                    
                    # Import via API
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    success = loop.run_until_complete(
                        st.session_state.halo_client.import_annotations(
                            st.session_state.selected_slide['id'],
                            annotations,
                            layer_name
                        )
                    )
                    
                    if success:
                        st.success(f"✅ Imported {len(annotations)} annotations to Halo!")
                    else:
                        st.error("❌ Import failed")
                
                except Exception as e:
                    st.error(f"❌ Error importing to Halo: {e}")
                    logger.exception(e)


def main():
    """Main application entry point"""
    # Initialize session state
    init_session_state()
    
    # Sidebar configuration
    params = sidebar_config()
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Slide Selection",
        "🎯 Segmentation",
        "💾 Export/Import",
        "ℹ️ About"
    ])
    
    with tab1:
        slide_selection_tab()
    
    with tab2:
        segmentation_tab(params)
    
    with tab3:
        export_tab(params)
    
    with tab4:
        st.header("About Halo AI Workflow")
        st.markdown("""
        ### 🔬 Halo AI Workflow
        
        A web-based GUI for digital pathology analysis with Halo API integration.
        
        **Key Features:**
        - 🔌 Integrate with Halo's GraphQL API to export WSIs/ROIs
        - 🤖 Run external ML models (e.g., MedSAM segmentation) in Python
        - 📊 Import results back to Halo for visualization
        - 🌐 OS-agnostic workflows outside vendor tools
        - 🎨 Interactive visualization of segmentation results
        - 📄 Export results as GeoJSON for interoperability
        
        **Technology Stack:**
        - Streamlit for web interface
        - GQL for GraphQL API communication
        - Large_image for handling whole slide images
        - MedSAM (Medical Segment Anything Model) for segmentation
        - PyTorch for deep learning inference
        
        **Getting Started:**
        1. Configure Halo API connection in the sidebar
        2. Initialize MedSAM predictor (or use default)
        3. Select or upload an image
        4. Run segmentation
        5. Export results or import back to Halo
        
        **Documentation:**
        - [GitHub Repository](https://github.com/eisascience/XHaloPathAnalyzer)
        - [Setup Instructions](https://github.com/eisascience/XHaloPathAnalyzer#setup)
        
        ---
        
        Built for exploratory AI in digital pathology.
        """)


if __name__ == "__main__":
    main()
