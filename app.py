"""
XHaloPathAnalyzer - Main Streamlit Application

Web-based GUI for custom image analysis on Halo digital pathology slides.
Provides interface for:
- Authentication with Halo API
- Slide selection and metadata viewing
- ROI export and image processing
- MedSAM segmentation analysis
- GeoJSON export for Halo import
"""

import streamlit as st
import asyncio
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime
import traceback

from config import Config
from utils.halo_api import HaloAPI
from utils.image_proc import (
    load_image_from_bytes,
    preprocess_for_medsam,
    postprocess_mask,
    overlay_mask_on_image,
    compute_mask_statistics
)
from utils.ml_models import MedSAMPredictor
from utils.geojson_utils import (
    mask_to_polygons,
    polygons_to_geojson,
    save_geojson
)

# Page configuration
st.set_page_config(
    page_title="XHaloPathAnalyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'api' not in st.session_state:
        st.session_state.api = None
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'selected_slide' not in st.session_state:
        st.session_state.selected_slide = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'current_mask' not in st.session_state:
        st.session_state.current_mask = None

init_session_state()


def authentication_page():
    """Authentication and configuration page"""
    st.markdown('<h1 class="main-header">🔬 XHaloPathAnalyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Web-Based GUI for Halo Digital Pathology Analysis")
    
    st.markdown("---")
    
    st.subheader("🔐 Authentication")
    st.write("Connect to your Halo digital pathology instance")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        endpoint = st.text_input(
            "Halo API Endpoint",
            value=Config.HALO_API_ENDPOINT,
            placeholder="https://your-halo-instance.com/graphql",
            help="URL of your Halo GraphQL API endpoint"
        )
        
        token = st.text_input(
            "API Token",
            value=Config.HALO_API_TOKEN,
            type="password",
            placeholder="Enter your API token",
            help="API authentication token from Halo settings"
        )
        
        if st.button("🔌 Connect", type="primary", use_container_width=True):
            if not endpoint or not token:
                st.error("❌ Please provide both endpoint and token")
            else:
                with st.spinner("Testing connection..."):
                    try:
                        # Create API instance
                        api = HaloAPI(endpoint, token)
                        
                        # Test connection
                        success = asyncio.run(api.test_connection())
                        
                        if success:
                            st.session_state.api = api
                            st.session_state.authenticated = True
                            st.success("✅ Connected successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Connection test failed")
                            
                    except Exception as e:
                        st.error(f"❌ Connection failed: {str(e)}")
    
    with col2:
        st.info("""
        **How to get API token:**
        1. Log into Halo
        2. Go to Settings → API
        3. Create new token
        4. Copy and paste here
        """)
    
    st.markdown("---")
    st.markdown("### 📋 Features")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔍 Slide Selection**")
        st.write("Browse and select slides from your Halo instance")
    with col2:
        st.markdown("**🤖 AI Analysis**")
        st.write("Run MedSAM segmentation on regions of interest")
    with col3:
        st.markdown("**📤 Export Results**")
        st.write("Generate GeoJSON annotations for Halo import")


def slide_selection_page():
    """Slide selection and browsing interface"""
    st.title("🔬 Slide Selection")
    
    if st.session_state.api is None:
        st.warning("⚠️ Please authenticate first")
        return
    
    # Fetch slides
    with st.spinner("Loading slides from Halo..."):
        try:
            slides = asyncio.run(st.session_state.api.get_slides(limit=100))
        except Exception as e:
            st.error(f"❌ Failed to fetch slides: {str(e)}")
            return
    
    if not slides:
        st.warning("⚠️ No slides found in your Halo instance")
        return
    
    st.success(f"✅ Found {len(slides)} slides")
    
    # Convert to DataFrame for display
    df = pd.DataFrame(slides)
    
    # Add filters
    st.subheader("🔍 Filter Slides")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        name_filter = st.text_input("Filter by name", "")
    with col2:
        study_filter = st.text_input("Filter by study ID", "")
    
    # Apply filters
    if name_filter:
        df = df[df['name'].str.contains(name_filter, case=False, na=False)]
    if study_filter:
        df = df[df['studyId'].str.contains(study_filter, case=False, na=False)]
    
    st.markdown("---")
    
    # Display slides table
    st.subheader("📊 Available Slides")
    
    if len(df) == 0:
        st.info("No slides match the filter criteria")
        return
    
    # Select slide from dropdown
    slide_names = df['name'].tolist()
    selected_name = st.selectbox(
        "Select a slide",
        slide_names,
        help="Choose a slide to analyze"
    )
    
    # Get selected slide data
    selected_idx = slide_names.index(selected_name)
    selected_slide = slides[selected_idx]
    
    # Display slide details
    st.markdown("---")
    st.subheader("📄 Slide Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Name", selected_slide['name'])
        st.metric("Width", f"{selected_slide['width']:,} px")
    with col2:
        st.metric("ID", selected_slide['id'][:16] + "...")
        st.metric("Height", f"{selected_slide['height']:,} px")
    with col3:
        mpp = selected_slide.get('mpp', 'N/A')
        st.metric("MPP", f"{mpp}" if mpp != 'N/A' else "N/A")
        st.metric("Format", selected_slide.get('format', 'Unknown'))
    
    # Save to session state
    if st.button("✅ Select This Slide", type="primary", use_container_width=True):
        st.session_state.selected_slide = selected_slide
        st.success(f"✅ Selected: {selected_slide['name']}")


def analysis_page():
    """Analysis interface with MedSAM segmentation"""
    st.title("🤖 AI-Powered Analysis")
    
    if st.session_state.selected_slide is None:
        st.warning("⚠️ Please select a slide first")
        return
    
    slide = st.session_state.selected_slide
    st.info(f"📊 Analyzing: **{slide['name']}**")
    
    # ROI selection
    st.subheader("📍 Region of Interest (ROI)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x = st.number_input("X coordinate (pixels)", min_value=0, max_value=slide['width'], value=0, step=100)
        width = st.number_input("Width (pixels)", min_value=1, max_value=slide['width'], value=1024, step=100)
    
    with col2:
        y = st.number_input("Y coordinate (pixels)", min_value=0, max_value=slide['height'], value=0, step=100)
        height = st.number_input("Height (pixels)", min_value=1, max_value=slide['height'], value=1024, step=100)
    
    # Validate ROI
    if x + width > slide['width']:
        st.error(f"❌ ROI extends beyond slide width ({slide['width']} px)")
        return
    if y + height > slide['height']:
        st.error(f"❌ ROI extends beyond slide height ({slide['height']} px)")
        return
    
    st.markdown("---")
    
    # Analysis settings
    st.subheader("⚙️ Analysis Settings")
    
    use_prompts = st.checkbox("Use point/box prompts", value=False, 
                             help="Enable interactive prompts for segmentation")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                try:
                    # Download region (note: this is a blocking operation)
                    st.info("⏳ Downloading region from Halo... This may take a moment for large regions.")
                    region_data = st.session_state.api.download_region(
                        slide['id'], x, y, width, height
                    )
                    
                    if not region_data:
                        st.error("❌ Failed to download region - no data received")
                        return
                    
                    # Load image
                    st.info("⏳ Loading image...")
                    image = load_image_from_bytes(region_data)
                    st.session_state.current_image = image
                    
                    # Initialize predictor if needed
                    if st.session_state.predictor is None:
                        st.info("⏳ Loading MedSAM model...")
                        st.session_state.predictor = MedSAMPredictor(
                            Config.MEDSAM_CHECKPOINT,
                            model_type=Config.MODEL_TYPE,
                            device=Config.DEVICE
                        )
                    
                    # Preprocess
                    st.info("⏳ Preprocessing image...")
                    preprocessed, metadata = preprocess_for_medsam(image, Config.DEFAULT_TARGET_SIZE)
                    
                    # Run inference
                    st.info("⏳ Running MedSAM segmentation...")
                    mask = st.session_state.predictor.predict(preprocessed)
                    
                    # Postprocess
                    st.info("⏳ Postprocessing results...")
                    final_mask = postprocess_mask(mask, metadata)
                    st.session_state.current_mask = final_mask
                    
                    # Compute statistics
                    mpp = slide.get('mpp')
                    stats = compute_mask_statistics(final_mask, mpp)
                    
                    # Store results
                    st.session_state.analysis_results = {
                        'image': image,
                        'mask': final_mask,
                        'roi': (x, y, width, height),
                        'statistics': stats,
                        'slide_id': slide['id'],
                        'slide_name': slide['name'],
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    st.success("✅ Analysis complete!")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.code(traceback.format_exc())
    
    with col2:
        if st.button("🔄 Clear Results"):
            st.session_state.analysis_results = None
            st.session_state.current_image = None
            st.session_state.current_mask = None
            st.success("✅ Cleared")
    
    # Display results
    if st.session_state.analysis_results is not None:
        st.markdown("---")
        st.subheader("📊 Results")
        
        results = st.session_state.analysis_results
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        stats = results['statistics']
        
        with col1:
            st.metric("Positive Pixels", f"{stats['num_positive_pixels']:,}")
        with col2:
            st.metric("Coverage", f"{stats['coverage_percent']:.2f}%")
        with col3:
            if 'area_um2' in stats:
                st.metric("Area", f"{stats['area_um2']:.2f} µm²")
        with col4:
            if 'area_mm2' in stats:
                st.metric("Area", f"{stats['area_mm2']:.4f} mm²")
        
        # Visualizations
        st.markdown("### 🖼️ Visualization")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(results['image'], caption="Original Image", use_column_width=True)
        
        with col2:
            st.image(results['mask'], caption="Segmentation Mask", use_column_width=True)
        
        with col3:
            overlay = overlay_mask_on_image(
                results['image'],
                results['mask'],
                color=(255, 0, 0),
                alpha=0.5
            )
            st.image(overlay, caption="Overlay", use_column_width=True)


def export_page():
    """Export results to GeoJSON format"""
    st.title("📤 Export Results")
    
    if st.session_state.analysis_results is None:
        st.warning("⚠️ No analysis results to export. Please run analysis first.")
        return
    
    results = st.session_state.analysis_results
    
    st.success(f"✅ Results ready for export from: **{results['slide_name']}**")
    
    # Export settings
    st.subheader("⚙️ Export Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        classification = st.text_input(
            "Classification Label",
            value="detected_object",
            help="Label for the detected objects"
        )
        
        min_area = st.number_input(
            "Minimum Polygon Area (pixels)",
            min_value=1,
            value=Config.MIN_POLYGON_AREA,
            help="Filter out small polygons"
        )
    
    with col2:
        simplify = st.checkbox(
            "Simplify Polygons",
            value=True,
            help="Reduce polygon complexity"
        )
        
        if simplify:
            tolerance = st.slider(
                "Simplification Tolerance",
                min_value=0.1,
                max_value=5.0,
                value=Config.SIMPLIFY_TOLERANCE,
                help="Higher = more simplified"
            )
        else:
            tolerance = 1.0
    
    st.markdown("---")
    
    if st.button("🔄 Generate GeoJSON", type="primary", use_container_width=True):
        with st.spinner("Converting mask to GeoJSON..."):
            try:
                # Convert mask to polygons
                st.info("⏳ Extracting polygons from mask...")
                polygons = mask_to_polygons(results['mask'], min_area=min_area)
                
                if len(polygons) == 0:
                    st.warning("⚠️ No polygons found. Try reducing minimum area.")
                    return
                
                # Create GeoJSON
                st.info(f"⏳ Creating GeoJSON with {len(polygons)} features...")
                geojson = polygons_to_geojson(
                    polygons,
                    properties={"classification": classification},
                    simplify=simplify,
                    tolerance=tolerance
                )
                
                # Save to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"annotations_{results['slide_name']}_{timestamp}.geojson"
                output_path = Config.get_temp_path(filename)
                
                save_geojson(geojson, str(output_path))
                
                # Store in session
                st.session_state.geojson = geojson
                st.session_state.geojson_path = output_path
                
                st.success(f"✅ Exported {len(polygons)} polygons to GeoJSON!")
                
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
                st.code(traceback.format_exc())
    
    # Display and download
    if 'geojson' in st.session_state:
        st.markdown("---")
        st.subheader("📄 GeoJSON Preview")
        
        # Statistics
        num_features = len(st.session_state.geojson['features'])
        st.metric("Number of Features", num_features)
        
        # Preview
        with st.expander("View GeoJSON"):
            st.json(st.session_state.geojson)
        
        # Download button
        with open(st.session_state.geojson_path, 'r') as f:
            geojson_str = f.read()
        
        st.download_button(
            label="💾 Download GeoJSON",
            data=geojson_str,
            file_name=st.session_state.geojson_path.name,
            mime="application/json",
            use_container_width=True
        )
        
        st.info("""
        **Next Steps:**
        1. Download the GeoJSON file
        2. Open your slide in Halo
        3. Import the GeoJSON as annotations
        4. Visualize and refine results
        """)


def import_page():
    """Import annotations to Halo (optional feature)"""
    st.title("📥 Import to Halo")
    
    st.info("🚧 This feature requires additional Halo API permissions")
    
    st.markdown("""
    ### Manual Import Instructions
    
    1. **Download GeoJSON** from the Export page
    2. **Open Halo** and navigate to your slide
    3. **Import Annotations**: 
       - File → Import → GeoJSON
       - Select the downloaded file
    4. **Review and Save** the imported annotations
    
    ### Programmatic Import (Coming Soon)
    
    Automatic upload of annotations via API will be available in a future release.
    """)


def main():
    """Main application"""
    st.sidebar.title("🔬 XHaloPathAnalyzer")
    st.sidebar.markdown("---")
    
    if not st.session_state.authenticated:
        # Show only authentication
        authentication_page()
    else:
        # Show navigation
        st.sidebar.success("✅ Connected to Halo")
        
        page = st.sidebar.radio(
            "Navigation",
            [
                "🔬 Slide Selection",
                "🤖 Analysis",
                "📤 Export",
                "📥 Import",
                "⚙️ Settings"
            ]
        )
        
        st.sidebar.markdown("---")
        
        # Show current slide info
        if st.session_state.selected_slide:
            st.sidebar.info(f"**Current Slide:**\n{st.session_state.selected_slide['name']}")
        
        # Logout button
        if st.sidebar.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Route to pages
        if page == "🔬 Slide Selection":
            slide_selection_page()
        elif page == "🤖 Analysis":
            analysis_page()
        elif page == "📤 Export":
            export_page()
        elif page == "📥 Import":
            import_page()
        elif page == "⚙️ Settings":
            st.title("⚙️ Settings")
            st.info("Configuration settings coming soon")
            st.write(f"**Device:** {Config.DEVICE}")
            st.write(f"**Model:** {Config.MODEL_TYPE}")
            st.write(f"**Checkpoint:** {Config.MEDSAM_CHECKPOINT}")


if __name__ == "__main__":
    main()
