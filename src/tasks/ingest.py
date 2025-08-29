"""Ingestion task implementation.

Reads the reviews CSV file, loads associated images, performs basic validation 
and deduplication, and outputs raw parquet files ready for normalization.
"""

import pandas as pd
import numpy as np
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image

from ..orchestrator import task
from ..orchestrator.logging import get_logger


@task(
    name="ingest_all",
    inputs=lambda params: [
        "configs/base.yaml",
        params['ingest']['sources'][0]['path']
    ],
    outputs=[
        "data/raw/restaurant_reviews/reviews_text_only.parquet",
        "data/raw/restaurant_reviews/reviews_with_images.parquet", 
        "data/raw/restaurant_reviews/image_arrays.pkl"
    ],
)
def ingest_all(params: Dict[str, Any]):
    """Read source CSV, load images, deduplicate, and write to raw parquet format.
    
    Creates two datasets:
    1. Text-only reviews for fast processing
    2. Reviews with image metadata for multimodal analysis
    """
    logger = get_logger("ingest_all")
    logger.info("Starting data ingestion")
    
    # Get paths from config
    source_config = params['ingest']['sources'][0]
    csv_path = Path(source_config['path'])
    dataset_slug = params['dataset']['slug']
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Source CSV not found: {csv_path}")
    
    logger.info(f"Reading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} raw reviews")
    
    # Basic data validation
    required_columns = ['business_name', 'author_name', 'text', 'rating']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Use config for image processing
    image_config = params.get('ingest', {}).get('image', {})
    resize_dims = tuple(image_config.get('resize_dimensions', [224, 224]))
    img_format = image_config.get('format', 'RGB')
    
    # Load images and create unified schema
    df_processed, image_arrays = _load_images_and_create_schema(
        df, logger, resize_dims, img_format, params
    )
    
    # Deduplicate reviews
    df_deduplicated, image_arrays_deduplicated = _deduplicate_reviews_with_images(
        df_processed, image_arrays, logger
    )
    
    # Split into text-only and with-images datasets
    df_text_only, df_with_images = _split_datasets(df_deduplicated, logger)
    
    # Write outputs
    _write_outputs(df_text_only, df_with_images, image_arrays_deduplicated, logger, dataset_slug)
    
    logger.info("Ingestion completed successfully")


def _load_images_and_create_schema(
    df: pd.DataFrame, 
    logger, 
    resize_dims: Tuple[int, int],
    img_format: str,
    params: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Load images and transform the CSV data into a unified schema format."""
    logger.info("Loading images and creating unified schema")
    
    # Get image base path from config or use default
    image_base_path = Path(params.get('ingest', {}).get('image', {}).get('base_path', 'data/archive/dataset'))
    
    # Create a hash for each review as a unique identifier
    def create_review_id(row):
        content = f"{row['business_name']}{row['author_name']}{row['text']}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
    
    # Load images and store as numpy arrays
    image_arrays = {}
    image_load_stats = {'found': 0, 'missing': 0, 'error': 0}
    
    for idx, row in df.iterrows():
        review_id = create_review_id(row)
        image_path = row.get('photo', '')
        
        if pd.isna(image_path) or image_path == '':
            image_load_stats['missing'] += 1
            continue
            
        # Construct full path using config
        full_image_path = image_base_path / image_path
        
        if full_image_path.exists():
            try:
                with Image.open(full_image_path) as img:
                    if img.mode != img_format:
                        img = img.convert(img_format)
                    
                    # Use config dimensions
                    img_resized = img.resize(resize_dims, Image.Resampling.LANCZOS)
                    
                    # Convert to numpy array
                    img_array = np.array(img_resized, dtype=np.uint8)
                    image_arrays[review_id] = img_array
                    
                image_load_stats['found'] += 1
                
            except Exception as e:
                logger.warning(f"Error loading image {full_image_path}: {e}")
                image_load_stats['error'] += 1
        else:
            image_load_stats['missing'] += 1
    
    logger.info(f"Image loading stats: {image_load_stats}")
    
    # Map to unified schema
    df_unified = pd.DataFrame({
        'review_id': df.apply(create_review_id, axis=1),
        'place_name': df['business_name'].str.lower().str.replace(' ', '_').str.replace('-', '_'),
        'user_name': df['author_name'].str.lower().str.replace(' ', '_'),
        'text': df['text'].fillna(''),
        'language': 'en',  # All reviews appear to be in English
        'rating': df['rating'].astype(int),
        # 'created_at': None,  # Not available in this dataset
        'has_image': df.apply(lambda row: create_review_id(row) in image_arrays, axis=1),
        'image_path': df['photo'].fillna(''),
        'metadata': df.apply(lambda row: {
            'source': 'restaurant_reviews',
            'business_name': row['business_name'],
            'author_name': row['author_name'],
            'photo': row.get('photo', ''),
            'rating_category': row.get('rating_category', ''),
            'original_business_name': row['business_name']
        }, axis=1)
    })
    
    logger.info(f"Created unified schema with {len(df_unified)} records, {len(image_arrays)} with images")
    return df_unified, image_arrays


def _deduplicate_reviews_with_images(
    df: pd.DataFrame, 
    image_arrays: Dict[str, np.ndarray], 
    logger
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Remove duplicate reviews and corresponding image arrays."""
    logger.info("Starting deduplication")
    
    initial_count = len(df)
    initial_image_count = len(image_arrays)
    
    # Create content hash for deduplication
    df['text_hash'] = df['text'].apply(
        lambda x: hashlib.md5(str(x).encode('utf-8')).hexdigest()
    )
    
    # Remove duplicates based on (place_name, user_name, text_hash)
    df_deduplicated = df.drop_duplicates(
        subset=['place_name', 'user_name', 'text_hash'], 
        keep='first'
    )
    
    # Get the review_ids that were kept
    kept_review_ids = set(df_deduplicated['review_id'].values)
    
    # Filter image arrays to only include kept reviews
    image_arrays_deduplicated = {
        review_id: img_array 
        for review_id, img_array in image_arrays.items()
        if review_id in kept_review_ids
    }
    
    # Remove the temporary text_hash column
    df_deduplicated = df_deduplicated.drop('text_hash', axis=1)
    
    duplicates_removed = initial_count - len(df_deduplicated)
    images_removed = initial_image_count - len(image_arrays_deduplicated)
    
    logger.info(f"Removed {duplicates_removed} duplicate reviews")
    logger.info(f"Removed {images_removed} duplicate images")
    logger.info(f"Final dataset contains {len(df_deduplicated)} unique reviews")
    logger.info(f"Final image set contains {len(image_arrays_deduplicated)} images")
    
    return df_deduplicated, image_arrays_deduplicated


def _split_datasets(df: pd.DataFrame, logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into text-only and with-images datasets."""
    logger.info("Splitting datasets")
    
    # Text-only dataset (all reviews, but without image-specific columns)
    df_text_only = df.drop(['has_image', 'image_path'], axis=1).copy()
    
    # With-images dataset (only reviews that have images)
    df_with_images = df[df['has_image'] == True].copy()
    
    logger.info(f"Text-only dataset: {len(df_text_only)} reviews")
    logger.info(f"With-images dataset: {len(df_with_images)} reviews")
    
    return df_text_only, df_with_images


def _write_outputs(
    df_text_only: pd.DataFrame,
    df_with_images: pd.DataFrame,
    image_arrays: Dict[str, np.ndarray],
    logger,
    dataset_slug: str
):
    """Write all outputs to their respective files."""
    logger.info("Writing outputs")
    
    # Create output directory - remove 'src/' prefix to match README convention
    output_dir = Path(f"data/raw/{dataset_slug}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write outputs using config-derived paths
    text_only_path = output_dir / "reviews_text_only.parquet"
    with_images_path = output_dir / "reviews_with_images.parquet"
    images_path = output_dir / "image_arrays.pkl"
    
    # Write text-only dataset
    df_text_only.to_parquet(text_only_path, index=False)
    logger.info(f"Wrote text-only dataset to {text_only_path}")
    
    # Write with-images dataset
    df_with_images.to_parquet(with_images_path, index=False)
    logger.info(f"Wrote with-images dataset to {with_images_path}")
    
    # Write image arrays as pickle file
    with open(images_path, 'wb') as f:
        pickle.dump(image_arrays, f)
    logger.info(f"Wrote {len(image_arrays)} image arrays to {images_path}")
    
    # Write summary stats
    summary = {
        'total_reviews': len(df_text_only),
        'reviews_with_images': len(df_with_images),
        'image_coverage': len(df_with_images) / len(df_text_only) if len(df_text_only) > 0 else 0,
        'unique_places': df_text_only['place_name'].nunique(),
        'unique_users': df_text_only['user_name'].nunique(),
    }
    
    logger.info(f"Ingestion summary: {summary}")