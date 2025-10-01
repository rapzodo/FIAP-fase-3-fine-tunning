#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import streamlit as st
from typing import Callable, List, Dict, Any, Optional

def stream_with_progress(file_handle, max_samples=None, min_content_length=10, process_function=None):
    """
    Process a file with a progress bar.
    This function is extracted for future use and not currently used in the main app.

    Args:
        file_handle: The open file handle
        max_samples: Maximum number of samples to process
        min_content_length: Minimum content length to include
        process_function: Function to process each item

    Returns:
        List of processed items
    """
    data = []

    # Get file size for progress calculation
    file_handle.seek(0, os.SEEK_END)
    file_size = file_handle.tell()
    file_handle.seek(0)

    # Check first character to determine if it's a JSON array
    first_char = file_handle.read(1)
    file_handle.seek(0)

    progress_bar = st.progress(0)

    if first_char == '[':
        # Process as JSON array with ijson
        import ijson
        items = ijson.items(file_handle, 'item')

        i = 0
        for item in items:
            if max_samples and i >= max_samples:
                break

            if process_function:
                processed = process_function(item)
                if processed:
                    data.append(processed)

            i += 1
            if i % 100 == 0:
                progress = min(file_handle.tell() / file_size, 1.0)
                progress_bar.progress(progress)
    else:
        # Process line by line
        import json
        i = 0
        for line in file_handle:
            if max_samples and i >= max_samples:
                break

            try:
                item = json.loads(line.strip())
                if process_function:
                    processed = process_function(item)
                    if processed:
                        data.append(processed)
            except json.JSONDecodeError:
                continue

            i += 1
            if i % 100 == 0:
                progress = min(file_handle.tell() / file_size, 1.0)
                progress_bar.progress(progress)

    progress_bar.progress(1.0)
    return data

def process_json_with_progress(json_data: List[Dict[str, Any]],
                              process_function: Callable[[Dict[str, Any]], Any],
                              max_samples: Optional[int] = None,
                              update_frequency: int = 100) -> List[Any]:
    """
    Process a JSON array with a streamlit progress bar.

    Args:
        json_data: List of JSON items to process
        process_function: Function to process each item, should take an item and return processed item or None
        max_samples: Maximum number of samples to process (None for all)
        update_frequency: How often to update the progress bar

    Returns:
        List of processed items
    """
    progress_bar = st.progress(0)
    data = []
    total = len(json_data)

    for i, item in enumerate(json_data):
        if max_samples and i >= max_samples:
            break

        result = process_function(item)
        if result is not None:
            data.append(result)

        if i % update_frequency == 0:
            progress_bar.progress(min(i / total, 1.0))

    progress_bar.progress(1.0)
    return data

def iterate_with_progress(items: List[Any],
                         process_function: Callable[[Any, int], Any],
                         description: str = "Processing",
                         update_frequency: int = 100) -> List[Any]:
    """
    General purpose iterator with progress bar.

    Args:
        items: List of items to iterate over
        process_function: Function to apply to each item, takes (item, index) and returns result
        description: Description to display during processing
        update_frequency: How often to update the progress bar

    Returns:
        List of results
    """
    with st.spinner(description):
        progress_bar = st.progress(0)
        results = []
        total = len(items)

        for i, item in enumerate(items):
            results.append(process_function(item, i))

            if i % update_frequency == 0:
                progress_bar.progress(min(i / total, 1.0))

        progress_bar.progress(1.0)

    return results
