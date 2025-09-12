"""
Common utilities shared across CommentSense modules
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)

def load_config(config_source: Union[str, Path, Dict[str, Any]], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load configuration from file or dict, merging with default configuration.
    
    Args:
        config_source: Configuration file path, Path object, or dict
        default_config: Default configuration to merge with
        
    Returns:
        Merged configuration dictionary
    """
    try:
        if isinstance(config_source, dict):
            # Config passed as dictionary directly
            config = config_source.copy()
        else:
            # Config passed as file path
            config_path = Path(config_source)
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
            else:
                logger.warning(f"Configuration file not found: {config_path}, using defaults")
                config = {}
        
        # Deep merge with default configuration
        merged_config = _deep_merge(default_config.copy(), config)
        return merged_config
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}, using defaults")
        return default_config.copy()

def _deep_merge(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with update_dict taking precedence.
    
    Args:
        base_dict: Base dictionary
        update_dict: Dictionary to merge in (takes precedence)
        
    Returns:
        Merged dictionary
    """
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            base_dict[key] = _deep_merge(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict
