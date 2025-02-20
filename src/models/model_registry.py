"""
Model registry for managing and versioning machine learning models.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, registry_path: str = "models"):
        """
        Initialize the model registry.
        
        Args:
            registry_path: Path to store model files and metadata
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / "metadata.json"
        self._load_metadata()
        
    def _load_metadata(self):
        """Load model metadata from JSON file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}
            
    def _save_metadata(self):
        """Save model metadata to JSON file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            
    def register_model(
        self,
        model_id: str,
        model_type: str,
        version: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        path: str
    ) -> bool:
        """
        Register a new model or update existing model metadata.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model (e.g., 'sentiment', 'price_prediction')
            version: Model version string
            params: Model parameters and hyperparameters
            metrics: Model performance metrics
            path: Path to the model file
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            model_info = {
                'model_type': model_type,
                'version': version,
                'params': params,
                'metrics': metrics,
                'path': path,
                'registered_at': datetime.now().isoformat(),
                'last_used': None,
                'is_active': True
            }
            
            if model_id not in self.metadata:
                self.metadata[model_id] = {'versions': {}}
                
            self.metadata[model_id]['versions'][version] = model_info
            self._save_metadata()
            
            logger.info(f"Registered model {model_id} version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {model_id}: {e}")
            return False
            
    def get_model_info(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get model metadata.
        
        Args:
            model_id: Model identifier
            version: Optional specific version, latest if None
            
        Returns:
            Model metadata dictionary or None if not found
        """
        try:
            if model_id not in self.metadata:
                return None
                
            versions = self.metadata[model_id]['versions']
            if not versions:
                return None
                
            if version is None:
                # Get latest version
                version = max(versions.keys())
                
            return versions.get(version)
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_id}: {e}")
            return None
            
    def list_models(
        self,
        model_type: Optional[str] = None,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List registered models with optional filtering.
        
        Args:
            model_type: Optional filter by model type
            active_only: Only return active models if True
            
        Returns:
            List of model metadata dictionaries
        """
        try:
            models = []
            for model_id, model_data in self.metadata.items():
                for version, info in model_data['versions'].items():
                    if model_type and info['model_type'] != model_type:
                        continue
                    if active_only and not info['is_active']:
                        continue
                    models.append({
                        'model_id': model_id,
                        'version': version,
                        **info
                    })
            return models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
            
    def deactivate_model(self, model_id: str, version: str) -> bool:
        """
        Deactivate a model version.
        
        Args:
            model_id: Model identifier
            version: Model version
            
        Returns:
            True if deactivation successful, False otherwise
        """
        try:
            if (model_id not in self.metadata or
                version not in self.metadata[model_id]['versions']):
                return False
                
            self.metadata[model_id]['versions'][version]['is_active'] = False
            self._save_metadata()
            
            logger.info(f"Deactivated model {model_id} version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deactivate model {model_id}: {e}")
            return False
            
    def update_last_used(self, model_id: str, version: str) -> bool:
        """
        Update the last used timestamp for a model.
        
        Args:
            model_id: Model identifier
            version: Model version
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            if (model_id not in self.metadata or
                version not in self.metadata[model_id]['versions']):
                return False
                
            self.metadata[model_id]['versions'][version]['last_used'] = datetime.now().isoformat()
            self._save_metadata()
            return True
            
        except Exception as e:
            logger.error(f"Failed to update last used for {model_id}: {e}")
            return False
            
    def cleanup_old_models(self, max_age_days: int = 90) -> int:
        """
        Deactivate models that haven't been used recently.
        
        Args:
            max_age_days: Maximum age in days before deactivation
            
        Returns:
            Number of models deactivated
        """
        try:
            deactivated = 0
            cutoff = datetime.now() - timedelta(days=max_age_days)
            
            for model_id, model_data in self.metadata.items():
                for version, info in model_data['versions'].items():
                    if not info['is_active']:
                        continue
                        
                    last_used = info.get('last_used')
                    if last_used:
                        last_used = datetime.fromisoformat(last_used)
                        if last_used < cutoff:
                            if self.deactivate_model(model_id, version):
                                deactivated += 1
                                
            return deactivated
            
        except Exception as e:
            logger.error(f"Failed to cleanup old models: {e}")
            return 0
