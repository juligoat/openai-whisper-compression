from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""

    model_id: str
    num_samples: int
    model_nickname: Optional[str] = None

    def get_model_name(self) -> str:
        """Get a clean model name for file naming"""
        return self.model_nickname or self.model_id.split("/")[-1]


class ProjectConfig:
    """Configuration class to manage paths and settings"""

    def __init__(self):
        # Get the project root (parent of the directory containing this file)
        self.project_root = Path(__file__).parent.parent.absolute()

        # Create timestamp for unique run identification
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Define directory structure
        self.dirs = {
            "models": self.project_root / "models",
            "results": self.project_root / "results",
            "plots": self.project_root / "results" / "plots",
            "logs": self.project_root / "logs",
        }

        # Create all directories
        self.ensure_dirs_exist()

    def ensure_dirs_exist(self):
        """Create necessary directories if they don't exist"""
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_run_dir(self, model_name: str, num_samples: int) -> Path:
        """Get directory for this specific run

        Args:
            model_name: Name of the model
            num_samples: Number of samples used

        Returns:
            Path object for the run directory
        """
        return self.dirs["results"] / f"{model_name}_samples{num_samples}_{self.timestamp}"

    def get_model_dir(self, model_name: str, num_samples: int) -> Path:
        """Get directory for saving model artifacts

        Args:
            model_name: Name of the model
            num_samples: Number of samples used

        Returns:
            Path object for the model directory
        """
        return self.dirs["models"] / f"{model_name}_samples{num_samples}_{self.timestamp}"
