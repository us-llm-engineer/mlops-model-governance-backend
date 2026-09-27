"""
MLOps interop framework: bridges to external systems.

MlflowBridge mirrors registry/run state to MLflow. (casbin is no longer here: it is a
test-time oracle for the policy layer, see tests/oracle/casbin_oracle.py.)
"""

from mlops.interop.mlflow_bridge import MlflowBridge

__all__ = ["MlflowBridge"]
