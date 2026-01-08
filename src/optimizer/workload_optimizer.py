"""
Workload Optimizer Module

This module implements ML-based workload prediction and optimization for GPU scheduling.
It uses historical telemetry data to predict resource needs and optimal placement strategies.

Based on research from:
- Google Autopilot (EuroSys '20): ML-based resource prediction
- Microsoft Gandiva (OSDI '18): Introspective scheduling for DL workloads
- NVIDIA MLPerf benchmarks: Workload characterization
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkloadType(Enum):
    """Types of GPU workloads with distinct resource characteristics."""
    TRAINING = "Training"
    INFERENCE = "Inference"
    FINE_TUNING = "FineTuning"
    DATA_PREPROCESSING = "DataPreprocessing"
    REINFORCEMENT = "Reinforcement"
    VISUALIZATION = "Visualization"


class MLFramework(Enum):
    """Supported ML frameworks with known performance profiles."""
    PYTORCH = "PyTorch"
    TENSORFLOW = "TensorFlow"
    JAX = "JAX"
    MXNET = "MXNet"
    TRITON = "Triton"


class DistributionStrategy(Enum):
    """Distribution strategies for multi-GPU training."""
    DATA_PARALLEL = "DataParallel"
    MODEL_PARALLEL = "ModelParallel"
    PIPELINE_PARALLEL = "PipelineParallel"
    HYBRID = "Hybrid"
    FSDP = "FSDP"
    DEEPSPEED = "DeepSpeed"


@dataclass
class WorkloadProfile:
    """
    Captures characteristic patterns of a workload for prediction.

    Based on features identified in "MLaaS in the Wild" (OSDI '22):
    - Memory access patterns
    - Compute utilization curves
    - Communication patterns for distributed workloads
    """
    workload_id: str
    workload_type: WorkloadType
    framework: Optional[MLFramework] = None

    # Resource usage patterns
    avg_gpu_utilization: float = 0.0
    peak_gpu_utilization: float = 0.0
    avg_memory_utilization: float = 0.0
    peak_memory_utilization: float = 0.0

    # Temporal patterns
    utilization_variance: float = 0.0
    memory_growth_rate: float = 0.0  # MB/s

    # Communication patterns (for distributed)
    avg_nvlink_bandwidth_usage: float = 0.0  # GB/s
    communication_compute_ratio: float = 0.0

    # Duration statistics
    avg_duration_seconds: float = 0.0
    duration_variance: float = 0.0

    # Feature vector for ML model
    feature_vector: List[float] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    sample_count: int = 0


@dataclass
class ResourcePrediction:
    """Predicted resource requirements from the optimizer."""
    recommended_gpus: int
    recommended_memory_gb: int
    expected_utilization: float
    expected_duration_seconds: float
    confidence: float

    # Placement hints
    prefer_nvlink: bool = False
    prefer_same_numa: bool = False
    recommended_mig_profile: Optional[str] = None

    # Cost estimation
    estimated_cost_per_hour: float = 0.0

    # Reasoning
    reasoning: List[str] = field(default_factory=list)


@dataclass
class PlacementHint:
    """Optimized placement recommendation."""
    preferred_nodes: List[str]
    preferred_gpus: Dict[str, List[str]]  # node -> gpu uuids
    score: int
    reasoning: str

    # Alternative placements
    alternatives: List['PlacementHint'] = field(default_factory=list)


@dataclass
class TelemetryDataPoint:
    """Single telemetry measurement from DCGM/NVML."""
    timestamp: datetime
    gpu_uuid: str
    gpu_utilization: float
    memory_used_bytes: int
    memory_total_bytes: int
    nvlink_bandwidth_gbps: float
    power_watts: float
    temperature_celsius: float
    workload_id: Optional[str] = None


class WorkloadClassifier:
    """
    Classifies workloads based on observed behavior patterns.

    Uses techniques from "Tiresias: A GPU Cluster Manager" (NSDI '19)
    to identify workload characteristics from early execution traces.
    """

    # Known workload signatures based on research
    WORKLOAD_SIGNATURES = {
        WorkloadType.TRAINING: {
            "min_gpu_util": 60,
            "memory_pattern": "growing",
            "duration_pattern": "long",
            "communication_heavy": True
        },
        WorkloadType.INFERENCE: {
            "min_gpu_util": 20,
            "memory_pattern": "stable",
            "duration_pattern": "short",
            "communication_heavy": False
        },
        WorkloadType.FINE_TUNING: {
            "min_gpu_util": 50,
            "memory_pattern": "growing",
            "duration_pattern": "medium",
            "communication_heavy": True
        },
        WorkloadType.DATA_PREPROCESSING: {
            "min_gpu_util": 30,
            "memory_pattern": "variable",
            "duration_pattern": "variable",
            "communication_heavy": False
        }
    }

    def __init__(self):
        self.history: Dict[str, List[TelemetryDataPoint]] = defaultdict(list)

    def add_sample(self, workload_id: str, datapoint: TelemetryDataPoint):
        """Add a telemetry sample for a workload."""
        datapoint.workload_id = workload_id
        self.history[workload_id].append(datapoint)

    def classify(self, workload_id: str) -> Tuple[WorkloadType, float]:
        """
        Classify a workload based on observed behavior.

        Returns:
            Tuple of (predicted WorkloadType, confidence score)
        """
        samples = self.history.get(workload_id, [])
        if len(samples) < 5:
            return WorkloadType.TRAINING, 0.3  # Default with low confidence

        # Extract features from samples
        gpu_utils = [s.gpu_utilization for s in samples]
        memory_utils = [s.memory_used_bytes / s.memory_total_bytes * 100 for s in samples]

        avg_gpu_util = np.mean(gpu_utils)
        memory_trend = self._calculate_trend(memory_utils)

        # Match against known signatures
        best_match = WorkloadType.TRAINING
        best_score = 0.0

        for wtype, signature in self.WORKLOAD_SIGNATURES.items():
            score = self._match_signature(
                avg_gpu_util, memory_trend, len(samples), signature
            )
            if score > best_score:
                best_score = score
                best_match = wtype

        return best_match, min(best_score, 0.95)

    def _calculate_trend(self, values: List[float]) -> str:
        """Determine if values are growing, stable, or variable."""
        if len(values) < 3:
            return "stable"

        diffs = np.diff(values)
        avg_diff = np.mean(diffs)
        variance = np.var(values)

        if avg_diff > 1.0:
            return "growing"
        elif variance > 100:
            return "variable"
        return "stable"

    def _match_signature(
        self,
        avg_gpu_util: float,
        memory_trend: str,
        sample_count: int,
        signature: dict
    ) -> float:
        """Calculate match score against a signature."""
        score = 0.0

        if avg_gpu_util >= signature["min_gpu_util"]:
            score += 0.3
        elif avg_gpu_util >= signature["min_gpu_util"] * 0.8:
            score += 0.15

        if memory_trend == signature["memory_pattern"]:
            score += 0.3

        # Duration inference from sample count
        duration_map = {"short": 10, "medium": 50, "long": 100}
        expected = duration_map.get(signature["duration_pattern"], 50)
        if abs(sample_count - expected) < expected * 0.3:
            score += 0.2

        # Confidence bonus for more samples
        score += min(sample_count / 100, 0.2)

        return score


class ResourcePredictor:
    """
    Predicts optimal resource allocation using ML techniques.

    Implements concepts from:
    - Google Autopilot: Vertical pod autoscaling
    - Microsoft Optimus: DL job resource optimization
    """

    # Historical benchmarks for different model sizes (parameters in billions)
    MODEL_RESOURCE_MAP = {
        # (min_params, max_params): (gpus, memory_gb, nvlink_required)
        (0, 0.1): (1, 10, False),      # Small models
        (0.1, 1): (1, 20, False),       # Medium models
        (1, 7): (1, 40, False),         # 7B class models
        (7, 13): (2, 40, True),         # 13B class models
        (13, 30): (4, 40, True),        # 30B class models
        (30, 70): (8, 80, True),        # 70B class models
        (70, 180): (16, 80, True),      # 180B class models
        (180, 500): (32, 80, True),     # 500B+ class models
    }

    # Framework-specific memory overhead factors
    FRAMEWORK_OVERHEAD = {
        MLFramework.PYTORCH: 1.0,
        MLFramework.TENSORFLOW: 1.1,
        MLFramework.JAX: 0.95,
        MLFramework.TRITON: 0.8,
    }

    # Distribution strategy efficiency factors
    STRATEGY_EFFICIENCY = {
        DistributionStrategy.DATA_PARALLEL: 0.85,
        DistributionStrategy.MODEL_PARALLEL: 0.75,
        DistributionStrategy.PIPELINE_PARALLEL: 0.80,
        DistributionStrategy.FSDP: 0.90,
        DistributionStrategy.DEEPSPEED: 0.92,
    }

    def __init__(self):
        self.profiles: Dict[str, WorkloadProfile] = {}
        self.classifier = WorkloadClassifier()

    def update_profile(
        self,
        workload_id: str,
        telemetry: List[TelemetryDataPoint]
    ) -> WorkloadProfile:
        """Update workload profile with new telemetry data."""
        profile = self.profiles.get(workload_id)
        if profile is None:
            wtype, _ = self.classifier.classify(workload_id)
            profile = WorkloadProfile(
                workload_id=workload_id,
                workload_type=wtype
            )
            self.profiles[workload_id] = profile

        # Update statistics
        gpu_utils = [t.gpu_utilization for t in telemetry]
        mem_utils = [t.memory_used_bytes / t.memory_total_bytes * 100 for t in telemetry]

        profile.avg_gpu_utilization = np.mean(gpu_utils)
        profile.peak_gpu_utilization = np.max(gpu_utils)
        profile.avg_memory_utilization = np.mean(mem_utils)
        profile.peak_memory_utilization = np.max(mem_utils)
        profile.utilization_variance = np.var(gpu_utils)

        if len(telemetry) >= 2:
            time_span = (telemetry[-1].timestamp - telemetry[0].timestamp).total_seconds()
            if time_span > 0:
                memory_diff = telemetry[-1].memory_used_bytes - telemetry[0].memory_used_bytes
                profile.memory_growth_rate = memory_diff / time_span / 1024 / 1024

        # Update feature vector for ML model
        profile.feature_vector = self._extract_features(profile, telemetry)
        profile.updated_at = datetime.now()
        profile.sample_count += len(telemetry)

        return profile

    def _extract_features(
        self,
        profile: WorkloadProfile,
        telemetry: List[TelemetryDataPoint]
    ) -> List[float]:
        """Extract feature vector for ML prediction."""
        features = [
            profile.avg_gpu_utilization / 100.0,
            profile.peak_gpu_utilization / 100.0,
            profile.avg_memory_utilization / 100.0,
            profile.peak_memory_utilization / 100.0,
            min(profile.utilization_variance / 1000.0, 1.0),
            min(abs(profile.memory_growth_rate) / 1000.0, 1.0),
            1.0 if profile.workload_type == WorkloadType.TRAINING else 0.0,
            1.0 if profile.workload_type == WorkloadType.INFERENCE else 0.0,
        ]

        # Temporal features
        if len(telemetry) >= 2:
            time_span = (telemetry[-1].timestamp - telemetry[0].timestamp).total_seconds()
            features.append(min(time_span / 3600.0, 1.0))  # Normalized duration
        else:
            features.append(0.0)

        return features

    def predict_resources(
        self,
        workload_id: str,
        model_params_billions: Optional[float] = None,
        framework: Optional[MLFramework] = None,
        strategy: Optional[DistributionStrategy] = None,
        batch_size: Optional[int] = None
    ) -> ResourcePrediction:
        """
        Predict optimal resource allocation for a workload.

        Uses a combination of:
        1. Historical profile data (if available)
        2. Model size heuristics
        3. Framework-specific adjustments
        """
        profile = self.profiles.get(workload_id)
        reasoning = []

        # Base prediction from model size
        gpus, memory_gb, nvlink_required = self._predict_from_model_size(
            model_params_billions
        )
        reasoning.append(
            f"Base allocation from model size ({model_params_billions}B params): "
            f"{gpus} GPUs, {memory_gb}GB memory"
        )

        # Adjust based on historical profile
        if profile and profile.sample_count >= 10:
            if profile.peak_memory_utilization > 90:
                memory_gb = int(memory_gb * 1.25)
                reasoning.append("Increased memory due to high historical usage")
            elif profile.peak_memory_utilization < 50:
                memory_gb = max(10, int(memory_gb * 0.75))
                reasoning.append("Reduced memory due to low historical usage")

            if profile.avg_gpu_utilization < 30 and profile.workload_type == WorkloadType.INFERENCE:
                # Good candidate for MIG
                reasoning.append("Low utilization inference workload - MIG recommended")

        # Framework overhead adjustment
        if framework and framework in self.FRAMEWORK_OVERHEAD:
            overhead = self.FRAMEWORK_OVERHEAD[framework]
            memory_gb = int(memory_gb * overhead)
            reasoning.append(f"Applied {framework.value} overhead factor: {overhead}")

        # Distribution strategy adjustment
        if strategy and gpus > 1:
            efficiency = self.STRATEGY_EFFICIENCY.get(strategy, 0.85)
            reasoning.append(
                f"Distribution strategy {strategy.value} efficiency: {efficiency:.0%}"
            )

        # Calculate expected utilization
        expected_util = self._estimate_utilization(profile, gpus)

        # Calculate expected duration
        expected_duration = self._estimate_duration(profile, gpus)

        # Calculate confidence
        confidence = self._calculate_confidence(profile)

        # Determine MIG recommendation
        mig_profile = None
        if profile and profile.avg_gpu_utilization < 40:
            if memory_gb <= 10:
                mig_profile = "1g.10gb"
            elif memory_gb <= 20:
                mig_profile = "2g.20gb"
            elif memory_gb <= 40:
                mig_profile = "3g.40gb"

        # Cost estimation (based on cloud GPU pricing)
        # Approximate H100 pricing: $3/GPU-hour
        cost_per_hour = gpus * 3.0

        return ResourcePrediction(
            recommended_gpus=gpus,
            recommended_memory_gb=memory_gb,
            expected_utilization=expected_util,
            expected_duration_seconds=expected_duration,
            confidence=confidence,
            prefer_nvlink=nvlink_required,
            prefer_same_numa=gpus <= 4,
            recommended_mig_profile=mig_profile,
            estimated_cost_per_hour=cost_per_hour,
            reasoning=reasoning
        )

    def _predict_from_model_size(
        self,
        model_params_billions: Optional[float]
    ) -> Tuple[int, int, bool]:
        """Predict resources from model parameter count."""
        if model_params_billions is None:
            return 1, 20, False

        for (min_p, max_p), (gpus, mem, nvlink) in self.MODEL_RESOURCE_MAP.items():
            if min_p <= model_params_billions < max_p:
                return gpus, mem, nvlink

        # Very large models
        return 64, 80, True

    def _estimate_utilization(
        self,
        profile: Optional[WorkloadProfile],
        gpus: int
    ) -> float:
        """Estimate expected GPU utilization."""
        if profile and profile.sample_count >= 5:
            # Use historical data
            base = profile.avg_gpu_utilization
            # Multi-GPU typically has lower per-GPU utilization
            if gpus > 1:
                base *= (0.85 ** (np.log2(gpus)))
            return min(base, 95.0)
        return 70.0  # Default estimate

    def _estimate_duration(
        self,
        profile: Optional[WorkloadProfile],
        gpus: int
    ) -> float:
        """Estimate expected duration in seconds."""
        if profile and profile.avg_duration_seconds > 0:
            # Scale by GPU count (sublinear scaling)
            return profile.avg_duration_seconds / (gpus ** 0.7)
        return 3600.0  # Default 1 hour

    def _calculate_confidence(self, profile: Optional[WorkloadProfile]) -> float:
        """Calculate prediction confidence based on available data."""
        if profile is None:
            return 0.3

        # More samples = higher confidence
        sample_factor = min(profile.sample_count / 100, 0.4)

        # Lower variance = higher confidence
        variance_factor = max(0, 0.3 - profile.utilization_variance / 1000)

        # Recency factor
        age = (datetime.now() - profile.updated_at).total_seconds() / 3600
        recency_factor = max(0, 0.3 - age / 24)

        return min(0.95, 0.3 + sample_factor + variance_factor + recency_factor)


class PlacementOptimizer:
    """
    Optimizes GPU placement based on topology and workload characteristics.

    Implements topology-aware scheduling concepts from:
    - "Efficient Large-Scale Language Model Training" (SC '21)
    - NVIDIA DGX best practices
    """

    def __init__(self, resource_predictor: ResourcePredictor):
        self.predictor = resource_predictor

    def get_optimal_placement(
        self,
        workload_id: str,
        cluster_topology: dict,
        requirements: dict
    ) -> PlacementHint:
        """
        Determine optimal GPU placement for a workload.

        Args:
            workload_id: Unique workload identifier
            cluster_topology: Current cluster GPU topology
            requirements: Workload resource requirements

        Returns:
            PlacementHint with preferred placement
        """
        gpu_count = requirements.get("gpu_count", 1)
        prefer_nvlink = requirements.get("prefer_nvlink", False)

        nodes = cluster_topology.get("nodes", {})
        if not nodes:
            return PlacementHint(
                preferred_nodes=[],
                preferred_gpus={},
                score=0,
                reasoning="No nodes available"
            )

        # Score each node
        node_scores = []
        for node_name, node_info in nodes.items():
            score, selected_gpus = self._score_node(
                node_info, gpu_count, prefer_nvlink
            )
            node_scores.append((node_name, score, selected_gpus))

        # Sort by score
        node_scores.sort(key=lambda x: x[1], reverse=True)

        if not node_scores or node_scores[0][1] == 0:
            return PlacementHint(
                preferred_nodes=[],
                preferred_gpus={},
                score=0,
                reasoning="No suitable placement found"
            )

        best_node, best_score, best_gpus = node_scores[0]

        # Build reasoning
        reasoning_parts = []
        if best_score >= 80:
            reasoning_parts.append("Optimal NVLink topology")
        elif best_score >= 60:
            reasoning_parts.append("Good topology with partial NVLink")
        else:
            reasoning_parts.append("PCIe-based placement")

        if gpu_count > 1:
            reasoning_parts.append(f"Allocated {gpu_count} GPUs on single node")

        # Build alternatives
        alternatives = []
        for node_name, score, gpus in node_scores[1:3]:
            if score > 0:
                alternatives.append(PlacementHint(
                    preferred_nodes=[node_name],
                    preferred_gpus={node_name: gpus},
                    score=score,
                    reasoning=f"Alternative placement (score: {score})"
                ))

        return PlacementHint(
            preferred_nodes=[best_node],
            preferred_gpus={best_node: best_gpus},
            score=best_score,
            reasoning=", ".join(reasoning_parts),
            alternatives=alternatives
        )

    def _score_node(
        self,
        node_info: dict,
        gpu_count: int,
        prefer_nvlink: bool
    ) -> Tuple[int, List[str]]:
        """Score a node for workload placement."""
        gpus = node_info.get("gpus", [])
        if len(gpus) < gpu_count:
            return 0, []

        # Filter available GPUs
        available = [g for g in gpus if g.get("health", {}).get("status") == "Healthy"]
        if len(available) < gpu_count:
            return 0, []

        score = 50  # Base score
        selected = []

        if gpu_count == 1:
            # Single GPU, pick healthiest with most free memory
            available.sort(
                key=lambda g: g.get("memory", {}).get("freeBytes", 0),
                reverse=True
            )
            selected = [available[0].get("uuid")]
            score = 80
        else:
            # Multi-GPU: find NVLink-connected group
            if prefer_nvlink or gpu_count >= 4:
                nvlink_group = self._find_nvlink_group(available, gpu_count)
                if len(nvlink_group) >= gpu_count:
                    selected = nvlink_group[:gpu_count]
                    score = 90

            if not selected:
                # Fallback: any available GPUs
                selected = [g.get("uuid") for g in available[:gpu_count]]
                score = 50

        return score, selected

    def _find_nvlink_group(
        self,
        gpus: List[dict],
        size: int
    ) -> List[str]:
        """Find a group of NVLink-connected GPUs."""
        # Build adjacency from NVLink connections
        adjacency = defaultdict(set)
        for gpu in gpus:
            uuid = gpu.get("uuid")
            nvlinks = gpu.get("topology", {}).get("nvlinkConnections", {})
            for peer_uuid in nvlinks.keys():
                adjacency[uuid].add(peer_uuid)
                adjacency[peer_uuid].add(uuid)

        # Greedy search for connected group
        for gpu in gpus:
            start = gpu.get("uuid")
            group = {start}
            candidates = list(adjacency[start])

            while len(group) < size and candidates:
                # Pick candidate with most connections to group
                best = max(
                    candidates,
                    key=lambda c: len(adjacency[c] & group)
                )
                if adjacency[best] & group:  # Must connect to group
                    group.add(best)
                    candidates.extend(
                        c for c in adjacency[best]
                        if c not in group and c not in candidates
                    )
                candidates.remove(best)

            if len(group) >= size:
                return list(group)[:size]

        return []


class WorkloadOptimizer:
    """
    Main optimizer class that combines prediction and placement optimization.

    This is the primary interface for the KGWE scheduler to get
    workload-aware optimization recommendations.
    """

    def __init__(self):
        self.predictor = ResourcePredictor()
        self.placement_optimizer = PlacementOptimizer(self.predictor)
        self.telemetry_buffer: Dict[str, List[TelemetryDataPoint]] = defaultdict(list)

    def ingest_telemetry(
        self,
        workload_id: str,
        datapoint: TelemetryDataPoint
    ):
        """Ingest telemetry data for continuous learning."""
        self.telemetry_buffer[workload_id].append(datapoint)
        self.predictor.classifier.add_sample(workload_id, datapoint)

        # Update profile periodically
        if len(self.telemetry_buffer[workload_id]) >= 10:
            self.predictor.update_profile(
                workload_id,
                self.telemetry_buffer[workload_id]
            )
            # Keep last 100 samples
            self.telemetry_buffer[workload_id] = \
                self.telemetry_buffer[workload_id][-100:]

    def predict_resources(
        self,
        workload_id: str,
        hints: Optional[dict] = None
    ) -> ResourcePrediction:
        """
        Predict optimal resources for a workload.

        Args:
            workload_id: Unique workload identifier
            hints: Optional hints (model_size, framework, etc.)

        Returns:
            ResourcePrediction with recommendations
        """
        hints = hints or {}
        return self.predictor.predict_resources(
            workload_id,
            model_params_billions=hints.get("model_params_billions"),
            framework=hints.get("framework"),
            strategy=hints.get("strategy"),
            batch_size=hints.get("batch_size")
        )

    def get_optimal_placement(
        self,
        workload_id: str,
        cluster_topology: dict,
        requirements: dict
    ) -> PlacementHint:
        """
        Get optimal GPU placement for a workload.

        Args:
            workload_id: Unique workload identifier
            cluster_topology: Current cluster topology
            requirements: Resource requirements

        Returns:
            PlacementHint with placement recommendation
        """
        return self.placement_optimizer.get_optimal_placement(
            workload_id, cluster_topology, requirements
        )

    def get_profile(self, workload_id: str) -> Optional[WorkloadProfile]:
        """Get the current profile for a workload."""
        return self.predictor.profiles.get(workload_id)

    def export_metrics(self) -> dict:
        """Export optimizer metrics for monitoring."""
        return {
            "total_profiles": len(self.predictor.profiles),
            "total_samples": sum(
                len(buf) for buf in self.telemetry_buffer.values()
            ),
            "profiles": {
                wid: {
                    "workload_type": p.workload_type.value,
                    "avg_gpu_util": p.avg_gpu_utilization,
                    "sample_count": p.sample_count,
                    "updated_at": p.updated_at.isoformat()
                }
                for wid, p in self.predictor.profiles.items()
            }
        }


# gRPC service interface for integration with Go components
class OptimizerService:
    """
    gRPC-compatible service for the workload optimizer.

    Exposes optimization capabilities to the Go-based scheduler.
    """

    def __init__(self):
        self.optimizer = WorkloadOptimizer()

    def PredictResources(self, request: dict) -> dict:
        """Handle resource prediction request."""
        workload_id = request.get("workload_id", "unknown")
        hints = request.get("hints", {})

        prediction = self.optimizer.predict_resources(workload_id, hints)

        return {
            "recommended_gpus": prediction.recommended_gpus,
            "recommended_memory_gb": prediction.recommended_memory_gb,
            "expected_utilization": prediction.expected_utilization,
            "expected_duration_seconds": prediction.expected_duration_seconds,
            "confidence": prediction.confidence,
            "prefer_nvlink": prediction.prefer_nvlink,
            "recommended_mig_profile": prediction.recommended_mig_profile,
            "estimated_cost_per_hour": prediction.estimated_cost_per_hour,
            "reasoning": prediction.reasoning
        }

    def GetPlacement(self, request: dict) -> dict:
        """Handle placement optimization request."""
        workload_id = request.get("workload_id", "unknown")
        topology = request.get("cluster_topology", {})
        requirements = request.get("requirements", {})

        hint = self.optimizer.get_optimal_placement(
            workload_id, topology, requirements
        )

        return {
            "preferred_nodes": hint.preferred_nodes,
            "preferred_gpus": hint.preferred_gpus,
            "score": hint.score,
            "reasoning": hint.reasoning,
            "alternatives": [
                {
                    "preferred_nodes": alt.preferred_nodes,
                    "score": alt.score
                }
                for alt in hint.alternatives
            ]
        }

    def IngestTelemetry(self, request: dict) -> dict:
        """Handle telemetry ingestion."""
        workload_id = request.get("workload_id", "unknown")
        datapoints = request.get("datapoints", [])

        for dp in datapoints:
            self.optimizer.ingest_telemetry(
                workload_id,
                TelemetryDataPoint(
                    timestamp=datetime.fromisoformat(dp["timestamp"]),
                    gpu_uuid=dp["gpu_uuid"],
                    gpu_utilization=dp["gpu_utilization"],
                    memory_used_bytes=dp["memory_used_bytes"],
                    memory_total_bytes=dp["memory_total_bytes"],
                    nvlink_bandwidth_gbps=dp.get("nvlink_bandwidth_gbps", 0),
                    power_watts=dp.get("power_watts", 0),
                    temperature_celsius=dp.get("temperature_celsius", 0)
                )
            )

        return {"status": "ok", "samples_ingested": len(datapoints)}

    def GetMetrics(self, request: dict) -> dict:
        """Return optimizer metrics."""
        return self.optimizer.export_metrics()


if __name__ == "__main__":
    # Example usage and testing
    optimizer = WorkloadOptimizer()

    # Simulate telemetry ingestion
    for i in range(20):
        optimizer.ingest_telemetry(
            "training-job-001",
            TelemetryDataPoint(
                timestamp=datetime.now() - timedelta(minutes=20-i),
                gpu_uuid="GPU-abc123",
                gpu_utilization=70 + np.random.normal(0, 5),
                memory_used_bytes=int(60e9 + np.random.normal(0, 1e9)),
                memory_total_bytes=int(80e9),
                nvlink_bandwidth_gbps=400,
                power_watts=350,
                temperature_celsius=65
            )
        )

    # Get resource prediction
    prediction = optimizer.predict_resources(
        "training-job-001",
        hints={
            "model_params_billions": 7.0,
            "framework": MLFramework.PYTORCH,
            "strategy": DistributionStrategy.FSDP
        }
    )

    print("Resource Prediction:")
    print(f"  GPUs: {prediction.recommended_gpus}")
    print(f"  Memory: {prediction.recommended_memory_gb} GB")
    print(f"  Expected Utilization: {prediction.expected_utilization:.1f}%")
    print(f"  Confidence: {prediction.confidence:.2f}")
    print(f"  Cost: ${prediction.estimated_cost_per_hour}/hour")
    print(f"  Reasoning: {prediction.reasoning}")

    # Simulate cluster topology
    cluster_topology = {
        "nodes": {
            "gpu-node-1": {
                "gpus": [
                    {
                        "uuid": f"GPU-{i}",
                        "health": {"status": "Healthy"},
                        "memory": {"freeBytes": 70e9},
                        "topology": {
                            "nvlinkConnections": {
                                f"GPU-{(i+1)%8}": {"bandwidthGBps": 900}
                            }
                        }
                    }
                    for i in range(8)
                ]
            }
        }
    }

    # Get placement hint
    hint = optimizer.get_optimal_placement(
        "training-job-001",
        cluster_topology,
        {"gpu_count": 4, "prefer_nvlink": True}
    )

    print("\nPlacement Hint:")
    print(f"  Node: {hint.preferred_nodes}")
    print(f"  GPUs: {hint.preferred_gpus}")
    print(f"  Score: {hint.score}")
    print(f"  Reasoning: {hint.reasoning}")
