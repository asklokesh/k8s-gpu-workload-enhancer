// Package scheduler implements topology-aware GPU scheduling for Kubernetes.
// It extends the default Kubernetes scheduler with GPU-specific optimizations
// based on NVIDIA hardware topology, NVLink connectivity, and workload characteristics.
package scheduler

import (
	"time"

	"github.com/nvidia/kgwe/src/discovery"
)

// GPUWorkload represents a GPU workload specification
type GPUWorkload struct {
	// Metadata
	Name      string `json:"name"`
	Namespace string `json:"namespace"`
	UID       string `json:"uid"`

	// GPU Requirements
	Requirements GPURequirements `json:"requirements"`

	// Workload characteristics
	WorkloadSpec WorkloadSpec `json:"workloadSpec"`

	// Scheduling constraints
	Constraints SchedulingConstraints `json:"constraints"`

	// Priority for preemption decisions
	Priority int32 `json:"priority"`

	// Creation timestamp
	CreatedAt time.Time `json:"createdAt"`
}

// GPURequirements specifies GPU resource requirements
type GPURequirements struct {
	// GPUCount is the number of GPUs required
	GPUCount int `json:"gpuCount"`

	// MinMemoryGB is minimum GPU memory per device
	MinMemoryGB int `json:"minMemoryGb"`

	// MinBandwidthGBps is minimum inter-GPU bandwidth
	MinBandwidthGBps float64 `json:"minBandwidthGbps"`

	// TopologyPreference indicates topology requirements
	TopologyPreference TopologyPreference `json:"topologyPreference"`

	// MIGRequirements for MIG-based allocation
	MIGRequirements *MIGRequirements `json:"migRequirements,omitempty"`

	// GPUArchitecture specifies required architecture
	GPUArchitecture discovery.GPUArchitecture `json:"gpuArchitecture,omitempty"`

	// GPUModel specifies exact GPU model (e.g., "H100")
	GPUModel string `json:"gpuModel,omitempty"`
}

// TopologyPreference specifies GPU topology requirements
type TopologyPreference string

const (
	// TopologyPreferenceNone - no topology preference
	TopologyPreferenceNone TopologyPreference = "None"

	// TopologyPreferenceNVLinkOptimal - prefer full NVLink connectivity
	TopologyPreferenceNVLinkOptimal TopologyPreference = "NVLinkOptimal"

	// TopologyPreferenceNVLinkRequired - require NVLink connectivity
	TopologyPreferenceNVLinkRequired TopologyPreference = "NVLinkRequired"

	// TopologyPreferenceSameNUMA - prefer same NUMA node
	TopologyPreferenceSameNUMA TopologyPreference = "SameNUMA"

	// TopologyPreferenceSamePCIeSwitch - prefer same PCIe switch
	TopologyPreferenceSamePCIeSwitch TopologyPreference = "SamePCIeSwitch"
)

// MIGRequirements specifies MIG instance requirements
type MIGRequirements struct {
	// Profile is the required MIG profile
	Profile string `json:"profile"`

	// Count is the number of MIG instances required
	Count int `json:"count"`

	// AllowPartialPlacement allows placing across multiple GPUs
	AllowPartialPlacement bool `json:"allowPartialPlacement"`
}

// WorkloadSpec describes workload characteristics
type WorkloadSpec struct {
	// Type categorizes the workload
	Type WorkloadType `json:"type"`

	// Framework identifies the ML framework
	Framework MLFramework `json:"framework,omitempty"`

	// DistributedConfig for distributed training
	DistributedConfig *DistributedConfig `json:"distributedConfig,omitempty"`

	// ExpectedDuration is estimated runtime
	ExpectedDuration time.Duration `json:"expectedDuration,omitempty"`

	// IsPreemptible indicates if workload can be preempted
	IsPreemptible bool `json:"isPreemptible"`

	// MemoryProfile describes memory access patterns
	MemoryProfile MemoryProfile `json:"memoryProfile,omitempty"`
}

// WorkloadType categorizes GPU workloads
type WorkloadType string

const (
	WorkloadTypeTraining           WorkloadType = "Training"
	WorkloadTypeInference          WorkloadType = "Inference"
	WorkloadTypeFineTuning         WorkloadType = "FineTuning"
	WorkloadTypeDataPreprocessing  WorkloadType = "DataPreprocessing"
	WorkloadTypeReinforcement      WorkloadType = "Reinforcement"
	WorkloadTypeVisualization      WorkloadType = "Visualization"
)

// MLFramework identifies machine learning frameworks
type MLFramework string

const (
	FrameworkPyTorch    MLFramework = "PyTorch"
	FrameworkTensorFlow MLFramework = "TensorFlow"
	FrameworkJAX        MLFramework = "JAX"
	FrameworkMXNet      MLFramework = "MXNet"
	FrameworkTriton     MLFramework = "Triton"
)

// DistributedConfig specifies distributed training configuration
type DistributedConfig struct {
	// Strategy is the distribution strategy
	Strategy DistributionStrategy `json:"strategy"`

	// WorldSize is the total number of processes
	WorldSize int `json:"worldSize"`

	// LocalRank is this process's local rank
	LocalRank int `json:"localRank"`

	// MasterAddr is the master node address
	MasterAddr string `json:"masterAddr,omitempty"`

	// MasterPort is the master node port
	MasterPort int `json:"masterPort,omitempty"`

	// Backend is the communication backend
	Backend CommunicationBackend `json:"backend"`
}

// DistributionStrategy for distributed training
type DistributionStrategy string

const (
	StrategyDataParallel  DistributionStrategy = "DataParallel"
	StrategyModelParallel DistributionStrategy = "ModelParallel"
	StrategyPipelineParallel DistributionStrategy = "PipelineParallel"
	StrategyHybrid        DistributionStrategy = "Hybrid"
	StrategyFSDP          DistributionStrategy = "FSDP"
	StrategyDeepSpeed     DistributionStrategy = "DeepSpeed"
)

// CommunicationBackend for distributed training
type CommunicationBackend string

const (
	BackendNCCL  CommunicationBackend = "NCCL"
	BackendGloo  CommunicationBackend = "Gloo"
	BackendMPI   CommunicationBackend = "MPI"
)

// MemoryProfile describes memory usage patterns
type MemoryProfile string

const (
	MemoryProfileLow      MemoryProfile = "Low"       // < 30% GPU memory
	MemoryProfileMedium   MemoryProfile = "Medium"    // 30-70% GPU memory
	MemoryProfileHigh     MemoryProfile = "High"      // 70-90% GPU memory
	MemoryProfileMaximum  MemoryProfile = "Maximum"   // > 90% GPU memory
)

// SchedulingConstraints specifies scheduling constraints
type SchedulingConstraints struct {
	// NodeSelector for node selection
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`

	// NodeAffinity rules
	NodeAffinity *NodeAffinity `json:"nodeAffinity,omitempty"`

	// Tolerations for taints
	Tolerations []Toleration `json:"tolerations,omitempty"`

	// ExcludeNodes lists nodes to exclude
	ExcludeNodes []string `json:"excludeNodes,omitempty"`

	// ColocateWith specifies pods to colocate with
	ColocateWith []PodReference `json:"colocateWith,omitempty"`

	// AntiAffinity specifies pods to avoid
	AntiAffinity []PodReference `json:"antiAffinity,omitempty"`

	// MaxSchedulingLatency is maximum acceptable wait time
	MaxSchedulingLatency time.Duration `json:"maxSchedulingLatency,omitempty"`
}

// NodeAffinity specifies node affinity rules
type NodeAffinity struct {
	// RequiredDuringScheduling is hard requirement
	RequiredDuringScheduling []NodeSelectorTerm `json:"requiredDuringScheduling,omitempty"`

	// PreferredDuringScheduling is soft preference
	PreferredDuringScheduling []WeightedNodeSelectorTerm `json:"preferredDuringScheduling,omitempty"`
}

// NodeSelectorTerm is a node selector term
type NodeSelectorTerm struct {
	MatchExpressions []NodeSelectorRequirement `json:"matchExpressions"`
}

// WeightedNodeSelectorTerm is a weighted node selector
type WeightedNodeSelectorTerm struct {
	Weight   int32            `json:"weight"`
	Selector NodeSelectorTerm `json:"selector"`
}

// NodeSelectorRequirement is a node selector requirement
type NodeSelectorRequirement struct {
	Key      string   `json:"key"`
	Operator string   `json:"operator"`
	Values   []string `json:"values"`
}

// Toleration is a taint toleration
type Toleration struct {
	Key      string `json:"key"`
	Operator string `json:"operator"`
	Value    string `json:"value,omitempty"`
	Effect   string `json:"effect,omitempty"`
}

// PodReference identifies a pod
type PodReference struct {
	Namespace string `json:"namespace"`
	Name      string `json:"name"`
}

// SchedulingDecision represents the scheduler's placement decision
type SchedulingDecision struct {
	// Workload being scheduled
	WorkloadUID string `json:"workloadUid"`

	// SelectedNode is the chosen node
	SelectedNode string `json:"selectedNode"`

	// SelectedGPUs lists allocated GPU UUIDs
	SelectedGPUs []string `json:"selectedGpus"`

	// MIGInstances lists allocated MIG instances (if MIG mode)
	MIGInstances []MIGInstanceAllocation `json:"migInstances,omitempty"`

	// Score is the placement quality score (0-100)
	Score int `json:"score"`

	// Reasons explains the placement decision
	Reasons []string `json:"reasons"`

	// EstimatedBandwidth is expected inter-GPU bandwidth
	EstimatedBandwidth float64 `json:"estimatedBandwidth"`

	// Timestamp of decision
	Timestamp time.Time `json:"timestamp"`
}

// MIGInstanceAllocation represents an allocated MIG instance
type MIGInstanceAllocation struct {
	// ParentGPUUID is the parent GPU UUID
	ParentGPUUID string `json:"parentGpuUuid"`

	// InstanceUUID is the MIG instance UUID
	InstanceUUID string `json:"instanceUuid"`

	// Profile is the MIG profile
	Profile string `json:"profile"`

	// DeviceIndex for CUDA_VISIBLE_DEVICES
	DeviceIndex int `json:"deviceIndex"`
}

// NodeScore represents a node's fitness for a workload
type NodeScore struct {
	// NodeName is the node name
	NodeName string `json:"nodeName"`

	// TotalScore is the aggregate score (0-100)
	TotalScore int `json:"totalScore"`

	// TopologyScore is the topology fit score
	TopologyScore int `json:"topologyScore"`

	// ResourceScore is the resource availability score
	ResourceScore int `json:"resourceScore"`

	// LocalityScore is the data locality score
	LocalityScore int `json:"localityScore"`

	// BalanceScore is the load balancing score
	BalanceScore int `json:"balanceScore"`

	// SelectedGPUs are the GPUs that would be used
	SelectedGPUs []string `json:"selectedGpus"`

	// Reasons explains the scoring
	Reasons []string `json:"reasons"`
}

// SchedulerMetrics tracks scheduling performance
type SchedulerMetrics struct {
	// TotalScheduled is count of scheduled workloads
	TotalScheduled int64 `json:"totalScheduled"`

	// TotalFailed is count of scheduling failures
	TotalFailed int64 `json:"totalFailed"`

	// AverageLatencyMs is average scheduling latency
	AverageLatencyMs float64 `json:"averageLatencyMs"`

	// P99LatencyMs is P99 scheduling latency
	P99LatencyMs float64 `json:"p99LatencyMs"`

	// TopologyOptimalPlacements is count of optimal topology placements
	TopologyOptimalPlacements int64 `json:"topologyOptimalPlacements"`

	// MIGAllocations is count of MIG allocations
	MIGAllocations int64 `json:"migAllocations"`

	// PreemptionCount is count of preemptions
	PreemptionCount int64 `json:"preemptionCount"`
}

// SchedulerConfig holds scheduler configuration
type SchedulerConfig struct {
	// SchedulerName is the name registered with Kubernetes
	SchedulerName string `json:"schedulerName"`

	// EnableTopologyAwareness enables topology-aware scheduling
	EnableTopologyAwareness bool `json:"enableTopologyAwareness"`

	// EnableMIGSupport enables MIG allocation
	EnableMIGSupport bool `json:"enableMigSupport"`

	// EnablePreemption enables workload preemption
	EnablePreemption bool `json:"enablePreemption"`

	// TopologyWeight is the weight for topology scoring (0-100)
	TopologyWeight int `json:"topologyWeight"`

	// ResourceWeight is the weight for resource scoring (0-100)
	ResourceWeight int `json:"resourceWeight"`

	// BalanceWeight is the weight for balance scoring (0-100)
	BalanceWeight int `json:"balanceWeight"`

	// MaxSchedulingAttempts before failure
	MaxSchedulingAttempts int `json:"maxSchedulingAttempts"`

	// SchedulingTimeout is the maximum scheduling time
	SchedulingTimeout time.Duration `json:"schedulingTimeout"`

	// EnableGangScheduling enables gang scheduling
	EnableGangScheduling bool `json:"enableGangScheduling"`
}

// DefaultSchedulerConfig returns default configuration
func DefaultSchedulerConfig() SchedulerConfig {
	return SchedulerConfig{
		SchedulerName:           "kgwe-scheduler",
		EnableTopologyAwareness: true,
		EnableMIGSupport:        true,
		EnablePreemption:        true,
		TopologyWeight:          40,
		ResourceWeight:          35,
		BalanceWeight:           25,
		MaxSchedulingAttempts:   3,
		SchedulingTimeout:       30 * time.Second,
		EnableGangScheduling:    true,
	}
}

// PreemptionCandidate represents a preemption target
type PreemptionCandidate struct {
	// Pod is the candidate pod
	Pod PodReference `json:"pod"`

	// NodeName where pod is running
	NodeName string `json:"nodeName"`

	// GPUs that would be freed
	GPUs []string `json:"gpus"`

	// Priority of the candidate
	Priority int32 `json:"priority"`

	// Age of the workload
	Age time.Duration `json:"age"`

	// PreemptionCost estimates impact of preemption
	PreemptionCost float64 `json:"preemptionCost"`
}

// GangSchedulingGroup represents a gang of pods
type GangSchedulingGroup struct {
	// GroupID is the gang identifier
	GroupID string `json:"groupId"`

	// MinMembers is minimum members required
	MinMembers int `json:"minMembers"`

	// Members lists pod references
	Members []PodReference `json:"members"`

	// Status of the gang
	Status GangStatus `json:"status"`

	// CreatedAt timestamp
	CreatedAt time.Time `json:"createdAt"`

	// ScheduledAt timestamp (if scheduled)
	ScheduledAt *time.Time `json:"scheduledAt,omitempty"`
}

// GangStatus represents gang scheduling status
type GangStatus string

const (
	GangStatusPending   GangStatus = "Pending"
	GangStatusScheduled GangStatus = "Scheduled"
	GangStatusRunning   GangStatus = "Running"
	GangStatusFailed    GangStatus = "Failed"
)
