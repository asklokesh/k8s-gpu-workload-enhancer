// Package discovery provides GPU topology discovery and resource mapping capabilities.
// It interfaces with NVIDIA Management Library (NVML) to gather real-time GPU information
// including interconnect topology, memory, and compute capabilities.
package discovery

import (
	"time"
)

// GPUDevice represents a single GPU device with its full specifications
type GPUDevice struct {
	// UUID is the unique identifier for the GPU (e.g., "GPU-a1b2c3d4-...")
	UUID string `json:"uuid"`

	// Index is the device index as reported by NVML (0-based)
	Index int `json:"index"`

	// Name is the product name (e.g., "NVIDIA H100 80GB HBM3")
	Name string `json:"name"`

	// Architecture represents the GPU architecture generation
	Architecture GPUArchitecture `json:"architecture"`

	// Memory specifications
	Memory GPUMemory `json:"memory"`

	// Compute specifications
	Compute GPUCompute `json:"compute"`

	// Topology information for this device
	Topology DeviceTopology `json:"topology"`

	// MIG configuration if enabled
	MIGConfig *MIGConfiguration `json:"migConfig,omitempty"`

	// Current utilization metrics
	Utilization GPUUtilization `json:"utilization"`

	// Health status
	Health GPUHealth `json:"health"`

	// NodeName is the Kubernetes node hosting this GPU
	NodeName string `json:"nodeName"`

	// LastUpdated timestamp
	LastUpdated time.Time `json:"lastUpdated"`
}

// GPUArchitecture represents NVIDIA GPU architecture generations
type GPUArchitecture string

const (
	ArchitectureVolta   GPUArchitecture = "Volta"
	ArchitectureTuring  GPUArchitecture = "Turing"
	ArchitectureAmpere  GPUArchitecture = "Ampere"
	ArchitectureHopper  GPUArchitecture = "Hopper"
	ArchitectureAda     GPUArchitecture = "Ada"
	ArchitectureBlackwell GPUArchitecture = "Blackwell"
)

// GPUMemory contains GPU memory specifications
type GPUMemory struct {
	// TotalBytes is the total GPU memory in bytes
	TotalBytes uint64 `json:"totalBytes"`

	// FreeBytes is the available GPU memory in bytes
	FreeBytes uint64 `json:"freeBytes"`

	// UsedBytes is the used GPU memory in bytes
	UsedBytes uint64 `json:"usedBytes"`

	// BandwidthGBps is the memory bandwidth in GB/s
	BandwidthGBps float64 `json:"bandwidthGBps"`

	// MemoryType (e.g., "HBM2e", "HBM3", "GDDR6X")
	MemoryType string `json:"memoryType"`

	// ECCEnabled indicates if ECC memory is enabled
	ECCEnabled bool `json:"eccEnabled"`
}

// GPUCompute contains GPU compute specifications
type GPUCompute struct {
	// SMCount is the number of Streaming Multiprocessors
	SMCount int `json:"smCount"`

	// CUDACores is the number of CUDA cores
	CUDACores int `json:"cudaCores"`

	// TensorCores is the number of Tensor Cores
	TensorCores int `json:"tensorCores"`

	// ClockMHz is the current GPU clock in MHz
	ClockMHz int `json:"clockMHz"`

	// MaxClockMHz is the maximum GPU clock in MHz
	MaxClockMHz int `json:"maxClockMHz"`

	// ComputeCapability (e.g., "9.0" for Hopper)
	ComputeCapability string `json:"computeCapability"`

	// FP16TFlops is the theoretical FP16 performance
	FP16TFlops float64 `json:"fp16TFlops"`

	// FP32TFlops is the theoretical FP32 performance
	FP32TFlops float64 `json:"fp32TFlops"`

	// TF32TFlops is the theoretical TF32 performance (Ampere+)
	TF32TFlops float64 `json:"tf32TFlops"`

	// FP8TFlops is the theoretical FP8 performance (Hopper+)
	FP8TFlops float64 `json:"fp8TFlops"`
}

// DeviceTopology contains interconnect and placement information
type DeviceTopology struct {
	// NVLinkConnections maps peer GPU UUIDs to NVLink bandwidth
	NVLinkConnections map[string]NVLinkInfo `json:"nvlinkConnections"`

	// PCIeInfo contains PCIe topology information
	PCIeInfo PCIeTopology `json:"pcieInfo"`

	// NUMANode is the NUMA node affinity (-1 if unknown)
	NUMANode int `json:"numaNode"`

	// CPUAffinity lists the CPU cores with best affinity to this GPU
	CPUAffinity []int `json:"cpuAffinity"`

	// IsNVSwitchConnected indicates if GPU is connected via NVSwitch
	IsNVSwitchConnected bool `json:"isNVSwitchConnected"`
}

// NVLinkInfo contains NVLink connection details
type NVLinkInfo struct {
	// LinkCount is the number of NVLink connections to peer
	LinkCount int `json:"linkCount"`

	// BandwidthGBps is the aggregate bandwidth to peer
	BandwidthGBps float64 `json:"bandwidthGBps"`

	// Version is the NVLink version (e.g., 4 for H100)
	Version int `json:"version"`

	// IsActive indicates if the link is active
	IsActive bool `json:"isActive"`
}

// PCIeTopology contains PCIe placement information
type PCIeTopology struct {
	// BusID is the PCIe bus ID (e.g., "0000:3B:00.0")
	BusID string `json:"busId"`

	// Generation is the PCIe generation (e.g., 5)
	Generation int `json:"generation"`

	// Width is the PCIe link width (e.g., 16)
	Width int `json:"width"`

	// BandwidthGBps is the theoretical PCIe bandwidth
	BandwidthGBps float64 `json:"bandwidthGBps"`

	// SwitchID identifies the PCIe switch (if behind one)
	SwitchID string `json:"switchId,omitempty"`
}

// MIGConfiguration represents Multi-Instance GPU configuration
type MIGConfiguration struct {
	// Enabled indicates if MIG mode is enabled
	Enabled bool `json:"enabled"`

	// MaxInstances is the maximum number of MIG instances
	MaxInstances int `json:"maxInstances"`

	// CurrentInstances lists active MIG instances
	CurrentInstances []MIGInstance `json:"currentInstances"`

	// AvailableProfiles lists creatable profiles
	AvailableProfiles []MIGProfile `json:"availableProfiles"`
}

// MIGInstance represents an active MIG instance
type MIGInstance struct {
	// InstanceID is the MIG instance identifier
	InstanceID int `json:"instanceId"`

	// UUID is the unique identifier for this MIG instance
	UUID string `json:"uuid"`

	// Profile is the MIG profile (e.g., "1g.10gb")
	Profile MIGProfile `json:"profile"`

	// GIProfileID is the GPU Instance profile ID
	GIProfileID int `json:"giProfileId"`

	// CIProfileID is the Compute Instance profile ID
	CIProfileID int `json:"ciProfileId"`

	// MemoryBytes is allocated memory
	MemoryBytes uint64 `json:"memoryBytes"`

	// SMCount is allocated SM count
	SMCount int `json:"smCount"`

	// InUse indicates if instance is allocated to a workload
	InUse bool `json:"inUse"`

	// PodReference is the pod using this instance (if any)
	PodReference *PodReference `json:"podReference,omitempty"`
}

// MIGProfile represents a MIG profile specification
type MIGProfile struct {
	// Name is the profile name (e.g., "1g.10gb", "3g.40gb")
	Name string `json:"name"`

	// GPUInstanceProfileID is the GI profile ID
	GPUInstanceProfileID int `json:"giProfileId"`

	// ComputeInstanceProfileID is the CI profile ID
	ComputeInstanceProfileID int `json:"ciProfileId"`

	// MemoryGB is the memory allocation in GB
	MemoryGB int `json:"memoryGb"`

	// SMFraction is the fraction of SMs (1/7, 2/7, etc.)
	SMFraction float64 `json:"smFraction"`

	// MaxInstanceCount is how many of this profile can exist
	MaxInstanceCount int `json:"maxInstanceCount"`
}

// Common MIG profiles for H100
var (
	MIGProfile1g10gb = MIGProfile{Name: "1g.10gb", GPUInstanceProfileID: 19, ComputeInstanceProfileID: 0, MemoryGB: 10, SMFraction: 1.0 / 7.0, MaxInstanceCount: 7}
	MIGProfile2g20gb = MIGProfile{Name: "2g.20gb", GPUInstanceProfileID: 14, ComputeInstanceProfileID: 0, MemoryGB: 20, SMFraction: 2.0 / 7.0, MaxInstanceCount: 3}
	MIGProfile3g40gb = MIGProfile{Name: "3g.40gb", GPUInstanceProfileID: 9, ComputeInstanceProfileID: 0, MemoryGB: 40, SMFraction: 3.0 / 7.0, MaxInstanceCount: 2}
	MIGProfile4g40gb = MIGProfile{Name: "4g.40gb", GPUInstanceProfileID: 5, ComputeInstanceProfileID: 0, MemoryGB: 40, SMFraction: 4.0 / 7.0, MaxInstanceCount: 1}
	MIGProfile7g80gb = MIGProfile{Name: "7g.80gb", GPUInstanceProfileID: 0, ComputeInstanceProfileID: 0, MemoryGB: 80, SMFraction: 1.0, MaxInstanceCount: 1}
)

// GPUUtilization contains real-time utilization metrics
type GPUUtilization struct {
	// GPUPercent is the GPU SM utilization (0-100)
	GPUPercent float64 `json:"gpuPercent"`

	// MemoryPercent is the memory utilization (0-100)
	MemoryPercent float64 `json:"memoryPercent"`

	// EncoderPercent is the encoder utilization (0-100)
	EncoderPercent float64 `json:"encoderPercent"`

	// DecoderPercent is the decoder utilization (0-100)
	DecoderPercent float64 `json:"decoderPercent"`

	// PowerWatts is current power consumption
	PowerWatts float64 `json:"powerWatts"`

	// TemperatureCelsius is the current GPU temperature
	TemperatureCelsius float64 `json:"temperatureCelsius"`

	// FanSpeedPercent is the fan speed (if applicable)
	FanSpeedPercent float64 `json:"fanSpeedPercent"`

	// Timestamp when metrics were collected
	Timestamp time.Time `json:"timestamp"`
}

// GPUHealth contains health and status information
type GPUHealth struct {
	// Status is the overall health status
	Status HealthStatus `json:"status"`

	// Reasons provides details if not healthy
	Reasons []string `json:"reasons,omitempty"`

	// RetiredPages tracks memory retirement
	RetiredPages RetiredPageInfo `json:"retiredPages"`

	// XIDErrors lists recent XID errors
	XIDErrors []XIDError `json:"xidErrors,omitempty"`

	// ThrottlingReasons lists active throttling causes
	ThrottlingReasons []string `json:"throttlingReasons,omitempty"`
}

// HealthStatus represents GPU health state
type HealthStatus string

const (
	HealthStatusHealthy   HealthStatus = "Healthy"
	HealthStatusDegraded  HealthStatus = "Degraded"
	HealthStatusUnhealthy HealthStatus = "Unhealthy"
	HealthStatusUnknown   HealthStatus = "Unknown"
)

// RetiredPageInfo tracks GPU memory page retirement
type RetiredPageInfo struct {
	// SingleBitCount is pages retired due to single-bit ECC errors
	SingleBitCount int `json:"singleBitCount"`

	// DoubleBitCount is pages retired due to double-bit ECC errors
	DoubleBitCount int `json:"doubleBitCount"`

	// PendingRetirement indicates if retirement is pending reboot
	PendingRetirement bool `json:"pendingRetirement"`
}

// XIDError represents an NVIDIA XID error
type XIDError struct {
	// XID is the error code
	XID int `json:"xid"`

	// Message is the error description
	Message string `json:"message"`

	// Timestamp when error occurred
	Timestamp time.Time `json:"timestamp"`

	// Count is occurrences since last reset
	Count int `json:"count"`
}

// PodReference identifies a Kubernetes pod
type PodReference struct {
	// Namespace of the pod
	Namespace string `json:"namespace"`

	// Name of the pod
	Name string `json:"name"`

	// UID of the pod
	UID string `json:"uid"`
}

// ClusterTopology represents the GPU topology of the entire cluster
type ClusterTopology struct {
	// Nodes maps node names to NodeTopology
	Nodes map[string]*NodeTopology `json:"nodes"`

	// TotalGPUs is the total GPU count in cluster
	TotalGPUs int `json:"totalGpus"`

	// TotalMemoryBytes is aggregate GPU memory
	TotalMemoryBytes uint64 `json:"totalMemoryBytes"`

	// LastUpdated timestamp
	LastUpdated time.Time `json:"lastUpdated"`
}

// NodeTopology represents GPU topology on a single node
type NodeTopology struct {
	// NodeName is the Kubernetes node name
	NodeName string `json:"nodeName"`

	// GPUs lists all GPUs on this node
	GPUs []*GPUDevice `json:"gpus"`

	// TopologyMatrix is the pairwise GPU topology
	TopologyMatrix TopologyMatrix `json:"topologyMatrix"`

	// NVSwitchInfo contains NVSwitch details (if present)
	NVSwitchInfo *NVSwitchInfo `json:"nvSwitchInfo,omitempty"`

	// SystemInfo contains host system information
	SystemInfo SystemInfo `json:"systemInfo"`
}

// TopologyMatrix represents pairwise GPU connectivity
type TopologyMatrix struct {
	// DeviceUUIDs lists GPU UUIDs in matrix order
	DeviceUUIDs []string `json:"deviceUuids"`

	// ConnectionTypes is the NxN matrix of connection types
	// Values: "NVL" (NVLink), "PIX" (PCIe), "PHB" (PCIe Host Bridge), "SOC" (SoC)
	ConnectionTypes [][]string `json:"connectionTypes"`

	// BandwidthGBps is the NxN matrix of bandwidths
	BandwidthGBps [][]float64 `json:"bandwidthGBps"`
}

// NVSwitchInfo contains NVSwitch details
type NVSwitchInfo struct {
	// Present indicates if NVSwitch is available
	Present bool `json:"present"`

	// Version is the NVSwitch version
	Version string `json:"version"`

	// SwitchCount is the number of NVSwitches
	SwitchCount int `json:"switchCount"`

	// TotalBandwidthGBps is aggregate switch bandwidth
	TotalBandwidthGBps float64 `json:"totalBandwidthGBps"`
}

// SystemInfo contains host system information
type SystemInfo struct {
	// DriverVersion is the NVIDIA driver version
	DriverVersion string `json:"driverVersion"`

	// CUDAVersion is the CUDA toolkit version
	CUDAVersion string `json:"cudaVersion"`

	// OS is the operating system
	OS string `json:"os"`

	// Hostname is the system hostname
	Hostname string `json:"hostname"`

	// CPUCount is the number of CPU cores
	CPUCount int `json:"cpuCount"`

	// MemoryBytes is total system memory
	MemoryBytes uint64 `json:"memoryBytes"`

	// NUMANodes is the number of NUMA nodes
	NUMANodes int `json:"numaNodes"`
}

// TopologyHint provides scheduling hints based on topology
type TopologyHint struct {
	// PreferredNode is the recommended node
	PreferredNode string `json:"preferredNode"`

	// PreferredGPUs lists recommended GPU UUIDs
	PreferredGPUs []string `json:"preferredGpus"`

	// Score indicates the quality of this placement (0-100)
	Score int `json:"score"`

	// Reasons explains the recommendation
	Reasons []string `json:"reasons"`

	// BandwidthEstimate is estimated inter-GPU bandwidth
	BandwidthEstimate float64 `json:"bandwidthEstimate"`
}
