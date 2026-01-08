// Package discovery provides GPU topology discovery and resource mapping capabilities.
package discovery

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// DiscoveryService manages GPU discovery and topology tracking
type DiscoveryService struct {
	mu sync.RWMutex

	// clusterTopology is the cached cluster-wide GPU topology
	clusterTopology *ClusterTopology

	// nvmlClient interfaces with NVIDIA Management Library
	nvmlClient NVMLClient

	// k8sClient interfaces with Kubernetes API
	k8sClient KubernetesClient

	// refreshInterval controls how often topology is updated
	refreshInterval time.Duration

	// eventChan receives topology change events
	eventChan chan TopologyEvent

	// stopChan signals shutdown
	stopChan chan struct{}
}

// NVMLClient defines the interface for NVML operations
type NVMLClient interface {
	// Initialize initializes NVML
	Initialize() error

	// Shutdown cleans up NVML resources
	Shutdown() error

	// GetDeviceCount returns the number of GPUs
	GetDeviceCount() (int, error)

	// GetDeviceByIndex returns GPU info by index
	GetDeviceByIndex(index int) (*GPUDevice, error)

	// GetDeviceByUUID returns GPU info by UUID
	GetDeviceByUUID(uuid string) (*GPUDevice, error)

	// GetTopologyMatrix returns pairwise GPU topology
	GetTopologyMatrix() (*TopologyMatrix, error)

	// GetNVLinkInfo returns NVLink connections for a device
	GetNVLinkInfo(uuid string) (map[string]NVLinkInfo, error)

	// GetMIGConfig returns MIG configuration for a device
	GetMIGConfig(uuid string) (*MIGConfiguration, error)

	// CreateMIGInstance creates a MIG instance
	CreateMIGInstance(uuid string, profile MIGProfile) (*MIGInstance, error)

	// DestroyMIGInstance destroys a MIG instance
	DestroyMIGInstance(uuid string, instanceID int) error

	// GetUtilization returns current utilization metrics
	GetUtilization(uuid string) (*GPUUtilization, error)

	// GetHealth returns device health status
	GetHealth(uuid string) (*GPUHealth, error)
}

// KubernetesClient defines the interface for Kubernetes operations
type KubernetesClient interface {
	// GetNodes returns all GPU-enabled nodes
	GetNodes(ctx context.Context) ([]string, error)

	// GetNodeLabels returns labels for a node
	GetNodeLabels(ctx context.Context, nodeName string) (map[string]string, error)

	// UpdateNodeLabels updates GPU-related labels on a node
	UpdateNodeLabels(ctx context.Context, nodeName string, labels map[string]string) error

	// GetPodsOnNode returns pods scheduled on a node
	GetPodsOnNode(ctx context.Context, nodeName string) ([]PodReference, error)

	// WatchNodes watches for node changes
	WatchNodes(ctx context.Context) (<-chan NodeEvent, error)
}

// TopologyEvent represents a change in GPU topology
type TopologyEvent struct {
	// Type of event
	Type TopologyEventType

	// NodeName affected (if applicable)
	NodeName string

	// GPU UUID affected (if applicable)
	GPUUID string

	// Timestamp of event
	Timestamp time.Time

	// Details provides additional context
	Details string
}

// TopologyEventType categorizes topology events
type TopologyEventType string

const (
	TopologyEventNodeAdded      TopologyEventType = "NodeAdded"
	TopologyEventNodeRemoved    TopologyEventType = "NodeRemoved"
	TopologyEventGPUAdded       TopologyEventType = "GPUAdded"
	TopologyEventGPURemoved     TopologyEventType = "GPURemoved"
	TopologyEventMIGChanged     TopologyEventType = "MIGChanged"
	TopologyEventHealthChanged  TopologyEventType = "HealthChanged"
)

// NodeEvent represents a Kubernetes node event
type NodeEvent struct {
	Type     string
	NodeName string
}

// Config holds discovery service configuration
type Config struct {
	// RefreshInterval for topology updates
	RefreshInterval time.Duration

	// EnableMIGDiscovery enables MIG instance discovery
	EnableMIGDiscovery bool

	// EnableHealthMonitoring enables GPU health monitoring
	EnableHealthMonitoring bool

	// TopologyUpdateCallback is called on topology changes
	TopologyUpdateCallback func(*ClusterTopology)
}

// DefaultConfig returns default configuration
func DefaultConfig() Config {
	return Config{
		RefreshInterval:        30 * time.Second,
		EnableMIGDiscovery:     true,
		EnableHealthMonitoring: true,
	}
}

// NewDiscoveryService creates a new discovery service
func NewDiscoveryService(nvmlClient NVMLClient, k8sClient KubernetesClient, config Config) (*DiscoveryService, error) {
	if err := nvmlClient.Initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize NVML: %w", err)
	}

	return &DiscoveryService{
		nvmlClient:      nvmlClient,
		k8sClient:       k8sClient,
		refreshInterval: config.RefreshInterval,
		clusterTopology: &ClusterTopology{
			Nodes: make(map[string]*NodeTopology),
		},
		eventChan: make(chan TopologyEvent, 100),
		stopChan:  make(chan struct{}),
	}, nil
}

// Start begins the discovery service
func (d *DiscoveryService) Start(ctx context.Context) error {
	// Initial discovery
	if err := d.RefreshTopology(ctx); err != nil {
		return fmt.Errorf("initial topology discovery failed: %w", err)
	}

	// Start background refresh
	go d.refreshLoop(ctx)

	// Start watching for Kubernetes events
	go d.watchNodes(ctx)

	return nil
}

// Stop shuts down the discovery service
func (d *DiscoveryService) Stop() error {
	close(d.stopChan)
	return d.nvmlClient.Shutdown()
}

// GetClusterTopology returns the current cluster topology
func (d *DiscoveryService) GetClusterTopology() *ClusterTopology {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.clusterTopology
}

// GetNodeTopology returns topology for a specific node
func (d *DiscoveryService) GetNodeTopology(nodeName string) (*NodeTopology, bool) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	node, ok := d.clusterTopology.Nodes[nodeName]
	return node, ok
}

// GetGPUByUUID returns a GPU device by UUID
func (d *DiscoveryService) GetGPUByUUID(uuid string) (*GPUDevice, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	for _, node := range d.clusterTopology.Nodes {
		for _, gpu := range node.GPUs {
			if gpu.UUID == uuid {
				return gpu, nil
			}
		}
	}
	return nil, fmt.Errorf("GPU with UUID %s not found", uuid)
}

// GetTopologyHint returns scheduling hints for GPU requirements
func (d *DiscoveryService) GetTopologyHint(req GPURequirements) (*TopologyHint, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	hint := &TopologyHint{
		Score: 0,
	}

	// Find best node and GPUs for the requirements
	for nodeName, node := range d.clusterTopology.Nodes {
		score, gpus := d.scoreNodeForRequirements(node, req)
		if score > hint.Score {
			hint.Score = score
			hint.PreferredNode = nodeName
			hint.PreferredGPUs = gpus
			hint.BandwidthEstimate = d.estimateBandwidth(node, gpus)
		}
	}

	if hint.Score == 0 {
		return nil, fmt.Errorf("no suitable placement found for requirements")
	}

	hint.Reasons = d.explainPlacement(hint)
	return hint, nil
}

// GPURequirements specifies requirements for GPU allocation
type GPURequirements struct {
	// Count is the number of GPUs required
	Count int

	// MinMemoryGB is minimum memory per GPU
	MinMemoryGB int

	// PreferNVLink indicates preference for NVLink connectivity
	PreferNVLink bool

	// RequireMIG indicates if MIG instance is required
	RequireMIG bool

	// MIGProfile specifies required MIG profile (if RequireMIG)
	MIGProfile string

	// PreferSameNUMA indicates preference for same NUMA node
	PreferSameNUMA bool

	// Architecture specifies required GPU architecture
	Architecture GPUArchitecture

	// MaxPowerWatts is maximum power budget
	MaxPowerWatts float64

	// WorkloadType hints at the workload characteristics
	WorkloadType WorkloadType
}

// WorkloadType categorizes GPU workloads
type WorkloadType string

const (
	WorkloadTypeTraining     WorkloadType = "Training"
	WorkloadTypeInference    WorkloadType = "Inference"
	WorkloadTypeDataProcessing WorkloadType = "DataProcessing"
	WorkloadTypeVisualization WorkloadType = "Visualization"
)

// RefreshTopology updates the cluster topology
func (d *DiscoveryService) RefreshTopology(ctx context.Context) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	nodes, err := d.k8sClient.GetNodes(ctx)
	if err != nil {
		return fmt.Errorf("failed to get nodes: %w", err)
	}

	newTopology := &ClusterTopology{
		Nodes:       make(map[string]*NodeTopology),
		LastUpdated: time.Now(),
	}

	for _, nodeName := range nodes {
		nodeTopology, err := d.discoverNodeTopology(ctx, nodeName)
		if err != nil {
			// Log error but continue with other nodes
			continue
		}
		newTopology.Nodes[nodeName] = nodeTopology
		newTopology.TotalGPUs += len(nodeTopology.GPUs)
		for _, gpu := range nodeTopology.GPUs {
			newTopology.TotalMemoryBytes += gpu.Memory.TotalBytes
		}
	}

	d.clusterTopology = newTopology
	return nil
}

// discoverNodeTopology discovers GPU topology on a single node
func (d *DiscoveryService) discoverNodeTopology(ctx context.Context, nodeName string) (*NodeTopology, error) {
	deviceCount, err := d.nvmlClient.GetDeviceCount()
	if err != nil {
		return nil, fmt.Errorf("failed to get device count: %w", err)
	}

	node := &NodeTopology{
		NodeName: nodeName,
		GPUs:     make([]*GPUDevice, 0, deviceCount),
	}

	for i := 0; i < deviceCount; i++ {
		gpu, err := d.nvmlClient.GetDeviceByIndex(i)
		if err != nil {
			continue
		}
		gpu.NodeName = nodeName

		// Get NVLink info
		nvlinkInfo, err := d.nvmlClient.GetNVLinkInfo(gpu.UUID)
		if err == nil {
			gpu.Topology.NVLinkConnections = nvlinkInfo
		}

		// Get MIG config
		migConfig, err := d.nvmlClient.GetMIGConfig(gpu.UUID)
		if err == nil {
			gpu.MIGConfig = migConfig
		}

		// Get utilization
		util, err := d.nvmlClient.GetUtilization(gpu.UUID)
		if err == nil {
			gpu.Utilization = *util
		}

		// Get health
		health, err := d.nvmlClient.GetHealth(gpu.UUID)
		if err == nil {
			gpu.Health = *health
		}

		gpu.LastUpdated = time.Now()
		node.GPUs = append(node.GPUs, gpu)
	}

	// Get topology matrix
	topoMatrix, err := d.nvmlClient.GetTopologyMatrix()
	if err == nil {
		node.TopologyMatrix = *topoMatrix
	}

	return node, nil
}

// scoreNodeForRequirements calculates placement score for a node
func (d *DiscoveryService) scoreNodeForRequirements(node *NodeTopology, req GPURequirements) (int, []string) {
	availableGPUs := d.getAvailableGPUs(node)
	if len(availableGPUs) < req.Count {
		return 0, nil
	}

	// Filter by memory requirements
	filteredGPUs := make([]*GPUDevice, 0)
	for _, gpu := range availableGPUs {
		memGB := int(gpu.Memory.FreeBytes / (1024 * 1024 * 1024))
		if memGB >= req.MinMemoryGB {
			filteredGPUs = append(filteredGPUs, gpu)
		}
	}

	if len(filteredGPUs) < req.Count {
		return 0, nil
	}

	// Score based on topology
	score := 50 // Base score
	selectedGPUs := make([]string, 0, req.Count)

	// Find best GPU combination
	if req.Count > 1 && req.PreferNVLink {
		// Find GPUs with NVLink connectivity
		nvlinkGroups := d.findNVLinkGroups(filteredGPUs, req.Count)
		if len(nvlinkGroups) > 0 {
			score += 30
			selectedGPUs = nvlinkGroups[0]
		}
	}

	if len(selectedGPUs) == 0 {
		for i := 0; i < req.Count && i < len(filteredGPUs); i++ {
			selectedGPUs = append(selectedGPUs, filteredGPUs[i].UUID)
		}
	}

	// Bonus for same NUMA node
	if req.PreferSameNUMA && d.areOnSameNUMA(node, selectedGPUs) {
		score += 10
	}

	// Bonus for matching architecture
	if req.Architecture != "" {
		for _, uuid := range selectedGPUs {
			for _, gpu := range node.GPUs {
				if gpu.UUID == uuid && gpu.Architecture == req.Architecture {
					score += 5
				}
			}
		}
	}

	return score, selectedGPUs
}

// getAvailableGPUs returns GPUs not fully allocated
func (d *DiscoveryService) getAvailableGPUs(node *NodeTopology) []*GPUDevice {
	available := make([]*GPUDevice, 0)
	for _, gpu := range node.GPUs {
		// Check if GPU has capacity
		if gpu.Health.Status == HealthStatusHealthy {
			if gpu.MIGConfig != nil && gpu.MIGConfig.Enabled {
				// For MIG, check for free instances
				for _, inst := range gpu.MIGConfig.CurrentInstances {
					if !inst.InUse {
						available = append(available, gpu)
						break
					}
				}
			} else {
				// For non-MIG, check utilization
				if gpu.Utilization.GPUPercent < 90 {
					available = append(available, gpu)
				}
			}
		}
	}
	return available
}

// findNVLinkGroups finds groups of GPUs connected via NVLink
func (d *DiscoveryService) findNVLinkGroups(gpus []*GPUDevice, size int) [][]string {
	groups := make([][]string, 0)

	// Simple greedy grouping - in production, use graph algorithms
	for _, gpu := range gpus {
		group := []string{gpu.UUID}
		for peerUUID := range gpu.Topology.NVLinkConnections {
			if len(group) >= size {
				break
			}
			// Check if peer is in available list
			for _, peer := range gpus {
				if peer.UUID == peerUUID {
					group = append(group, peerUUID)
					break
				}
			}
		}
		if len(group) >= size {
			groups = append(groups, group[:size])
		}
	}

	return groups
}

// areOnSameNUMA checks if all GPUs are on the same NUMA node
func (d *DiscoveryService) areOnSameNUMA(node *NodeTopology, uuids []string) bool {
	numaNode := -2 // Invalid initial value
	for _, uuid := range uuids {
		for _, gpu := range node.GPUs {
			if gpu.UUID == uuid {
				if numaNode == -2 {
					numaNode = gpu.Topology.NUMANode
				} else if numaNode != gpu.Topology.NUMANode {
					return false
				}
			}
		}
	}
	return true
}

// estimateBandwidth estimates inter-GPU bandwidth for a placement
func (d *DiscoveryService) estimateBandwidth(node *NodeTopology, uuids []string) float64 {
	if len(uuids) < 2 {
		return 0
	}

	totalBandwidth := 0.0
	count := 0

	for i, uuid1 := range uuids {
		for j, uuid2 := range uuids {
			if i >= j {
				continue
			}
			// Find bandwidth between GPUs
			for _, gpu := range node.GPUs {
				if gpu.UUID == uuid1 {
					if nvlink, ok := gpu.Topology.NVLinkConnections[uuid2]; ok {
						totalBandwidth += nvlink.BandwidthGBps
					} else {
						// Fallback to PCIe bandwidth
						totalBandwidth += gpu.Topology.PCIeInfo.BandwidthGBps
					}
					count++
					break
				}
			}
		}
	}

	if count == 0 {
		return 0
	}
	return totalBandwidth / float64(count)
}

// explainPlacement generates human-readable placement explanation
func (d *DiscoveryService) explainPlacement(hint *TopologyHint) []string {
	reasons := make([]string, 0)

	if hint.Score >= 80 {
		reasons = append(reasons, "Optimal NVLink connectivity between selected GPUs")
	} else if hint.Score >= 60 {
		reasons = append(reasons, "Good topology placement with partial NVLink connectivity")
	} else {
		reasons = append(reasons, "PCIe-based connectivity (NVLink not available)")
	}

	if hint.BandwidthEstimate > 500 {
		reasons = append(reasons, fmt.Sprintf("High aggregate bandwidth: %.0f GB/s", hint.BandwidthEstimate))
	}

	return reasons
}

// refreshLoop periodically refreshes topology
func (d *DiscoveryService) refreshLoop(ctx context.Context) {
	ticker := time.NewTicker(d.refreshInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if err := d.RefreshTopology(ctx); err != nil {
				// Log error
			}
		case <-d.stopChan:
			return
		case <-ctx.Done():
			return
		}
	}
}

// watchNodes watches for Kubernetes node events
func (d *DiscoveryService) watchNodes(ctx context.Context) {
	events, err := d.k8sClient.WatchNodes(ctx)
	if err != nil {
		return
	}

	for {
		select {
		case event := <-events:
			switch event.Type {
			case "ADDED", "MODIFIED":
				d.RefreshTopology(ctx)
				d.eventChan <- TopologyEvent{
					Type:      TopologyEventNodeAdded,
					NodeName:  event.NodeName,
					Timestamp: time.Now(),
				}
			case "DELETED":
				d.mu.Lock()
				delete(d.clusterTopology.Nodes, event.NodeName)
				d.mu.Unlock()
				d.eventChan <- TopologyEvent{
					Type:      TopologyEventNodeRemoved,
					NodeName:  event.NodeName,
					Timestamp: time.Now(),
				}
			}
		case <-d.stopChan:
			return
		case <-ctx.Done():
			return
		}
	}
}

// Events returns the topology event channel
func (d *DiscoveryService) Events() <-chan TopologyEvent {
	return d.eventChan
}
