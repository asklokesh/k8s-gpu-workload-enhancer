// Package sharing implements GPU sharing mechanisms including MIG and MPS.
// MIG (Multi-Instance GPU) enables spatial partitioning of NVIDIA GPUs.
// MPS (Multi-Process Service) enables temporal sharing for inference workloads.
package sharing

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/nvidia/kgwe/src/discovery"
)

// MIGController manages Multi-Instance GPU configurations
type MIGController struct {
	mu sync.RWMutex

	// nvmlClient interfaces with NVIDIA Management Library
	nvmlClient discovery.NVMLClient

	// strategies maps strategy names to configurations
	strategies map[string]*MIGStrategy

	// allocations tracks current MIG allocations
	allocations map[string]*MIGAllocation

	// pendingOperations tracks in-flight MIG operations
	pendingOperations map[string]*MIGOperation

	// config holds controller configuration
	config MIGControllerConfig

	// eventChan emits MIG-related events
	eventChan chan MIGEvent
}

// MIGControllerConfig holds controller configuration
type MIGControllerConfig struct {
	// EnableAutoRebalancing enables automatic partition rebalancing
	EnableAutoRebalancing bool

	// RebalanceInterval controls rebalancing frequency
	RebalanceInterval time.Duration

	// MinUtilizationThreshold triggers rebalancing when below this
	MinUtilizationThreshold float64

	// MaxReconfigurationTime limits MIG reconfiguration duration
	MaxReconfigurationTime time.Duration

	// PreferredProfiles lists profiles in order of preference
	PreferredProfiles []string

	// EnablePrewarming creates common profiles proactively
	EnablePrewarming bool
}

// DefaultMIGControllerConfig returns sensible defaults
func DefaultMIGControllerConfig() MIGControllerConfig {
	return MIGControllerConfig{
		EnableAutoRebalancing:   true,
		RebalanceInterval:       5 * time.Minute,
		MinUtilizationThreshold: 0.3,
		MaxReconfigurationTime:  60 * time.Second,
		PreferredProfiles:       []string{"1g.10gb", "2g.20gb", "3g.40gb"},
		EnablePrewarming:        true,
	}
}

// MIGStrategy defines a MIG partitioning strategy
type MIGStrategy struct {
	// Name is the strategy identifier
	Name string `json:"name"`

	// Description explains the strategy purpose
	Description string `json:"description"`

	// NodeSelector identifies target nodes
	NodeSelector map[string]string `json:"nodeSelector"`

	// GPUSelector identifies target GPUs
	GPUSelector GPUSelector `json:"gpuSelector"`

	// ProfileDistribution defines desired partition distribution
	ProfileDistribution map[string]float64 `json:"profileDistribution"`

	// AllowDynamicReconfig allows runtime reconfiguration
	AllowDynamicReconfig bool `json:"allowDynamicReconfig"`

	// Priority for conflict resolution
	Priority int `json:"priority"`
}

// GPUSelector specifies GPU selection criteria
type GPUSelector struct {
	// Model filters by GPU model (e.g., "H100")
	Model string `json:"model,omitempty"`

	// MinMemoryGB minimum GPU memory
	MinMemoryGB int `json:"minMemoryGb,omitempty"`

	// Architecture filters by GPU architecture
	Architecture string `json:"architecture,omitempty"`

	// MIGCapable must be MIG-capable
	MIGCapable bool `json:"migCapable"`
}

// MIGAllocation represents an allocated MIG instance
type MIGAllocation struct {
	// ID is the allocation identifier
	ID string `json:"id"`

	// WorkloadUID is the owning workload
	WorkloadUID string `json:"workloadUid"`

	// GPUUID is the parent GPU
	GPUUID string `json:"gpuUuid"`

	// NodeName is the hosting node
	NodeName string `json:"nodeName"`

	// Instance is the MIG instance details
	Instance discovery.MIGInstance `json:"instance"`

	// AllocatedAt timestamp
	AllocatedAt time.Time `json:"allocatedAt"`

	// ExpiresAt optional expiration (for time-boxed allocations)
	ExpiresAt *time.Time `json:"expiresAt,omitempty"`

	// Utilization current utilization metrics
	Utilization MIGUtilization `json:"utilization"`
}

// MIGUtilization tracks MIG instance utilization
type MIGUtilization struct {
	// GPUPercent SM utilization
	GPUPercent float64 `json:"gpuPercent"`

	// MemoryPercent memory utilization
	MemoryPercent float64 `json:"memoryPercent"`

	// LastUpdated timestamp
	LastUpdated time.Time `json:"lastUpdated"`
}

// MIGOperation represents an in-flight MIG operation
type MIGOperation struct {
	// ID is the operation identifier
	ID string `json:"id"`

	// Type of operation
	Type MIGOperationType `json:"type"`

	// GPUUID target GPU
	GPUUID string `json:"gpuUuid"`

	// Profile target profile (for create)
	Profile string `json:"profile,omitempty"`

	// InstanceID target instance (for destroy)
	InstanceID int `json:"instanceId,omitempty"`

	// Status of the operation
	Status OperationStatus `json:"status"`

	// StartedAt timestamp
	StartedAt time.Time `json:"startedAt"`

	// CompletedAt timestamp
	CompletedAt *time.Time `json:"completedAt,omitempty"`

	// Error message if failed
	Error string `json:"error,omitempty"`
}

// MIGOperationType categorizes MIG operations
type MIGOperationType string

const (
	MIGOpCreate   MIGOperationType = "Create"
	MIGOpDestroy  MIGOperationType = "Destroy"
	MIGOpReconfig MIGOperationType = "Reconfigure"
)

// OperationStatus represents operation state
type OperationStatus string

const (
	OpStatusPending   OperationStatus = "Pending"
	OpStatusRunning   OperationStatus = "Running"
	OpStatusCompleted OperationStatus = "Completed"
	OpStatusFailed    OperationStatus = "Failed"
)

// MIGEvent represents a MIG-related event
type MIGEvent struct {
	// Type of event
	Type MIGEventType `json:"type"`

	// GPUUID affected GPU
	GPUUID string `json:"gpuUuid"`

	// InstanceID affected instance
	InstanceID int `json:"instanceId,omitempty"`

	// Profile involved
	Profile string `json:"profile,omitempty"`

	// Timestamp of event
	Timestamp time.Time `json:"timestamp"`

	// Details additional context
	Details string `json:"details,omitempty"`
}

// MIGEventType categorizes events
type MIGEventType string

const (
	MIGEventInstanceCreated   MIGEventType = "InstanceCreated"
	MIGEventInstanceDestroyed MIGEventType = "InstanceDestroyed"
	MIGEventInstanceAllocated MIGEventType = "InstanceAllocated"
	MIGEventInstanceReleased  MIGEventType = "InstanceReleased"
	MIGEventRebalanceStarted  MIGEventType = "RebalanceStarted"
	MIGEventRebalanceComplete MIGEventType = "RebalanceComplete"
)

// NewMIGController creates a new MIG controller
func NewMIGController(nvmlClient discovery.NVMLClient, config MIGControllerConfig) *MIGController {
	return &MIGController{
		nvmlClient:        nvmlClient,
		strategies:        make(map[string]*MIGStrategy),
		allocations:       make(map[string]*MIGAllocation),
		pendingOperations: make(map[string]*MIGOperation),
		config:            config,
		eventChan:         make(chan MIGEvent, 100),
	}
}

// RegisterStrategy registers a MIG partitioning strategy
func (c *MIGController) RegisterStrategy(strategy *MIGStrategy) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Validate strategy
	if err := c.validateStrategy(strategy); err != nil {
		return fmt.Errorf("invalid strategy: %w", err)
	}

	c.strategies[strategy.Name] = strategy
	return nil
}

// validateStrategy validates a MIG strategy
func (c *MIGController) validateStrategy(strategy *MIGStrategy) error {
	if strategy.Name == "" {
		return fmt.Errorf("strategy name is required")
	}

	// Validate profile distribution sums to <= 100%
	total := 0.0
	for _, pct := range strategy.ProfileDistribution {
		total += pct
	}
	if total > 1.01 { // Allow small float error
		return fmt.Errorf("profile distribution exceeds 100%%")
	}

	// Validate profile names
	validProfiles := map[string]bool{
		"1g.5gb":  true, // A30
		"1g.10gb": true, // H100
		"2g.10gb": true, // A30
		"2g.20gb": true, // H100
		"3g.20gb": true, // A30
		"3g.40gb": true, // H100
		"4g.20gb": true, // A30
		"4g.40gb": true, // H100
		"7g.40gb": true, // A30
		"7g.80gb": true, // H100
	}

	for profile := range strategy.ProfileDistribution {
		if !validProfiles[profile] {
			return fmt.Errorf("invalid MIG profile: %s", profile)
		}
	}

	return nil
}

// AllocateMIGInstance allocates a MIG instance for a workload
func (c *MIGController) AllocateMIGInstance(
	ctx context.Context,
	workloadUID string,
	profile string,
	preferredNode string,
) (*MIGAllocation, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Find available instance matching profile
	instance, gpuUUID, nodeName, err := c.findAvailableInstance(profile, preferredNode)
	if err != nil {
		// Try to create a new instance
		instance, gpuUUID, nodeName, err = c.createInstance(ctx, profile, preferredNode)
		if err != nil {
			return nil, fmt.Errorf("failed to allocate MIG instance: %w", err)
		}
	}

	// Create allocation
	allocation := &MIGAllocation{
		ID:          fmt.Sprintf("mig-%s-%d", gpuUUID[:8], instance.InstanceID),
		WorkloadUID: workloadUID,
		GPUUID:      gpuUUID,
		NodeName:    nodeName,
		Instance:    *instance,
		AllocatedAt: time.Now(),
	}

	c.allocations[allocation.ID] = allocation

	c.eventChan <- MIGEvent{
		Type:       MIGEventInstanceAllocated,
		GPUUID:     gpuUUID,
		InstanceID: instance.InstanceID,
		Profile:    profile,
		Timestamp:  time.Now(),
		Details:    fmt.Sprintf("Allocated to workload %s", workloadUID),
	}

	return allocation, nil
}

// findAvailableInstance finds an existing unallocated MIG instance
func (c *MIGController) findAvailableInstance(
	profile string,
	preferredNode string,
) (*discovery.MIGInstance, string, string, error) {
	// Get all MIG-enabled GPUs
	// In production, this would query the discovery service
	// For now, return not found to trigger creation
	return nil, "", "", fmt.Errorf("no available instance found")
}

// createInstance creates a new MIG instance
func (c *MIGController) createInstance(
	ctx context.Context,
	profile string,
	preferredNode string,
) (*discovery.MIGInstance, string, string, error) {
	// Find GPU with capacity for this profile
	gpuUUID, nodeName, err := c.findGPUWithCapacity(profile, preferredNode)
	if err != nil {
		return nil, "", "", err
	}

	// Get the profile specification
	migProfile, err := c.getProfileSpec(profile)
	if err != nil {
		return nil, "", "", err
	}

	// Create operation
	opID := fmt.Sprintf("op-%d", time.Now().UnixNano())
	op := &MIGOperation{
		ID:        opID,
		Type:      MIGOpCreate,
		GPUUID:    gpuUUID,
		Profile:   profile,
		Status:    OpStatusRunning,
		StartedAt: time.Now(),
	}
	c.pendingOperations[opID] = op

	// Create the MIG instance via NVML
	instance, err := c.nvmlClient.CreateMIGInstance(gpuUUID, migProfile)
	if err != nil {
		op.Status = OpStatusFailed
		op.Error = err.Error()
		now := time.Now()
		op.CompletedAt = &now
		return nil, "", "", fmt.Errorf("failed to create MIG instance: %w", err)
	}

	// Update operation status
	now := time.Now()
	op.Status = OpStatusCompleted
	op.CompletedAt = &now

	c.eventChan <- MIGEvent{
		Type:       MIGEventInstanceCreated,
		GPUUID:     gpuUUID,
		InstanceID: instance.InstanceID,
		Profile:    profile,
		Timestamp:  time.Now(),
	}

	return instance, gpuUUID, nodeName, nil
}

// findGPUWithCapacity finds a GPU that can host the requested profile
func (c *MIGController) findGPUWithCapacity(
	profile string,
	preferredNode string,
) (string, string, error) {
	// In production, query discovery service for MIG-enabled GPUs
	// Check each GPU's current partitioning to find capacity
	// For now, simulate finding a GPU
	return "", "", fmt.Errorf("no GPU with capacity found for profile %s", profile)
}

// getProfileSpec returns the MIG profile specification
func (c *MIGController) getProfileSpec(profile string) (discovery.MIGProfile, error) {
	profiles := map[string]discovery.MIGProfile{
		"1g.10gb": discovery.MIGProfile1g10gb,
		"2g.20gb": discovery.MIGProfile2g20gb,
		"3g.40gb": discovery.MIGProfile3g40gb,
		"4g.40gb": discovery.MIGProfile4g40gb,
		"7g.80gb": discovery.MIGProfile7g80gb,
	}

	if p, ok := profiles[profile]; ok {
		return p, nil
	}
	return discovery.MIGProfile{}, fmt.Errorf("unknown profile: %s", profile)
}

// ReleaseMIGAllocation releases a MIG allocation
func (c *MIGController) ReleaseMIGAllocation(allocationID string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	allocation, ok := c.allocations[allocationID]
	if !ok {
		return fmt.Errorf("allocation not found: %s", allocationID)
	}

	delete(c.allocations, allocationID)

	c.eventChan <- MIGEvent{
		Type:       MIGEventInstanceReleased,
		GPUUID:     allocation.GPUUID,
		InstanceID: allocation.Instance.InstanceID,
		Profile:    allocation.Instance.Profile.Name,
		Timestamp:  time.Now(),
	}

	// Optionally destroy the instance if not needed
	// This depends on the strategy's reuse policy

	return nil
}

// GetAllocation returns an allocation by ID
func (c *MIGController) GetAllocation(allocationID string) (*MIGAllocation, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	alloc, ok := c.allocations[allocationID]
	return alloc, ok
}

// ListAllocations returns all current allocations
func (c *MIGController) ListAllocations() []*MIGAllocation {
	c.mu.RLock()
	defer c.mu.RUnlock()

	result := make([]*MIGAllocation, 0, len(c.allocations))
	for _, alloc := range c.allocations {
		result = append(result, alloc)
	}
	return result
}

// Rebalance triggers MIG partition rebalancing based on strategies
func (c *MIGController) Rebalance(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.eventChan <- MIGEvent{
		Type:      MIGEventRebalanceStarted,
		Timestamp: time.Now(),
	}

	// Analyze current allocation patterns
	profileUsage := make(map[string]int)
	for _, alloc := range c.allocations {
		profileUsage[alloc.Instance.Profile.Name]++
	}

	// Compare against strategies and rebalance if needed
	for _, strategy := range c.strategies {
		if !strategy.AllowDynamicReconfig {
			continue
		}

		// Calculate desired vs actual distribution
		// In production, implement sophisticated rebalancing logic
		// considering workload patterns, time of day, etc.
	}

	c.eventChan <- MIGEvent{
		Type:      MIGEventRebalanceComplete,
		Timestamp: time.Now(),
	}

	return nil
}

// Events returns the event channel
func (c *MIGController) Events() <-chan MIGEvent {
	return c.eventChan
}

// GetMetrics returns MIG controller metrics
func (c *MIGController) GetMetrics() MIGMetrics {
	c.mu.RLock()
	defer c.mu.RUnlock()

	metrics := MIGMetrics{
		TotalAllocations: len(c.allocations),
		AllocationsByProfile: make(map[string]int),
	}

	for _, alloc := range c.allocations {
		metrics.AllocationsByProfile[alloc.Instance.Profile.Name]++
	}

	return metrics
}

// MIGMetrics contains MIG controller metrics
type MIGMetrics struct {
	TotalAllocations     int            `json:"totalAllocations"`
	AllocationsByProfile map[string]int `json:"allocationsByProfile"`
	PendingOperations    int            `json:"pendingOperations"`
	FailedOperations     int            `json:"failedOperations"`
}

// MPSController manages CUDA Multi-Process Service for GPU sharing
type MPSController struct {
	mu sync.RWMutex

	// daemonStatus tracks MPS daemon status per node
	daemonStatus map[string]*MPSDaemonStatus

	// allocations tracks MPS allocations
	allocations map[string]*MPSAllocation

	// config holds controller configuration
	config MPSControllerConfig
}

// MPSControllerConfig holds MPS controller configuration
type MPSControllerConfig struct {
	// EnableActiveThreadPercentage enables thread percentage limiting
	EnableActiveThreadPercentage bool

	// DefaultThreadPercentage default per-client thread limit
	DefaultThreadPercentage int

	// EnableMemoryLimit enables per-client memory limits
	EnableMemoryLimit bool

	// MaxClientsPerGPU maximum concurrent MPS clients
	MaxClientsPerGPU int
}

// DefaultMPSControllerConfig returns sensible defaults
func DefaultMPSControllerConfig() MPSControllerConfig {
	return MPSControllerConfig{
		EnableActiveThreadPercentage: true,
		DefaultThreadPercentage:      25,
		EnableMemoryLimit:            true,
		MaxClientsPerGPU:             8,
	}
}

// MPSDaemonStatus tracks MPS daemon on a node
type MPSDaemonStatus struct {
	NodeName    string    `json:"nodeName"`
	GPUUID      string    `json:"gpuUuid"`
	Running     bool      `json:"running"`
	PID         int       `json:"pid"`
	StartedAt   time.Time `json:"startedAt"`
	ClientCount int       `json:"clientCount"`
}

// MPSAllocation represents an MPS client allocation
type MPSAllocation struct {
	ID                 string    `json:"id"`
	WorkloadUID        string    `json:"workloadUid"`
	GPUUID             string    `json:"gpuUuid"`
	NodeName           string    `json:"nodeName"`
	ThreadPercentage   int       `json:"threadPercentage"`
	MemoryLimitMB      int       `json:"memoryLimitMb"`
	AllocatedAt        time.Time `json:"allocatedAt"`
}

// NewMPSController creates a new MPS controller
func NewMPSController(config MPSControllerConfig) *MPSController {
	return &MPSController{
		daemonStatus: make(map[string]*MPSDaemonStatus),
		allocations:  make(map[string]*MPSAllocation),
		config:       config,
	}
}

// EnsureMPSDaemon ensures MPS daemon is running on a GPU
func (c *MPSController) EnsureMPSDaemon(ctx context.Context, gpuUUID string, nodeName string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	key := fmt.Sprintf("%s/%s", nodeName, gpuUUID)
	if status, ok := c.daemonStatus[key]; ok && status.Running {
		return nil // Already running
	}

	// In production, this would execute nvidia-cuda-mps-control commands
	// via the node agent
	c.daemonStatus[key] = &MPSDaemonStatus{
		NodeName:  nodeName,
		GPUUID:    gpuUUID,
		Running:   true,
		StartedAt: time.Now(),
	}

	return nil
}

// AllocateMPSClient allocates an MPS client for a workload
func (c *MPSController) AllocateMPSClient(
	ctx context.Context,
	workloadUID string,
	gpuUUID string,
	nodeName string,
	threadPercentage int,
	memoryLimitMB int,
) (*MPSAllocation, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Ensure daemon is running
	key := fmt.Sprintf("%s/%s", nodeName, gpuUUID)
	status, ok := c.daemonStatus[key]
	if !ok || !status.Running {
		return nil, fmt.Errorf("MPS daemon not running on %s", key)
	}

	// Check client limit
	if status.ClientCount >= c.config.MaxClientsPerGPU {
		return nil, fmt.Errorf("MPS client limit reached on %s", key)
	}

	// Apply defaults
	if threadPercentage == 0 {
		threadPercentage = c.config.DefaultThreadPercentage
	}

	allocation := &MPSAllocation{
		ID:               fmt.Sprintf("mps-%s-%d", gpuUUID[:8], time.Now().UnixNano()),
		WorkloadUID:      workloadUID,
		GPUUID:           gpuUUID,
		NodeName:         nodeName,
		ThreadPercentage: threadPercentage,
		MemoryLimitMB:    memoryLimitMB,
		AllocatedAt:      time.Now(),
	}

	c.allocations[allocation.ID] = allocation
	status.ClientCount++

	return allocation, nil
}

// ReleaseMPSClient releases an MPS client allocation
func (c *MPSController) ReleaseMPSClient(allocationID string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	allocation, ok := c.allocations[allocationID]
	if !ok {
		return fmt.Errorf("allocation not found: %s", allocationID)
	}

	key := fmt.Sprintf("%s/%s", allocation.NodeName, allocation.GPUUID)
	if status, ok := c.daemonStatus[key]; ok {
		status.ClientCount--
	}

	delete(c.allocations, allocationID)
	return nil
}

// GPUSharingManager coordinates MIG and MPS sharing strategies
type GPUSharingManager struct {
	migController *MIGController
	mpsController *MPSController

	// sharingPolicy determines which sharing method to use
	sharingPolicy SharingPolicy
}

// SharingPolicy defines GPU sharing preferences
type SharingPolicy struct {
	// PreferMIG prefers MIG over MPS when both are available
	PreferMIG bool

	// MIGProfiles profiles to use for MIG sharing
	MIGProfiles []string

	// MPSThreadPercentage default MPS thread allocation
	MPSThreadPercentage int

	// WorkloadTypePolicy maps workload types to sharing methods
	WorkloadTypePolicy map[string]SharingMethod
}

// SharingMethod identifies a GPU sharing approach
type SharingMethod string

const (
	SharingMethodNone      SharingMethod = "None"
	SharingMethodMIG       SharingMethod = "MIG"
	SharingMethodMPS       SharingMethod = "MPS"
	SharingMethodTimeSlice SharingMethod = "TimeSlice"
)

// NewGPUSharingManager creates a new sharing manager
func NewGPUSharingManager(
	migController *MIGController,
	mpsController *MPSController,
	policy SharingPolicy,
) *GPUSharingManager {
	return &GPUSharingManager{
		migController: migController,
		mpsController: mpsController,
		sharingPolicy: policy,
	}
}

// AllocateSharedGPU allocates GPU resources using appropriate sharing method
func (m *GPUSharingManager) AllocateSharedGPU(
	ctx context.Context,
	workloadUID string,
	workloadType string,
	requirements GPUSharingRequirements,
) (*GPUSharingAllocation, error) {
	// Determine sharing method based on policy
	method := m.determineSharingMethod(workloadType, requirements)

	switch method {
	case SharingMethodMIG:
		migAlloc, err := m.migController.AllocateMIGInstance(
			ctx,
			workloadUID,
			requirements.PreferredMIGProfile,
			requirements.PreferredNode,
		)
		if err != nil {
			return nil, err
		}
		return &GPUSharingAllocation{
			Method:        SharingMethodMIG,
			MIGAllocation: migAlloc,
		}, nil

	case SharingMethodMPS:
		mpsAlloc, err := m.mpsController.AllocateMPSClient(
			ctx,
			workloadUID,
			requirements.GPUUID,
			requirements.PreferredNode,
			requirements.ThreadPercentage,
			requirements.MemoryLimitMB,
		)
		if err != nil {
			return nil, err
		}
		return &GPUSharingAllocation{
			Method:        SharingMethodMPS,
			MPSAllocation: mpsAlloc,
		}, nil

	default:
		return nil, fmt.Errorf("unsupported sharing method: %s", method)
	}
}

// determineSharingMethod selects the best sharing method
func (m *GPUSharingManager) determineSharingMethod(
	workloadType string,
	requirements GPUSharingRequirements,
) SharingMethod {
	// Check policy for workload type
	if method, ok := m.sharingPolicy.WorkloadTypePolicy[workloadType]; ok {
		return method
	}

	// Default logic
	if requirements.RequireIsolation {
		return SharingMethodMIG
	}

	if m.sharingPolicy.PreferMIG && requirements.PreferredMIGProfile != "" {
		return SharingMethodMIG
	}

	return SharingMethodMPS
}

// GPUSharingRequirements specifies sharing requirements
type GPUSharingRequirements struct {
	// PreferredNode for placement
	PreferredNode string

	// GPUUID specific GPU (for MPS)
	GPUUID string

	// PreferredMIGProfile for MIG allocation
	PreferredMIGProfile string

	// RequireIsolation requires hardware isolation (MIG)
	RequireIsolation bool

	// ThreadPercentage for MPS
	ThreadPercentage int

	// MemoryLimitMB for MPS
	MemoryLimitMB int
}

// GPUSharingAllocation represents an allocation using any sharing method
type GPUSharingAllocation struct {
	Method        SharingMethod
	MIGAllocation *MIGAllocation
	MPSAllocation *MPSAllocation
}

// Release releases the sharing allocation
func (a *GPUSharingAllocation) Release(manager *GPUSharingManager) error {
	switch a.Method {
	case SharingMethodMIG:
		if a.MIGAllocation != nil {
			return manager.migController.ReleaseMIGAllocation(a.MIGAllocation.ID)
		}
	case SharingMethodMPS:
		if a.MPSAllocation != nil {
			return manager.mpsController.ReleaseMPSClient(a.MPSAllocation.ID)
		}
	}
	return nil
}
