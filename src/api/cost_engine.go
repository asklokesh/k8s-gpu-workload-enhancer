// Package api provides the cost engine and chargeback system for GPU workloads.
// It implements accurate cost tracking, attribution, and optimization recommendations
// based on actual GPU utilization and cloud provider pricing models.
package api

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

// CostEngine provides GPU cost tracking and optimization
type CostEngine struct {
	mu sync.RWMutex

	// pricingModels maps GPU types to pricing information
	pricingModels map[string]*GPUPricingModel

	// usageRecords stores historical usage data
	usageRecords map[string][]*UsageRecord

	// budgets tracks namespace/team budgets
	budgets map[string]*Budget

	// alerts active budget alerts
	alerts []*BudgetAlert

	// config holds engine configuration
	config CostEngineConfig

	// metricsCollector for telemetry
	metricsCollector MetricsCollector
}

// CostEngineConfig holds configuration
type CostEngineConfig struct {
	// DefaultCurrency for cost calculations
	DefaultCurrency string

	// BillingGranularity minimum billing increment
	BillingGranularity time.Duration

	// RetentionPeriod for historical data
	RetentionPeriod time.Duration

	// EnableSpotOptimization enables spot instance recommendations
	EnableSpotOptimization bool

	// EnableRightsizing enables resource rightsizing recommendations
	EnableRightsizing bool

	// AlertThresholds for budget warnings
	AlertThresholds []float64
}

// DefaultCostEngineConfig returns sensible defaults
func DefaultCostEngineConfig() CostEngineConfig {
	return CostEngineConfig{
		DefaultCurrency:        "USD",
		BillingGranularity:     time.Second,
		RetentionPeriod:        90 * 24 * time.Hour,
		EnableSpotOptimization: true,
		EnableRightsizing:      true,
		AlertThresholds:        []float64{0.5, 0.75, 0.9, 1.0},
	}
}

// GPUPricingModel defines pricing for a GPU type
type GPUPricingModel struct {
	// GPUType identifier (e.g., "nvidia-h100-80gb")
	GPUType string `json:"gpuType"`

	// OnDemandPricePerHour base on-demand price
	OnDemandPricePerHour float64 `json:"onDemandPricePerHour"`

	// SpotPricePerHour current spot price
	SpotPricePerHour float64 `json:"spotPricePerHour"`

	// ReservedPricePerHour 1-year reserved price
	ReservedPricePerHour float64 `json:"reservedPricePerHour"`

	// MIGPricing maps MIG profiles to fractional prices
	MIGPricing map[string]float64 `json:"migPricing"`

	// Provider cloud provider name
	Provider string `json:"provider"`

	// Region pricing region
	Region string `json:"region"`

	// LastUpdated pricing update timestamp
	LastUpdated time.Time `json:"lastUpdated"`
}

// UsageRecord tracks GPU usage for billing
type UsageRecord struct {
	// ID unique record identifier
	ID string `json:"id"`

	// WorkloadUID workload identifier
	WorkloadUID string `json:"workloadUid"`

	// Namespace Kubernetes namespace
	Namespace string `json:"namespace"`

	// TeamID for cost attribution
	TeamID string `json:"teamId"`

	// GPUType type of GPU used
	GPUType string `json:"gpuType"`

	// GPUCount number of GPUs
	GPUCount int `json:"gpuCount"`

	// MIGProfile if using MIG
	MIGProfile string `json:"migProfile,omitempty"`

	// StartTime when usage began
	StartTime time.Time `json:"startTime"`

	// EndTime when usage ended (nil if ongoing)
	EndTime *time.Time `json:"endTime,omitempty"`

	// DurationSeconds total usage duration
	DurationSeconds float64 `json:"durationSeconds"`

	// Utilization average GPU utilization
	Utilization GPUUtilizationMetrics `json:"utilization"`

	// PricingTier (OnDemand, Spot, Reserved)
	PricingTier PricingTier `json:"pricingTier"`

	// RawCost before adjustments
	RawCost float64 `json:"rawCost"`

	// AdjustedCost after utilization-based adjustments
	AdjustedCost float64 `json:"adjustedCost"`

	// Currency for costs
	Currency string `json:"currency"`

	// Labels additional metadata
	Labels map[string]string `json:"labels,omitempty"`
}

// GPUUtilizationMetrics for usage-based billing
type GPUUtilizationMetrics struct {
	// AvgGPUPercent average SM utilization
	AvgGPUPercent float64 `json:"avgGpuPercent"`

	// AvgMemoryPercent average memory utilization
	AvgMemoryPercent float64 `json:"avgMemoryPercent"`

	// PeakGPUPercent peak SM utilization
	PeakGPUPercent float64 `json:"peakGpuPercent"`

	// PeakMemoryPercent peak memory utilization
	PeakMemoryPercent float64 `json:"peakMemoryPercent"`

	// IdleSeconds seconds with <5% utilization
	IdleSeconds float64 `json:"idleSeconds"`
}

// PricingTier represents pricing category
type PricingTier string

const (
	PricingTierOnDemand PricingTier = "OnDemand"
	PricingTierSpot     PricingTier = "Spot"
	PricingTierReserved PricingTier = "Reserved"
)

// Budget defines spending limits
type Budget struct {
	// ID unique budget identifier
	ID string `json:"id"`

	// Name human-readable name
	Name string `json:"name"`

	// Scope (namespace, team, project)
	Scope BudgetScope `json:"scope"`

	// ScopeID namespace/team/project identifier
	ScopeID string `json:"scopeId"`

	// LimitAmount spending limit
	LimitAmount float64 `json:"limitAmount"`

	// Currency for the limit
	Currency string `json:"currency"`

	// Period budget period
	Period BudgetPeriod `json:"period"`

	// CurrentSpend current period spend
	CurrentSpend float64 `json:"currentSpend"`

	// PeriodStart current period start
	PeriodStart time.Time `json:"periodStart"`

	// AlertsEnabled enable budget alerts
	AlertsEnabled bool `json:"alertsEnabled"`

	// EnforcementPolicy what to do when exceeded
	EnforcementPolicy EnforcementPolicy `json:"enforcementPolicy"`
}

// BudgetScope defines what the budget applies to
type BudgetScope string

const (
	BudgetScopeNamespace BudgetScope = "Namespace"
	BudgetScopeTeam      BudgetScope = "Team"
	BudgetScopeProject   BudgetScope = "Project"
	BudgetScopeCluster   BudgetScope = "Cluster"
)

// BudgetPeriod defines budget time window
type BudgetPeriod string

const (
	BudgetPeriodDaily   BudgetPeriod = "Daily"
	BudgetPeriodWeekly  BudgetPeriod = "Weekly"
	BudgetPeriodMonthly BudgetPeriod = "Monthly"
)

// EnforcementPolicy defines budget enforcement
type EnforcementPolicy string

const (
	EnforcementPolicyAlert    EnforcementPolicy = "Alert"
	EnforcementPolicyThrottle EnforcementPolicy = "Throttle"
	EnforcementPolicyBlock    EnforcementPolicy = "Block"
)

// BudgetAlert represents a budget warning
type BudgetAlert struct {
	// BudgetID associated budget
	BudgetID string `json:"budgetId"`

	// Severity (Warning, Critical)
	Severity AlertSeverity `json:"severity"`

	// ThresholdPercent that triggered alert
	ThresholdPercent float64 `json:"thresholdPercent"`

	// CurrentSpend at time of alert
	CurrentSpend float64 `json:"currentSpend"`

	// Message alert message
	Message string `json:"message"`

	// Timestamp when alert was created
	Timestamp time.Time `json:"timestamp"`

	// Acknowledged if alert was acknowledged
	Acknowledged bool `json:"acknowledged"`
}

// AlertSeverity levels
type AlertSeverity string

const (
	AlertSeverityInfo     AlertSeverity = "Info"
	AlertSeverityWarning  AlertSeverity = "Warning"
	AlertSeverityCritical AlertSeverity = "Critical"
)

// MetricsCollector interface for telemetry
type MetricsCollector interface {
	// RecordCost records a cost metric
	RecordCost(namespace, team string, amount float64, labels map[string]string)

	// RecordUtilization records utilization metric
	RecordUtilization(gpuUUID string, utilization float64)
}

// NewCostEngine creates a new cost engine
func NewCostEngine(config CostEngineConfig, metricsCollector MetricsCollector) *CostEngine {
	engine := &CostEngine{
		pricingModels:    make(map[string]*GPUPricingModel),
		usageRecords:     make(map[string][]*UsageRecord),
		budgets:          make(map[string]*Budget),
		alerts:           make([]*BudgetAlert, 0),
		config:           config,
		metricsCollector: metricsCollector,
	}

	// Initialize default pricing models
	engine.initDefaultPricing()

	return engine
}

// initDefaultPricing sets up default GPU pricing
func (e *CostEngine) initDefaultPricing() {
	// H100 pricing (approximate, varies by provider/region)
	e.pricingModels["nvidia-h100-80gb"] = &GPUPricingModel{
		GPUType:              "nvidia-h100-80gb",
		OnDemandPricePerHour: 3.00,
		SpotPricePerHour:     1.20,
		ReservedPricePerHour: 2.10,
		MIGPricing: map[string]float64{
			"1g.10gb": 0.43, // ~1/7 of full GPU
			"2g.20gb": 0.86, // ~2/7 of full GPU
			"3g.40gb": 1.29, // ~3/7 of full GPU
			"4g.40gb": 1.72, // ~4/7 of full GPU
			"7g.80gb": 3.00, // Full GPU
		},
		Provider:    "generic",
		Region:      "us-west",
		LastUpdated: time.Now(),
	}

	// A100 pricing
	e.pricingModels["nvidia-a100-80gb"] = &GPUPricingModel{
		GPUType:              "nvidia-a100-80gb",
		OnDemandPricePerHour: 2.50,
		SpotPricePerHour:     1.00,
		ReservedPricePerHour: 1.75,
		MIGPricing: map[string]float64{
			"1g.10gb": 0.36,
			"2g.20gb": 0.72,
			"3g.40gb": 1.08,
			"4g.40gb": 1.44,
			"7g.80gb": 2.50,
		},
		Provider:    "generic",
		Region:      "us-west",
		LastUpdated: time.Now(),
	}

	// L40S pricing
	e.pricingModels["nvidia-l40s"] = &GPUPricingModel{
		GPUType:              "nvidia-l40s",
		OnDemandPricePerHour: 1.50,
		SpotPricePerHour:     0.60,
		ReservedPricePerHour: 1.05,
		Provider:             "generic",
		Region:               "us-west",
		LastUpdated:          time.Now(),
	}
}

// StartUsageTracking begins tracking usage for a workload
func (e *CostEngine) StartUsageTracking(
	workloadUID string,
	namespace string,
	teamID string,
	gpuType string,
	gpuCount int,
	migProfile string,
	pricingTier PricingTier,
	labels map[string]string,
) (*UsageRecord, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	record := &UsageRecord{
		ID:          fmt.Sprintf("usage-%d", time.Now().UnixNano()),
		WorkloadUID: workloadUID,
		Namespace:   namespace,
		TeamID:      teamID,
		GPUType:     gpuType,
		GPUCount:    gpuCount,
		MIGProfile:  migProfile,
		StartTime:   time.Now(),
		PricingTier: pricingTier,
		Currency:    e.config.DefaultCurrency,
		Labels:      labels,
	}

	e.usageRecords[workloadUID] = append(e.usageRecords[workloadUID], record)
	return record, nil
}

// UpdateUsageMetrics updates utilization metrics for a record
func (e *CostEngine) UpdateUsageMetrics(
	workloadUID string,
	metrics GPUUtilizationMetrics,
) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	records, ok := e.usageRecords[workloadUID]
	if !ok || len(records) == 0 {
		return fmt.Errorf("no usage record found for workload %s", workloadUID)
	}

	// Update the most recent record
	record := records[len(records)-1]
	if record.EndTime != nil {
		return fmt.Errorf("usage record already finalized")
	}

	record.Utilization = metrics
	return nil
}

// FinalizeUsage completes usage tracking and calculates cost
func (e *CostEngine) FinalizeUsage(workloadUID string) (*UsageRecord, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	records, ok := e.usageRecords[workloadUID]
	if !ok || len(records) == 0 {
		return nil, fmt.Errorf("no usage record found for workload %s", workloadUID)
	}

	record := records[len(records)-1]
	if record.EndTime != nil {
		return nil, fmt.Errorf("usage record already finalized")
	}

	now := time.Now()
	record.EndTime = &now
	record.DurationSeconds = now.Sub(record.StartTime).Seconds()

	// Calculate cost
	record.RawCost = e.calculateRawCost(record)
	record.AdjustedCost = e.calculateAdjustedCost(record)

	// Update budget
	e.updateBudgetSpend(record)

	// Record metrics
	if e.metricsCollector != nil {
		e.metricsCollector.RecordCost(
			record.Namespace,
			record.TeamID,
			record.AdjustedCost,
			record.Labels,
		)
	}

	return record, nil
}

// calculateRawCost calculates base cost without adjustments
func (e *CostEngine) calculateRawCost(record *UsageRecord) float64 {
	pricing, ok := e.pricingModels[record.GPUType]
	if !ok {
		// Use default pricing if model not found
		pricing = e.pricingModels["nvidia-h100-80gb"]
	}

	var hourlyRate float64

	// Check for MIG pricing
	if record.MIGProfile != "" {
		if migRate, ok := pricing.MIGPricing[record.MIGProfile]; ok {
			hourlyRate = migRate
		} else {
			hourlyRate = pricing.OnDemandPricePerHour
		}
	} else {
		// Full GPU pricing based on tier
		switch record.PricingTier {
		case PricingTierSpot:
			hourlyRate = pricing.SpotPricePerHour
		case PricingTierReserved:
			hourlyRate = pricing.ReservedPricePerHour
		default:
			hourlyRate = pricing.OnDemandPricePerHour
		}
	}

	hours := record.DurationSeconds / 3600.0
	return hourlyRate * float64(record.GPUCount) * hours
}

// calculateAdjustedCost applies utilization-based adjustments
func (e *CostEngine) calculateAdjustedCost(record *UsageRecord) float64 {
	rawCost := record.RawCost

	// No adjustment for very short durations
	if record.DurationSeconds < 60 {
		return rawCost
	}

	// Calculate utilization factor
	avgUtil := (record.Utilization.AvgGPUPercent + record.Utilization.AvgMemoryPercent) / 2

	// Apply idle penalty (optional, configurable)
	idleRatio := record.Utilization.IdleSeconds / record.DurationSeconds
	if idleRatio > 0.5 {
		// Add idle surcharge for very underutilized workloads
		// This encourages right-sizing
		rawCost *= (1 + idleRatio*0.1)
	}

	// For very high utilization, provide a small discount
	if avgUtil > 80 {
		rawCost *= 0.95
	}

	return math.Round(rawCost*100) / 100 // Round to cents
}

// updateBudgetSpend updates relevant budget with new spend
func (e *CostEngine) updateBudgetSpend(record *UsageRecord) {
	for _, budget := range e.budgets {
		if e.budgetApplies(budget, record) {
			budget.CurrentSpend += record.AdjustedCost
			e.checkBudgetAlerts(budget)
		}
	}
}

// budgetApplies checks if budget applies to the record
func (e *CostEngine) budgetApplies(budget *Budget, record *UsageRecord) bool {
	switch budget.Scope {
	case BudgetScopeNamespace:
		return budget.ScopeID == record.Namespace
	case BudgetScopeTeam:
		return budget.ScopeID == record.TeamID
	default:
		return false
	}
}

// checkBudgetAlerts checks and creates budget alerts
func (e *CostEngine) checkBudgetAlerts(budget *Budget) {
	if !budget.AlertsEnabled {
		return
	}

	spendRatio := budget.CurrentSpend / budget.LimitAmount

	for _, threshold := range e.config.AlertThresholds {
		if spendRatio >= threshold {
			// Check if alert already exists for this threshold
			exists := false
			for _, alert := range e.alerts {
				if alert.BudgetID == budget.ID && alert.ThresholdPercent == threshold {
					exists = true
					break
				}
			}

			if !exists {
				severity := AlertSeverityInfo
				if threshold >= 0.9 {
					severity = AlertSeverityCritical
				} else if threshold >= 0.75 {
					severity = AlertSeverityWarning
				}

				e.alerts = append(e.alerts, &BudgetAlert{
					BudgetID:         budget.ID,
					Severity:         severity,
					ThresholdPercent: threshold,
					CurrentSpend:     budget.CurrentSpend,
					Message: fmt.Sprintf("Budget '%s' has reached %.0f%% of limit ($%.2f of $%.2f)",
						budget.Name, threshold*100, budget.CurrentSpend, budget.LimitAmount),
					Timestamp: time.Now(),
				})
			}
		}
	}
}

// CreateBudget creates a new budget
func (e *CostEngine) CreateBudget(budget *Budget) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if budget.ID == "" {
		budget.ID = fmt.Sprintf("budget-%d", time.Now().UnixNano())
	}

	budget.PeriodStart = time.Now()
	budget.CurrentSpend = 0

	e.budgets[budget.ID] = budget
	return nil
}

// GetBudget returns a budget by ID
func (e *CostEngine) GetBudget(budgetID string) (*Budget, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	budget, ok := e.budgets[budgetID]
	return budget, ok
}

// GetCostSummary returns cost summary for a scope
func (e *CostEngine) GetCostSummary(
	scope BudgetScope,
	scopeID string,
	start, end time.Time,
) *CostSummary {
	e.mu.RLock()
	defer e.mu.RUnlock()

	summary := &CostSummary{
		Scope:       scope,
		ScopeID:     scopeID,
		StartTime:   start,
		EndTime:     end,
		Currency:    e.config.DefaultCurrency,
		ByGPUType:   make(map[string]float64),
		ByWorkload:  make(map[string]float64),
		ByPricingTier: make(map[PricingTier]float64),
	}

	for _, records := range e.usageRecords {
		for _, record := range records {
			if !e.recordInScope(record, scope, scopeID) {
				continue
			}
			if record.EndTime == nil || record.EndTime.Before(start) || record.StartTime.After(end) {
				continue
			}

			summary.TotalCost += record.AdjustedCost
			summary.TotalGPUHours += record.DurationSeconds / 3600.0 * float64(record.GPUCount)
			summary.RecordCount++

			summary.ByGPUType[record.GPUType] += record.AdjustedCost
			summary.ByWorkload[record.WorkloadUID] += record.AdjustedCost
			summary.ByPricingTier[record.PricingTier] += record.AdjustedCost

			// Track utilization
			summary.AvgUtilization += record.Utilization.AvgGPUPercent
		}
	}

	if summary.RecordCount > 0 {
		summary.AvgUtilization /= float64(summary.RecordCount)
		summary.CostPerGPUHour = summary.TotalCost / summary.TotalGPUHours
	}

	return summary
}

// recordInScope checks if record belongs to scope
func (e *CostEngine) recordInScope(record *UsageRecord, scope BudgetScope, scopeID string) bool {
	switch scope {
	case BudgetScopeNamespace:
		return record.Namespace == scopeID
	case BudgetScopeTeam:
		return record.TeamID == scopeID
	case BudgetScopeCluster:
		return true
	default:
		return false
	}
}

// CostSummary aggregates cost information
type CostSummary struct {
	Scope           BudgetScope            `json:"scope"`
	ScopeID         string                 `json:"scopeId"`
	StartTime       time.Time              `json:"startTime"`
	EndTime         time.Time              `json:"endTime"`
	TotalCost       float64                `json:"totalCost"`
	TotalGPUHours   float64                `json:"totalGpuHours"`
	CostPerGPUHour  float64                `json:"costPerGpuHour"`
	AvgUtilization  float64                `json:"avgUtilization"`
	RecordCount     int                    `json:"recordCount"`
	Currency        string                 `json:"currency"`
	ByGPUType       map[string]float64     `json:"byGpuType"`
	ByWorkload      map[string]float64     `json:"byWorkload"`
	ByPricingTier   map[PricingTier]float64 `json:"byPricingTier"`
}

// GetOptimizationRecommendations returns cost optimization suggestions
func (e *CostEngine) GetOptimizationRecommendations(
	ctx context.Context,
	scope BudgetScope,
	scopeID string,
) []OptimizationRecommendation {
	e.mu.RLock()
	defer e.mu.RUnlock()

	recommendations := make([]OptimizationRecommendation, 0)

	// Analyze usage patterns
	workloadStats := make(map[string]*workloadUsageStats)

	for _, records := range e.usageRecords {
		for _, record := range records {
			if !e.recordInScope(record, scope, scopeID) {
				continue
			}

			stats, ok := workloadStats[record.WorkloadUID]
			if !ok {
				stats = &workloadUsageStats{
					workloadUID: record.WorkloadUID,
				}
				workloadStats[record.WorkloadUID] = stats
			}

			stats.totalCost += record.AdjustedCost
			stats.totalDuration += record.DurationSeconds
			stats.avgUtilization = (stats.avgUtilization*float64(stats.recordCount) +
				record.Utilization.AvgGPUPercent) / float64(stats.recordCount+1)
			stats.recordCount++
			stats.gpuType = record.GPUType
			stats.gpuCount = record.GPUCount
			stats.pricingTier = record.PricingTier
		}
	}

	// Generate recommendations
	for _, stats := range workloadStats {
		// Recommendation: Switch to spot instances
		if e.config.EnableSpotOptimization && stats.pricingTier == PricingTierOnDemand {
			pricing := e.pricingModels[stats.gpuType]
			if pricing != nil {
				savings := (pricing.OnDemandPricePerHour - pricing.SpotPricePerHour) *
					(stats.totalDuration / 3600.0) * float64(stats.gpuCount)
				if savings > 10 { // Only recommend if meaningful savings
					recommendations = append(recommendations, OptimizationRecommendation{
						Type:            RecommendationTypeSpot,
						WorkloadUID:     stats.workloadUID,
						CurrentCost:     stats.totalCost,
						ProjectedSaving: savings,
						SavingPercent:   (savings / stats.totalCost) * 100,
						Description:     fmt.Sprintf("Switch to spot instances for %s", stats.workloadUID),
						Confidence:      0.8,
					})
				}
			}
		}

		// Recommendation: Right-size (use smaller GPU or MIG)
		if e.config.EnableRightsizing && stats.avgUtilization < 40 {
			migSavings := stats.totalCost * 0.6 // Estimate 60% savings with MIG
			recommendations = append(recommendations, OptimizationRecommendation{
				Type:            RecommendationTypeRightsize,
				WorkloadUID:     stats.workloadUID,
				CurrentCost:     stats.totalCost,
				ProjectedSaving: migSavings,
				SavingPercent:   60,
				Description: fmt.Sprintf("Low utilization (%.0f%%) - consider MIG partitioning for %s",
					stats.avgUtilization, stats.workloadUID),
				Confidence: 0.7,
			})
		}

		// Recommendation: Consolidate workloads
		if stats.avgUtilization < 30 && stats.recordCount > 5 {
			recommendations = append(recommendations, OptimizationRecommendation{
				Type:            RecommendationTypeConsolidate,
				WorkloadUID:     stats.workloadUID,
				CurrentCost:     stats.totalCost,
				ProjectedSaving: stats.totalCost * 0.5,
				SavingPercent:   50,
				Description: fmt.Sprintf("Very low utilization (%.0f%%) - consolidate with other workloads",
					stats.avgUtilization),
				Confidence: 0.6,
			})
		}
	}

	// Sort by potential savings
	sort.Slice(recommendations, func(i, j int) bool {
		return recommendations[i].ProjectedSaving > recommendations[j].ProjectedSaving
	})

	return recommendations
}

type workloadUsageStats struct {
	workloadUID    string
	totalCost      float64
	totalDuration  float64
	avgUtilization float64
	recordCount    int
	gpuType        string
	gpuCount       int
	pricingTier    PricingTier
}

// OptimizationRecommendation represents a cost optimization suggestion
type OptimizationRecommendation struct {
	Type            RecommendationType `json:"type"`
	WorkloadUID     string             `json:"workloadUid"`
	CurrentCost     float64            `json:"currentCost"`
	ProjectedSaving float64            `json:"projectedSaving"`
	SavingPercent   float64            `json:"savingPercent"`
	Description     string             `json:"description"`
	Confidence      float64            `json:"confidence"`
}

// RecommendationType categorizes optimization suggestions
type RecommendationType string

const (
	RecommendationTypeSpot        RecommendationType = "UseSpotInstances"
	RecommendationTypeRightsize   RecommendationType = "Rightsize"
	RecommendationTypeConsolidate RecommendationType = "Consolidate"
	RecommendationTypeSchedule    RecommendationType = "OptimizeSchedule"
	RecommendationTypeReserved    RecommendationType = "UseReservedCapacity"
)

// GetAlerts returns active budget alerts
func (e *CostEngine) GetAlerts() []*BudgetAlert {
	e.mu.RLock()
	defer e.mu.RUnlock()

	alerts := make([]*BudgetAlert, len(e.alerts))
	copy(alerts, e.alerts)
	return alerts
}

// AcknowledgeAlert marks an alert as acknowledged
func (e *CostEngine) AcknowledgeAlert(budgetID string, threshold float64) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	for _, alert := range e.alerts {
		if alert.BudgetID == budgetID && alert.ThresholdPercent == threshold {
			alert.Acknowledged = true
			return nil
		}
	}
	return fmt.Errorf("alert not found")
}

// ExportChargebackReport generates a chargeback report
func (e *CostEngine) ExportChargebackReport(
	start, end time.Time,
	groupBy BudgetScope,
) *ChargebackReport {
	e.mu.RLock()
	defer e.mu.RUnlock()

	report := &ChargebackReport{
		StartTime:  start,
		EndTime:    end,
		GroupBy:    groupBy,
		Currency:   e.config.DefaultCurrency,
		LineItems:  make([]ChargebackLineItem, 0),
		GeneratedAt: time.Now(),
	}

	// Aggregate by group
	groups := make(map[string]*ChargebackLineItem)

	for _, records := range e.usageRecords {
		for _, record := range records {
			if record.EndTime == nil || record.EndTime.Before(start) || record.StartTime.After(end) {
				continue
			}

			var groupID string
			switch groupBy {
			case BudgetScopeNamespace:
				groupID = record.Namespace
			case BudgetScopeTeam:
				groupID = record.TeamID
			default:
				groupID = "cluster"
			}

			item, ok := groups[groupID]
			if !ok {
				item = &ChargebackLineItem{
					GroupID: groupID,
				}
				groups[groupID] = item
			}

			item.TotalCost += record.AdjustedCost
			item.GPUHours += record.DurationSeconds / 3600.0 * float64(record.GPUCount)
			item.WorkloadCount++
		}
	}

	for _, item := range groups {
		if item.GPUHours > 0 {
			item.AvgCostPerGPUHour = item.TotalCost / item.GPUHours
		}
		report.LineItems = append(report.LineItems, *item)
		report.TotalCost += item.TotalCost
	}

	// Sort by cost descending
	sort.Slice(report.LineItems, func(i, j int) bool {
		return report.LineItems[i].TotalCost > report.LineItems[j].TotalCost
	})

	return report
}

// ChargebackReport contains cost attribution data
type ChargebackReport struct {
	StartTime   time.Time            `json:"startTime"`
	EndTime     time.Time            `json:"endTime"`
	GroupBy     BudgetScope          `json:"groupBy"`
	TotalCost   float64              `json:"totalCost"`
	Currency    string               `json:"currency"`
	LineItems   []ChargebackLineItem `json:"lineItems"`
	GeneratedAt time.Time            `json:"generatedAt"`
}

// ChargebackLineItem is a single line in the report
type ChargebackLineItem struct {
	GroupID           string  `json:"groupId"`
	TotalCost         float64 `json:"totalCost"`
	GPUHours          float64 `json:"gpuHours"`
	AvgCostPerGPUHour float64 `json:"avgCostPerGpuHour"`
	WorkloadCount     int     `json:"workloadCount"`
}
