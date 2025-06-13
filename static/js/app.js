// Golf Prediction System JavaScript

// Global variables
let currentData = null;
let charts = {};

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    loadInitialData();
});

function initializeApp() {
    console.log('Golf Prediction System initialized');
    
    // Add fade-in animation to main content
    const mainContent = document.querySelector('main');
    if (mainContent) {
        mainContent.classList.add('fade-in');
    }
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

function setupEventListeners() {
    // Refresh button functionality
    const refreshButtons = document.querySelectorAll('[data-action="refresh"]');
    refreshButtons.forEach(button => {
        button.addEventListener('click', function() {
            refreshData();
        });
    });
    
    // Export functionality
    const exportButtons = document.querySelectorAll('[data-action="export"]');
    exportButtons.forEach(button => {
        button.addEventListener('click', function() {
            const format = this.dataset.format || 'csv';
            exportData(format);
        });
    });
    
    // Search functionality
    const searchInput = document.getElementById('playerSearch');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            filterPlayers(this.value);
        });
    }
}

function loadInitialData() {
    // Load any initial data needed for the current page
    const currentPage = getCurrentPage();
    
    switch(currentPage) {
        case 'predictions':
            loadPredictionsData();
            break;
        case 'value-picks':
            loadValuePicksData();
            break;
        case 'analytics':
            loadAnalyticsData();
            break;
        default:
            loadDashboardData();
    }
}

function getCurrentPage() {
    const path = window.location.pathname;
    if (path.includes('predictions')) return 'predictions';
    if (path.includes('value-picks')) return 'value-picks';
    if (path.includes('analytics')) return 'analytics';
    return 'dashboard';
}

// Data loading functions
async function loadDashboardData() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        if (data.status === 'success') {
            currentData = data;
            updateDashboardStats(data);
        }
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showError('Failed to load dashboard data');
    }
}

async function loadPredictionsData() {
    try {
        showLoading('predictionsTable');
        
        const response = await fetch('/api/predictions?limit=25');
        const data = await response.json();
        
        if (data.status === 'success') {
            currentData = data;
            updatePredictionsTable(data.predictions);
        }
    } catch (error) {
        console.error('Error loading predictions:', error);
        showError('Failed to load predictions');
    } finally {
        hideLoading('predictionsTable');
    }
}

async function loadValuePicksData() {
    try {
        const response = await fetch('/api/value-picks');
        const data = await response.json();
        
        if (data.status === 'success') {
            currentData = data;
            updateValuePicksTable(data.value_picks);
        }
    } catch (error) {
        console.error('Error loading value picks:', error);
        showError('Failed to load value picks');
    }
}

async function loadAnalyticsData() {
    try {
        const [statsResponse, eliteResponse] = await Promise.all([
            fetch('/api/stats'),
            fetch('/api/elite-analysis')
        ]);
        
        const statsData = await statsResponse.json();
        const eliteData = await eliteResponse.json();
        
        if (statsData.status === 'success' && eliteData.status === 'success') {
            currentData = { stats: statsData, elite: eliteData };
            updateAnalyticsCharts(statsData, eliteData);
        }
    } catch (error) {
        console.error('Error loading analytics data:', error);
        showError('Failed to load analytics data');
    }
}

// UI update functions
function updateDashboardStats(data) {
    // Update statistics cards if they exist
    const statsElements = {
        'total-players': data.tournament_summary?.total_players,
        'with-predictions': data.tournament_summary?.with_predictions,
        'professionals': data.tournament_summary?.professionals,
        'countries': data.tournament_summary?.countries
    };
    
    Object.entries(statsElements).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element && value !== undefined) {
            element.textContent = value;
        }
    });
}

function updatePredictionsTable(predictions) {
    const tableBody = document.querySelector('#predictionsTable tbody');
    if (!tableBody) return;
    
    tableBody.innerHTML = '';
    
    predictions.forEach((player, index) => {
        const row = createPredictionRow(player, index + 1);
        tableBody.appendChild(row);
    });
}

function createPredictionRow(player, rank) {
    const row = document.createElement('tr');
    row.innerHTML = `
        <td><strong class="text-primary">${rank}</strong></td>
        <td><strong>${player.player_name}</strong></td>
        <td><span class="badge bg-secondary">${player.country || 'N/A'}</span></td>
        <td>#${player.datagolf_rank || 'N/A'}</td>
        <td><span class="badge bg-primary fs-6">${player.final_prediction_score.toFixed(3)}</span></td>
        <td>${player.course_fit_score.toFixed(3)}</td>
        <td>${getFitCategoryBadge(player.fit_category)}</td>
        <td>${getPenaltyText(player.fit_multiplier)}</td>
        <td>${player.general_form_score.toFixed(3)}</td>
        <td>${player.sg_total ? player.sg_total.toFixed(2) : 'N/A'}</td>
    `;
    
    return row;
}

function getFitCategoryBadge(category) {
    const badgeClasses = {
        'Good Fit': 'bg-success',
        'Average Fit': 'bg-warning text-dark',
        'Poor Fit': 'bg-danger',
        'Very Poor Fit': 'bg-dark'
    };
    
    const className = badgeClasses[category] || 'bg-secondary';
    return `<span class="badge ${className}">${category}</span>`;
}

function getPenaltyText(penalty) {
    if (penalty < 0.95) {
        return `<span class="text-danger">${penalty.toFixed(3)}</span>`;
    } else if (penalty < 0.99) {
        return `<span class="text-warning">${penalty.toFixed(3)}</span>`;
    } else {
        return `<span class="text-success">${penalty.toFixed(3)}</span>`;
    }
}

// Utility functions
function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.add('loading');
    }
}

function hideLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.remove('loading');
    }
}

function showError(message) {
    // Create or update error alert
    let errorAlert = document.getElementById('error-alert');
    if (!errorAlert) {
        errorAlert = document.createElement('div');
        errorAlert.id = 'error-alert';
        errorAlert.className = 'alert alert-danger alert-dismissible fade show';
        errorAlert.innerHTML = `
            <strong>Error:</strong> <span id="error-message"></span>
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        const container = document.querySelector('main .container') || document.querySelector('main');
        container.insertBefore(errorAlert, container.firstChild);
    }
    
    document.getElementById('error-message').textContent = message;
}

function refreshData() {
    loadInitialData();
}

function exportData(format = 'csv') {
    if (!currentData) {
        showError('No data available to export');
        return;
    }
    
    const currentPage = getCurrentPage();
    let endpoint = '/api/predictions';
    
    switch(currentPage) {
        case 'value-picks':
            endpoint = '/api/value-picks';
            break;
        case 'analytics':
            endpoint = '/api/elite-analysis';
            break;
    }
    
    window.open(endpoint, '_blank');
}

function filterPlayers(searchTerm) {
    const tableRows = document.querySelectorAll('#predictionsTable tbody tr');
    const term = searchTerm.toLowerCase();
    
    tableRows.forEach(row => {
        const playerName = row.cells[1].textContent.toLowerCase();
        const country = row.cells[2].textContent.toLowerCase();
        
        if (playerName.includes(term) || country.includes(term)) {
            row.style.display = '';
        } else {
            row.style.display = 'none';
        }
    });
}

// Chart utilities
function createChart(canvasId, config) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    
    const ctx = canvas.getContext('2d');
    
    // Destroy existing chart if it exists
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }
    
    charts[canvasId] = new Chart(ctx, config);
    return charts[canvasId];
}

// API health check
async function checkApiHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (data.status === 'healthy') {
            console.log('API is healthy');
            return true;
        } else {
            console.warn('API health check failed:', data);
            return false;
        }
    } catch (error) {
        console.error('API health check error:', error);
        return false;
    }
}

// Periodic health check
setInterval(checkApiHealth, 300000); // Check every 5 minutes

// Export functions for global access
window.GolfPredictionApp = {
    refreshData,
    exportData,
    filterPlayers,
    checkApiHealth,
    createChart
};
