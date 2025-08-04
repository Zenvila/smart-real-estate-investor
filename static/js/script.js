// Global variables
let availableOptions = {};
let currentResults = [];

// DOM elements
const searchForm = document.getElementById('searchForm');
const resultsSection = document.getElementById('results');
const resultsGrid = document.getElementById('resultsGrid');
const resultsSummary = document.getElementById('resultsSummary');
const loadingSpinner = document.getElementById('loadingSpinner');
const noResults = document.getElementById('noResults');
const propertyModal = document.getElementById('propertyModal');
const propertyDetails = document.getElementById('propertyDetails');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    loadAvailableOptions();
});

// Initialize the application
function initializeApp() {
    console.log('SmartReal Estate Pro - Initializing...');
    
    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Active navigation highlighting
    window.addEventListener('scroll', highlightActiveNav);
}

// Setup event listeners
function setupEventListeners() {
    // Search form submission
    searchForm.addEventListener('submit', handleSearch);
    
    // Modal close functionality
    const closeBtn = document.querySelector('.close');
    if (closeBtn) {
        closeBtn.addEventListener('click', closeModal);
    }
    
    // Close modal when clicking outside
    window.addEventListener('click', function(event) {
        if (event.target === propertyModal) {
            closeModal();
        }
    });
    
    // City change handler
    const citySelect = document.getElementById('target_city');
    if (citySelect) {
        citySelect.addEventListener('change', handleCityChange);
    }
}

// Load available options from the API
async function loadAvailableOptions() {
    try {
        const response = await fetch('/api/available_options');
        const data = await response.json();
        
        if (data.success) {
            availableOptions = data.data;
            populateDropdowns();
        } else {
            console.error('Failed to load options:', data.error);
            showNotification('Error loading options', 'error');
        }
    } catch (error) {
        console.error('Error loading options:', error);
        showNotification('Network error loading options', 'error');
    }
}

// Populate dropdowns with available options
function populateDropdowns() {
    const citySelect = document.getElementById('target_city');
    const locationSelect = document.getElementById('target_location');
    
    if (availableOptions.cities && citySelect) {
        availableOptions.cities.forEach(city => {
            const option = document.createElement('option');
            option.value = city;
            option.textContent = city;
            citySelect.appendChild(option);
        });
    }
    
    if (availableOptions.locations && locationSelect) {
        // Add "All Locations" option first
        const allOption = document.createElement('option');
        allOption.value = "";
        allOption.textContent = "All Locations";
        locationSelect.appendChild(allOption);
        
        // Add all locations
        availableOptions.locations.forEach(location => {
            const option = document.createElement('option');
            option.value = location;
            option.textContent = location;
            locationSelect.appendChild(option);
        });
    }
}

// Handle city change to filter locations
function handleCityChange() {
    const citySelect = document.getElementById('target_city');
    const locationSelect = document.getElementById('target_location');
    
    if (!citySelect || !locationSelect) return;
    
    const selectedCity = citySelect.value;
    
    // Clear current locations
    locationSelect.innerHTML = '<option value="">All Locations</option>';
    
    if (selectedCity && availableOptions.locations_by_city && availableOptions.locations_by_city[selectedCity]) {
        // Add city-specific locations
        availableOptions.locations_by_city[selectedCity].forEach(location => {
            const option = document.createElement('option');
            option.value = location;
            option.textContent = location;
            locationSelect.appendChild(option);
        });
    } else if (selectedCity) {
        // If no specific locations found for this city, add a general option
        const option = document.createElement('option');
        option.value = selectedCity;
        option.textContent = `All ${selectedCity} Locations`;
        locationSelect.appendChild(option);
    }
    
    // Also add "All Cities" option to show locations from all cities
    const allCitiesOption = document.createElement('option');
    allCitiesOption.value = "all_cities";
    allCitiesOption.textContent = "All Cities - All Locations";
    locationSelect.appendChild(allCitiesOption);
}

// Handle search form submission
async function handleSearch(event) {
    event.preventDefault();
    
    const formData = new FormData(searchForm);
    const searchData = {
        min_budget: formData.get('min_budget') || 0,
        max_budget: formData.get('max_budget') || Number.MAX_SAFE_INTEGER,
        target_city: formData.get('target_city') || null,
        target_location: formData.get('target_location') || null,
        property_category: formData.get('property_category') || null,
        purpose: formData.get('purpose') || null
    };
    
    // Validate form data
    if (searchData.min_budget > searchData.max_budget) {
        showNotification('Minimum budget cannot be greater than maximum budget', 'error');
        return;
    }
    
    // Show loading state
    showLoading();
    
    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(searchData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentResults = data.data;
            displayResults(data);
        } else {
            showNotification(data.error || 'Search failed', 'error');
            showNoResults();
        }
    } catch (error) {
        console.error('Search error:', error);
        showNotification('Network error during search', 'error');
        showNoResults();
    } finally {
        hideLoading();
    }
}

// Display search results
function displayResults(data) {
    if (!data.data || data.data.length === 0) {
        showNoResults();
        return;
    }
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Display summary
    displaySummary(data.summary);
    
    // Display property cards
    displayPropertyCards(data.data);
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Display results summary
function displaySummary(summary) {
    if (!summary) return;
    
    const summaryHTML = `
        <div class="summary-grid">
            <div class="summary-item">
                <i class="fas fa-home"></i>
                <span class="summary-value">${summary.total_properties}</span>
                <span class="summary-label">Properties Found</span>
            </div>
            <div class="summary-item">
                <i class="fas fa-coins"></i>
                <span class="summary-value">${summary.avg_price}</span>
                <span class="summary-label">Average Price</span>
            </div>
            <div class="summary-item">
                <i class="fas fa-chart-line"></i>
                <span class="summary-value">${summary.avg_roi}</span>
                <span class="summary-label">Average ROI</span>
            </div>
            <div class="summary-item">
                <i class="fas fa-arrows-alt-h"></i>
                <span class="summary-value">${summary.price_range.min} - ${summary.price_range.max}</span>
                <span class="summary-label">Price Range</span>
            </div>
        </div>
    `;
    
    resultsSummary.innerHTML = summaryHTML;
}

// Display property cards
function displayPropertyCards(properties) {
    resultsGrid.innerHTML = '';
    
    properties.forEach(property => {
        const card = createPropertyCard(property);
        resultsGrid.appendChild(card);
    });
}

// Create a property card element
function createPropertyCard(property) {
    const card = document.createElement('div');
    card.className = 'property-card';
    card.onclick = () => showPropertyDetails(property.id);
    
    const recommendationClass = getRecommendationClass(property.recommendation);
    
    card.innerHTML = `
        <div class="property-header">
            <div>
                <div class="property-title">${property.title}</div>
                <div class="property-location">${property.location}, ${property.city}</div>
            </div>
            <div class="property-price">${property.price}</div>
        </div>
        
        <div class="property-details">
            <div class="detail-item">
                <i class="fas fa-building"></i>
                <span>${property.property_category}</span>
            </div>
            <div class="detail-item">
                <i class="fas fa-tag"></i>
                <span>${property.purpose}</span>
            </div>
            <div class="detail-item">
                <i class="fas fa-ruler-combined"></i>
                <span>${property.area}</span>
            </div>
            <div class="detail-item">
                <i class="fas fa-map-marker-alt"></i>
                <span>${property.location}</span>
            </div>
        </div>
        
        <div class="property-metrics">
            <div class="metric">
                <div class="metric-value">${property.roi_prediction}</div>
                <div class="metric-label">ROI Prediction</div>
            </div>
            <div class="metric">
                <div class="metric-value">${property.investment_score}</div>
                <div class="metric-label">Investment Score</div>
            </div>
            <div class="metric">
                <div class="metric-value">${property.risk_level}</div>
                <div class="metric-label">Risk Level</div>
            </div>
        </div>
        

        
        <div class="property-recommendation ${recommendationClass}">
            ${property.recommendation}
        </div>
    `;
    
    return card;
}

// Get recommendation CSS class
function getRecommendationClass(recommendation) {
    const rec = recommendation.toLowerCase();
    if (rec.includes('excellent')) return 'recommendation-excellent';
    if (rec.includes('good')) return 'recommendation-good';
    if (rec.includes('fair')) return 'recommendation-fair';
    if (rec.includes('poor')) return 'recommendation-poor';
    return 'recommendation-fair';
}

// Show property details modal
async function showPropertyDetails(propertyId) {
    try {
        const response = await fetch(`/api/property/${propertyId}`);
        const data = await response.json();
        
        if (data.success) {
            displayPropertyModal(data.data);
        } else {
            showNotification('Failed to load property details', 'error');
        }
    } catch (error) {
        console.error('Error loading property details:', error);
        showNotification('Network error loading property details', 'error');
    }
}

// Display property modal
function displayPropertyModal(property) {
    const recommendationClass = getRecommendationClass(property.recommendation);
    
    propertyDetails.innerHTML = `
        <h2>${property.title}</h2>
        
        <div class="modal-property-info">
            <div class="modal-section">
                <h3><i class="fas fa-info-circle"></i> Property Information</h3>
                <div class="info-grid">
                    <div class="info-item">
                        <span class="info-label">Price:</span>
                        <span class="info-value">${property.price}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Area:</span>
                        <span class="info-value">${property.area}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Location:</span>
                        <span class="info-value">${property.location}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">City:</span>
                        <span class="info-value">${property.city}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Type:</span>
                        <span class="info-value">${property.property_category}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Purpose:</span>
                        <span class="info-value">${property.purpose}</span>
                    </div>
                </div>
            </div>
            
            <div class="modal-section">
                <h3><i class="fas fa-chart-line"></i> Investment Analysis</h3>
                <div class="analysis-grid">
                    <div class="analysis-item">
                        <span class="analysis-label">ROI Prediction:</span>
                        <span class="analysis-value">${property.roi_prediction}</span>
                    </div>
                    <div class="analysis-item">
                        <span class="analysis-label">Risk Level:</span>
                        <span class="analysis-value">${property.risk_level}</span>
                    </div>
                    <div class="analysis-item">
                        <span class="analysis-label">Investment Score:</span>
                        <span class="analysis-value">${property.investment_score}</span>
                    </div>
                    <div class="analysis-item">
                        <span class="analysis-label">Risk-Adjusted ROI:</span>
                        <span class="analysis-value">${property.risk_adjusted_roi}</span>
                    </div>
                    <div class="analysis-item">
                        <span class="analysis-label">Investment Category:</span>
                        <span class="analysis-value">${property.investment_category}</span>
                    </div>
                </div>
            </div>
            

            
            <div class="modal-section">
                <h3><i class="fas fa-lightbulb"></i> Recommendation</h3>
                <div class="recommendation-box ${recommendationClass}">
                    ${property.recommendation}
                </div>
            </div>
        </div>
    `;
    
    propertyModal.style.display = 'block';
}

// Close modal
function closeModal() {
    propertyModal.style.display = 'none';
}

// Clear form
function clearForm() {
    searchForm.reset();
    const locationSelect = document.getElementById('target_location');
    if (locationSelect) {
        locationSelect.innerHTML = '<option value="">All Locations</option>';
    }
}

// Show loading state
function showLoading() {
    loadingSpinner.style.display = 'block';
    resultsGrid.style.display = 'none';
    noResults.style.display = 'none';
}

// Hide loading state
function hideLoading() {
    loadingSpinner.style.display = 'none';
    resultsGrid.style.display = 'grid';
}

// Show no results
function showNoResults() {
    noResults.style.display = 'block';
    resultsGrid.style.display = 'none';
    resultsSummary.innerHTML = '';
}

// Highlight active navigation
function highlightActiveNav() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');
    
    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        if (window.pageYOffset >= sectionTop - 200) {
            current = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
}

// Show notification
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
        <span>${message}</span>
        <button onclick="this.parentElement.remove()">&times;</button>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

// Add notification styles
const notificationStyles = `
    .notification {
        position: fixed;
        top: 100px;
        right: 20px;
        background: white;
        border-radius: 10px;
        padding: 15px 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        z-index: 3000;
        display: flex;
        align-items: center;
        gap: 10px;
        animation: slideIn 0.3s ease;
    }
    
    .notification-info {
        border-left: 4px solid #667eea;
    }
    
    .notification-error {
        border-left: 4px solid #dc3545;
    }
    
    .notification button {
        background: none;
        border: none;
        font-size: 1.2rem;
        cursor: pointer;
        color: #666;
    }
    
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
`;

// Add styles to head
const styleSheet = document.createElement('style');
styleSheet.textContent = notificationStyles;
document.head.appendChild(styleSheet);

// Add modal styles
const modalStyles = `
    .modal-property-info {
        margin-top: 20px;
    }
    
    .modal-section {
        margin-bottom: 30px;
    }
    
    .modal-section h3 {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #667eea;
        margin-bottom: 15px;
        font-size: 1.2rem;
    }
    
    .info-grid,
    .analysis-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 15px;
    }
    
    .info-item,
    .analysis-item {
        display: flex;
        justify-content: space-between;
        padding: 10px;
        background: #f8f9ff;
        border-radius: 8px;
    }
    
    .info-label,
    .analysis-label {
        font-weight: 600;
        color: #333;
    }
    
    .info-value,
    .analysis-value {
        color: #667eea;
        font-weight: 600;
    }
    
    .recommendation-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
    }
    
    .summary-item {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #f8f9ff 0%, #e8ecff 100%);
        border-radius: 15px;
    }
    
    .summary-item i {
        font-size: 2rem;
        color: #667eea;
        margin-bottom: 10px;
    }
    
    .summary-value {
        display: block;
        font-size: 1.5rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 5px;
    }
    
    .summary-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
`;

// Add modal styles to head
const modalStyleSheet = document.createElement('style');
modalStyleSheet.textContent = modalStyles;
document.head.appendChild(modalStyleSheet); 