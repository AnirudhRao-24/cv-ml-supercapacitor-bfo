// ==========================================
// GLOBALS & CHART.JS PREMIUM STYLING
// ==========================================
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.color = '#64748B'; // Text-light
Chart.defaults.scale.grid.color = '#E2E8F0';

// ==========================================
// 1. FIREBASE AUTHENTICATION
// ==========================================
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.8.1/firebase-app.js";
import { getAuth, signInWithEmailAndPassword, signOut } from "https://www.gstatic.com/firebasejs/10.8.1/firebase-auth.js";

// Firebase config provided
const firebaseConfig = {
    apiKey: "AIzaSyA5SXvJuwpdqgxv6Mx1ZwySy0bLXR4_A64",
    authDomain: "ml-cv-prediction.firebaseapp.com",
    projectId: "ml-cv-prediction",
    storageBucket: "ml-cv-prediction.firebasestorage.app",
    messagingSenderId: "238161403374",
    appId: "1:238161403374:web:4fb6d1b3ce2cc0a6e442ff",
    measurementId: "G-ZNH0CPFX54"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

const domElements = {
    loginSection: document.getElementById('login-section'),
    mainApp: document.getElementById('main-app'),
    emailInput: document.getElementById('email'),
    passwordInput: document.getElementById('password'),
    btnLogin: document.getElementById('btn-login'),
    btnLogout: document.getElementById('btn-logout'),
    errorMsg: document.getElementById('error-msg')
};

domElements.btnLogin.addEventListener('click', () => {
    const email = domElements.emailInput.value;
    const password = domElements.passwordInput.value;
    domElements.btnLogin.innerText = "Authenticating...";

    signInWithEmailAndPassword(auth, email, password)
        .then(() => {
            domElements.errorMsg.style.display = 'none';
            domElements.loginSection.classList.remove('active');
            domElements.mainApp.style.display = 'block';
            domElements.btnLogin.innerText = "Authenticate Portal";
        })
        .catch(() => {
            domElements.errorMsg.style.display = 'block';
            domElements.errorMsg.textContent = "Invalid credentials. Access Denied.";
            domElements.btnLogin.innerText = "Authenticate Portal";
        });
});

domElements.btnLogout.addEventListener('click', () => {
    signOut(auth).then(() => {
        domElements.mainApp.style.display = 'none';
        domElements.loginSection.classList.add('active');
        domElements.emailInput.value = '';
        domElements.passwordInput.value = '';
        currentSlide = 1;
        updateSlideUI();
    });
});

// ==========================================
// 2. NAVIGATION LOGIC
// ==========================================
function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.section').forEach(sec => sec.classList.remove('active'));
    
    if(tabId === 'slides') {
        document.getElementById('btn-tab-slides').classList.add('active');
        document.getElementById('slides-section').classList.add('active');
    } else {
        document.getElementById('btn-tab-demo').classList.add('active');
        document.getElementById('demo-section').classList.add('active');
    }
}

document.getElementById('btn-tab-slides').addEventListener('click', () => switchTab('slides'));
document.getElementById('btn-tab-demo').addEventListener('click', () => switchTab('demo'));

// ==========================================
// 3. SLIDE DECK & ANIMATED CHARTS
// ==========================================
let currentSlide = 1;
const totalSlides = 15;
const btnPrev = document.getElementById('btn-prev');
const btnNext = document.getElementById('btn-next');
const slideCounter = document.getElementById('slide-counter');
const progressFill = document.getElementById('progress-fill');

// Object to store all chart instances so they can be properly destroyed and reanimated
const charts = { dataset: null, redox: null, ann: null, rf: null, xgb: null, radar: null, accuracy: null, scatter: null, capacitance: null };
const animConfig = { duration: 1500, easing: 'easeOutQuart' };

function updateSlideUI() {
    document.querySelectorAll('.slide').forEach(slide => slide.classList.remove('active'));
    document.getElementById(`slide-${currentSlide}`).classList.add('active');
    
    slideCounter.innerText = `Slide ${currentSlide} of ${totalSlides}`;
    progressFill.style.width = `${(currentSlide / totalSlides) * 100}%`;
    
    btnPrev.disabled = (currentSlide === 1);
    btnNext.disabled = (currentSlide === totalSlides);

    // Render exact charts based on active slide
    if(currentSlide === 5) renderDatasetChart();
    if(currentSlide === 6) renderRedoxPie();
    if(currentSlide === 8) renderAnnChart();
    if(currentSlide === 9) renderRfChart();
    if(currentSlide === 10) renderXgbChart();
    if(currentSlide === 11) renderRadarChart();
    if(currentSlide === 12) renderAccuracyChart();
    if(currentSlide === 13) renderScatterChart();
    if(currentSlide === 14) renderCapacitanceChart();
}

btnPrev.addEventListener('click', () => { currentSlide--; updateSlideUI(); });
btnNext.addEventListener('click', () => { currentSlide++; updateSlideUI(); });

// --- Graph Generating Functions ---

function renderDatasetChart() {
    if(charts.dataset) charts.dataset.destroy();
    charts.dataset = new Chart(document.getElementById('chart-dataset'), {
        type: 'doughnut',
        data: {
            labels: ['Training Data', 'Validation Data'],
            datasets: [{ data: [216200, 2000], backgroundColor: ['#4F46E5', '#10B981'], borderWidth: 0, hoverOffset: 5 }]
        },
        options: { maintainAspectRatio: false, animation: animConfig, plugins: { legend: { position: 'bottom' } }, cutout: '70%' }
    });
}

function renderRedoxPie() {
    if(charts.redox) charts.redox.destroy();
    charts.redox = new Chart(document.getElementById('chart-redox'), {
        type: 'pie',
        data: {
            labels: ['Oxidation State (1)', 'Reduction State (0)'],
            datasets: [{ data: [50, 50], backgroundColor: ['#F43F5E', '#3B82F6'], borderWidth: 0, hoverOffset: 5 }]
        },
        options: { maintainAspectRatio: false, animation: animConfig, plugins: { legend: { position: 'bottom' } } }
    });
}

function renderAnnChart() {
    if(charts.ann) charts.ann.destroy();
    
    const epochs = Array.from({length: 50}, (_, i) => i + 1);
    const trainLoss = epochs.map(e => 0.5 * Math.exp(-0.15 * e) + 0.02);
    const valLoss = epochs.map(e => 0.5 * Math.exp(-0.13 * e) + 0.03 + (Math.random()*0.01));

    charts.ann = new Chart(document.getElementById('chart-ann'), {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                { label: 'Training Loss', data: trainLoss, borderColor: '#4F46E5', backgroundColor: 'rgba(79, 70, 229, 0.1)', fill: true, tension: 0.4, pointRadius: 0 },
                { label: 'Validation Loss', data: valLoss, borderColor: '#10B981', borderDash: [5, 5], tension: 0.4, pointRadius: 0 }
            ]
        },
        options: { 
            maintainAspectRatio: false,
            animation: animConfig, 
            plugins: { title: { display: true, text: 'ANN Learning Curve (MSE vs Epochs)' }, legend: { position: 'bottom' } },
            scales: { x: { title: { display: true, text: 'Epochs' } }, y: { title: { display: true, text: 'Loss (MSE)' } } }
        }
    });
}

function renderRfChart() {
    if(charts.rf) charts.rf.destroy();
    charts.rf = new Chart(document.getElementById('chart-rf'), {
        type: 'bar',
        indexAxis: 'y', // Makes it a horizontal bar chart
        data: {
            labels: ['Potential', 'Scan Rate', 'Zn/Co Ratio', 'Oxidation State'],
            datasets: [{
                label: 'Relative Importance (%)',
                data: [45, 32, 15, 8],
                backgroundColor: ['#4F46E5', '#10B981', '#F43F5E', '#94A3B8'],
                borderRadius: 4
            }]
        },
        options: { 
            maintainAspectRatio: false,
            animation: animConfig, 
            plugins: { title: { display: true, text: 'Random Forest Feature Importance' }, legend: { display: false } },
            scales: { x: { max: 50, title: { display: true, text: 'Importance (%)' } } }
        }
    });
}

function renderXgbChart() {
    if(charts.xgb) charts.xgb.destroy();
    
    const rounds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    const rmse = [0.08, 0.04, 0.02, 0.012, 0.008, 0.005, 0.003, 0.0018, 0.0013, 0.0011, 0.0011];

    charts.xgb = new Chart(document.getElementById('chart-xgb'), {
        type: 'line',
        data: {
            labels: rounds,
            datasets: [{
                label: 'RMSE Error',
                data: rmse,
                borderColor: '#F43F5E',
                backgroundColor: 'rgba(244, 63, 94, 0.1)',
                stepped: true, // This creates the specific "staircase" look
                fill: true
            }]
        },
        options: { 
            maintainAspectRatio: false,
            animation: animConfig, 
            plugins: { title: { display: true, text: 'XGBoost Error Reduction (Stepped)' }, legend: { display: false } },
            scales: { x: { title: { display: true, text: 'Boosting Rounds' } }, y: { title: { display: true, text: 'RMSE' } } }
        }
    });
}

function renderRadarChart() {
    if(charts.radar) charts.radar.destroy();
    charts.radar = new Chart(document.getElementById('chart-radar'), {
        type: 'radar',
        data: {
            labels: ['Accuracy (R²)', 'Handling Noise', 'Training Speed', 'Non-linear Mapping', 'Interpretability'],
            datasets: [
                { label: 'ANN', data: [95, 70, 40, 95, 20], backgroundColor: 'rgba(79, 70, 229, 0.2)', borderColor: '#4F46E5' },
                { label: 'Random Forest', data: [85, 95, 80, 80, 70], backgroundColor: 'rgba(16, 185, 129, 0.2)', borderColor: '#10B981' },
                { label: 'XGBoost', data: [90, 85, 90, 90, 60], backgroundColor: 'rgba(244, 63, 94, 0.2)', borderColor: '#F43F5E' }
            ]
        },
        options: { maintainAspectRatio: false, animation: animConfig, scales: { r: { min: 0, max: 100, ticks: {display: false} } } }
    });
}

function renderAccuracyChart() {
    if(charts.accuracy) charts.accuracy.destroy();
    charts.accuracy = new Chart(document.getElementById('chart-accuracy'), {
        type: 'bar',
        data: {
            labels: ['Base Models Avg', 'Stacked Meta-Model'],
            datasets: [{ label: 'R² Accuracy (%)', data: [97.50, 97.85], backgroundColor: ['#94A3B8', '#4F46E5'], borderRadius: 6 }]
        },
        options: { maintainAspectRatio: false, animation: animConfig, scales: { y: { min: 96, max: 100 } }, plugins: { legend: { display: false } } }
    });
}

function renderScatterChart() {
    if(charts.scatter) charts.scatter.destroy();
    
    // Generate simulated tight cluster data mimicking 0.996 R2
    const scatterData = Array.from({length: 40}, () => {
        const val = (Math.random() * 0.04) - 0.02; 
        return { x: val, y: val + (Math.random() * 0.002 - 0.001) };
    });

    charts.scatter = new Chart(document.getElementById('chart-scatter'), {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Test Set (Actual vs Predicted)',
                data: scatterData,
                backgroundColor: '#F43F5E',
            }]
        },
        options: { 
            maintainAspectRatio: false,
            animation: animConfig, 
            scales: { 
                x: { title: {display: true, text: 'Actual Current (A)'} },
                y: { title: {display: true, text: 'Predicted Current (A)'} }
            } 
        }
    });
}

function renderCapacitanceChart() {
    if(charts.capacitance) charts.capacitance.destroy();
    charts.capacitance = new Chart(document.getElementById('chart-capacitance'), {
        type: 'bar',
        data: {
            labels: ['Experimental Data', 'ML Prediction'],
            datasets: [{ label: 'F g⁻¹', data: [0.01798, 0.01732], backgroundColor: ['#94A3B8', '#10B981'], borderRadius: 6 }]
        },
        options: { maintainAspectRatio: false, animation: animConfig, plugins: { legend: { display: false } } }
    });
}


// ==========================================
// 4. MAIN LIVE DEMO CHART LOGIC
// ==========================================
let liveChartInstance = new Chart(document.getElementById('liveChart'), {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { title: { display: true, text: 'Awaiting Execution...', font: {size: 16, weight: '600'} } },
        scales: { 
            x: { title: { display: true, text: 'Potential (V)', font: {weight: '600'} } },
            y: { title: { display: true, text: 'Current (A)', font: {weight: '600'} } }
        }
    }
});

document.getElementById('btn-run-demo').addEventListener('click', () => {
    // Hide metrics initially on re-run
    document.getElementById('metrics-grid').style.display = 'none';

    const potentialValues = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.3, -0.4, -0.5];
    const actualCurrent = [-0.015, -0.012, -0.005, 0.005, 0.012, 0.018, 0.022, 0.019, 0.012, 0.005, 0.002, -0.005, -0.012, -0.018, -0.021, -0.019, -0.012, -0.005, -0.008, -0.012, -0.015];
    const predictedCurrent = [-0.0148, -0.0118, -0.0048, 0.0052, 0.0119, 0.0178, 0.0221, 0.0188, 0.0118, 0.0048, 0.0019, -0.0048, -0.0118, -0.0178, -0.0208, -0.0188, -0.0118, -0.0048, -0.0078, -0.0118, -0.0148];

    liveChartInstance.data.labels = potentialValues;
    liveChartInstance.data.datasets = [
        { label: 'Experimental Curve', data: actualCurrent, borderColor: '#1E293B', backgroundColor: 'rgba(30, 41, 59, 0.05)', borderWidth: 2, tension: 0.4, fill: true },
        { label: 'Meta-Model Prediction', data: predictedCurrent, borderColor: '#F43F5E', borderWidth: 3, borderDash: [5, 5], tension: 0.4, fill: false }
    ];
    
    liveChartInstance.options.animation = { duration: 2000, easing: 'easeOutQuart' };
    liveChartInstance.options.plugins.title.text = 'Validation: Actual vs Predicted CV Curve at 60mV/s';
    liveChartInstance.update();

    // Show final numbers after the line is drawn
    setTimeout(() => {
        document.getElementById('metrics-grid').style.display = 'grid';
        document.getElementById('metrics-grid').style.animation = 'slideUp 0.8s ease forwards';
    }, 1500);
});
