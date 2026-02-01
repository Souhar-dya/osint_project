/**
 * OSINT Monitor - Popup Script
 * Handles popup UI interactions and settings
 */

document.addEventListener('DOMContentLoaded', init);

async function init() {
  // Load settings
  await loadSettings();
  
  // Check backend status
  await checkStatus();
  
  // Set up event listeners
  setupEventListeners();
}

/**
 * Load settings from storage
 */
async function loadSettings() {
  const settings = await chrome.storage.sync.get({
    enabled: true,
    showOverlay: true,
    anonymize: true,
    includeBaseline: true,
    analyzePublicOnly: true,
    apiUrl: 'http://localhost:8000/api',
  });
  
  document.getElementById('setting-enabled').checked = settings.enabled;
  document.getElementById('setting-overlay').checked = settings.showOverlay;
  document.getElementById('setting-anonymize').checked = settings.anonymize;
  document.getElementById('setting-baseline').checked = settings.includeBaseline;
  document.getElementById('setting-public-only').checked = settings.analyzePublicOnly;
  document.getElementById('api-url').value = settings.apiUrl;
}

/**
 * Save settings to storage
 */
async function saveSettings() {
  const settings = {
    enabled: document.getElementById('setting-enabled').checked,
    showOverlay: document.getElementById('setting-overlay').checked,
    anonymize: document.getElementById('setting-anonymize').checked,
    includeBaseline: document.getElementById('setting-baseline').checked,
    analyzePublicOnly: document.getElementById('setting-public-only').checked,
    apiUrl: document.getElementById('api-url').value,
  };
  
  await chrome.storage.sync.set(settings);
  showToast('Settings saved');
  
  // Notify content scripts
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tabs[0]) {
    chrome.tabs.sendMessage(tabs[0].id, { type: 'REFRESH_SETTINGS' });
  }
}

/**
 * Check backend connection status
 */
async function checkStatus() {
  const statusEl = document.getElementById('status');
  const statusText = statusEl.querySelector('.status-text');
  
  try {
    const result = await chrome.runtime.sendMessage({ type: 'HEALTH_CHECK' });
    
    if (result.healthy) {
      statusEl.className = 'status connected';
      statusText.textContent = 'Connected';
    } else {
      statusEl.className = 'status disconnected';
      statusText.textContent = 'Disconnected';
    }
  } catch (error) {
    statusEl.className = 'status disconnected';
    statusText.textContent = 'Error';
  }
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
  // Analyze button
  document.getElementById('analyze-btn').addEventListener('click', analyzeText);
  
  // Test connection button
  document.getElementById('test-connection').addEventListener('click', testConnection);
  
  // Settings checkboxes
  const checkboxes = document.querySelectorAll('input[type="checkbox"]');
  checkboxes.forEach(cb => {
    cb.addEventListener('change', saveSettings);
  });
  
  // API URL change
  document.getElementById('api-url').addEventListener('change', saveSettings);
  
  // Text input - analyze on Enter
  document.getElementById('text-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      analyzeText();
    }
  });
}

/**
 * Analyze text from input
 */
async function analyzeText() {
  const textInput = document.getElementById('text-input');
  const text = textInput.value.trim();
  
  if (!text) {
    showToast('Please enter text to analyze');
    return;
  }
  
  if (text.length < 10) {
    showToast('Text too short (min 10 characters)');
    return;
  }
  
  const btn = document.getElementById('analyze-btn');
  const resultsSection = document.getElementById('results-section');
  const resultsEl = document.getElementById('results');
  
  // Show loading
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span> Analyzing...';
  resultsSection.style.display = 'block';
  resultsEl.innerHTML = '<div class="loading">Analyzing text...</div>';
  
  try {
    const result = await chrome.runtime.sendMessage({
      type: 'ANALYZE_TEXT',
      data: { text, source: 'manual' },
    });
    
    if (result.error) {
      throw new Error(result.error);
    }
    
    displayResults(result);
    
  } catch (error) {
    resultsEl.innerHTML = `
      <div style="padding: 16px; color: #dc2626; text-align: center;">
        ❌ Analysis failed: ${error.message}
      </div>
    `;
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">🔍</span> Analyze Text';
  }
}

/**
 * Display analysis results
 */
function displayResults(result) {
  const resultsEl = document.getElementById('results');
  
  const sentiment = result.sentiment || {};
  const topics = result.topics || {};
  const framing = result.framing || {};
  const misinfo = result.misinformation || {};
  const baseline = result.baseline || {};
  const explanation = result.explanation || {};
  
  let html = `
    <div class="result-row">
      <span class="result-label">Sentiment</span>
      <span class="result-value ${sentiment.label}">${sentiment.label} (${Math.round(sentiment.score * 100)}%)</span>
    </div>
    <div class="result-row">
      <span class="result-label">Topic</span>
      <span class="result-value">${topics.topic_label || 'Unknown'}</span>
    </div>
    <div class="result-row">
      <span class="result-label">Frame</span>
      <span class="result-value">${framing.frame || 'Unknown'}</span>
    </div>
    <div class="result-row">
      <span class="result-label">Misinfo Risk</span>
      <span class="result-value ${misinfo.risk_level}">${(misinfo.risk_level || 'unknown').toUpperCase()} (${Math.round((misinfo.risk_score || 0) * 100)}%)</span>
    </div>
    <div class="result-row">
      <span class="result-label">Narrative Distance</span>
      <span class="result-value">${Math.round((baseline.narrative_distance || 0) * 100)}% ${baseline.deviation_type ? `(${baseline.deviation_type})` : ''}</span>
    </div>
  `;
  
  // Flags
  if (explanation.flags && explanation.flags.length > 0) {
    html += `
      <div class="result-flags">
        ${explanation.flags.map(f => `<span class="result-flag">${f}</span>`).join('')}
      </div>
    `;
  }
  
  // Explanation
  if (explanation.reasoning) {
    html += `
      <div class="result-explanation">
        ${explanation.reasoning}
      </div>
    `;
  }
  
  resultsEl.innerHTML = html;
}

/**
 * Test backend connection
 */
async function testConnection() {
  const btn = document.getElementById('test-connection');
  btn.disabled = true;
  btn.textContent = 'Testing...';
  
  await checkStatus();
  
  const statusEl = document.getElementById('status');
  if (statusEl.classList.contains('connected')) {
    showToast('✅ Connection successful');
  } else {
    showToast('❌ Connection failed');
  }
  
  btn.disabled = false;
  btn.textContent = 'Test Connection';
}

/**
 * Show toast notification
 */
function showToast(message) {
  // Remove existing toast
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();
  
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.textContent = message;
  document.body.appendChild(toast);
  
  setTimeout(() => toast.remove(), 3000);
}
