/**
 * OSINT Monitor - Background Service Worker
 * Handles API communication and manages extension state
 */

// Configuration
const CONFIG = {
  API_URL: 'http://localhost:8000/api',
  CACHE_DURATION: 5 * 60 * 1000, // 5 minutes
};

// Cache for analysis results
const analysisCache = new Map();

/**
 * Handle messages from content script and popup
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'ANALYZE_TEXT') {
    handleAnalyzeText(message.data)
      .then(sendResponse)
      .catch(error => sendResponse({ error: error.message }));
    return true; // Keep channel open for async response
  }
  
  if (message.type === 'ANALYZE_IMAGE') {
    handleAnalyzeImage(message.data)
      .then(sendResponse)
      .catch(error => sendResponse({ error: error.message }));
    return true;
  }
  
  if (message.type === 'GET_SETTINGS') {
    getSettings().then(sendResponse);
    return true;
  }
  
  if (message.type === 'SAVE_SETTINGS') {
    saveSettings(message.data).then(sendResponse);
    return true;
  }
  
  if (message.type === 'HEALTH_CHECK') {
    checkBackendHealth().then(sendResponse);
    return true;
  }
  
  // New handlers for thread and claim verification
  if (message.type === 'ANALYZE_THREAD') {
    handleAnalyzeThread(message.data)
      .then(sendResponse)
      .catch(error => sendResponse({ error: error.message }));
    return true;
  }
  
  if (message.type === 'VERIFY_CLAIM') {
    handleVerifyClaim(message.data)
      .then(sendResponse)
      .catch(error => sendResponse({ error: error.message }));
    return true;
  }
  
  if (message.type === 'ANALYZE_QUOTE') {
    handleAnalyzeQuote(message.data)
      .then(sendResponse)
      .catch(error => sendResponse({ error: error.message }));
    return true;
  }
});

/**
 * Analyze text via backend API
 */
async function handleAnalyzeText(data) {
  const { text, source } = data;
  
  // Check cache first
  const cacheKey = hashText(text);
  if (analysisCache.has(cacheKey)) {
    const cached = analysisCache.get(cacheKey);
    if (Date.now() - cached.timestamp < CONFIG.CACHE_DURATION) {
      return { ...cached.result, cached: true };
    }
  }
  
  // Get settings
  const settings = await getSettings();
  
  // Call API
  try {
    const response = await fetch(`${settings.apiUrl || CONFIG.API_URL}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: text,
        source: source || 'unknown',
        anonymize: settings.anonymize !== false,
        include_baseline: settings.includeBaseline !== false,
      }),
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    const result = await response.json();
    
    // Cache result
    analysisCache.set(cacheKey, {
      result,
      timestamp: Date.now(),
    });
    
    return result;
  } catch (error) {
    console.error('Analysis error:', error);
    throw error;
  }
}

/**
 * Analyze image via backend API
 */
async function handleAnalyzeImage(data) {
  const { image_url, text, source } = data;
  
  // Check cache first (using image URL as key)
  const cacheKey = hashText(image_url + (text || ''));
  if (analysisCache.has(cacheKey)) {
    const cached = analysisCache.get(cacheKey);
    if (Date.now() - cached.timestamp < CONFIG.CACHE_DURATION) {
      return { ...cached.result, cached: true };
    }
  }
  
  // Get settings
  const settings = await getSettings();
  
  // Call Image Analysis API
  try {
    const response = await fetch(`${settings.apiUrl || CONFIG.API_URL}/analyze-image`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_url: image_url,
        text: text || null,
        source: source || 'unknown',
        extract_text: true,
        analyze_content: true,
      }),
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    const result = await response.json();
    
    // Cache result
    analysisCache.set(cacheKey, {
      result,
      timestamp: Date.now(),
    });
    
    return result;
  } catch (error) {
    console.error('Image analysis error:', error);
    throw error;
  }
}

/**
 * Check backend health
 */
async function checkBackendHealth() {
  const settings = await getSettings();
  
  // Health endpoint is at root, not under /api
  const baseUrl = (settings.apiUrl || CONFIG.API_URL).replace('/api', '');
  
  try {
    const response = await fetch(`${baseUrl}/health`);
    const data = await response.json();
    return { healthy: data.status === 'healthy' || data.status === 'degraded', data };
  } catch (error) {
    return { healthy: false, error: error.message };
  }
}

/**
 * Get settings from storage
 */
async function getSettings() {
  return new Promise((resolve) => {
    chrome.storage.sync.get({
      // Defaults
      enabled: true,
      apiUrl: CONFIG.API_URL,
      anonymize: true,
      includeBaseline: true,
      showOverlay: true,
      autoAnalyze: false,
      analyzePublicOnly: true,
    }, resolve);
  });
}

/**
 * Save settings to storage
 */
async function saveSettings(settings) {
  return new Promise((resolve) => {
    chrome.storage.sync.set(settings, () => {
      resolve({ success: true });
    });
  });
}

/**
 * Simple hash function for cache keys
 */
function hashText(text) {
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    const char = text.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return hash.toString(36);
}

/**
 * Analyze thread (original + replies) via backend API
 */
async function handleAnalyzeThread(data) {
  const { original_text, replies } = data;
  const settings = await getSettings();
  
  try {
    const response = await fetch(`${settings.apiUrl || CONFIG.API_URL}/analyze-thread`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        original_text: original_text,
        replies: replies,
      }),
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Thread analysis error:', error);
    throw error;
  }
}

/**
 * Verify claim via backend API
 */
async function handleVerifyClaim(data) {
  const { claim, use_external_api } = data;
  const settings = await getSettings();
  
  try {
    const response = await fetch(`${settings.apiUrl || CONFIG.API_URL}/verify-claim-quick`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        claim: claim,
        use_external_api: use_external_api || false,
      }),
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Claim verification error:', error);
    throw error;
  }
}

/**
 * Analyze quote tweet via backend API
 */
async function handleAnalyzeQuote(data) {
  const { quoted_text, commentary } = data;
  const settings = await getSettings();
  
  try {
    const response = await fetch(`${settings.apiUrl || CONFIG.API_URL}/analyze-quote`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        quoted_text: quoted_text,
        commentary: commentary,
      }),
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Quote analysis error:', error);
    throw error;
  }
}

/**
 * Clear old cache entries periodically
 */
setInterval(() => {
  const now = Date.now();
  for (const [key, value] of analysisCache.entries()) {
    if (now - value.timestamp > CONFIG.CACHE_DURATION) {
      analysisCache.delete(key);
    }
  }
}, CONFIG.CACHE_DURATION);

console.log('OSINT Monitor background service started');
