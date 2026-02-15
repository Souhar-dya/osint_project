/**
 * OSINT Monitor - Content Script
 * Extracts text from social media posts and displays analysis overlays
 */

(function() {
  'use strict';
  
  // Platform-specific selectors
  const PLATFORM_SELECTORS = {
    twitter: {
      posts: '[data-testid="tweet"]',
      text: '[data-testid="tweetText"]',
      username: '[data-testid="User-Name"]',
      images: '[data-testid="tweetPhoto"] img, [data-testid="tweet"] img[src*="media"]',
    },
    x: {
      posts: '[data-testid="tweet"]',
      text: '[data-testid="tweetText"]',
      username: '[data-testid="User-Name"]',
      images: '[data-testid="tweetPhoto"] img, [data-testid="tweet"] img[src*="media"]',
    },
    instagram: {
      posts: 'article',
      text: 'span[class*="x193iq5w"]',
      username: 'a[href*="/"] span',
      images: 'article img[src*="instagram"], article img[srcset]',
    },
    youtube: {
      posts: 'ytd-comment-renderer',
      text: '#content-text',
      username: '#author-text',
      images: null, // YouTube comments don't have images
    },
    facebook: {
      posts: '[data-ad-comet-preview="message"]',
      text: '[data-ad-comet-preview="message"]',
      username: 'strong',
      images: 'img[src*="scontent"]',
    },
    reddit: {
      posts: 'shreddit-post, [data-testid="post-container"]',
      text: '[slot="text-body"], [data-click-id="text"]',
      username: '[data-testid="post_author_link"]',
      images: 'img[src*="redd.it"], img[src*="reddit"]',
    },
  };
  
  // Detect current platform
  function detectPlatform() {
    const hostname = window.location.hostname;
    if (hostname.includes('twitter.com')) return 'twitter';
    if (hostname.includes('x.com')) return 'x';
    if (hostname.includes('instagram.com')) return 'instagram';
    if (hostname.includes('youtube.com')) return 'youtube';
    if (hostname.includes('facebook.com')) return 'facebook';
    if (hostname.includes('reddit.com')) return 'reddit';
    return 'unknown';
  }
  
  const platform = detectPlatform();
  const selectors = PLATFORM_SELECTORS[platform] || {};
  
  // Track analyzed posts
  const analyzedPosts = new WeakSet();
  
  // Settings
  let settings = {
    enabled: true,
    showOverlay: true,
    autoAnalyze: false,
  };
  
  /**
   * Initialize the content script
   */
  async function init() {
    console.log(`OSINT Monitor initialized on ${platform}`);
    
    // Load settings
    settings = await loadSettings();
    
    if (!settings.enabled) {
      console.log('OSINT Monitor is disabled');
      return;
    }
    
    // Add analyze buttons to posts
    observePosts();
    
    // Initial scan
    scanPosts();
  }
  
  /**
   * Load settings from storage
   */
  async function loadSettings() {
    return new Promise((resolve) => {
      chrome.storage.sync.get({
        enabled: true,
        showOverlay: true,
        autoAnalyze: false,
      }, resolve);
    });
  }
  
  /**
   * Observe DOM for new posts
   */
  function observePosts() {
    const observer = new MutationObserver((mutations) => {
      let shouldScan = false;
      for (const mutation of mutations) {
        if (mutation.addedNodes.length > 0) {
          shouldScan = true;
          break;
        }
      }
      if (shouldScan) {
        scanPosts();
      }
    });
    
    observer.observe(document.body, {
      childList: true,
      subtree: true,
    });
  }
  
  /**
   * Scan for posts and add analyze buttons
   */
  function scanPosts() {
    if (!selectors.posts) return;
    
    const posts = document.querySelectorAll(selectors.posts);
    
    posts.forEach((post) => {
      if (analyzedPosts.has(post)) return;
      if (post.querySelector('.osint-analyze-btn')) return;
      
      // Add analyze button
      addAnalyzeButton(post);
      
      // Auto-analyze if enabled
      if (settings.autoAnalyze) {
        analyzePost(post);
      }
    });
  }
  
  /**
   * Analyze a post
   */
  async function analyzePost(post) {
    // Extract text
    const text = extractText(post);
    
    // Extract image URLs
    const imageUrls = extractImages(post);
    
    // Check if we have content to analyze
    if ((!text || text.length < 10) && imageUrls.length === 0) {
      showNotification(post, 'No content to analyze', 'warning');
      return;
    }
    
    // Show loading state
    showLoading(post, true);
    
    try {
      let result;
      
      // If we have images, use the image analysis endpoint
      if (imageUrls.length > 0) {
        result = await chrome.runtime.sendMessage({
          type: 'ANALYZE_IMAGE',
          data: {
            image_url: imageUrls[0], // Analyze first image
            text: text,
            source: platform,
          },
        });
      } else {
        // Text-only analysis
        result = await chrome.runtime.sendMessage({
          type: 'ANALYZE_TEXT',
          data: {
            text: text,
            source: platform,
          },
        });
      }
      
      if (result.error) {
        throw new Error(result.error);
      }
      
      // Display results
      displayResults(post, result, imageUrls.length > 0);
      analyzedPosts.add(post);
      
    } catch (error) {
      console.error('Analysis error:', error);
      showNotification(post, `Analysis failed: ${error.message}`, 'error');
    } finally {
      showLoading(post, false);
    }
  }
  
  /**
   * Extract image URLs from a post
   */
  function extractImages(post) {
    if (!selectors.images) {
      return [];
    }
    
    const images = post.querySelectorAll(selectors.images);
    const urls = [];
    
    images.forEach(img => {
      // Get the best quality image URL
      let url = img.src;
      
      // For srcset, get the largest image
      if (img.srcset) {
        const srcset = img.srcset.split(',');
        const largest = srcset[srcset.length - 1].trim().split(' ')[0];
        if (largest) url = largest;
      }
      
      // Skip small images (likely icons/avatars)
      if (img.width && img.width < 100) return;
      if (img.height && img.height < 100) return;
      
      // Skip data URIs that are too small (placeholder images)
      if (url.startsWith('data:') && url.length < 1000) return;
      
      if (url && !urls.includes(url)) {
        urls.push(url);
      }
    });
    
    return urls;
  }
  
  /**
   * Extract text from a post
   */
  function extractText(post) {
    if (!selectors.text) {
      return post.innerText.trim();
    }
    
    const textElement = post.querySelector(selectors.text);
    if (textElement) {
      return textElement.innerText.trim();
    }
    
    return post.innerText.trim();
  }
  
  /**
   * Show/hide loading indicator
   */
  function showLoading(post, isLoading) {
    const btn = post.querySelector('.osint-analyze-btn');
    if (btn) {
      btn.disabled = isLoading;
      btn.innerHTML = isLoading ? '⏳ Analyzing...' : '🔍 Analyze';
    }
  }
  
  /**
   * Display analysis results as overlay
   */
  function displayResults(post, result, isImageAnalysis = false) {
    // Remove any existing overlays
    document.querySelectorAll('.osint-overlay, .osint-backdrop').forEach(el => el.remove());
    
    // Create backdrop
    const backdrop = document.createElement('div');
    backdrop.className = 'osint-backdrop';
    backdrop.addEventListener('click', () => {
      backdrop.remove();
      overlay.remove();
    });
    document.body.appendChild(backdrop);
    
    // Create overlay
    const overlay = document.createElement('div');
    overlay.className = 'osint-overlay';
    
    // Build content based on analysis type
    if (isImageAnalysis) {
      overlay.innerHTML = buildImageOverlayContent(result);
    } else {
      overlay.innerHTML = buildOverlayContent(result);
    }
    
    // Add close button handler
    const closeBtn = overlay.querySelector('.osint-close');
    if (closeBtn) {
      closeBtn.addEventListener('click', () => {
        backdrop.remove();
        overlay.remove();
      });
    }
    
    // Append to body (fixed positioning)
    document.body.appendChild(overlay);
  }
  
  /**
   * Build overlay HTML content for image analysis
   */
  function buildImageOverlayContent(result) {
    const content = result.content || {};
    const ocr = result.ocr || {};
    const textAnalysis = result.text_analysis || {};
    const sentiment = textAnalysis.sentiment || {};
    const misinfo = textAnalysis.misinformation || {};
    const framing = textAnalysis.framing || {};
    
    // Risk color
    let riskColor = 'green';
    if (result.overall_risk === 'high') {
      riskColor = 'red';
    } else if (result.overall_risk === 'medium') {
      riskColor = 'orange';
    }
    
    // Build risk factors HTML
    const riskFactorsHtml = (result.risk_factors || [])
      .map(f => `<span class="osint-flag">${f}</span>`)
      .join('');
    
    // OCR text preview
    const ocrText = ocr.text || result.extracted_text || '';
    const ocrPreview = ocrText.length > 100 ? ocrText.substring(0, 100) + '...' : ocrText;
    
    return `
      <div class="osint-header">
        <span class="osint-title">🔍 OSINT Analysis (Image)</span>
        <button class="osint-close">✕</button>
      </div>
      
      <div class="osint-content">
        <div class="osint-row">
          <span class="osint-label">Image Type:</span>
          <span class="osint-value">${result.image_type || 'Unknown'}</span>
        </div>
        
        ${result.is_manipulated ? `
        <div class="osint-row">
          <span class="osint-label">⚠️ Manipulation:</span>
          <span class="osint-value osint-risk-high">DETECTED</span>
        </div>
        ` : ''}
        
        ${ocrPreview ? `
        <div class="osint-row">
          <span class="osint-label">OCR Text:</span>
          <span class="osint-value" style="font-size: 11px; max-width: 180px; overflow: hidden; text-overflow: ellipsis;">${ocrPreview}</span>
        </div>
        ` : ''}
        
        ${sentiment.label ? `
        <div class="osint-row">
          <span class="osint-label">Sentiment:</span>
          <span class="osint-value osint-sentiment-${sentiment.label}">
            ${sentiment.label} (${Math.round((sentiment.score || 0) * 100)}%)
          </span>
        </div>
        ` : ''}
        
        ${framing.frame ? `
        <div class="osint-row">
          <span class="osint-label">Frame:</span>
          <span class="osint-value">${framing.frame}</span>
        </div>
        ` : ''}
        
        ${misinfo.risk_level ? `
        <div class="osint-row">
          <span class="osint-label">Text Misinfo:</span>
          <span class="osint-value osint-risk-${misinfo.risk_level}">
            ${misinfo.risk_level.toUpperCase()} (${Math.round((misinfo.risk_score || 0) * 100)}%)
          </span>
        </div>
        ` : ''}
        
        <div class="osint-row">
          <span class="osint-label">Overall Risk:</span>
          <span class="osint-value" style="color: ${riskColor}; font-weight: bold;">
            ${(result.overall_risk || 'low').toUpperCase()}
          </span>
        </div>
        
        ${riskFactorsHtml ? `<div class="osint-flags">${riskFactorsHtml}</div>` : ''}
      </div>
    `;
  }
  
  /**
   * Build overlay HTML content
   */
  function buildOverlayContent(result) {
    const sentiment = result.sentiment || {};
    const misinfo = result.misinformation || {};
    const baseline = result.baseline || {};
    const explanation = result.explanation || {};
    
    // Determine overall risk color
    let riskColor = 'green';
    if (misinfo.risk_level === 'high' || baseline.narrative_distance > 0.7) {
      riskColor = 'red';
    } else if (misinfo.risk_level === 'medium' || baseline.narrative_distance > 0.4) {
      riskColor = 'orange';
    }
    
    // Build flags HTML
    const flagsHtml = (explanation.flags || [])
      .map(flag => `<span class="osint-flag">${flag}</span>`)
      .join('');
    
    return `
      <div class="osint-header">
        <span class="osint-title">🔍 OSINT Analysis</span>
        <button class="osint-close">✕</button>
      </div>
      
      <div class="osint-content">
        <div class="osint-row">
          <span class="osint-label">Sentiment:</span>
          <span class="osint-value osint-sentiment-${sentiment.label}">
            ${sentiment.label} (${Math.round(sentiment.score * 100)}%)
          </span>
        </div>
        
        <div class="osint-row">
          <span class="osint-label">Topic:</span>
          <span class="osint-value">${result.topics?.topic_label || 'Unknown'}</span>
        </div>
        
        <div class="osint-row">
          <span class="osint-label">Frame:</span>
          <span class="osint-value">${result.framing?.frame || 'Unknown'}</span>
        </div>
        
        <div class="osint-row">
          <span class="osint-label">Misinfo Risk:</span>
          <span class="osint-value osint-risk-${misinfo.risk_level}">
            ${misinfo.risk_level?.toUpperCase()} (${Math.round(misinfo.risk_score * 100)}%)
          </span>
        </div>
        
        ${misinfo.verification_verdict ? `
        <div class="osint-row">
          <span class="osint-label">Verification:</span>
          <span class="osint-value" style="color: ${misinfo.verification_verdict?.includes('AUTHENTIC') ? '#27ae60' : misinfo.verification_verdict?.includes('FAKE') ? '#e74c3c' : '#f39c12'}; font-weight:600;">
            ${misinfo.verification_verdict?.replace(/_/g, ' ')}
          </span>
        </div>
        ` : ''}
        
        ${misinfo.credibility_score != null ? `
        <div class="osint-row">
          <span class="osint-label">Credibility:</span>
          <span class="osint-value" style="color: ${misinfo.credibility_score > 0.6 ? '#27ae60' : misinfo.credibility_score > 0.35 ? '#f39c12' : '#e74c3c'}">
            ${Math.round(misinfo.credibility_score * 100)}%
          </span>
        </div>
        ` : ''}
        
        ${(misinfo.verified_sources && misinfo.verified_sources.length > 0) ? `
        <div class="osint-section" style="margin-top:10px;">
          <div style="font-size:12px; font-weight:600; color:#94a3b8; margin-bottom:6px;">📰 Verified Against:</div>
          ${misinfo.verified_sources.slice(0, 4).map(s => `
            <div style="display:flex; align-items:center; gap:6px; padding:4px 0; font-size:11px; border-bottom:1px solid rgba(255,255,255,0.05);">
              <span style="color:${s.trust_level === 'trusted' ? '#27ae60' : s.trust_level === 'unreliable' ? '#e74c3c' : '#f39c12'};">●</span>
              <a href="${s.url}" target="_blank" rel="noopener" style="color:#93c5fd; text-decoration:none; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:220px;" title="${s.title}">
                ${s.source || s.domain}
              </a>
              <span style="color:#64748b; font-size:10px; margin-left:auto;">${s.trust_level}</span>
            </div>
          `).join('')}
        </div>
        ` : ''}
        
        <div class="osint-row">
          <span class="osint-label">Narrative Distance:</span>
          <span class="osint-value" style="color: ${riskColor}">
            ${Math.round(baseline.narrative_distance * 100)}%
            ${baseline.deviation_type ? `(${baseline.deviation_type})` : ''}
          </span>
        </div>
        
        ${flagsHtml ? `<div class="osint-flags">${flagsHtml}</div>` : ''}
        
        <div class="osint-explanation">
          ${explanation.reasoning || ''}
        </div>
      </div>
    `;
  }
  
  /**
   * Show a notification on a post
   */
  function showNotification(post, message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `osint-notification osint-notification-${type}`;
    notification.textContent = message;
    
    post.style.position = 'relative';
    post.appendChild(notification);
    
    setTimeout(() => notification.remove(), 3000);
  }
  
  /**
   * Extract a Twitter/X thread (original + replies)
   * Collects visible replies on the page (Twitter lazy-loads, so scroll for more)
   */
  function extractThread(originalPost) {
    const thread = {
      original: extractText(originalPost),
      replies: []
    };
    
    // Get all tweets on the page after the original
    const allPosts = document.querySelectorAll(selectors.posts);
    let foundOriginal = false;
    const seenTexts = new Set(); // Avoid duplicates
    
    allPosts.forEach((post, index) => {
      if (post === originalPost) {
        foundOriginal = true;
        return;
      }
      
      if (foundOriginal && thread.replies.length < 50) { // Increased limit to 50
        const text = extractText(post);
        if (text && text.length > 10 && !seenTexts.has(text)) {
          seenTexts.add(text);
          thread.replies.push(text);
        }
      }
    });
    
    return thread;
  }
  
  /**
   * Extract quote tweet data
   */
  function extractQuoteTweet(post) {
    // Find quoted tweet within this tweet
    const quotedTweet = post.querySelector('[data-testid="quoteTweet"]') || 
                        post.querySelector('div[role="link"] > div > div');
    
    if (!quotedTweet) return null;
    
    // Get the main tweet text (commentary)
    const mainTextEl = post.querySelector(selectors.text);
    const commentary = mainTextEl ? mainTextEl.innerText.trim() : '';
    
    // Get the quoted tweet text
    const quotedTextEl = quotedTweet.querySelector('[data-testid="tweetText"], span');
    const quotedText = quotedTextEl ? quotedTextEl.innerText.trim() : '';
    
    if (commentary && quotedText && commentary !== quotedText) {
      return {
        quoted_text: quotedText,
        commentary: commentary
      };
    }
    
    return null;
  }
  
  /**
   * Analyze a thread (original + replies)
   */
  async function analyzeThread(post) {
    showLoading(post, true);
    
    try {
      const thread = extractThread(post);
      
      if (thread.replies.length === 0) {
        showNotification(post, 'No replies found. Scroll down to load replies first.', 'warning');
        return;
      }
      
      // Show how many replies will be analyzed
      console.log(`Analyzing ${thread.replies.length} visible replies (scroll to load more)`);
      
      const result = await chrome.runtime.sendMessage({
        type: 'ANALYZE_THREAD',
        data: {
          original_text: thread.original,
          replies: thread.replies
        }
      });
      
      if (result.error) {
        throw new Error(result.error);
      }
      
      // Add note about limited replies if applicable
      if (thread.replies.length < 20) {
        result.note = `Only ${thread.replies.length} replies were visible. Scroll down for more replies before analyzing.`;
      }
      
      displayThreadResults(post, result);
      
    } catch (error) {
      console.error('Thread analysis error:', error);
      showNotification(post, `Thread analysis failed: ${error.message}`, 'error');
    } finally {
      showLoading(post, false);
    }
  }
  
  /**
   * Verify a claim from a post
   */
  async function verifyClaim(post) {
    const text = extractText(post);
    
    if (!text || text.length < 10) {
      showNotification(post, 'No claim to verify', 'warning');
      return;
    }
    
    showLoading(post, true);
    
    try {
      const result = await chrome.runtime.sendMessage({
        type: 'VERIFY_CLAIM',
        data: {
          claim: text,
          use_external_api: false
        }
      });
      
      if (result.error) {
        throw new Error(result.error);
      }
      
      displayClaimVerificationResults(post, result);
      
    } catch (error) {
      console.error('Claim verification error:', error);
      showNotification(post, `Verification failed: ${error.message}`, 'error');
    } finally {
      showLoading(post, false);
    }
  }
  
  /**
   * Analyze quote tweet
   */
  async function analyzeQuoteTweet(post) {
    showLoading(post, true);
    
    try {
      // Try to find quoted content within the post
      const quotedElement = post.querySelector('[data-testid="quoteTweet"]') || 
                           post.querySelector('article[role="article"] article') ||
                           post.querySelector('[data-testid="tweetText"] + div [data-testid="tweetText"]');
      
      const mainText = extractText(post);
      let quotedText = '';
      
      if (quotedElement) {
        const quotedTextEl = quotedElement.querySelector('[data-testid="tweetText"]');
        if (quotedTextEl) {
          quotedText = quotedTextEl.innerText.trim();
        }
      }
      
      if (!quotedText) {
        // Fallback: analyze as regular post with quote detection hint
        showNotification(post, 'No quote found. Analyzing as regular post.', 'info');
        analyzePost(post);
        return;
      }
      
      const result = await chrome.runtime.sendMessage({
        type: 'ANALYZE_QUOTE',
        data: {
          original_text: quotedText,
          quote_text: mainText
        }
      });
      
      if (result.error) {
        throw new Error(result.error);
      }
      
      displayQuoteResults(post, result, mainText, quotedText);
      
    } catch (error) {
      console.error('Quote analysis error:', error);
      showNotification(post, `Quote analysis failed: ${error.message}`, 'error');
    } finally {
      showLoading(post, false);
    }
  }
  
  /**
   * Display quote analysis results
   */
  function displayQuoteResults(post, result, quoteText, originalText) {
    document.querySelectorAll('.osint-overlay, .osint-backdrop').forEach(el => el.remove());
    
    const backdrop = document.createElement('div');
    backdrop.className = 'osint-backdrop';
    backdrop.addEventListener('click', () => {
      backdrop.remove();
      overlay.remove();
    });
    document.body.appendChild(backdrop);
    
    const overlay = document.createElement('div');
    overlay.className = 'osint-overlay';
    
    const stanceColor = result.stance === 'agree' ? '#27ae60' : 
                       result.stance === 'disagree' ? '#e74c3c' : 
                       result.stance === 'discuss' ? '#3498db' : '#6c757d';
    
    overlay.innerHTML = `
      <div class="osint-header">
        <span class="osint-title">💬 Quote Tweet Analysis</span>
        <button class="osint-close">✕</button>
      </div>
      <div class="osint-content">
        <div class="osint-row">
          <span class="osint-label">Stance</span>
          <span class="osint-value" style="color: ${stanceColor}; text-transform: capitalize;">
            ${result.stance || 'Unknown'} (${Math.round((result.stance_confidence || 0) * 100)}%)
          </span>
        </div>
        
        <div class="osint-section">
          <div class="osint-section-title">Original Tweet</div>
          <div style="padding: 10px; background: #1a1a2e; border-radius: 6px; font-size: 13px; color: #a0aec0;">
            "${originalText.substring(0, 150)}${originalText.length > 150 ? '...' : ''}"
          </div>
        </div>
        
        <div class="osint-section">
          <div class="osint-section-title">Quote Commentary</div>
          <div style="padding: 10px; background: #1a1a2e; border-radius: 6px; font-size: 13px; color: #e2e8f0;">
            "${quoteText.substring(0, 150)}${quoteText.length > 150 ? '...' : ''}"
          </div>
        </div>
        
        ${result.analysis ? `
        <div class="osint-section">
          <div class="osint-section-title">Analysis</div>
          <div style="font-size: 13px; color: #a0aec0;">${result.analysis}</div>
        </div>
        ` : ''}
      </div>
    `;
    
    overlay.querySelector('.osint-close').addEventListener('click', () => {
      backdrop.remove();
      overlay.remove();
    });
    
    document.body.appendChild(overlay);
  }
  
  /**
   * Display thread analysis results
   */
  function displayThreadResults(post, result) {
    document.querySelectorAll('.osint-overlay, .osint-backdrop').forEach(el => el.remove());
    
    const backdrop = document.createElement('div');
    backdrop.className = 'osint-backdrop';
    backdrop.addEventListener('click', () => {
      backdrop.remove();
      overlay.remove();
    });
    document.body.appendChild(backdrop);
    
    const overlay = document.createElement('div');
    overlay.className = 'osint-overlay';
    
    const consensus = result.consensus || {};
    const breakdown = result.stance_breakdown || {};
    
    // Determine color based on engagement quality
    let qualityColor = '#6c757d';
    if (consensus.engagement_quality === 'controversial') qualityColor = '#e74c3c';
    else if (consensus.engagement_quality === 'supportive') qualityColor = '#27ae60';
    else if (consensus.engagement_quality === 'debated') qualityColor = '#f39c12';
    
    overlay.innerHTML = `
      <div class="osint-header">
        <span class="osint-title">🧵 Thread Analysis</span>
        <button class="osint-close">✕</button>
      </div>
      
      <div class="osint-content">
        <div class="osint-row">
          <span class="osint-label">Replies Analyzed:</span>
          <span class="osint-value">${result.reply_count}</span>
        </div>
        
        <div class="osint-row">
          <span class="osint-label">Engagement Quality:</span>
          <span class="osint-value" style="color: ${qualityColor}; text-transform: capitalize;">
            ${consensus.engagement_quality?.replace('_', ' ') || 'Unknown'}
          </span>
        </div>
        
        <div class="osint-row">
          <span class="osint-label">Dominant Stance:</span>
          <span class="osint-value" style="text-transform: capitalize;">
            ${consensus.dominant_stance || 'Mixed'}
          </span>
        </div>
        
        <div class="osint-section">
          <div class="osint-section-title">Stance Breakdown</div>
          <div class="osint-stance-bars">
            <div class="osint-bar-row">
              <span class="osint-bar-label">👍 Agree</span>
              <div class="osint-bar-container">
                <div class="osint-bar osint-bar-agree" style="width: ${(consensus.agreement_ratio || 0) * 100}%"></div>
              </div>
              <span class="osint-bar-value">${breakdown.agree || 0}</span>
            </div>
            <div class="osint-bar-row">
              <span class="osint-bar-label">👎 Disagree</span>
              <div class="osint-bar-container">
                <div class="osint-bar osint-bar-disagree" style="width: ${(consensus.disagreement_ratio || 0) * 100}%"></div>
              </div>
              <span class="osint-bar-value">${breakdown.disagree || 0}</span>
            </div>
            <div class="osint-bar-row">
              <span class="osint-bar-label">💬 Discuss</span>
              <div class="osint-bar-container">
                <div class="osint-bar osint-bar-discuss" style="width: ${(consensus.discuss_ratio || 0) * 100}%"></div>
              </div>
              <span class="osint-bar-value">${breakdown.discuss || 0}</span>
            </div>
            <div class="osint-bar-row">
              <span class="osint-bar-label">🔇 Unrelated</span>
              <div class="osint-bar-container">
                <div class="osint-bar osint-bar-unrelated" style="width: ${(consensus.unrelated_ratio || 0) * 100}%"></div>
              </div>
              <span class="osint-bar-value">${breakdown.unrelated || 0}</span>
            </div>
          </div>
        </div>
        
        ${consensus.controversial ? `
          <div class="osint-flags">
            <span class="osint-flag">⚠️ Controversial Topic</span>
          </div>
        ` : ''}
        
        <div class="osint-explanation">
          ${consensus.credibility_signal === 'likely_disputed' ? 
            '⚠️ This claim is being disputed by many replies. Verify before sharing.' :
            consensus.credibility_signal === 'generally_accepted' ?
            '✅ Replies generally support this claim.' :
            '💡 Mixed reactions - verify independently.'}
        </div>
      </div>
    `;
    
    overlay.querySelector('.osint-close').addEventListener('click', () => {
      backdrop.remove();
      overlay.remove();
    });
    
    document.body.appendChild(overlay);
  }
  
  /**
   * Display claim verification results
   */
  function displayClaimVerificationResults(post, result) {
    document.querySelectorAll('.osint-overlay, .osint-backdrop').forEach(el => el.remove());
    
    const backdrop = document.createElement('div');
    backdrop.className = 'osint-backdrop';
    backdrop.addEventListener('click', () => {
      backdrop.remove();
      overlay.remove();
    });
    document.body.appendChild(backdrop);
    
    const overlay = document.createElement('div');
    overlay.className = 'osint-overlay';
    
    // Determine verdict color and icon
    let verdictColor = '#6c757d';
    let verdictIcon = '❓';
    const verdict = result.verdict || 'UNVERIFIED';
    
    if (verdict.includes('FALSE') || verdict.includes('MISLEADING')) {
      verdictColor = '#e74c3c';
      verdictIcon = '❌';
    } else if (verdict.includes('TRUE') || verdict === 'NO MISINFO DETECTED') {
      verdictColor = '#27ae60';
      verdictIcon = '✅';
    } else if (verdict === 'DISPUTED' || verdict.includes('POSSIBLY')) {
      verdictColor = '#f39c12';
      verdictIcon = '⚠️';
    } else if (verdict === 'LOW RISK') {
      verdictColor = '#3498db';
      verdictIcon = '✓';
    }
    
    // Build ML analysis section if available
    let mlAnalysisHTML = '';
    if (result.ml_analysis) {
      const ml = result.ml_analysis;
      const misinfoColor = ml.misinfo_confidence > 0.6 ? '#e74c3c' : ml.misinfo_confidence > 0.4 ? '#f39c12' : '#27ae60';
      mlAnalysisHTML = `
        <div class="osint-section">
          <div class="osint-section-title">🤖 ML Model Analysis</div>
          <div class="osint-row">
            <span class="osint-label">Misinfo Risk:</span>
            <span class="osint-value" style="color: ${misinfoColor}">
              ${ml.misinfo_risk || 'N/A'} (${Math.round((ml.misinfo_confidence || 0) * 100)}%)
            </span>
          </div>
          ${ml.sentiment ? `
          <div class="osint-row">
            <span class="osint-label">Sentiment:</span>
            <span class="osint-value">${ml.sentiment}</span>
          </div>
          ` : ''}
        </div>
      `;
    }
    
    // Sources section
    let sourcesHTML = '';
    if (result.sources && result.sources.length > 0) {
      sourcesHTML = `
        <div class="osint-section">
          <div class="osint-section-title">📚 Sources</div>
          <div style="font-size: 12px; color: #a0aec0;">
            ${result.sources.join(', ')}
          </div>
        </div>
      `;
    }
    
    overlay.innerHTML = `
      <div class="osint-header">
        <span class="osint-title">🔎 Claim Verification</span>
        <button class="osint-close">✕</button>
      </div>
      
      <div class="osint-content">
        <div class="osint-verdict" style="text-align: center; padding: 15px; margin-bottom: 15px; background: ${verdictColor}22; border-radius: 8px;">
          <span style="font-size: 32px;">${verdictIcon}</span>
          <div style="font-size: 20px; font-weight: bold; color: ${verdictColor}; margin-top: 10px;">
            ${verdict}
          </div>
          <div style="font-size: 14px; color: #888; margin-top: 5px;">
            Confidence: ${Math.round((result.confidence || 0) * 100)}%
          </div>
        </div>
        
        ${result.matched ? `
          <div class="osint-section">
            <div style="font-size: 13px; color: #e74c3c; padding: 10px; background: rgba(231,76,60,0.1); border-radius: 4px; border-left: 3px solid #e74c3c;">
              ⚠️ Matched known misinformation pattern
            </div>
          </div>
        ` : ''}
        
        ${mlAnalysisHTML}
        
        <div class="osint-section">
          <div class="osint-section-title">📝 Explanation</div>
          <div style="font-size: 13px; color: #e2e8f0; line-height: 1.6;">
            ${result.explanation || result.reasoning || 'Unable to verify this claim.'}
          </div>
        </div>
        
        ${sourcesHTML}
        
        <div class="osint-recommendation" style="margin-top: 15px; padding: 12px; background: rgba(59,130,246,0.1); border-left: 3px solid #3b82f6; border-radius: 0 6px 6px 0; font-size: 13px;">
          ${result.recommendation || '❓ Unable to verify. Check reputable news sources.'}
        </div>
      </div>
    `;
    
    overlay.querySelector('.osint-close').addEventListener('click', () => {
      backdrop.remove();
      overlay.remove();
    });
    
    document.body.appendChild(overlay);
  }
  
  /**
   * Show dropdown overlay for analysis options
   */
  function showDropdownOverlay(post) {
    // Remove any existing dropdown overlay
    document.querySelectorAll('.osint-dropdown-overlay, .osint-dropdown-backdrop').forEach(el => el.remove());
    
    // Save current scroll position BEFORE anything changes
    const savedScrollX = window.scrollX;
    const savedScrollY = window.scrollY;
    
    // Create backdrop
    const backdrop = document.createElement('div');
    backdrop.className = 'osint-dropdown-backdrop';
    backdrop.style.cssText = 'position:fixed !important; top:0 !important; left:0 !important; right:0 !important; bottom:0 !important; width:100vw !important; height:100vh !important; background:rgba(0,0,0,0.7) !important; z-index:2147483646 !important;';
    
    // Create overlay
    const overlay = document.createElement('div');
    overlay.className = 'osint-dropdown-overlay';
    overlay.style.cssText = 'position:fixed !important; top:50% !important; left:50% !important; transform:translate(-50%,-50%) !important; background:#1e293b !important; border-radius:16px !important; box-shadow:0 16px 48px rgba(0,0,0,0.5) !important; z-index:2147483647 !important; min-width:320px !important; max-width:90vw !important; border:1px solid rgba(255,255,255,0.1) !important; overflow:visible !important;';
    overlay.innerHTML = `
      <div class="osint-dropdown-header" style="display:flex; justify-content:space-between; align-items:center; padding:16px 20px; border-bottom:1px solid rgba(255,255,255,0.1); background:linear-gradient(135deg,#1e40af 0%,#2563eb 100%); border-radius:16px 16px 0 0;">
        <span style="font-size:16px; font-weight:600; color:#f1f5f9;">Analysis Options</span>
        <button class="osint-dropdown-close" style="display:flex; align-items:center; justify-content:center; width:28px; height:28px; background:rgba(255,255,255,0.1); border:none; border-radius:50%; color:white; font-size:16px; cursor:pointer;">✕</button>
      </div>
      <button class="osint-dropdown-item" data-action="full" style="display:flex; align-items:center; gap:12px; width:100%; padding:14px 20px; background:#1e293b; border:none; color:#e2e8f0; font-size:15px; text-align:left; cursor:pointer;">
        <span class="icon" style="font-size:24px;">🔍</span>
        <span class="text" style="display:flex; flex-direction:column; gap:2px;">
          <strong style="font-size:14px; font-weight:600; color:#f1f5f9;">Full Analysis</strong>
          <small style="font-size:12px; color:#94a3b8;">Sentiment, topic, misinfo risk & more</small>
        </span>
      </button>
      <button class="osint-dropdown-item" data-action="verify" style="display:flex; align-items:center; gap:12px; width:100%; padding:14px 20px; background:#1e293b; border:none; color:#e2e8f0; font-size:15px; text-align:left; cursor:pointer;">
        <span class="icon" style="font-size:24px;">🔎</span>
        <span class="text" style="display:flex; flex-direction:column; gap:2px;">
          <strong style="font-size:14px; font-weight:600; color:#f1f5f9;">Verify Claim</strong>
          <small style="font-size:12px; color:#94a3b8;">Check against misinformation patterns + ML</small>
        </span>
      </button>
      <button class="osint-dropdown-item" data-action="thread" style="display:flex; align-items:center; gap:12px; width:100%; padding:14px 20px; background:#1e293b; border:none; color:#e2e8f0; font-size:15px; text-align:left; cursor:pointer;">
        <span class="icon" style="font-size:24px;">🧵</span>
        <span class="text" style="display:flex; flex-direction:column; gap:2px;">
          <strong style="font-size:14px; font-weight:600; color:#f1f5f9;">Analyze Thread</strong>
          <small style="font-size:12px; color:#94a3b8;">Analyze replies and stance distribution</small>
        </span>
      </button>
      <button class="osint-dropdown-item" data-action="quote" style="display:flex; align-items:center; gap:12px; width:100%; padding:14px 20px; background:#1e293b; border:none; color:#e2e8f0; font-size:15px; text-align:left; cursor:pointer;">
        <span class="icon" style="font-size:24px;">💬</span>
        <span class="text" style="display:flex; flex-direction:column; gap:2px;">
          <strong style="font-size:14px; font-weight:600; color:#f1f5f9;">Quote Tweet Analysis</strong>
          <small style="font-size:12px; color:#94a3b8;">Compare original vs quote sentiment</small>
        </span>
      </button>
    `;
    
    // Close function
    const closeOverlay = () => {
      backdrop.remove();
      overlay.remove();
    };
    
    // Close button
    overlay.querySelector('.osint-dropdown-close').addEventListener('click', () => closeOverlay());
    
    // Close on backdrop click
    backdrop.addEventListener('click', () => closeOverlay());
    
    // Handle dropdown actions
    overlay.querySelectorAll('.osint-dropdown-item').forEach(item => {
      item.addEventListener('click', () => {
        closeOverlay();
        const action = item.dataset.action;
        if (action === 'full') {
          analyzePost(post);
        } else if (action === 'verify') {
          verifyClaim(post);
        } else if (action === 'thread') {
          analyzeThread(post);
        } else if (action === 'quote') {
          analyzeQuoteTweet(post);
        }
      });
    });
    
    document.body.appendChild(backdrop);
    document.body.appendChild(overlay);
    
    // Restore scroll position (prevent scroll-to-top)
    window.scrollTo(savedScrollX, savedScrollY);
  }
  
  /**
   * Add enhanced analyze button with dropdown
   */
  function addAnalyzeButton(post) {
    // Check if button already exists
    if (post.querySelector('.osint-analyze-btn')) return;
    
    const container = document.createElement('div');
    container.className = 'osint-btn-container';
    
    // Main analyze button
    const btn = document.createElement('button');
    btn.className = 'osint-analyze-btn';
    btn.innerHTML = '🔍 Analyze';
    btn.title = 'Analyze this post with OSINT Monitor';
    btn.setAttribute('type', 'button');
    btn.setAttribute('role', 'button');
    btn.setAttribute('tabindex', '0');
    btn.setAttribute('data-osint-btn', 'analyze');
    
    // Store post reference
    const postRef = post;
    
    // Use onclick property
    btn.onclick = function(e) {
      if (e) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
      }
      analyzePost(postRef);
      return false;
    };
    
    // Block all pointer/mouse events at capture phase
    ['mousedown', 'mouseup', 'pointerdown', 'pointerup', 'touchstart', 'touchend'].forEach(eventType => {
      btn.addEventListener(eventType, (e) => {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        return false;
      }, true);
    });
    
    // Dropdown for additional options (only on Twitter/X)
    if (platform === 'twitter' || platform === 'x') {
      const dropdownBtn = document.createElement('button');
      dropdownBtn.className = 'osint-dropdown-btn';
      dropdownBtn.title = 'More options';
      dropdownBtn.innerHTML = '▼';
      dropdownBtn.setAttribute('type', 'button');
      dropdownBtn.setAttribute('role', 'button');
      dropdownBtn.setAttribute('tabindex', '0');
      dropdownBtn.setAttribute('data-osint-btn', 'dropdown');
      
      // Store post reference for later
      const postRef = post;
      
      // Use onclick property (more reliable than addEventListener for stopping Twitter)
      dropdownBtn.onclick = function(e) {
        if (e) {
          e.preventDefault();
          e.stopPropagation();
          e.stopImmediatePropagation();
        }
        showDropdownOverlay(postRef);
        return false;
      };
      
      // Block all pointer/mouse events at capture phase
      ['mousedown', 'mouseup', 'pointerdown', 'pointerup', 'touchstart', 'touchend'].forEach(eventType => {
        dropdownBtn.addEventListener(eventType, (e) => {
          e.preventDefault();
          e.stopPropagation();
          e.stopImmediatePropagation();
          return false;
        }, true);
      });
      
      container.appendChild(btn);
      container.appendChild(dropdownBtn);
    } else {
      container.appendChild(btn);
    }
    
    // Find a good place to insert
    const textElement = post.querySelector(selectors.text);
    if (textElement) {
      textElement.parentElement.appendChild(container);
    } else {
      post.appendChild(container);
    }
  }
  
  // Listen for messages from popup
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'ANALYZE_SELECTED') {
      const selection = window.getSelection().toString().trim();
      if (selection) {
        chrome.runtime.sendMessage({
          type: 'ANALYZE_TEXT',
          data: { text: selection, source: platform },
        }).then(sendResponse);
        return true;
      }
    }
    
    if (message.type === 'TOGGLE_EXTENSION') {
      settings.enabled = message.enabled;
      sendResponse({ success: true });
    }
    
    if (message.type === 'REFRESH_SETTINGS') {
      loadSettings().then((newSettings) => {
        settings = newSettings;
        sendResponse({ success: true });
      });
      return true;
    }
    
    // New: Handle verify claim from popup
    if (message.type === 'VERIFY_CLAIM_RESULT') {
      displayClaimVerificationResults(null, message.result);
      sendResponse({ success: true });
    }
  });
  
  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
  
})();
