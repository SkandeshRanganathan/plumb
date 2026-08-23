const cricProfileCache = {};

async function extractDeepMatchContext() {
    let context = {
        bowler: "Unknown",
        batter: "Unknown",
        format: "T20",
        pressure_index: 50.0,
        right_bat: true,
        last_over_string: "",
        last_commentary: "",
        dew_pct: typeof window.currentDewPct !== 'undefined' ? window.currentDewPct : 30,
        wickets: 3,
        req_rate: 8.5,
        bowling_angle: typeof window.currentBowlingAngle !== 'undefined' ? window.currentBowlingAngle : "Over the wicket",
        scraped_style: "FAST_SEAM"
    };

    try {
        const url = window.location.href;
        
        if (url.includes("cricbuzz.com")) {
            // Smarter DOM scraping using Regex on raw text
            const allText = document.body.innerText || "";
            // Regex to find names before a star: e.g. "Cameron Green *" or "Taskin Ahmed *"
            const starMatches = [...allText.matchAll(/([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s*\*/g)];
            
            if (starMatches.length > 0) {
                context.batter = starMatches[0][1].trim();
                if (starMatches.length > 1) {
                    context.bowler = starMatches[1][1].trim();
                }
            } else {
                // Fallback to title parsing if no star is found
                const title = document.title || "";
                if (title.includes(" vs ")) {
                    const parts = title.split('(');
                    if (parts.length > 2) {
                        const playerPart = parts[2].split(')')[0];
                        const players = playerPart.split(/[0-9]+\s*\([0-9]+\)/).map(p => p.trim()).filter(p => p.length > 0);
                        if (players.length > 0) {
                            context.batter = players[players.length - 1].replace(/[0-9]+$/, '').trim();
                        }
                    }
                }
            }
            
            // 2. Ultra-robust Scrape of live commentary for Next Ball Prediction context
            // We search the entire raw text of the page for the first line that looks like "19.5 Bowler to Batter, result"
            const allTextLines = document.body.innerText || "";
            // Regex matches: over number (0.1 to 999.6) -> space -> Commentary Text
            const commMatch = allTextLines.match(/(?:^|\n)\s*(\d{1,3}\.[1-6])\s+([A-Z][^\n]+)/);
            if (commMatch) {
                context.last_over_string = commMatch[1].trim();
                context.last_commentary = commMatch[2].trim();
                
                // Extract Bowler and Batter perfectly synchronously from the commentary!
                // Format: "Bowler to Batter, result"
                const playersMatch = context.last_commentary.match(/^([A-Za-z\s'-]+)\s+to\s+([A-Za-z\s'-]+),/);
                if (playersMatch) {
                    context.bowler = playersMatch[1].trim();
                    context.batter = playersMatch[2].trim();
                }
            }
            
            // 3. Match Pressure (heuristic based on scorecard text)
            const scoreStr = document.body.innerText.substring(0, 500); // Check top of page
            if (scoreStr.toLowerCase().includes("need")) {
                context.pressure_index = 85; // High pressure chase
            } else if (scoreStr.toLowerCase().includes("trail")) {
                context.format = "Test";
                context.pressure_index = 60;
            }
            
            // 4. Tactical AI Scraping: Wickets and Req Rate
            const scoreMatch = allTextLines.match(/(\d{1,3})\s*[\/\-]\s*([0-9]|10)/);
            if (scoreMatch) {
                context.wickets = parseInt(scoreMatch[2]);
                if (context.wickets >= 7) context.pressure_index += 10;
            }
            
            const reqMatch = allTextLines.match(/Req(?:uired)?\s*R(?:un)?\s*R(?:ate)?\s*(?:\:|\-)?\s*(\d+\.\d+)/i);
            if (reqMatch) {
                context.req_rate = parseFloat(reqMatch[1]);
            }
        } // Closing if (url.includes("cricbuzz.com"))
        
        // --- 5. Stealth Profile Scraping ---
        if (typeof window.currentBowlingStyleOverride !== 'undefined' && window.currentBowlingStyleOverride !== "Auto-Detect") {
            context.scraped_style = window.currentBowlingStyleOverride;
            console.log(`Cricket AI: Style OVERRIDE applied -> ${context.scraped_style}`);
        } else if (context.bowler !== "Unknown") {
            if (cricProfileCache[context.bowler]) {
                context.scraped_style = cricProfileCache[context.bowler];
            } else {
                // Look for bowler profile link in DOM
                const links = Array.from(document.querySelectorAll("a[href*='/profiles/']"));
                // Find a link that contains the bowler's last name or first name
                const bowlerParts = context.bowler.split(" ");
                const lastName = bowlerParts[bowlerParts.length - 1];
                const firstName = bowlerParts[0];
                const profileLink = links.find(a => a.innerText.includes(lastName) || a.innerText.includes(firstName) || a.innerText.includes(context.bowler));
                
                if (profileLink) {
                    console.log(`Cricket AI: Fetching profile for ${context.bowler} from ${profileLink.href}`);
                    const resp = await fetch(profileLink.href);
                    const html = await resp.text();
                    // Regex to extract bowling style
                    const styleMatch = html.match(/Bowling Style.*?<div[^>]*>([^<]+)<\/div>/si);
                    if (styleMatch && styleMatch[1]) {
                        const style = styleMatch[1].trim();
                        cricProfileCache[context.bowler] = style;
                        context.scraped_style = style;
                        console.log(`Cricket AI: Profile Scraped! ${context.bowler} is ${style}`);
                    }
                }
            }
        }
        
    } catch(e) {
        console.error("Cricket AI: Error extracting deep context", e);
    }
    
    return context;
}

function injectSidebar() {
    const existingSidebar = document.getElementById('cric-ai-panel');
    if (existingSidebar) {
        if (existingSidebar.style.display === 'none') {
            existingSidebar.style.display = 'flex';
        } else {
            existingSidebar.style.display = 'none';
        }
        return;
    }

    const sidebar = document.createElement('div');
    sidebar.id = 'cric-ai-panel';
    sidebar.innerHTML = `
        <div class="cric-ai-header">
            <span style="font-size:18px; font-weight:900; letter-spacing:0.5px; color:#fff;">LIVE BROADCAST COMPANION</span>
            <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
                <button id="btn-vision" style="background:#8b5cf6; color:#fff; border:none; padding:4px 8px; border-radius:4px; font-size:10px; cursor:pointer; font-weight:700;">ENABLE VISION</button>
                <button id="test-end-match" style="background:#f43f5e; color:#fff; border:none; padding:4px 8px; border-radius:4px; font-size:10px; cursor:pointer; font-weight:700;">TEST POST-MATCH</button>
                <span id="cric-ai-close" style="cursor:pointer; font-size:20px; opacity:0.7; transition: 0.2s;">&times;</span>
            </div>
        </div>
        
        <!-- CV Vision Overlay Data -->
        <div id="vision-results" class="cric-ai-hidden" style="background:#1e1b4b; margin:10px; padding:10px; border-radius:6px; border-left:4px solid #8b5cf6; font-size:11px;">
            <div style="color:#a78bfa; font-weight:700; margin-bottom:4px; display:flex; justify-content:space-between;">
                <span>BROADCAST VISION ACTIVE</span>
                <span id="vision-status" style="color:#34d399;">Scanning...</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                <span style="color:#d1d5db;">Detected Pitch:</span>
                <span id="vis-pitch" style="color:#fff; font-weight:bold;">--</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                <span style="color:#d1d5db;">Field Setup:</span>
                <span id="vis-field" style="color:#fff; font-weight:bold;">--</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:2px; border-top:1px solid #3730a3; padding-top:4px; margin-top:4px;">
                <span style="color:#d1d5db;">Par Score:</span>
                <span id="vis-par" style="color:#fcd34d; font-weight:bold;">--</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                <span style="color:#d1d5db;">Toss/Win Prob:</span>
                <span id="vis-toss" style="color:#34d399; font-weight:bold;">--</span>
            </div>
            <div style="margin-top:4px; background:rgba(0,0,0,0.3); padding:6px; border-radius:4px; border-left:2px solid #38bdf8;">
                <span style="color:#9ca3af; display:block; margin-bottom:2px;">Optimal Length:</span>
                <span id="vis-length" style="color:#bae6fd; font-style:italic;">--</span>
            </div>
            <canvas id="hidden-vid-canvas" style="display:none;"></canvas>
            <video id="hidden-vid-stream" style="display:none;" autoplay></video>
        </div>
        
        <!-- Scrollable Content Area -->
        <div style="flex:1; overflow-y:auto; padding-right:8px; margin-bottom:16px;">
            <div id="ai-auto-status" style="text-align:center; margin-bottom:16px; font-size:11px; color:#34d399; font-weight:700; letter-spacing:1px; display:flex; align-items:center; justify-content:center; gap:6px;">
                <div style="width:6px; height:6px; background:#34d399; border-radius:50%; box-shadow:0 0 6px #34d399; animation: pulse 2s infinite;"></div>
                LIVE POLLING ACTIVE
            </div>

            <div class="cric-ai-card">
                <div class="cric-ai-title-sm">Match Context</div>
                <div class="cric-ai-row">
                    <span>Bowler:</span>
                    <span id="ai-bowler" style="color:#60a5fa; font-weight: 700;">Loading...</span>
                </div>
                <div class="cric-ai-row">
                    <span>Batter:</span>
                    <span id="ai-batter" style="color:#fcd34d; font-weight: 700;">Loading...</span>
                </div>
                <div class="cric-ai-row" style="align-items:flex-start;">
                    <span>Last Ball:</span>
                    <span id="ai-last-ball" style="color:#d1d5db; font-size: 13px; max-width: 250px; text-align:right; line-height:1.4;">--</span>
                </div>
                <div class="cric-ai-hidden" id="plan-container" style="margin-top:12px; padding-top:12px; border-top:1px solid #374151;">
                    <div style="font-size:12px; font-weight:700; color:#34d399; margin-bottom:4px;">Captain's Game Plan</div>
                    <div id="ai-plan" style="font-size:13px; color:#d1d5db; line-height:1.4;">--</div>
                </div>
            </div>

            <div id="ai-results" class="cric-ai-hidden">
                <!-- Evaluation Block -->
                <div id="eval-card" class="cric-eval-card cric-ai-hidden">
                    <div style="font-weight:700; color:#f59e0b; margin-bottom:6px;">Post-Ball Review</div>
                    <div id="eval-text" style="color:#d1d5db;">Loading evaluation...</div>
                </div>

                <div class="cric-ai-card">
                    <div class="cric-ai-title-sm">Live LBW Tracker (Current Ball)</div>
                    <div class="cric-ai-pitch-container" style="margin-bottom: 10px;">
                        <div class="cric-ai-pitch" id="pitch-lbw">
                            <div class="crease-bowling"></div>
                            <div class="crease-batting"></div>
                            <div class="stumps-container">
                                <div class="stump"></div>
                                <div class="stump"></div>
                                <div class="stump"></div>
                            </div>
                            <div class="ball"></div>
                        </div>
                    </div>
                    
                    <!-- Hawkeye Amenities Box -->
                    <div id="hawkeye-box" style="background:#1f2937; border-radius:6px; padding:10px; margin-top:15px; font-size:12px; border-left:3px solid #3b82f6;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                            <span style="color:#9ca3af;">Pitching</span>
                            <span id="hk-pitching" style="font-weight:700; color:#fff;">--</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                            <span style="color:#9ca3af;">Impact</span>
                            <span id="hk-impact" style="font-weight:700; color:#fff;">--</span>
                        </div>
                        <div style="display:flex; justify-content:space-between;">
                            <span style="color:#9ca3af;">Wickets</span>
                            <span id="hk-wickets" style="font-weight:700; color:#fff;">--</span>
                        </div>
                    </div>
                </div>

                <div class="cric-ai-card">
                    <div class="cric-ai-title-sm">Next Ball Prediction</div>
                    <div class="cric-ai-pitch-container">
                        <div class="cric-ai-pitch" id="pitch-next">
                            <div class="crease-bowling"></div>
                            <div class="crease-batting"></div>
                            <div class="stumps-container">
                                <div class="stump"></div>
                                <div class="stump"></div>
                                <div class="stump"></div>
                            </div>
                            <div class="ball ball-predictive"></div>
                        </div>
                    </div>
                    
                    <div class="cric-ai-row" style="margin-top:20px;">
                        <span style="font-weight:700;">Expected Delivery:</span>
                        <span id="pred-type" class="cric-ai-pill cric-ai-blue">...</span>
                    </div>
                    
                    <div class="cric-ai-row">
                        <span style="font-weight:700;">Batter Intent:</span>
                        <span id="batter-intent" class="cric-intent-badge intent-def">Loading...</span>
                    </div>
                    
                    <div class="cric-xw-container">
                        <div class="cric-xw-fill" id="xw-bar"></div>
                    </div>
                    <div class="cric-xw-text">
                        <span>Expected Wicket (xW)</span>
                        <span id="xw-val" style="font-weight:bold;">--%</span>
                    </div>
                    <div class="cric-ai-row">
                        <span style="font-weight:700;">Confidence:</span>
                        <span id="pred-conf" class="cric-ai-pill cric-ai-green">...</span>
                    </div>
                    
                    <div style="background:rgba(0,0,0,0.3); padding:10px; border-radius:6px; font-size:12px; border-left:3px solid #00e676; margin-top:12px;">
                        <strong style="color:#00e676;">⚡ AI EXECUTION PLAN:</strong><br>
                        
                        <!-- 2D Canvas Map here -->
                        <div id="cric-pitch-container">
                            <canvas id="cric-pitch-canvas" width="300" height="140"></canvas>
                        </div>
                        
                        <div style="margin-top:4px; display:flex; justify-content:space-between;">
                            <span style="color:#a3a3a3;">Angle:</span> <strong id="pred-angle" style="color:#fff; text-align:right;">...</strong>
                        </div>
                        <div style="margin-top:2px; display:flex; justify-content:space-between;">
                            <span style="color:#a3a3a3;">Pace:</span> <strong id="pred-pace" style="color:#fff; text-align:right;">...</strong>
                        </div>
                        <div style="margin-top:2px; display:flex; justify-content:space-between;">
                            <span style="color:#a3a3a3;">Field:</span> <strong id="pred-field" style="color:#fef3c7; text-align:right; font-size:11px;">...</strong>
                        </div>
                    </div>
                    
                    <!-- Tactical Field Radar -->
                    <div id="field-radar-container" class="cric-ai-hidden" style="margin-top: 15px; border-top: 1px solid #374151; padding-top: 10px;">
                        <span style="font-size:12px; font-weight:700; color:#9ca3af;">Tactical Radar Map:</span>
                        <div class="cric-field-ground" id="cric-field-ground">
                            <div class="cric-field-pitch"></div>
                            <!-- Fielders injected here -->
                        </div>
                    </div>
                    
                    <div id="pred-exp" style="margin-top:12px; font-size:12px; color:#9ca3af; line-height:1.5;"></div>
                </div>

                <div class="cric-ai-card">
                    <div class="cric-ai-title-sm" style="display:flex; justify-content:space-between;">
                        <span>Tactical Overrides</span>
                        <span style="font-size:10px; background:#3b82f6; padding:2px 6px; border-radius:10px;">PRO</span>
                    </div>
                    <div style="margin-top:10px;">
                        <label style="font-size:12px; color:#9ca3af; display:flex; justify-content:space-between;">
                            <span>Dew Factor:</span>
                            <span id="dew-val">30%</span>
                        </label>
                        <input type="range" id="dew-slider" min="0" max="100" value="30" style="width:100%; accent-color:#3b82f6;">
                    </div>
                    <div style="margin-top:10px; display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:12px; color:#9ca3af;">Bowling Angle:</span>
                        <select id="angle-select" style="background:#374151; color:#fff; border:1px solid #4b5563; padding:4px; border-radius:4px; font-size:11px;">
                            <option value="Over the wicket">Over the wicket</option>
                            <option value="Around the wicket">Around the wicket</option>
                        </select>
                    </div>
                    <div style="margin-top:10px; display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:12px; color:#9ca3af;">Bowling Style:</span>
                        <select id="style-override" style="background:#374151; color:#fff; border:1px solid #4b5563; padding:4px; border-radius:4px; font-size:11px;">
                            <option value="Auto-Detect">Auto-Detect</option>
                            <option value="SPIN_OVERRIDE">Spin</option>
                            <option value="PACE_OVERRIDE">Pace</option>
                        </select>
                    </div>
                </div>

                <div class="cric-ai-card">
                    <div class="cric-ai-title-sm">Bowler Analytics</div>
                    <div id="bowler-analytics" style="font-size:12px; color:#d1d5db; line-height: 1.6;">Loading stats...</div>
                </div>
            </div>
            
            <div id="chat-response" class="chat-response cric-ai-hidden"></div>
        </div>

        <!-- Floating Chat Button (Temporarily Disabled) -->
        <div id="cric-chat-fab" style="display:none; position:absolute; bottom:20px; right:20px; background:#3b82f6; width:50px; height:50px; border-radius:50%; align-items:center; justify-content:center; cursor:pointer; box-shadow: 0 4px 12px rgba(0,0,0,0.5); z-index: 10; transition: transform 0.2s;">
            <svg viewBox="0 0 24 24" width="24" height="24" stroke="#fff" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path></svg>
        </div>

        <!-- Modern Chatbot Container (Popup) -->
        <div class="cric-chat-wrapper cric-ai-hidden" id="cric-chat-popup" style="display:none; position:absolute; bottom:80px; right:20px; width:380px; z-index:10; box-shadow:0 10px 25px rgba(0,0,0,0.8);">
            <div class="cric-chat-header">Match Context Q&A <span id="cric-chat-close" style="float:right; cursor:pointer;">&times;</span></div>
            <div id="cric-chat-history" class="cric-chat-history">
                <div class="chat-bubble bot-bubble">Ask me anything about the bowler's stats, strategies, or predictions!</div>
            </div>
            <div class="cric-chat-input-area">
                <input type="text" id="chat-input" class="cric-chat-input" placeholder="Type your question...">
                <button id="chat-btn" class="cric-chat-btn">
                    <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                </button>
            </div>
        </div>
    `;
    
    // Inject natively docked to the right side of the screen
    document.body.appendChild(sidebar);
    
    document.getElementById('cric-ai-close').addEventListener('click', () => {
        sidebar.style.display = 'none';
    });
    
    // Chat FAB toggles popup
    document.getElementById('cric-chat-fab').addEventListener('click', () => {
        const popup = document.getElementById('cric-chat-popup');
        popup.classList.toggle('cric-ai-hidden');
    });
    document.getElementById('cric-chat-close').addEventListener('click', () => {
        document.getElementById('cric-chat-popup').classList.add('cric-ai-hidden');
    });
    
    // Broadcast Vision Logic
    let visionInterval = null;
    let videoStream = null;
    
    document.getElementById('btn-vision').addEventListener('click', async () => {
        if (visionInterval) {
            // Stop Vision
            clearInterval(visionInterval);
            visionInterval = null;
            if (videoStream) {
                videoStream.getTracks().forEach(t => t.stop());
            }
            document.getElementById('vision-results').classList.add('cric-ai-hidden');
            document.getElementById('btn-vision').innerText = '👁️ ENABLE VISION';
            document.getElementById('btn-vision').style.background = '#8b5cf6';
            return;
        }
        
        try {
            // Request user to select the broadcast tab
            videoStream = await navigator.mediaDevices.getDisplayMedia({
                video: { displaySurface: "browser" },
                audio: false
            });
            
            const videoEl = document.getElementById('hidden-vid-stream');
            videoEl.muted = true;
            videoEl.playsInline = true;
            videoEl.srcObject = videoStream;
            videoEl.play().catch(e => console.error("Video play failed:", e));
            
            document.getElementById('vision-results').classList.remove('cric-ai-hidden');
            document.getElementById('btn-vision').innerText = '🔴 STOP VISION';
            document.getElementById('btn-vision').style.background = '#ef4444';
            document.getElementById('vision-status').innerText = "Live";
            document.getElementById('vision-status').style.color = "#34d399";
            
            // Poll OpenCV backend every 4 seconds
            visionInterval = setInterval(async () => {
                if (!videoEl || videoEl.readyState < 2) {
                    console.log("Video not ready yet");
                    return;
                }
                
                const canvas = document.getElementById('hidden-vid-canvas');
                canvas.width = videoEl.videoWidth || 1280;
                canvas.height = videoEl.videoHeight || 720;
                
                const ctx = canvas.getContext('2d');
                try {
                    ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
                } catch(err) {
                    console.error("Canvas drawImage failed", err);
                    return;
                }
                
                const dataUrl = canvas.toDataURL('image/jpeg', 0.5);
                
                try {
                    const res = await fetch('http://localhost:8000/analyze_frame', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: dataUrl })
                    });
                    const data = await res.json();
                    if (data.status === 'success') {
                        document.getElementById('vision-status').innerText = "Live";
                        document.getElementById('vis-pitch').innerText = data.pitch_type;
                        document.getElementById('vis-field').innerText = data.field_setup;
                        
                        // Injecting Pitch Intelligence Data
                        if (data.par_score) {
                            document.getElementById('vis-par').innerText = data.par_score;
                            document.getElementById('vis-toss').innerText = `${data.toss_decision} (${data.win_prob})`;
                            document.getElementById('vis-length').innerText = data.optimal_length;
                        }
                    }
                } catch(err) {
                    console.error("Vision API Error", err);
                }
            }, 4000);
            
        } catch (e) {
            console.error("Vision Permission Denied", e);
            alert("Broadcast Vision requires permission to view the stream tab.");
        }
    });
    
    // Tactical Overrides listeners
    window.currentDewPct = 30;
    window.currentBowlingAngle = "Over the wicket";
    window.currentBowlingStyleOverride = "Auto-Detect";
    
    document.getElementById('dew-slider').addEventListener('input', (e) => {
        window.currentDewPct = parseInt(e.target.value);
        document.getElementById('dew-val').textContent = window.currentDewPct + '%';
    });
    
    document.getElementById('angle-select').addEventListener('change', (e) => {
        window.currentBowlingAngle = e.target.value;
    });
    
    document.getElementById('style-override').addEventListener('change', (e) => {
        window.currentBowlingStyleOverride = e.target.value;
    });



    // Auto-polling Engine
    let lastContextHash = "";
    
    async function runAutoPrediction() {
        const currentCtx = await extractDeepMatchContext();
        // Generate a hash to check if state changed (new ball bowled)
        const ctxHash = `${currentCtx.bowler}-${currentCtx.batter}-${currentCtx.last_over_string}`;
        
        // Update live context UI always
        document.getElementById('ai-bowler').innerText = currentCtx.bowler;
        document.getElementById('ai-batter').innerText = currentCtx.batter;
        
        if (ctxHash !== lastContextHash) {
            lastContextHash = ctxHash;
            
            // Temporarily show raw text while fetching new prediction
            if (currentCtx.last_over_string) {
                document.getElementById('ai-last-ball').innerText = `${currentCtx.last_over_string}: ${currentCtx.last_commentary}`;
            }
            
            try {
                const res = await fetch('http://localhost:8000/predict_next_ball', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(currentCtx)
                });
                const data = await res.json();
                
                if (!res.ok) {
                    console.error("API Error", data);
                    return;
                }
                
                document.getElementById('ai-results').classList.remove('cric-ai-hidden');
                
                // Helper for pill colors
                const getTypeColor = (type) => {
                    if (type.includes('Yorker') || type.includes('Death')) return 'cric-ai-red';
                    if (type.includes('Length')) return 'cric-ai-blue';
                    if (type.includes('Bouncer') || type.includes('Short')) return 'cric-ai-yellow';
                    return 'cric-ai-blue';
                };
                
                // Update Prediction UI
                document.getElementById('pred-type').innerText = data.predicted_type;
                document.getElementById('pred-type').className = `cric-ai-pill ${getTypeColor(data.predicted_type)}`;
                document.getElementById('pred-conf').innerText = data.confidence;
                if (document.getElementById('pred-exp')) document.getElementById('pred-exp').innerText = data.explanation;
                
                // Ultimate Tactical Engine Mapping
                if (data.rec_angle) document.getElementById('pred-angle').innerText = data.rec_angle;
                if (data.rec_pace) document.getElementById('pred-pace').innerText = data.rec_pace;
                if (data.field_pred) document.getElementById('pred-field').innerText = data.field_pred;
                
                // Intent & xW mapping
                if (data.batter_intent) {
                    const intentEl = document.getElementById('batter-intent');
                    intentEl.innerText = data.batter_intent.split('(')[0].trim();
                    if (data.batter_intent.includes('Aggressive') || data.batter_intent.includes('Attacking')) {
                        intentEl.className = 'cric-intent-badge intent-agg';
                    } else {
                        intentEl.className = 'cric-intent-badge intent-def';
                    }
                }
                
                if (data.xw) {
                    document.getElementById('xw-val').innerText = data.xw.toFixed(1) + '%';
                    document.getElementById('xw-bar').style.width = data.xw + '%';
                }
                
                // 2D HTML5 Canvas Target Drawer
                if (data.rec_x !== undefined && data.rec_y !== undefined) {
                    const canvas = document.getElementById('cric-pitch-canvas');
                    if (canvas) {
                        const ctx = canvas.getContext('2d');
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        
                        // Draw Pitch background
                        ctx.fillStyle = '#1f2937';
                        ctx.fillRect(40, 10, 220, 120);
                        
                        // Draw Crease lines
                        ctx.strokeStyle = '#fff';
                        ctx.lineWidth = 1.5;
                        ctx.beginPath();
                        ctx.moveTo(40, 20); ctx.lineTo(260, 20); // Bowling Crease
                        ctx.moveTo(40, 120); ctx.lineTo(260, 120); // Batting Crease
                        ctx.moveTo(150, 10); ctx.lineTo(150, 130); // Center line
                        ctx.stroke();
                        
                        // Draw Stumps
                        ctx.fillStyle = '#f59e0b';
                        ctx.fillRect(145, 118, 2, 4);
                        ctx.fillRect(149, 118, 2, 4);
                        ctx.fillRect(153, 118, 2, 4);
                        
                        // Calculate coordinates (API: x=0-100, y=0-100)
                        // x=0 is left (width: 40 to 260) -> 220 pixels
                        // y=0 is bowling crease (height: 20 to 120) -> 100 pixels
                        const drawX = 40 + (data.rec_x / 100) * 220;
                        const drawY = 20 + (data.rec_y / 100) * 100;
                        
                        // Draw Glowing Crosshair
                        ctx.beginPath();
                        ctx.arc(drawX, drawY, 8, 0, 2 * Math.PI);
                        ctx.fillStyle = 'rgba(239, 68, 68, 0.4)'; // Red glow
                        ctx.fill();
                        
                        ctx.beginPath();
                        ctx.arc(drawX, drawY, 3, 0, 2 * Math.PI);
                        ctx.fillStyle = '#ef4444'; // Solid Red Core
                        ctx.fill();
                        
                        ctx.beginPath();
                        ctx.moveTo(drawX - 12, drawY);
                        ctx.lineTo(drawX + 12, drawY);
                        ctx.moveTo(drawX, drawY - 12);
                        ctx.lineTo(drawX, drawY + 12);
                        ctx.strokeStyle = '#ef4444';
                        ctx.stroke();
                    }
                }
                
                // Map Tactical Field Radar
                if (data.field_map) {
                    const radar = document.getElementById('field-radar-container');
                    radar.classList.remove('cric-ai-hidden');
                    
                    const ground = document.getElementById('cric-field-ground');
                    // Remove existing fielders
                    ground.querySelectorAll('.fielder-dot').forEach(el => el.remove());
                    
                    data.field_map.forEach(fielder => {
                        const dot = document.createElement('div');
                        dot.className = `fielder-dot ${fielder.moved ? 'fielder-moved' : ''}`;
                        dot.style.left = fielder.x + '%';
                        dot.style.top = fielder.y + '%';
                        dot.title = fielder.role;
                        ground.appendChild(dot);
                    });
                }

                if (data.situational_plan) {
                    document.getElementById('plan-container').classList.remove('cric-ai-hidden');
                    document.getElementById('ai-plan').innerText = data.situational_plan;
                }
                
                // Map coordinates to CSS variables for LBW pitch
                if (data.lbw_anim) {
                    const pitchLbw = document.getElementById('pitch-lbw');
                    const lbwEl = pitchLbw.querySelector('.ball');
                    lbwEl.style.animation = 'none';
                    void lbwEl.offsetWidth; // Force reflow
                    
                    lbwEl.style.setProperty('--start-x', data.lbw_anim.start_x + '%');
                    lbwEl.style.setProperty('--start-y', data.lbw_anim.start_y + '%');
                    lbwEl.style.setProperty('--pitch-x', data.lbw_anim.pitch_x + '%');
                    lbwEl.style.setProperty('--pitch-y', data.lbw_anim.pitch_y + '%');
                    lbwEl.style.setProperty('--end-x', data.lbw_anim.end_x + '%');
                    lbwEl.style.setProperty('--end-y', data.lbw_anim.end_y + '%');
                    
                    lbwEl.style.animation = 'isoTrajectory 2s cubic-bezier(0.25, 1, 0.5, 1) infinite';
                    
                    // Render Over History dots
                    if (data.over_history) {
                        // Clear old history dots
                        pitchLbw.querySelectorAll('.history-dot').forEach(el => el.remove());
                        data.over_history.forEach(pt => {
                            const dot = document.createElement('div');
                            dot.className = 'history-dot';
                            dot.style.setProperty('--pitch-x', pt.x + '%');
                            dot.style.setProperty('--pitch-y', pt.y + '%');
                            pitchLbw.appendChild(dot);
                        });
                    }
                }
                
                // Update Hawkeye Amenities
                if (data.hawkeye) {
                    const getHkColor = (val) => {
                        if (val === 'Hitting' || val === 'In Line') return '#ef4444'; // Red for hitting
                        if (val === "Umpire's Call") return '#f59e0b'; // Orange
                        return '#10b981'; // Green for missing/outside
                    };
                    const pEl = document.getElementById('hk-pitching');
                    pEl.innerText = data.hawkeye.pitching;
                    pEl.style.color = getHkColor(data.hawkeye.pitching);
                    
                    const iEl = document.getElementById('hk-impact');
                    iEl.innerText = data.hawkeye.impact;
                    iEl.style.color = getHkColor(data.hawkeye.impact);
                    
                    const wEl = document.getElementById('hk-wickets');
                    wEl.innerText = data.hawkeye.wickets;
                    wEl.style.color = getHkColor(data.hawkeye.wickets);
                }
                
                // Map coordinates to CSS variables for Next Ball pitch
                if (data.next_anim) {
                    const nextEl = document.querySelector('#pitch-next .ball');
                    nextEl.style.animation = 'none';
                    void nextEl.offsetWidth; // Force reflow
                    
                    nextEl.style.setProperty('--start-x', data.next_anim.start_x + '%');
                    nextEl.style.setProperty('--start-y', data.next_anim.start_y + '%');
                    nextEl.style.setProperty('--pitch-x', data.next_anim.pitch_x + '%');
                    nextEl.style.setProperty('--pitch-y', data.next_anim.pitch_y + '%');
                    nextEl.style.setProperty('--end-x', data.next_anim.end_x + '%');
                    nextEl.style.setProperty('--end-y', data.next_anim.end_y + '%');
                    nextEl.style.animation = 'isoTrajectoryPredictive 2s cubic-bezier(0.25, 1, 0.5, 1) infinite';
                }
                
                // Update Last Ball with Unique AI Comment
                if (data.unique_comment) {
                    document.getElementById('ai-last-ball').innerText = `${currentCtx.last_over_string}: ${data.unique_comment}`;
                }
                
                // Update Bowler Analytics
                document.getElementById('bowler-analytics').innerText = data.bowler_analytics;
                
                // Show Post-Ball Evaluation if available
                if (data.evaluation) {
                    document.getElementById('eval-card').classList.remove('cric-ai-hidden');
                    document.getElementById('eval-text').innerText = data.evaluation;
                } else {
                    document.getElementById('eval-card').classList.add('cric-ai-hidden');
                }
                
            } catch (err) {
                console.error("AI Auto-Polling Error", err);
            }
        }
    }
    
    // Run immediately, then poll every 2 seconds
    runAutoPrediction();
    setInterval(runAutoPrediction, 2000);
    
    // Chat functionality
    function addChatBubble(text, sender) {
        const history = document.getElementById('cric-chat-history');
        const bubble = document.createElement('div');
        bubble.className = `chat-bubble ${sender === 'user' ? 'user-bubble' : 'bot-bubble'}`;
        bubble.innerHTML = text;
        history.appendChild(bubble);
        history.scrollTop = history.scrollHeight;
    }

    document.getElementById('chat-btn').addEventListener('click', async () => {
        const inputEl = document.getElementById('chat-input');
        const query = inputEl.value.trim();
        if (!query) return;
        
        addChatBubble(query, 'user');
        inputEl.value = "";
        
        const loadingId = 'loading-' + Date.now();
        const history = document.getElementById('cric-chat-history');
        history.insertAdjacentHTML('beforeend', `<div id="${loadingId}" class="chat-bubble bot-bubble" style="opacity: 0.5;">Thinking...</div>`);
        history.scrollTop = history.scrollHeight;
        
        try {
            const ctx = await extractDeepMatchContext();
            const res = await fetch('http://localhost:8000/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, context: ctx })
            });
            const data = await res.json();
            document.getElementById(loadingId).remove();
            addChatBubble(data.answer, 'bot');
        } catch (e) {
            document.getElementById(loadingId).remove();
            addChatBubble("Failed to connect to AI server.", 'bot');
        }
    });
    
    document.getElementById('chat-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') document.getElementById('chat-btn').click();
    });
    
    // Check for match end automatically in text
    setInterval(() => {
        const txt = document.body.innerText.substring(0, 1000).toLowerCase();
        if ((txt.includes("match ended") || txt.includes("stumps") || txt.includes("won by")) && !window.cricMatchEnded) {
            // triggerPostMatch(); // Auto-trigger disabled so it doesn't randomly trigger, user can click TEST STUMPS
        }
    }, 10000);
    
    document.getElementById('test-end-match').addEventListener('click', () => {
        triggerPostMatch();
    });
}

async function triggerPostMatch() {
    if (window.cricMatchEnded) return;
    window.cricMatchEnded = true;
    
    // Scrape Real Team Names from h1
    let ta = "Team A";
    let tb = "Team B";
    const h1 = document.querySelector('h1') ? document.querySelector('h1').innerText : (document.title || "");
    const matchStr = h1.replace("Cricket scorecard | ", "");
    if (matchStr.includes(" vs ")) {
        const teams = matchPart = matchStr.split(',')[0].split(' vs ');
        if(teams.length === 2) { 
            ta = teams[0].trim(); 
            tb = teams[1].trim(); 
        }
    }
    
    // Deep Scrape for Scores and Player of the Match
    let taScore = "150/4"; // Realistic fallbacks
    let tbScore = "148/8";
    let pom = "Top Scorer";
    let mvp_a = "Loading...";
    let mvp_b = "Loading...";
    
    // 1. Framework-Agnostic Player Scraping (Look for links to player profiles!)
    const playerNodes = Array.from(document.querySelectorAll('a[href*="/profiles/"], a[href*="/player/"]'));
    const playerLinks = playerNodes.map(a => a.innerText.trim()).filter(text => text.length > 2 && !text.includes("Match"));
    
    // Fallback to Live UI elements
    let liveBatter = document.getElementById('ai-batter') ? document.getElementById('ai-batter').innerText : "";
    let liveBowler = document.getElementById('ai-bowler') ? document.getElementById('ai-bowler').innerText : "";

    const uniquePlayers = [...new Set(playerLinks)];
    if (uniquePlayers.length >= 4) {
        mvp_a = "⭐ " + uniquePlayers[0] + "\n🔥 " + uniquePlayers[1];
        mvp_b = "⭐ " + uniquePlayers[2] + "\n🔥 " + uniquePlayers[3];
        pom = uniquePlayers[0];
    } else {
        mvp_a = "⭐ " + (liveBatter || ta + " Batter");
        mvp_b = "🔥 " + (liveBowler || tb + " Bowler");
        pom = liveBatter || ta + " Star";
    }
    
    // 2. Accurate Score Scraping (Tied directly to the Team Names to avoid garbage)
    const bodyText = document.body.innerText || "";
    // Match "Australia 140/8" or "AUS 140/8"
    const scoreRegex = new RegExp(`(?:${ta}|${tb}|[A-Z]{3})\\s+(\\d{1,3}\\/\\d{1,2}|\\d{1,3}-\\d{1,2})`, 'ig');
    const scoreMatches = [...bodyText.matchAll(scoreRegex)];
    
    if (scoreMatches.length >= 2) {
        tbScore = scoreMatches[0][1]; // Innings 1
        taScore = scoreMatches[1][1]; // Innings 2
    } else {
        // Fallback to any valid score looking text
        const fallback = [...bodyText.matchAll(/\b(\d{1,3}\/\d{1,2})\b/g)];
        if (fallback.length >= 2) {
            tbScore = fallback[0][1];
            taScore = fallback[1][1];
        }
    }
    
    // 3. Scrape Player of the Match surgically
    const pomMatch = bodyText.match(/PLAYER OF THE MATCH\s*([A-Za-z\s'-]+)\s*\n/i);
    if (pomMatch) pom = pomMatch[1].trim();
    
    // Inject Overlay HTML
    const overlay = document.createElement('div');
    overlay.id = 'cric-post-match-overlay';
    overlay.innerHTML = `
        <div id="cric-pm-close">&times;</div>
        <div class="cric-pm-title">Post-Match Strategic Debrief</div>
        <div class="cric-pm-dashboard" style="max-width: 1700px; gap: 20px;">
            
            <!-- Left: Pitch Maps -->
            <div class="cric-pm-panel" style="flex: 1.2;">
                <h2>Cumulative Pitch Maps</h2>
                <div style="display:flex; gap:20px; height:100%; align-items:center;">
                    <div style="flex:1; display:flex; flex-direction:column; align-items:center;">
                        <span style="font-size:12px; color:#9ca3af; margin-bottom:20px;" id="lbl-pitch-a">${ta} Bowling</span>
                        <div class="cric-ai-pitch" id="pm-pitch-a" style="transform: rotateX(60deg) scale(1.1); margin-top:20px;">
                            <div class="crease-bowling"></div><div class="crease-batting"></div>
                            <div class="stumps-container"><div class="stump"></div><div class="stump"></div><div class="stump"></div></div>
                        </div>
                    </div>
                    <div style="flex:1; display:flex; flex-direction:column; align-items:center;">
                        <span style="font-size:12px; color:#9ca3af; margin-bottom:20px;" id="lbl-pitch-b">${tb} Bowling</span>
                        <div class="cric-ai-pitch" id="pm-pitch-b" style="transform: rotateX(60deg) scale(1.1); margin-top:20px;">
                            <div class="crease-bowling"></div><div class="crease-batting"></div>
                            <div class="stumps-container"><div class="stump"></div><div class="stump"></div><div class="stump"></div></div>
                        </div>
                    </div>
                </div>
                <div style="display:flex; justify-content:center; gap:15px; margin-top:20px; font-size:12px;">
                    <span style="color:#ef4444;">● Short</span>
                    <span style="color:#3b82f6;">● Good Length</span>
                    <span style="color:#facc15;">● Full/Yorker</span>
                </div>
            </div>
            
            <!-- Middle: Big Data Grid -->
            <div class="cric-pm-panel" style="flex: 1.8;">
                <h2>Match Analytics Grid</h2>
                
                <div style="display:flex; gap: 20px; height: 100%;">
                    
                    <!-- Performers & Metrics -->
                    <div style="flex:1; display:flex; flex-direction:column; gap: 15px;">
                        <div class="cric-pm-stat-box" style="margin-bottom:0; flex:1;">
                            <div class="cric-pm-stat-title" id="lbl-mvp-a">${ta} Top Performers</div>
                            <div id="pm-mvp-a" style="color:#fcd34d; font-weight:700; white-space:pre-wrap;">Loading...</div>
                        </div>
                        <div class="cric-pm-stat-box" style="margin-bottom:0; flex:1;">
                            <div class="cric-pm-stat-title" id="lbl-mvp-b">${tb} Top Performers</div>
                            <div id="pm-mvp-b" style="color:#34d399; font-weight:700; white-space:pre-wrap;">Loading...</div>
                        </div>
                        <div class="cric-pm-stat-box" style="margin-bottom:0; display:flex; gap:20px;">
                            <div style="flex:1;">
                                <div class="cric-pm-stat-title" style="font-size:10px;">Dot Ball %</div>
                                <div style="font-size:24px; font-weight:900; color:#fff;" id="pm-dot">--%</div>
                            </div>
                            <div style="flex:1;">
                                <div class="cric-pm-stat-title" style="font-size:10px;">Boundary %</div>
                                <div style="font-size:24px; font-weight:900; color:#facc15;" id="pm-bound">--%</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Wagon Wheel & Phase Breakdown -->
                    <div style="flex:1; display:flex; flex-direction:column; gap: 15px;">
                        <div class="cric-pm-stat-box" style="margin-bottom:0; display:flex; flex-direction:column; align-items:center; justify-content:center;">
                            <div class="cric-pm-stat-title" style="margin-bottom:10px;">Wagon Wheel (Boundaries)</div>
                            <div class="wagon-wheel-container" id="pm-wagon-wheel" style="width: 120px; height: 120px;">
                                <div class="wagon-wheel-pitch"></div>
                            </div>
                        </div>
                        
                        <div class="cric-pm-stat-box" style="margin-bottom:0; flex:1;">
                            <div class="cric-pm-stat-title" style="margin-bottom:10px;">Phase Breakdown</div>
                            <table style="width:100%; font-size:12px; border-collapse:collapse; text-align:center;">
                                <tr style="color:#9ca3af; border-bottom:1px solid #4b5563;">
                                    <th style="padding-bottom:5px; text-align:left;">Phase</th>
                                    <th style="padding-bottom:5px;" id="lbl-ph-a">${ta}</th>
                                    <th style="padding-bottom:5px;" id="lbl-ph-b">${tb}</th>
                                </tr>
                                <tr style="border-bottom:1px solid #374151;">
                                    <td style="padding:8px 0; text-align:left; color:#facc15;">Powerplay</td>
                                    <td style="padding:8px 0;" id="ph-a-pp">--</td>
                                    <td style="padding:8px 0;" id="ph-b-pp">--</td>
                                </tr>
                                <tr style="border-bottom:1px solid #374151;">
                                    <td style="padding:8px 0; text-align:left; color:#3b82f6;">Middle</td>
                                    <td style="padding:8px 0;" id="ph-a-mid">--</td>
                                    <td style="padding:8px 0;" id="ph-b-mid">--</td>
                                </tr>
                                <tr>
                                    <td style="padding:8px 0; text-align:left; color:#ef4444;">Death</td>
                                    <td style="padding:8px 0;" id="ph-a-dth">--</td>
                                    <td style="padding:8px 0;" id="ph-b-dth">--</td>
                                </tr>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Right: Strategy -->
            <div class="cric-pm-panel" style="flex: 1.2;">
                <h2>The Boardroom (AI Strategy)</h2>
                <div class="cric-pm-stat-box" style="border-left: 4px solid #3b82f6;">
                    <div class="cric-pm-stat-title" id="lbl-strat-a">${ta} Debrief</div>
                    <div id="pm-team-a" style="color:#e5e7eb;">Loading...</div>
                </div>
                <div class="cric-pm-stat-box" style="border-left: 4px solid #ef4444;">
                    <div class="cric-pm-stat-title" id="lbl-strat-b">${tb} Debrief</div>
                    <div id="pm-team-b" style="color:#e5e7eb;">Loading...</div>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(overlay);
    
    document.getElementById('cric-pm-close').addEventListener('click', () => {
        overlay.remove();
        window.cricMatchEnded = false;
    });
    
    try {
        const res = await fetch('http://localhost:8000/post_match', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                team_a: ta, 
                team_b: tb,
                team_a_score: taScore,
                team_b_score: tbScore,
                team_a_mvp: mvp_a,
                team_b_mvp: mvp_b,
                pom: pom
            })
        });
        const data = await res.json();
        
        if (data.status === 'success') {
            document.getElementById('pm-mvp-a').innerText = data.team_a_mvp;
            document.getElementById('pm-mvp-b').innerText = data.team_b_mvp;
            document.getElementById('pm-team-a').innerText = data.team_a_strategy;
            document.getElementById('pm-team-b').innerText = data.team_b_strategy;
            
            document.getElementById('pm-dot').innerText = data.team_a_metrics.dot_pct + "%";
            document.getElementById('pm-bound').innerText = data.team_a_metrics.bound_pct + "%";
            
            document.getElementById('ph-a-pp').innerText = data.team_a_phases.pp;
            document.getElementById('ph-a-mid').innerText = data.team_a_phases.middle;
            document.getElementById('ph-a-dth').innerText = data.team_a_phases.death;
            
            document.getElementById('ph-b-pp').innerText = data.team_b_phases.pp;
            document.getElementById('ph-b-mid').innerText = data.team_b_phases.middle;
            document.getElementById('ph-b-dth').innerText = data.team_b_phases.death;
            
            const pitchA = document.getElementById('pm-pitch-a');
            data.team_a_pitch_map.forEach(pt => {
                const dot = document.createElement('div');
                dot.className = 'cum-pitch-dot';
                dot.style.left = pt.x + '%';
                dot.style.top = pt.y + '%';
                dot.style.color = pt.color;
                dot.style.backgroundColor = pt.color;
                pitchA.appendChild(dot);
            });
            
            const pitchB = document.getElementById('pm-pitch-b');
            data.team_b_pitch_map.forEach(pt => {
                const dot = document.createElement('div');
                dot.className = 'cum-pitch-dot';
                dot.style.left = pt.x + '%';
                dot.style.top = pt.y + '%';
                dot.style.color = pt.color;
                dot.style.backgroundColor = pt.color;
                pitchB.appendChild(dot);
            });
            
            const wheel = document.getElementById('pm-wagon-wheel');
            data.wagon_wheel.forEach(spoke => {
                const line = document.createElement('div');
                line.className = 'wagon-spoke wagon-' + spoke.type;
                line.style.transform = `translateY(-50%) rotate(${spoke.angle}deg)`;
                wheel.appendChild(line);
            });
        }
    } catch (e) {
        console.error("Failed to fetch post match stats", e);
    }
}

// Inject immediately when Chrome grants permission and loads the script
setTimeout(injectSidebar, 1500);
