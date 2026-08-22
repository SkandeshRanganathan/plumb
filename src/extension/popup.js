let currentContext = null;
let lastPrediction = null;

// 1. Ask content.js for live context when popup opens
chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    chrome.tabs.sendMessage(tabs[0].id, {action: "GET_CONTEXT"}, function(response) {
        const btn = document.getElementById('btn-predict');
        if (response) {
            currentContext = response;
            document.getElementById('val-bowler').innerText = response.bowler;
            document.getElementById('val-batter').innerText = response.batter;
            document.getElementById('val-pressure').innerText = response.pressure + " / 100";
            
            btn.innerText = "Run Live Prediction";
            btn.disabled = false;
        } else {
            document.getElementById('val-bowler').innerText = "Not found";
            document.getElementById('val-batter').innerText = "Not found";
            btn.innerText = "No Match Detected";
        }
    });
});

// 2. Fetch from FastAPI Backend
document.getElementById('btn-predict').addEventListener('click', async () => {
    if (!currentContext) return;
    
    document.getElementById('btn-predict').innerText = "Predicting...";
    
    try {
        const res = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentContext)
        });
        
        const data = await res.json();
        lastPrediction = data.prediction;
        
        // Show results
        document.getElementById('results').style.display = 'block';
        document.getElementById('btn-predict').style.display = 'none';
        
        // Populate Wide logic
        const wide = data.prediction.wide;
        const wBadge = document.getElementById('wide-badge');
        wBadge.innerText = wide.decision;
        wBadge.className = "badge " + (wide.decision === "WIDE" ? "badge-wide" : "badge-legal");
        document.getElementById('wide-prob').innerText = (wide.confidence * 100).toFixed(1) + "% Confident";
        document.getElementById('wide-exp').innerText = wide.explanation;
        
        // Populate LBW logic
        const lbw = data.prediction.lbw;
        const pEl = document.getElementById('lbw-pitch');
        const iEl = document.getElementById('lbw-impact');
        const wEl = document.getElementById('lbw-wkt');
        
        pEl.innerText = lbw.pitching;
        pEl.className = lbw.pitching.includes("LINE") || lbw.pitching.includes("OFF") ? "val-green" : "val-red";
        
        iEl.innerText = lbw.impact;
        iEl.className = lbw.impact.includes("LINE") ? "val-green" : "val-red";
        
        wEl.innerText = lbw.wickets;
        wEl.className = lbw.wickets === "HITTING" ? "val-green" : (lbw.wickets === "MISSING" ? "val-red" : "val-orange");
        
        document.getElementById('lbw-exp').innerText = lbw.explanation;
        
    } catch(err) {
        console.error(err);
        document.getElementById('btn-predict').innerText = "API Error! Is it running?";
    }
});

// 3. Feedback Loop
document.getElementById('btn-wrong').addEventListener('click', async () => {
    if (!currentContext || !lastPrediction) return;
    
    const feedbackData = {
        bowler: currentContext.bowler,
        batter: currentContext.batter,
        predicted_decision: lastPrediction.wide.decision,
        actual_decision: lastPrediction.wide.decision === "WIDE" ? "LEGAL" : "WIDE",
        margin_of_error: "User flagged prediction error"
    };
    
    try {
        await fetch('http://localhost:8000/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(feedbackData)
        });
        
        document.getElementById('btn-wrong').style.display = 'none';
        document.getElementById('feedback-msg').style.display = 'block';
    } catch(err) {
        console.error(err);
    }
});
