chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "fetch_espn") {
        const espnUrl = `https://www.espncricinfo.com/series/${request.seriesId}/game/${request.matchId}`;
        fetch(espnUrl)
            .then(res => res.text())
            .then(text => sendResponse({success: true, html: text}))
            .catch(err => sendResponse({success: false, error: err.message}));
        return true; // Keep message channel open for async response
    }
});
