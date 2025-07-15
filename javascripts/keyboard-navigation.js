// Enhanced keyboard navigation for MkDocs Material theme
document.addEventListener('DOMContentLoaded', function() {
    // Ensure smooth scrolling is enabled
    document.documentElement.style.scrollBehavior = 'smooth';
    
    // Add custom keyboard shortcuts
    keyboard$.subscribe(function(key) {
        if (key.mode === "global") {
            switch(key.type) {
                // Vim-style navigation
                case "j": // Scroll down
                    window.scrollBy({ top: 100, behavior: 'smooth' });
                    key.claim();
                    break;
                case "k": // Scroll up
                    window.scrollBy({ top: -100, behavior: 'smooth' });
                    key.claim();
                    break;
                case "g": // Go to top
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                    key.claim();
                    break;
                case "shift+g": // Go to bottom
                    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
                    key.claim();
                    break;
                case "d": // Scroll down half page
                    window.scrollBy({ top: window.innerHeight / 2, behavior: 'smooth' });
                    key.claim();
                    break;
                case "u": // Scroll up half page
                    window.scrollBy({ top: -window.innerHeight / 2, behavior: 'smooth' });
                    key.claim();
                    break;
            }
        }
    });
    
    // Ensure arrow keys work for scrolling
    document.addEventListener('keydown', function(event) {
        // Only process if not in an input field
        if (event.target.tagName !== 'INPUT' && 
            event.target.tagName !== 'TEXTAREA' && 
            !event.target.isContentEditable) {
            
            switch(event.key) {
                case 'ArrowDown':
                    event.preventDefault();
                    window.scrollBy({ top: 50, behavior: 'smooth' });
                    break;
                case 'ArrowUp':
                    event.preventDefault();
                    window.scrollBy({ top: -50, behavior: 'smooth' });
                    break;
                case 'PageDown':
                    event.preventDefault();
                    window.scrollBy({ top: window.innerHeight * 0.9, behavior: 'smooth' });
                    break;
                case 'PageUp':
                    event.preventDefault();
                    window.scrollBy({ top: -window.innerHeight * 0.9, behavior: 'smooth' });
                    break;
                case 'Home':
                    event.preventDefault();
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                    break;
                case 'End':
                    event.preventDefault();
                    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
                    break;
            }
        }
    });
    
    // Add visual feedback for keyboard navigation
    let scrollIndicator = document.createElement('div');
    scrollIndicator.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 10px;
        background: rgba(0, 0, 0, 0.7);
        color: white;
        border-radius: 5px;
        font-size: 12px;
        display: none;
        z-index: 9999;
    `;
    document.body.appendChild(scrollIndicator);
    
    // Show keyboard shortcuts help
    document.addEventListener('keydown', function(event) {
        if (event.key === '?' && !event.target.matches('input, textarea')) {
            scrollIndicator.innerHTML = `
                <strong>Keyboard Shortcuts:</strong><br>
                ↑/↓ or j/k - Scroll up/down<br>
                PgUp/PgDn or u/d - Scroll by half page<br>
                Home/End or g/G - Go to top/bottom<br>
                ←/→ or p/n - Previous/Next page<br>
                / or s - Search<br>
                ? - Show this help
            `;
            scrollIndicator.style.display = 'block';
            setTimeout(() => scrollIndicator.style.display = 'none', 5000);
        }
    });
}); 