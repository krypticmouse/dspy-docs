/**
 * "Copy page" button for MkDocs Material pages.
 *
 * Adds a clipboard button next to the existing "Edit this page" action.
 * Clicking the button fetches the raw Markdown source from GitHub and copies
 * it to the user's clipboard.
 */
document.addEventListener("DOMContentLoaded", function () {
  // The raw-content base URL derived from the repo + edit_uri in mkdocs.yml.
  // edit links look like:  .../blob/main/docs/docs/<path>.md
  // raw links look like:   .../raw/main/docs/docs/<path>.md
  var REPO_RAW_BASE =
    "https://raw.githubusercontent.com/stanfordnlp/dspy/main/docs/docs/";

  // Clipboard icon (Material Design "content_copy")
  var ICON =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">' +
    '<path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14' +
    "c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z" +
    '"/></svg>';

  function getMarkdownPath() {
    // Try to derive the .md path from the existing "Edit this page" link
    var editLink = document.querySelector('a[title="Edit this page"]');
    if (editLink) {
      var href = editLink.getAttribute("href") || "";
      // href example: https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/modules/RLM.md
      var marker = "docs/docs/";
      var idx = href.indexOf(marker);
      if (idx !== -1) {
        return href.substring(idx + marker.length);
      }
    }

    // Fallback: build the path from the current URL
    var path = window.location.pathname.replace(/\/$/, "");
    if (path === "" || path === "/") {
      return "index.md";
    }
    // /api/modules/RLM/ -> api/modules/RLM.md  (directory style)
    // /faqs/             -> faqs.md
    // Try stripping leading slash
    path = path.replace(/^\//, "");

    // If it already ends with .md, keep it
    if (path.endsWith(".md")) {
      return path;
    }

    // MkDocs can serve /foo/ from either foo.md or foo/index.md.
    // We try foo.md first, then foo/index.md as a fallback inside copyMarkdown().
    return path + ".md";
  }

  function createButton() {
    // Find the article actions area (where Edit lives)
    var actionsContainer = document.querySelector(".md-content__button");
    if (!actionsContainer) return;

    var wrapper = actionsContainer.parentElement;
    if (!wrapper) return;

    var btn = document.createElement("a");
    btn.className = "md-content__button md-icon copy-md-btn";
    btn.title = "Copy page as Markdown";
    btn.href = "#";
    btn.setAttribute("aria-label", "Copy page as Markdown");
    btn.innerHTML = ICON + " Copy page";

    btn.addEventListener("click", function (e) {
      e.preventDefault();
      copyMarkdown(btn);
    });

    // Insert before the first existing action button
    wrapper.insertBefore(btn, actionsContainer);
  }

  function copyMarkdown(btn) {
    var mdPath = getMarkdownPath();
    var url = REPO_RAW_BASE + mdPath;

    btn.classList.add("copy-md-btn--loading");

    fetch(url)
      .then(function (res) {
        if (!res.ok) {
          // Retry with /index.md suffix (MkDocs directory pages)
          var altPath = mdPath.replace(/\.md$/, "/index.md");
          return fetch(REPO_RAW_BASE + altPath).then(function (res2) {
            if (!res2.ok) {
              throw new Error("Could not fetch markdown source");
            }
            return res2.text();
          });
        }
        return res.text();
      })
      .then(function (text) {
        return navigator.clipboard.writeText(text);
      })
      .then(function () {
        showFeedback(btn, true);
      })
      .catch(function () {
        showFeedback(btn, false);
      });
  }

  function showFeedback(btn, success) {
    btn.classList.remove("copy-md-btn--loading");
    btn.classList.add(success ? "copy-md-btn--success" : "copy-md-btn--error");
    btn.innerHTML = ICON + (success ? " Copied!" : " Failed to copy");

    setTimeout(function () {
      btn.classList.remove("copy-md-btn--success", "copy-md-btn--error");
      btn.innerHTML = ICON + " Copy page";
    }, 2000);
  }

  // MkDocs Material uses instant loading — re-run on every page navigation
  if (typeof document$ !== "undefined") {
    document$.subscribe(function () {
      // Remove any previously injected button (instant navigation re-renders)
      var existing = document.querySelector(".copy-md-btn");
      if (existing) existing.remove();
      createButton();
    });
  } else {
    createButton();
  }
});
