/* BunkerVision — shell UI interactions (dependency-free vanilla JS).
   Handles the custom port-selector dropdown in the top app-bar. */
(function () {
  "use strict";

  function initPortSelect() {
    var root = document.querySelector("[data-port-select]");
    if (!root) return;

    var btn = root.querySelector("[data-port-toggle]");
    var menu = root.querySelector("[data-port-menu]");
    if (!btn || !menu) return;

    function open() {
      root.classList.add("open");
      btn.setAttribute("aria-expanded", "true");
    }
    function close() {
      root.classList.remove("open");
      btn.setAttribute("aria-expanded", "false");
    }
    function toggle(e) {
      e.preventDefault();
      e.stopPropagation();
      root.classList.contains("open") ? close() : open();
    }

    btn.addEventListener("click", toggle);

    // Close on outside click
    document.addEventListener("click", function (e) {
      if (root.classList.contains("open") && !root.contains(e.target)) close();
    });

    // Close on Escape
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && root.classList.contains("open")) {
        close();
        btn.focus();
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initPortSelect);
  } else {
    initPortSelect();
  }
})();
