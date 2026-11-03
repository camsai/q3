// Opens the second level of the documentation sidebar once Doxygen has built it, so the
// sections of each top-level page are visible without clicking, and highlights the entry of
// the page being viewed (styled by docs/doxygen-q3.css). docs.yml adds this script to every
// generated page.
(function () {
    "use strict";
    var CURRENT_PAGE_CLASS = "q3-current-page";

    // Doxygen scrolls the sidebar itself once its tree has loaded, after this script has
    // opened it, and can leave the current row out of view. Bring the row back for the
    // first two seconds, until the reader scrolls the sidebar themselves.
    function keepInView(row) {
        var sidebar = document.getElementById("nav-tree");
        if (sidebar === null) {
            return;
        }
        var readerScrolled = false;
        ["wheel", "touchstart", "mousedown", "keydown"].forEach(function (eventName) {
            sidebar.addEventListener(
                eventName,
                function () {
                    readerScrolled = true;
                },
                { once: true, passive: true }
            );
        });
        var checks = 0;
        var keeper = setInterval(function () {
            checks += 1;
            if (readerScrolled || checks > 20) {
                clearInterval(keeper);
                return;
            }
            var rowBox = row.getBoundingClientRect();
            var sidebarBox = sidebar.getBoundingClientRect();
            if (rowBox.top < sidebarBox.top || rowBox.bottom > sidebarBox.bottom) {
                row.scrollIntoView({ block: "center" });
            }
        }, 100);
        row.scrollIntoView({ block: "center" });
    }

    var attempts = 0;
    var timer = setInterval(function () {
        attempts += 1;
        var secondLevelItems = document.querySelectorAll("#nav-tree-contents > ul > li > ul > li");
        if (secondLevelItems.length === 0) {
            if (attempts > 50) {
                clearInterval(timer);
            }
            return;
        }
        clearInterval(timer);
        secondLevelItems.forEach(function (item) {
            var toggle = item.querySelector(":scope > div.item > a");
            var children = item.querySelector(":scope > ul");
            var isOpen = children !== null && children.style.display !== "none";
            if (toggle !== null && toggle.querySelector("span.arrow") !== null && !isOpen) {
                toggle.click();
            }
        });
        // Opening sections moves the sidebar; mark the current page and bring it back into view.
        setTimeout(function () {
            var selected = document.querySelector("#nav-tree-contents .selected");
            if (selected === null) {
                return;
            }
            // The selected row is the page itself or one of its sections; mark the page's row,
            // the nearest one whose link has no section anchor (a tutorial lesson sits a level
            // below the tutorials index).
            var item = selected.closest("#nav-tree-contents li");
            while (item !== null) {
                var row = item.querySelector(":scope > div.item");
                var link = row !== null ? row.querySelector("a[href]:not([href^='javascript'])") : null;
                if (link !== null && link.getAttribute("href").indexOf("#") === -1) {
                    row.classList.add(CURRENT_PAGE_CLASS);
                    break;
                }
                item = item.parentElement.closest("#nav-tree-contents li");
            }
            keepInView(selected);
        }, 300);
    }, 100);
})();
