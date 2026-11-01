// Opens the second level of the documentation sidebar once Doxygen has built it, so the
// sections of each top-level page are visible without clicking. docs.yml adds this script
// to every generated page.
(function () {
    "use strict";
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
        // Opening sections moves the sidebar; bring the current page back into view.
        setTimeout(function () {
            var selected = document.querySelector("#nav-tree-contents .selected");
            if (selected !== null) {
                selected.scrollIntoView({ block: "center" });
            }
        }, 300);
    }, 100);
})();
