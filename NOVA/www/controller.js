/**
 * @file controller.js
 * @brief Handles frontend interaction logic for the driver assistant UI.
 *        Exposes functions to be called from Python using Eel.
 *        Controls message display, assistant visual effects, and location sharing.
 */

$(document).ready(function () {

    // === Expose Python-callable function to display assistant messages ===
    eel.expose(DisplayMessage);
    function DisplayMessage(message) {
        $(".siri-message").text(message);                     // Set message content
        $('.siri-message').textillate('start');              // Trigger animation
    }

    // === Show SiriWave and hide static oval (Assistant is listening) ===
    eel.expose(ShowHood);
    function ShowHood() {
        $("#Oval").attr("hidden", true);
        $("#SiriWave").attr("hidden", false);
    }

    // === Hide SiriWave and show static oval (Assistant is idle) ===
    eel.expose(ExitHood);
    function ExitHood() {
        $("#Oval").attr("hidden", false);
        $("#SiriWave").attr("hidden", true);
    }

    // === Get user geolocation and send it to Python ===
    navigator.geolocation.getCurrentPosition(function (position) {
        const lat = position.coords.latitude;
        const lon = position.coords.longitude;
        eel.ReceiveLocation(lat, lon);  // Send coordinates back to Python
    });

});
