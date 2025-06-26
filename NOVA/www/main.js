/**
 * @file main.js
 * @brief Handles UI interactions for buttons, SiriWave animation, assistant microphone,
 *        GPS access, monitoring toggles, and dynamic settings windows.
 */

$(document).ready(function () {
    let isMonitoring = false;       // Global flag (reserved for future use)
    let pollingPaused = false;      // Prevent polling during modal interactions

    // === TEXT ANIMATION (Main header text) ===
    $('.text').textillate({
        loop: true,
        sync: true,
        in: { effect: "fadeIn" },
        out: { effect: "fadeOutUp" },
    });

    // === SIRIWAVE VISUALIZER SETUP ===
    var siriWave = new SiriWave({
        container: document.getElementById("siri-container"),
        width: 800,
        height: 200,
        style: "ios9",
        color: "#fff",
        speed: 0.2,
        amplitude: 1,
        autostart: true
    });

    // === ASSISTANT MESSAGE ANIMATION ===
    $('.siri-message').textillate({
        loop: true,
        sync: true,
        in: { effect: "fadeInUp", sync: true },
        out: { effect: "fadeOutUp", sync: true },
    });

    // === MICROPHONE BUTTON ===
    $("#MicBtn").click(function () {
        eel.playClickSound();                        // Play feedback sound
        $("#Oval").attr("hidden", true);             // Hide idle icon
        $("#SiriWave").attr("hidden", false);        // Show active wave
        eel.set_mic_pressed();                       // Notify backend
    });

    // === GPS BUTTON ===
    $("#GpsBtn").click(function () {
        eel.playClickSound();
        $("#Oval").attr("hidden", false);
        $("#SiriWave").attr("hidden", true);
        eel.OpenGps("gps");
    });

    // === SETTINGS TOGGLE ===
    $("#SettingsBtn").click(function () {
        $("#SettingsWindow").fadeToggle();   // Show/hide settings
    });

    $("#CloseSettings").click(function () {
        $("#SettingsWindow").fadeOut();      // Force hide
    });

    // === INSTRUCTIONS TOGGLE ===
    $("#InstructionsBtn").click(function () {
        $("#SettingsWindow").fadeOut();           // Close settings if open
        $("#InstructionsWindow").fadeToggle();    // Show/hide instructions
    });

    $("#CloseInstructionsBtn").click(function () {
        $("#InstructionsWindow").fadeOut();
    });

    // === MONITOR MODE TOGGLE LOGIC ===

    // Update toggle buttons visually based on backend state
    function updateMonitorButtons() {
        eel.get_monitor_mode()(function (state) {
            if (state === "on") {
                $("#MonitorOnBtn").addClass("selected-option");
                $("#MonitorOffBtn").removeClass("selected-option");
            } else {
                $("#MonitorOffBtn").addClass("selected-option");
                $("#MonitorOnBtn").removeClass("selected-option");
            }
        });
    }

    // Monitor On
    $("#MonitorOnBtn").click(function () {
        $("#Oval").attr("hidden", false);
        $("#SiriWave").attr("hidden", true);
        eel.Set_jason_flag();       // Enable monitoring in backend
        updateMonitorButtons();
    });

    // Monitor Off
    $("#MonitorOffBtn").click(function () {
        $("#Oval").attr("hidden", false);
        $("#SiriWave").attr("hidden", true);
        eel.Clear_jason_flag();     // Disable monitoring in backend
        updateMonitorButtons();
    });

    // === INITIALIZATION ===

    // On first load
    updateMonitorButtons();

    // Poll backend every 2 seconds to keep UI in sync
    setInterval(() => {
        if (!pollingPaused) updateMonitorButtons();
    }, 2000);
});
