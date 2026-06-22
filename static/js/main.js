// // navbar function 
// $(document).ready(function(){
//     $('.fa-bars').click(function(){
//         $(this).toggleClass('fa-times');
//         $('.navbar').toggleClass('nav-toggle');
//     });

//     $(Window).on('scroll load',function(){
//         $('fa-bars').removeClass('fa-times');
//         $('.navbar').removeClass('nav-toggle');

//         if($(Window).scrollTop() > 30){
//             $('header').addClass('header-active');
//         }
//         else{
//             $('header').removeClass('header-active');
//         }
//     });
//     // 1. Function to save language choice to Flask Session
//     function updateLanguages() {
//         var input_lang = $("#input_lang").val();
//         var output_lang = $("#output_lang").val();

//         $.ajax({
//             url: "/set_languages",
//             type: "POST",
//             data: {
//                 input_lang: input_lang,
//                 output_lang: output_lang
//             },
//             // success: function(response) {
//             // // If output is Urdu, make the chat box RTL
//             //     if (output_lang === 'ur') {
//             //         $("body").addClass("urdu-font");
//             //     } else {
//             //         $("body").removeClass("urdu-font");
//             //     }
//             success: function(response) {
//             // Check if selected output is a regional RTL language
//             const rtlLanguages = ['ur', 'pa', 'skr'];
            
//             if (rtlLanguages.includes(output_lang)) {
//                 // Add class for Right-to-Left alignment and Urdu/Arabic fonts
//                 $("body").addClass("rtl-mode");
//             } else {
//                 $("body").removeClass("rtl-mode");
//             }
//            }
//         });
//     }

// // 2. Trigger update when selection changes
//     $("#input_lang, #output_lang").on("change", function() {
//         updateLanguages();
//     });

// // 3. Update your existing Send Message AJAX call to include languages
//     $("#messageFormeight").on("submit", function(event) {
//         var rawText = $("#text").val();
//         var inputLang = $("#input_lang").val();
//         var outputLang = $("#output_lang").val();

//         $.ajax({
//             data: {
//                 msg: rawText,
//                 input_lang: inputLang,   // Add this
//                 output_lang: outputLang  // Add this
//             },
//             type: "POST",
//             url: "/get",
//         }).done(function(data) {
//         // ... your existing code to append botHtml ...
//         });
//         event.preventDefault();
//     });
// });

// navbar function 
$(document).ready(function(){
    $('.fa-bars').click(function(){
        $(this).toggleClass('fa-times');
        $('.navbar').toggleClass('nav-toggle');
    });

    $(Window).on('scroll load',function(){
        $('fa-bars').removeClass('fa-times');
        $('.navbar').removeClass('nav-toggle');

        if($(Window).scrollTop() > 30){
            $('header').addClass('header-active');
        }
        else{
            $('header').removeClass('header-active');
        }
    });

    // 1. Function to save language choice to Flask Session
    function updateLanguages() {
        var input_lang = $("#input_lang").val();
        var output_lang = $("#output_lang").val();

        $.ajax({
            url: "/set_languages",
            type: "POST",
            data: {
                input_lang: input_lang,
                output_lang: output_lang
            },
            success: function(response) {
                // Check if selected output is a regional RTL language
                const rtlLanguages = ['ur', 'pa', 'skr'];
                
                if (rtlLanguages.includes(output_lang)) {
                    // Add class for Right-to-Left alignment and Urdu/Arabic fonts
                    $("body").addClass("rtl-mode");
                } else {
                    $("body").removeClass("rtl-mode");
                }
            }
        });
    }

    // 2. Trigger update when selection changes
    $("#input_lang, #output_lang").on("change", function() {
        updateLanguages();
    });

    // 3. Update your existing Send Message AJAX call to include languages
    $("#messageFormeight").on("submit", function(event) {
        var rawText = $("#text").val();
        var inputLang = $("#input_lang").val();
        var outputLang = $("#output_lang").val();

        $.ajax({
            data: {
                msg: rawText,
                input_lang: inputLang,
                output_lang: outputLang
            },
            type: "POST",
            url: "/get",
        }).done(function(data) {
            // your existing code to append botHtml
        });
        event.preventDefault();
    });

});


// ─────────────────────────────────────────────────────────────────
// SPEECH TO TEXT
// Sits outside $(document).ready so it initializes independently
// ─────────────────────────────────────────────────────────────────

const SpeechRecognition = window.SpeechRecognition
                        || window.webkitSpeechRecognition;

if (!SpeechRecognition) {
    // Browser does not support speech — hide mic button silently
    document.addEventListener('DOMContentLoaded', function () {
        var btn = document.getElementById('micBtn');
        if (btn) btn.style.display = 'none';
    });

} else {

    const recognition = new SpeechRecognition();
    recognition.continuous     = false;   // Stop after one sentence
    recognition.interimResults = false;   // Only return final result

    // Map your existing language dropdown values to speech API codes
    const speechLangMap = {
        'en':  'en-US',
        'ur':  'ur-PK',
        'pa':  'pa-PK',
        'skr': 'ur-PK'    // Saraiki — closest supported code
    };

    document.addEventListener('DOMContentLoaded', function () {

        var micBtn = document.getElementById('micBtn');
        if (!micBtn) return;   // Safety check

        // ── When mic button is clicked ──────────────────────────────
        micBtn.addEventListener('click', function () {

            // Read whichever input language user has selected
            var selectedLang = document.getElementById('input_lang')
                               ? document.getElementById('input_lang').value
                               : 'en';

            recognition.lang = speechLangMap[selectedLang] || 'en-US';

            recognition.start();

            // Visual feedback — button pulses red while recording
            micBtn.classList.add('recording');
            micBtn.innerHTML = '<i class="fas fa-microphone-slash"></i>';
            micBtn.disabled  = true;
        });

        // ── When speech is detected and converted to text ───────────
        recognition.onresult = function (event) {
            var transcript = event.results[0][0].transcript;

            // Put the recognised text directly into the chat input box
            // User can then edit it or just hit Send
            document.getElementById('text').value = transcript;

            resetMicBtn();
        };

        // ── Handle any errors (mic blocked, no speech, etc.) ────────
        recognition.onerror = function (event) {
            console.error('Speech recognition error:', event.error);

            // Show a small alert only for common user-facing errors
            if (event.error === 'not-allowed') {
                alert('Microphone access was blocked. Please allow microphone permission in your browser settings.');
            } else if (event.error === 'no-speech') {
                alert('No speech was detected. Please try again.');
            }

            resetMicBtn();
        };

        // ── When recognition ends for any reason ────────────────────
        recognition.onend = function () {
            resetMicBtn();
        };

    }); // end DOMContentLoaded

    // ── Reset mic button back to normal state ───────────────────────
    function resetMicBtn() {
        var btn = document.getElementById('micBtn');
        if (btn) {
            btn.classList.remove('recording');
            btn.innerHTML = '<i class="fas fa-microphone"></i>';
            btn.disabled  = false;
        }
    }

} // end SpeechRecognition check