document.addEventListener("DOMContentLoaded", function() {
    const sendButton = document.getElementById("sendButton");
    const recordButton = document.getElementById("recordButton");

    const messageInput = document.getElementById("messageInput");
    const messagesContainer = document.getElementById("messagesContainer");

    // Create a new array to store user messages
    const userMessages = [];

    // Flag to track if the conversation has ended
    let conversationEnded = false;

    // Determine the base URL dynamically
    const baseURL = window.location.origin;

    const moodTrackerButton = document.querySelector('.dashboard-item:nth-child(1)');
    moodTrackerButton.addEventListener('click', function() {
        navigateToMoodTracker(userMessages);
    });

    function navigateToMoodTracker(userMessages) {
        const encodedMessages = encodeURIComponent(JSON.stringify(userMessages));
        window.location.href = `${baseURL}/mood_tracker?messages=${encodedMessages}`;
    }

    // Account dropdown functionality
    const accountItem = document.getElementById('accountItem');
    const accountDropdown = document.getElementById('accountDropdown');

    accountItem.addEventListener('click', function() {
        accountDropdown.style.display = accountDropdown.style.display === 'block' ? 'none' : 'block';
    });

    document.addEventListener('click', function(event) {
        if (!accountItem.contains(event.target) && !accountDropdown.contains(event.target)) {
            accountDropdown.style.display = 'none';
        }
    });

    const deleteButton = document.getElementById("deleteButton");
    
    if (deleteButton) {
        deleteButton.addEventListener('click', deleteAccount);
    }

    function deleteAccount() {
        fetch(`${baseURL}/delete_account`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({})
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                window.location.href = '/register'; // Redirect to the registration page
            } else {
                console.error('Failed to delete account');
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }

    function generateBotMessage(message) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", "bot-message");
    
        const botImage = document.createElement("img");
        const botImageSrc = "/static/images/bot.png";

        console.log("Bot image src:", botImageSrc);
        botImage.src = botImageSrc;
        botImage.alt = "Bot Icon";
        botImage.classList.add("bot-image");
    
        const messageText = document.createElement("div");
        messageText.textContent = message;
    
        messageElement.appendChild(botImage);
        messageElement.appendChild(messageText);
    
        messagesContainer.appendChild(messageElement);
        scrollToBottom();
    }
    
    function generateUserMessage(message) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", "user-message");
    
        const userImage = document.createElement("img");
        const userImageSrc = "/static/images/user.png";

        console.log("User image src:", userImageSrc);
        userImage.src = userImageSrc;
        userImage.alt = "User Icon";
        userImage.classList.add("user-image");
    
        const messageText = document.createElement("div");
        messageText.textContent = message;
    
        messageElement.appendChild(userImage);
        messageElement.appendChild(messageText);
    
        messagesContainer.appendChild(messageElement);
        scrollToBottom();
    }
    
    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function handleUserInput() {
        const message = messageInput.value.trim();
        if (message !== "") {
            generateUserMessage(message);
            userMessages.push(message);
    
            if (message.toLowerCase() === "quit") {
                conversationEnded = true;
                console.log("User messages:", userMessages);
                disableMessageInput();
                sendMessageToServer(message);
                showChartContainer(); // Show the chart container
            } else if (!conversationEnded) {
                messageInput.value = "";
                sendMessageToServer(message);
            }
        }
    }

    function showChartContainer() {
        const chartContainer = document.querySelector('.chart-container');
        chartContainer.style.display = 'block';
    
        fetch(`${baseURL}/get_emotion_probabilities`, {
            method: 'POST',
            body: JSON.stringify({ messages: userMessages }),
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            const emotionProbabilities = data.emotion_probabilities;
            console.log("Emotion Probabilities:", emotionProbabilities);
    
            // Create the pie chart
            const ctx = document.getElementById('emotionChart').getContext('2d');
            const emotionChart = new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: Object.keys(emotionProbabilities),
                    datasets: [{
                        data: Object.values(emotionProbabilities),
                        backgroundColor: [
                            '#FF6384',
                            '#36A2EB',
                            '#FFCE56',
                            '#4BC0C0',
                            '#9966FF',
                            '#FF9F40'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }

    function disableMessageInput() {
        messageInput.disabled = true;
        sendButton.disabled = true;
        recordButton.disabled = true;
    }

    function handleVoiceInput() {
        try {
            const recognition = new webkitSpeechRecognition();
            recognition.onresult = function(event) {
                const message = event.results[0][0].transcript;
                generateUserMessage(message);
                sendMessageToServer(message);
            };
            recognition.start();
        } catch (error) {
            console.error("Error handling voice input:", error);
        }
    }

    function sendMessageToServer(message) {
        fetch(`${baseURL}/get_response`, {
            method: 'POST',
            body: JSON.stringify({ user_input: message, userMessages: userMessages }),
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            const responseMessage = data.response;
            generateBotMessage(responseMessage);
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }

    function sendVoiceMessageToServer(message) {
        fetch(`${baseURL}/get_voice_response`, {
            method: 'POST',
            body: message, // Send the voice input directly as the request body
            headers: {
                'Content-Type': 'audio/wav' // Set the appropriate content type for the voice data
            }
        })
        .then(response => response.json())
        .then(data => {
            const responseMessage = data.response;
            generateBotMessage(responseMessage);
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }

    fetch(`${baseURL}/initiate_conversation`, {
        method: 'GET'
    })
    .then(response => response.json())
    .then(data => {
        const responseMessage = data.response;
        generateBotMessage(responseMessage);
    })
    .catch(error => {
        console.error('Error:', error);
    });

    recordButton.addEventListener("click", handleVoiceInput);
    sendButton.addEventListener("click", handleUserInput);

    messageInput.addEventListener("keypress", function(event) {
        if (event.key === "Enter") {
            handleUserInput();
        }
    });
});
