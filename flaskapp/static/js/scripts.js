// Default login
function defaultLogin() {
    const email = document.querySelector('input[name="email"]');
    const password = document.querySelector('input[name="password"]');

    email.value = "c2729751@gmail.com";
    password.value = "password123";
}

// Log out user
function confirmLogout() {
    // Show a confirmation dialog
    const confirmed = confirm("Are you sure you want to log out?");
    if (confirmed) {
        // If the user confirms, log out by redirecting to the logout route
        window.location.href = "/logout";
    }
}

// Change user details
function changeUserDetails() {
    // Function to handle the change form submission
    document.getElementById("changeForm").addEventListener("submit", function(event) {
        event.preventDefault();  // Prevent the form from reloading the page

        // Get the values from the form
        const newUsername = document.getElementById("newUsername").value;
        const newPassword = document.getElementById("newPassword").value;
        const confirmPassword = document.getElementById("confirmPassword").value;

        // Check if the passwords match
        if (newPassword !== confirmPassword) {
            alert("Passwords do not match.");
            return;
        }

        // Send a POST request to change the username and password
        fetch("/changeUserDetails", {
            method: "POST",
            headers: {
                "Content-Type": "application/x-www-form-urlencoded"
            },
            body: `newUsername=${encodeURIComponent(newUsername)}&newPassword=${encodeURIComponent(newPassword)}&confirmPassword=${encodeURIComponent(confirmPassword)}`
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);  // Show error message if any
            } else {
                alert(data.message);  // Show success message
            }
        })
        .catch(error => {
            console.error("Error:", error);
            alert("An error occurred. Please try again.");
        });
    });
}

// Function to handle the form submission
function submitFile() {
    const fileInput = document.getElementById("formFile");
    const imageResult = document.getElementById("imageResult");
    const videoResult = document.getElementById("videoResult")
    const noSrc = document.getElementById("noSrc");

    // Check if an image or video was uploaded
    if (!fileInput.files[0]) {
        alert("Please upload an image or video.");
        return;
    }

    // Check if the uploaded file is an image or video
    if (!fileInput.files[0].type.startsWith("image/") && !fileInput.files[0].type.startsWith("video/")) {
        alert("Please upload a valid image or video file.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    // Disable the submit button
    const submitBtn = document.getElementById("submitBtn");
    submitBtn.classList.add("disabled");

    // Disable the clear button
    const clearBtn = document.getElementById("clearBtn");
    clearBtn.classList.add("disabled");

    // Hide 'no media' text
    noSrc.style.display = 'none';

    // Scroll to the result section
    document.querySelector('#result').scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });

    // Start progress tracking only if it's a video
    if (fileInput.files[0].type.startsWith("video/")) {
        // Hide the result image/video and reset its source
        const imageResult = document.getElementById("imageResult");
        const videoResult = document.getElementById("videoResult")    
        imageResult.src = "";
        imageResult.style.display = 'none';
        imageResult.style.border = 'none';
        videoResult.src = "";
        videoResult.style.display = 'none';
        videoResult.style.border = 'none';

        initializeProgressTracking();
    }

    // Send the image to the server for inference
    fetch("/runDetection", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.file_type==="image") {
            // Show the image
            const imageUrl = data.output_url;
            imageResult.src = imageUrl;
            imageResult.style.display = 'inline';
            imageResult.style.border = '0.25rem solid #64a19d';

            videoResult.src = "";
            videoResult.style.display = 'none';
            videoResult.style.border = 'none';
        } else if (data.file_type==="video") {
            // Show the video
            const videoUrl = data.output_url;
            videoResult.src = videoUrl;
            videoResult.style.display = 'inline';
            videoResult.style.border = '0.25rem solid #64a19d';

            imageResult.src = "";
            imageResult.style.display = 'none';
            imageResult.style.border = 'none';
        }

        // Fetch image history again after submitting a new image
        fetchDetectionHistory();

        // Re enable submit and clear buttons
        submitBtn.classList.remove("disabled");
        clearBtn.classList.remove("disabled");
    })
    .catch(error => {
        alert("Error: " + error);
    });
}

// Function to clear the file input and the displayed file
function clearFile() {
    const fileInput = document.getElementById("formFile");
    const imageResult = document.getElementById("imageResult");
    const videoResult = document.getElementById("videoResult")
    const buttonGroup = document.getElementById('buttonGroup');
    const noSrc = document.getElementById("noSrc");

    // Clear the file input
    fileInput.value = "";

    // Hide the result image/video and reset its source
    imageResult.src = "";
    imageResult.style.display = 'none';
    imageResult.style.border = 'none';
    videoResult.src = "";
    videoResult.style.display = 'none';
    videoResult.style.border = 'none';

    // Show 'no file' text
    noSrc.style.display = 'inline';

    // Check if a file is selected
    if (fileInput.files.length > 0) {
        // Hide the button group if a file is selected
        buttonGroup.style.display = 'inline-flex';
    } else {
        // Show the button group if no file is selected
        buttonGroup.style.display = 'none';
    }
}

// Function to check if a file is selected
function checkFileInput() {
    const fileInput = document.getElementById('formFile');
    const buttonGroup = document.getElementById('buttonGroup');
    const submitBtn = document.getElementById("submitBtn");
    
    // Check if a file is selected
    if (fileInput.files.length > 0) {
        // Show the button group if a file is selected
        buttonGroup.style.display = 'inline-flex';

        // Remove 'disabled' class from the submit button
        submitBtn.classList.remove("disabled");
    } else {
        // Hide the button group if no file is selected
        buttonGroup.style.display = 'none';
    }
}

// Fetch image/video history from the server
function fetchDetectionHistory() {
    fetch("/detectionHistory")
        .then(response => response.json())
        .then(data => {
            const deleteBtn = document.getElementById("deleteHistBtn");
            const historyDiv = document.getElementById("imageHistory");
            historyDiv.innerHTML = ""; // Clear existing content     

            // Check if there are any files in the history
            if (data.files.length === 0) {
                // Display "No History" message if no files are found
                const noHistoryMessage = document.createElement("p");
                noHistoryMessage.classList.add("text-center", "text-muted");
                noHistoryMessage.textContent = "No History available.";
                historyDiv.appendChild(noHistoryMessage);

                // Hide delete history button if no files in the history
                deleteBtn.style.display = 'none';

                return; // Stop further processing if no files
            } else {
                // Show delete history button if there are files in the history
                deleteBtn.style.display = 'inline-block';
            }

            // Loop through each file (image or video) and create a row for it
            data.files.forEach(file => {
                const fileURL = file.file_url.slice(8); // Adjust this if needed

                // Create a new row
                const rowDiv = document.createElement("div");
                rowDiv.classList.add("row", "gx-0", "my-4", "justify-content-center", "history-row");

                // Left column with project description
                const colLeftDiv = document.createElement("div");
                colLeftDiv.classList.add("col-lg-4");

                const projectDiv = document.createElement("div");
                projectDiv.classList.add("text-center", "history-label");

                const flexDiv = document.createElement("div");
                flexDiv.classList.add("d-flex", "h-100");

                const textDiv = document.createElement("div");
                textDiv.classList.add("project-text", "w-100", "py-auto", "text-center", "text-lg-left");

                const pElement = document.createElement("p");
                pElement.classList.add("mb-0", "text-black-50");
                pElement.textContent = fileURL;

                textDiv.appendChild(pElement);
                flexDiv.appendChild(textDiv);
                projectDiv.appendChild(flexDiv);
                colLeftDiv.appendChild(projectDiv);

                // Right column with the appropriate media (image or video)
                const colRightDiv = document.createElement("div");
                colRightDiv.classList.add("col-lg-8");

                if (file.file_type === "image") {
                    // For images, create an img element
                    const imgElement = document.createElement("img");
                    imgElement.classList.add("img-fluid", "mx-auto", "d-flex", "align-items-center", "justify-content-center");
                    imgElement.src = file.file_url;
                    imgElement.alt = `Image taken at ${file.timestamp}`;
                    colRightDiv.appendChild(imgElement);
                } else if (file.file_type === "video") {
                    // For videos, create a video element
                    const videoElement = document.createElement("video");
                    videoElement.classList.add("img-fluid", "mx-auto", "d-flex", "align-items-center", "justify-content-center");
                    videoElement.src = file.file_url;
                    videoElement.controls = true; // Enable video controls
                    videoElement.alt = `Video taken at ${file.timestamp}`;
                    colRightDiv.appendChild(videoElement);
                }

                // Append columns to the row
                rowDiv.appendChild(colLeftDiv);
                rowDiv.appendChild(colRightDiv);

                // Finally, append the row to the history div
                historyDiv.appendChild(rowDiv);
            });
        })
        .catch(error => {
            console.error("Error fetching detection history:", error);
        });
}

// Delete detection history from the server
function deleteDetectionHistory() {
    // Show a confirmation dialog to the user before deleting
    const confirmed = confirm("Are you sure you want to delete all detection history?");
    
    if (confirmed) {
        // Send a POST request to the delete route using the Fetch API
        fetch("/deleteDetectionHistory", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",  // Specify the content type
            },
            credentials: "same-origin",  // To send cookies with the request (including CSRF token)
        })
        .then(response => response.json())
        .then(data => {
            if (data.message) {
                alert(data.message);  // Show success message
                location.reload();  // Optionally, reload the page to reflect the changes
            } else if (data.error) {
                alert(data.error);  // Show error message
            }
        })
        .catch(error => {
            alert("An error occurred: " + error);  // Handle any errors
        });
    }
}

// Function to initialize the socket connection and handle progress events
function initializeProgressTracking() {
    const progressTracking = document.getElementById("progressTracking");
    progressTracking.style.display = "inline";

    const socket = io.connect("http://localhost:8000");

    socket.on("progress", (data) => {
        const progressBar = document.getElementById("progressBar").children[0];
        
        const progress = data.progress;
        
        // Update the progress bar width
        progressBar.style.width = progress + "%";
        
        // Update the progress bar text
        progressBar.textContent = Math.round(progress) + "%";

        // Hide the spinner when progress is complete
        if (progress === 100) {
            setTimeout(() => {
                progressTracking.style.display = "none";
            }, 1000);
        }
    });
}

// Fetch detection history when the page loads
window.onload = function() {
    // Check if the current page is index.html (or the home page)
    if (window.location.pathname === "/index") {
        fetchDetectionHistory();
    }
};