
const API_URL = "http://127.0.0.1:8000";

const imageInput = document.getElementById("image-input");
const imagePreview = document.getElementById("image-preview");
const uploadPlaceholder = document.getElementById("upload-placeholder");
const uploadArea = document.getElementById("upload-area");

let selectedImage = null;

// --------------------------------------------------
// TAB NAVIGATION
// --------------------------------------------------

document.querySelectorAll(".tab").forEach(button => {
    button.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach(tab => {
            tab.classList.remove("active");
        });

        document.querySelectorAll(".panel").forEach(panel => {
            panel.classList.remove("active");
        });

        button.classList.add("active");

        document.getElementById(
            `${button.dataset.tab}-panel`
        ).classList.add("active");
    });
});


// --------------------------------------------------
// BACKEND CONNECTION STATUS
// --------------------------------------------------

async function checkBackend() {
    const dot = document.querySelector(".status-dot");
    const status = document.querySelector(".status");

    try {
        const response = await fetch(`${API_URL}/health`);

        if (!response.ok) {
            throw new Error("Backend unavailable");
        }

        dot.classList.add("connected");
        status.lastChild.textContent = " Backend connected";

    } catch (error) {
        dot.classList.remove("connected");
        status.lastChild.textContent = " Backend offline";
    }
}

checkBackend();


// --------------------------------------------------
// IMAGE UPLOAD AND PREVIEW
// --------------------------------------------------

function displayImage(file) {
    if (!file) return;

    const allowedTypes = [
        "image/jpeg",
        "image/png",
        "image/webp"
    ];

    if (!allowedTypes.includes(file.type)) {
        alert("Please upload a JPEG, PNG, or WEBP image.");
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        alert("Image must be smaller than 10 MB.");
        return;
    }

    selectedImage = file;

    const reader = new FileReader();

    reader.onload = event => {
        imagePreview.src = event.target.result;
        imagePreview.hidden = false;
        uploadPlaceholder.hidden = true;
    };

    reader.readAsDataURL(file);
}

imageInput.addEventListener("change", event => {
    displayImage(event.target.files[0]);
});

uploadArea.addEventListener("dragover", event => {
    event.preventDefault();
    uploadArea.classList.add("dragging");
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragging");
});

uploadArea.addEventListener("drop", event => {
    event.preventDefault();
    uploadArea.classList.remove("dragging");

    const file = event.dataTransfer.files[0];

    displayImage(file);
});


// --------------------------------------------------
// RESULT DISPLAY HELPERS
// --------------------------------------------------

function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, char => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
    })[char]);
}

function section(title, content) {
    return `
        <div class="result-section">
            <h4>${escapeHTML(title)}</h4>
            ${content}
        </div>
    `;
}

function renderList(items) {
    if (!Array.isArray(items) || items.length === 0) {
        return "<p>No evidence reported.</p>";
    }

    return `
        <ul>
            ${items.map(item =>
                `<li>${escapeHTML(item)}</li>`
            ).join("")}
        </ul>
    `;
}

function renderProbabilities(probabilities) {
    if (!probabilities) return "";

    return Object.entries(probabilities)
        .map(([label, probability]) => {
            const percentage = Math.max(
                0,
                Math.min(100, Number(probability) * 100)
            );

            return `
                <div class="probability">
                    <span>${escapeHTML(label.replaceAll("_", " "))}</span>
                    <strong>${percentage.toFixed(1)}%</strong>
                </div>
                <div class="bar">
                    <div
                        class="bar-fill"
                        style="width:${percentage}%"
                    ></div>
                </div>
            `;
        }).join("");
}

function renderCaptionResults(data) {
    const lean = data.political_lean;
    const stance = data.stance;

    return `
        <h3>Caption analysis results</h3>

        ${section(
            "Political lean",
            `
                <p class="prediction">
                    ${escapeHTML(lean.prediction.replaceAll("_", " "))}
                </p>
                <p>Experimental score: ${(lean.confidence * 100).toFixed(1)}%</p>
                ${renderProbabilities(lean.probabilities)}
            `
        )}

        ${section(
            "Communicative stance",
            `
                <p class="prediction">
                    ${escapeHTML(stance.prediction.replaceAll("_", " "))}
                </p>
                <p>Experimental score: ${(stance.confidence * 100).toFixed(1)}%</p>
                ${renderProbabilities(stance.probabilities)}
            `
        )}

        ${section(
            "Limitations",
            renderList(data.limitations)
        )}

        <p class="disclaimer">
            These are experimental model outputs, not validated measures
            of political ideology or classification accuracy.
        </p>
    `;
}


// --------------------------------------------------
// CAPTION-ONLY ANALYSIS
// --------------------------------------------------

document.getElementById("caption-form").addEventListener(
    "submit",
    async event => {
        event.preventDefault();

        const caption = document.getElementById(
            "caption-input"
        ).value.trim();

        const button = document.getElementById("caption-submit");
        const results = document.getElementById("caption-results");

        if (!caption) {
            alert("Please enter a caption.");
            return;
        }

        button.disabled = true;
        button.textContent = "Analyzing caption...";
        results.hidden = false;
        results.innerHTML = "<p>Running analysis. Please wait...</p>";

        try {
            const formData = new FormData();
            formData.append("caption", caption);

            const response = await fetch(
                `${API_URL}/analyze-caption`,
                {
                    method: "POST",
                    body: formData
                }
            );

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Caption analysis failed.");
            }

            results.innerHTML = renderCaptionResults(
                data.caption_analysis
            );

        } catch (error) {
            results.innerHTML = `
                <div class="error">
                    ${escapeHTML(error.message)}
                </div>
            `;

        } finally {
            button.disabled = false;
            button.innerHTML = 'Analyze caption <span>→</span>';
        }
    }
);


// --------------------------------------------------
// IMAGE + CAPTION ANALYSIS
// --------------------------------------------------

document.getElementById("image-form").addEventListener(
    "submit",
    async event => {
        event.preventDefault();

        const button = document.getElementById("image-submit");
        const results = document.getElementById("image-results");

        if (!selectedImage) {
            alert("Please upload an image first.");
            return;
        }

        const caption = document.getElementById(
            "image-caption"
        ).value.trim();

        const formData = new FormData();
        formData.append("image", selectedImage);
        formData.append("caption", caption);

        button.disabled = true;
        button.textContent = "Analyzing image...";
        results.hidden = false;
        results.innerHTML = "<p>Analyzing image. This may take a moment...</p>";

        try {
            const response = await fetch(
                `${API_URL}/analyze`,
                {
                    method: "POST",
                    body: formData
                }
            );

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Image analysis failed.");
            }

            const analysis = data.image_analysis;

            results.innerHTML = `
                <h3>Image analysis results</h3>

                ${section(
                    "Party affiliation",
                    `<p class="prediction">${escapeHTML(analysis.party_affiliation)}</p>`
                )}

                ${section(
                    "Ideological orientation",
                    `<p class="prediction">${escapeHTML(analysis.ideological_orientation)}</p>`
                )}

                ${section(
                    "Communicative stance",
                    `<p class="prediction">${escapeHTML(analysis.communicative_stance)}</p>
                     <p><strong>Target:</strong> ${escapeHTML(analysis.stance_target || "Not identified")}</p>`
                )}

                ${section(
                    "Visual evidence",
                    renderList(analysis.visual_evidence)
                )}

                ${section(
                    "Embedded text",
                    renderList(analysis.embedded_text_evidence)
                )}

                ${section(
                    "Caption analysis",
                    `<p>${escapeHTML(analysis.caption_analysis)}</p>`
                )}

                ${section(
                    "Reasoning",
                    `<p>${escapeHTML(analysis.reasoning)}</p>`
                )}

                ${section(
                    "Limitations",
                    renderList(analysis.limitations)
                )}

                <p class="disclaimer">
                    ${escapeHTML(analysis.confidence_note)}
                </p>

                ${
                    data.caption_analysis
                        ? `
                            <h3>Independent caption classifier</h3>
                            ${renderCaptionResults(data.caption_analysis)}
                        `
                        : ""
                }

                ${
                    data.caption_analysis_error
                        ? `
                            <div class="error">
                                Image analysis succeeded, but caption classification
                                failed: ${escapeHTML(data.caption_analysis_error)}
                            </div>
                        `
                        : ""
                }
            `;

        } catch (error) {
            results.innerHTML = `
                <div class="error">
                    ${escapeHTML(error.message)}
                </div>
            `;

        } finally {
            button.disabled = false;
            button.innerHTML = 'Analyze image and caption <span>→</span>';
        }
    }
);
