document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predictionForm");
    const button = document.getElementById("predictButton");
    const errorBox = document.getElementById("error");
    const resultBox = document.getElementById("result");

    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        errorBox.textContent = "";
        resultBox.hidden = true;
        button.disabled = true;
        button.textContent = "Calculating...";

        const payload = {
            longitude: document.getElementById("longitude").value,
            latitude: document.getElementById("latitude").value,
            age: document.getElementById("age").value,
            rooms: document.getElementById("rooms").value,
            bedrooms: document.getElementById("bedrooms").value,
            population: document.getElementById("population").value,
            households: document.getElementById("households").value,
            income: document.getElementById("income").value,
            proximity: document.getElementById("proximity").value
        };

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || "Prediction failed.");
            }

            document.getElementById("price").textContent =
                "₹" + Number(result.price).toLocaleString("en-IN", {
                    maximumFractionDigits: 2
                });

            resultBox.hidden = false;
            resultBox.scrollIntoView({behavior: "smooth", block: "nearest"});

        } catch (error) {
            errorBox.textContent = error.message;
        } finally {
            button.disabled = false;
            button.textContent = "Predict estimated value →";
        }
    });

    const data = window.houseData || {};

    if (typeof Chart === "undefined") {
        document.querySelectorAll(".chart-box").forEach(box => {
            box.innerHTML = "<p class='muted'>Charts ke liye internet connection chahiye.</p>";
        });
        return;
    }

    const chartText = "#d8e0f3";
    const gridColor = "#293650";

    const baseOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {labels: {color: chartText}}
        },
        scales: {
            x: {
                ticks: {color: chartText},
                grid: {color: gridColor}
            },
            y: {
                ticks: {color: chartText},
                grid: {color: gridColor}
            }
        }
    };

    new Chart(document.getElementById("categoryChart"), {
        type: "bar",
        data: {
            labels: data.categories || [],
            datasets: [{
                label: "Average target value",
                data: data.categoryPrices || [],
                backgroundColor: "#9d88ff",
                borderRadius: 7
            }]
        },
        options: baseOptions
    });

    new Chart(document.getElementById("actualChart"), {
        type: "scatter",
        data: {
            datasets: [{
                label: "Predicted vs actual",
                data: (data.actual || []).map((value, index) => ({
                    x: value,
                    y: (data.predicted || [])[index]
                })),
                backgroundColor: "#57e5d1",
                pointRadius: 4
            }]
        },
        options: {
            ...baseOptions,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: "Actual target value",
                        color: chartText
                    },
                    ticks: {color: chartText},
                    grid: {color: gridColor}
                },
                y: {
                    title: {
                        display: true,
                        text: "Predicted target value",
                        color: chartText
                    },
                    ticks: {color: chartText},
                    grid: {color: gridColor}
                }
            }
        }
    });
});