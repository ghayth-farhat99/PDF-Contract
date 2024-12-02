import React, { useState } from 'react';
import './settings.css';
function Settings() {
    // State to control modal visibility
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [temperature, setTemperature] = useState(0.2); // Default temperature value (midpoint)
    const [modelUsed, setModelUsed] = useState("llama3.1");
    // Function to toggle the modal
    const toggleModal = () => {
        setIsModalOpen(!isModalOpen);
    };

    // Handle temperature change (0 to 1 range)
    const handleTemperatureChange = (event) => {
        setTemperature(parseFloat(event.target.value));
    };

    // Function to handle save and send the data to the backend
    const handleSave = async () => {
        try {
            const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/configure_model`,
                {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ temperature }),
                });
            if (response.ok) {
                console.log('Temperature saved successfully');
            } else {
                console.error('Failed to save temperature');
            }
        } catch (error) {
            console.error('Error:', error);
        } finally {
            // Close the modal after saving
            toggleModal();
        }


    };

    return (
        <div>
            <div className="button_container">
                <button
                    type="button"
                    className="btn btn-icon btn-outline-secondary btn-sm"
                    onClick={toggleModal}
                >
                    <svg width="0" height="0" style={{ display: 'none' }}>
                        <symbol fill="currentColor" viewBox="0 0 1000 1000" id="settings">
                            <path d="M827.568 500.5c0-110.61 120.715-26.921 64.516-162.6-56.231-135.681-82.391 8.823-160.606-69.384s66.333-104.365-69.36-160.562c-135.718-56.2-52.024 64.481-162.61 64.481-110.619 0-26.921-120.68-162.612-64.481s8.827 82.349-69.391 160.562-104.372-66.3-160.575 69.384c-56.23 135.679 64.486 51.99 64.486 162.6S50.7 527.419 106.93 663.1c56.2 135.681 82.359-8.823 160.575 69.352 78.218 78.214-66.3 104.4 69.391 160.6s51.993-64.483 162.612-64.483c110.586 0 26.892 120.678 162.61 64.483 135.693-56.2-8.854-82.384 69.36-160.6 78.215-78.175 104.375 66.329 160.606-69.352 56.199-135.681-64.516-51.99-64.516-162.6M500 301.015c110.457 0 200 89.537 200 199.985s-89.543 199.985-200 199.985S300 611.448 300 501s89.543-199.985 200-199.985" style={{ fillRule: 'evenodd' }}></path>
                        </symbol>
                    </svg>

                    <svg width="1.875em" height="1.875em" aria-hidden="true" focusable="false">
                        <use href="#settings" />
                    </svg>
                </button>
            </div>

            {isModalOpen && (
                <>
                    <div className="modal-overlay" onClick={toggleModal}></div>

                    <div className="modal-content">
                        <h3>Settings</h3>
                        <div class="container">
                            <div class="slider-container">
                                <div class="label-top">
                                    <span class="label-left">Temperature</span>
                                    <span class="label-value">{temperature}</span>
                                </div>
                                <input
                                    type="range"
                                    class="form-range"
                                    value={temperature}
                                    onChange={handleTemperatureChange}
                                    min="0.0" max="1.0" step="0.1"
                                    id="temperature-slider"
                                ></input>
                                <div class="label-bottom">
                                    <span class="label-left">Precise</span>
                                    <span class="label-right">Creative</span>
                                </div>
                            </div>

                            {/* <!-- Model Dropdown Section --> */}

                            <div class="model-selection-container">
                                <div class="model-label">
                                    <span>Selected Model: {modelUsed}</span>
                                </div>

                                <div class="btn-group-2">
                                    <button class="btn btn-dropdown btn-lg dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">
                                        {modelUsed}
                                    </button>
                                    <ul class="dropdown-menu">
                                        <li><a class="dropdown-item" href="#" onClick={() => setModelUsed('LLAMA3.1')}>LLAMA3.1</a></li>
                                        <li><a class="dropdown-item" href="#" onClick={() => setModelUsed('GPT-3.5')}>GPT-3.5</a></li>
                                        <li><a class="dropdown-item" href="#" onClick={() => setModelUsed('GPT-4')}>GPT-4</a></li>
                                        <li><a class="dropdown-item" href="#" onClick={() => setModelUsed('BERT')}>BERT</a></li>
                                        <li><a class="dropdown-item" href="#" onClick={() => setModelUsed('T5')}>T5</a></li>
                                    </ul>
                                </div>
                            </div>

                        </div>

                        <div className='button-group'>
                            <button type="button" className="btn btn-outline-secondary btn-cancel" onClick={toggleModal}>
                                Annuler
                            </button>
                            <button type="button" className="btn btn-primary btn-save" onClick={handleSave}>
                                Save
                            </button>
                        </div>
                    </div>
                </>
            )}

        </div>
    );
}

export default Settings;
