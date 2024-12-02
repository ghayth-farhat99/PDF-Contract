import React from 'react';
import Settings from '../settings/settings';

const Step1 = ({ onNext, onBack, message, setMessage, loading }) => {
    return (
        <div className="second-step">
            <Settings />
            <p className="first_paragraph">Transform PDF Contracts into Smart Contracts</p>

            <div className="container_step">
                <nav className="stepped-process" aria-label="Progress navigation">
                    <ol className="stepped-process-list">
                        <li className="stepped-process-item active">
                            <span className="stepped-process-link">Extract information</span>
                        </li>
                        <li className="stepped-process-item">
                            <span className="stepped-process-link">Generate unit test</span>
                        </li>
                        <li className="stepped-process-item">
                            <span className="stepped-process-link">Generate and Analyse the smart contract</span>
                        </li>
                    </ol>
                </nav>

                <div className="textarea-container">
                    <label htmlFor="contractInfo" className="form-label">
                        Extracted Information
                    </label>
                    <textarea
                        id="contractInfo"
                        name="contractInfo"
                        rows="10"
                        className="form-control"
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        placeholder="Extracted information will be displayed here..."
                        required
                    />
                </div>
            </div>

            {loading && (
                <div style={{ marginBottom: "2vh" }} className="loading-indicator">
                    <div className="spinner-border text-primary" role="status">
                        <span className="visually-hidden">Loading...</span>
                    </div>
                </div>
            )}

            <div className="button-group">
                <button type="button" className="btn btn-outline-secondary" onClick={onBack}>
                    Back
                </button>
                <button type="button" className="btn btn-secondary" onClick={onNext} disabled={loading}>
                    {loading ? 'Processing...' : 'Next'}
                </button>
            </div>
        </div>
    );
};

export default Step1;
