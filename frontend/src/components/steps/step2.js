import React, { useState, } from 'react';
import Settings from '../settings/settings';

const Step2 = ({ onBack, onNext, messageUnitTest, setMessageUnitTest, loading, startEslint, eslintResult, RegenerateUnitTestBasedOnEslint }) => {

    const [showToast, setShowToast] = useState(false); // State to control toast visibility
    const [showToastContent, setShowToastContent] = useState(''); // State to control toast visibility

    const [toastMessage, setToastMessage] = useState(''); // Toast message state

    const generateLineNumbers = (text) => {
        const lines = text.split('\n');
        return Array.from({ length: lines.length }, (_, i) => i + 1);
    };

    const handleCopy = (code, title) => {
        navigator.clipboard.writeText(code).then(() => {
            setToastMessage('Code copied to clipboard!');
            setShowToast(true);
            setShowToastContent(title)
            setTimeout(() => setShowToast(false), 3000); // Hide toast after 3 seconds
        }).catch((err) => {
            console.error('Failed to copy: ', err);
        });
    };

    const handleScroll = (e) => {
        const lineNumberDiv = document.querySelector('#unitTest .form-control.bg-dark');
        lineNumberDiv.scrollTop = e.target.scrollTop;
    };

      // this function will extract the status of each results variable
    function extractStatus(resultString) {
        const lines = resultString.split("\n");
        const statusLine = lines.find(line => line.startsWith("status:"));
        return statusLine ? statusLine.split(":")[1].trim() : null;
    }


    return (
        <div className="third-step">
            <Settings />
            <p className="first_paragraph">Transform PDF Contracts into Smart Contracts</p>

            <div className="container_step">
                <nav className="stepped-process" aria-label="Progress navigation">
                    <ol className="stepped-process-list">
                        <li className="stepped-process-item">
                            <span className="stepped-process-link">Extract information</span>
                        </li>
                        <li className="stepped-process-item active">
                            <span className="stepped-process-link">Generate unit test</span>
                        </li>
                        <li className="stepped-process-item">
                            <span className="stepped-process-link">Generate and Analyse the smart contract</span>
                        </li>
                    </ol>
                </nav>

                <div id="unitTest">
                    <label htmlFor="unitTestInfo" className="form-label">
                        Generated Unit Test
                    </label>
                    <div className="d-flex" style={{ position: 'relative', width: '100%' }}>
                    <div
                        style={{
                            position: 'absolute',
                            top: '0',
                            right: '1.5vw', // Adjust this for global positioning
                            display: 'flex',
                            gap: '0.5rem', // Adds consistent spacing between buttons
                            marginTop: '1.5vh',
                        }}
                    >
                        {(eslintResult && extractStatus(eslintResult) === 'failed') && (
                            <button
                            onClick={RegenerateUnitTestBasedOnEslint}
                                type="button"
                                className="btn btn-success"
                                style={{
                                    '--bs-btn-padding-y': '.25rem',
                                    '--bs-btn-padding-x': '.5rem',
                                    '--bs-btn-font-size': '.75rem',
                                }}
                            >
                                Regenerate
                            </button>
                        )}

                        <button
                            onClick={() => handleCopy(messageUnitTest, 'Unit Test Code Copied')}
                            type="button"
                            className="btn btn-primary"
                            style={{
                                '--bs-btn-padding-y': '.25rem',
                                '--bs-btn-padding-x': '.5rem',
                                '--bs-btn-font-size': '.75rem',
                            }}
                        >
                            Copy
                        </button>
                    </div>


                        <div className="d-flex" style={{ width: '100%' }}>
                            {/* Line numbers column */}
                            <div
                                className="form-control bg-dark text-white "
                                style={{
                                    width: '50px',
                                    userSelect: 'none',
                                    height: '296px',
                                    flexDirection: 'column',
                                    overflowY: 'hidden',
                                }}
                            >
                                {generateLineNumbers(messageUnitTest).map((lineNumber) => (
                                    <div key={lineNumber} style={{ flex: '0 0 auto' }}>{lineNumber}</div>
                                ))}
                            </div>

                            {/* Code area column */}
                            <textarea
                                id="unitTestInfo"
                                name="unitTestInfo"
                                className="form-control bg-dark text-white "
                                value={messageUnitTest}
                                onChange={(e) => setMessageUnitTest(e.target.value)}
                                onScroll={handleScroll}
                                style={{
                                    fontSize: '16px',
                                    marginLeft: '0px',
                                    width: 'calc(100% - 40px)', // Account for line number width
                                    height: '300px',
                                    paddingLeft: '10px',
                                }}
                                placeholder="Generated unit test will be displayed here..."
                                rows="10"
                                required
                                disabled={loading}
                            />
                        </div>
                    </div>
                </div>

                <div style={{ width: '105%', margin: 0 }} className="card text-center custom-card mt-3">
                    <div className="card-header text-body bg-body">
                        <ul className="nav nav-pills card-header-pills">
                            <li className="nav-item">
                                ESlint Result

                                <button
                                    onClick={startEslint}
                                    type="button"
                                    className="btn btn-primary"
                                    style={{
                                        marginLeft: '1.5vw',
                                        '--bs-btn-padding-y': '.25rem',
                                        '--bs-btn-padding-x': '.5rem',
                                        '--bs-btn-font-size': '.75rem',
                                    }}
                                    disabled={loading}
                                >
                                    {loading ? 'Running...' : 'Start ESLint Test'}
                                </button>
                            </li>


                        </ul>
                    </div>
                    <div className="card-body">
                        <div
                            style={{
                                backgroundColor: '#1e1e1e',
                                color: '#d4d4d4',
                                fontFamily: 'monospace',
                                padding: '1rem',
                                borderRadius: '5px',
                                overflowY: 'auto',
                                maxHeight: '300px',
                                textAlign: 'left',
                            }}
                        >
                            <pre>{eslintResult}</pre>
                        </div>
                    </div>
                </div>

            </div>

            {loading && (
                <div style={{ marginBottom: "2vh" }} className="loadineslintResultg-indicator">
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
            {/* Custom Toast Notification */}
            {showToast && (
                <div className="toast-container position-fixed bottom-0 end-0 ">
                    <div className="toast show" role="alert" aria-live="assertive" aria-atomic="true">
                        <div className="toast-header">
                            <strong className="me-auto">{showToastContent}</strong>
                            <small>Just Now</small>
                            <button type="button" className="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
                        </div>
                        <div className="toast-body">
                            {toastMessage}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Step2;
